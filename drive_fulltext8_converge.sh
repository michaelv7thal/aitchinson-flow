#!/usr/bin/env bash
# Full-text8 DirichletFM convergence run (d1280/14L, L256).
#
# Sequence (gated after the current pipeline frees the GPU):
#   1. wait for HINGE-SWEEPS COMPLETE  (no GPU contention)
#   2. PROBE batch throughput (16..48) -> pick the windows/s-optimal batch.
#      Memory headroom (8.8/20 GB at b16) says a bigger batch FITS; the probe
#      decides whether it's actually FASTER (only true if compute-underutilised
#      at b16). We optimise for measured throughput, not assumption.
#   3. TRAIN on the FULL afmck/text8 split (--full-split, lazy CLR features) at
#      the probe batch, LR sqrt-scaled (3e-4·sqrt(B/16)) for the fewer updates,
#      to CONVERGENCE via early stopping (restores best-val ckpt) + 96 h cap.
#   4. EVAL: generation (KL/H_ratio/BPC) + recovery sweep, for apples-to-apples
#      comparison against the 50k run.
set -uo pipefail
# torch lives in the buildpack site-packages; scripts self-bootstrap src.
export PYTHONPATH="/layers/paketo-buildpacks_pip-install/packages/lib/python3.11/site-packages:/layers/paketo-buildpacks_pip/pip/lib/python3.11/site-packages:/layers/paketo-buildpacks_cpython/cpython:src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/renku/work/aitchinson-flow

SCALE=a100_20g_L256_d1280L14_full
ARM=DirichletFM
R="runs/sflm_bench_${SCALE}"
LOG="${R}/_driver"; mkdir -p "$LOG"
CKPT="${R}/${ARM}/epoch_final.pt"
HS="runs/ood_det_rerun/_driver/hinge_sweeps.out"            # hinge end marker
B50="runs/sflm_bench_a100_20g_L256_d1280L14_b16/_driver/bigdfm_50ep.out"  # train end
ts(){ date +%Y-%m-%d_%H:%M:%S; }

echo "### FULLTEXT8 START $(ts) — waiting for prior pipeline to free the GPU"
# Proceed when the hinge sweeps complete, OR the hinge driver has exited (chain
# ended any way) with the big training already done. The latter guards against
# the (minor) hinge sweeps failing and blocking this priority run forever.
while true; do
  grep -q "HINGE-SWEEPS COMPLETE" "$HS" 2>/dev/null && { echo "### [$(ts)] hinge sweeps complete"; break; }
  if ! pgrep -f 'drive_hinge_sweeps.sh' >/dev/null 2>&1 \
       && grep -q "BIGDFM-50EP COMPLETE" "$B50" 2>/dev/null; then
    echo "### [$(ts)] hinge driver ended w/o COMPLETE but training done — proceeding"; break
  fi
  sleep 120
done
# Extra safety: never start while any GPU trainer/sweeper is still alive.
while pgrep -f 'train_for_sflm_bench|sweep_dfm_svgp_corruption' >/dev/null 2>&1; do sleep 60; done
echo "### [$(ts)] GPU free — running batch-throughput probe"

# --- 2. probe -------------------------------------------------------------
PROBELOG="${LOG}/batch_probe.log"
python scripts/probe_batch_throughput.py \
    --scale "$SCALE" --arm "$ARM" --batches 16,24,32,40,48 \
    --warmup 6 --iters 25 --out "${R}/batch_probe.json" \
    > "$PROBELOG" 2>&1
echo "### [$(ts)] probe results:"; grep -E "B=|BEST" "$PROBELOG" | sed 's/^/    /'

B=$(grep -oE 'BEST_BATCH=[0-9]+' "$PROBELOG" | tail -1 | cut -d= -f2)
if [ -z "${B:-}" ]; then
  echo "### [$(ts)] probe found no fitting batch — ABORT (see $PROBELOG)"; exit 1
fi
LR=$(python -c "import math;print(f'{3e-4*math.sqrt($B/16):.6g}')")
echo "### [$(ts)] selected batch=$B  lr=$LR (sqrt-scaled from 3e-4 @ b16)"

# --- 3. train to convergence on the full split ----------------------------
echo "### [$(ts)] training $ARM @ $SCALE — FULL split, batch $B, lr $LR, early-stop"
python scripts/train_for_sflm_bench.py --scale "$SCALE" --only "$ARM" --force \
    --full-split --batch "$B" --lr "$LR" \
    --epochs 30 --val-eval --early-stop-patience 3 --es-min-delta 0.002 \
    --max-hours 96 \
    > "${LOG}/train_${ARM}.log" 2>&1
echo "### [$(ts)] train exit=$?  meta=$(tr -d '\n ' < "${R}/${ARM}/train_meta.json" 2>/dev/null)"

if [ ! -f "$CKPT" ]; then
  echo "### [$(ts)] no checkpoint at $CKPT — ABORT eval (see ${LOG}/train_${ARM}.log)"; exit 1
fi

# --- 4. eval (generation + recovery), mirroring the 50k driver ------------
echo "### [$(ts)] generation eval"
python scripts/eval_all.py --ckpt "$CKPT" --model-kind DirichletFM \
    --split test --n 256 --steps 200 --bpc-mc 8 \
    --out "${R}/${ARM}/eval_all.json" > "${LOG}/eval_${ARM}.log" 2>&1
python -c "import json;p='${R}/${ARM}/eval_all.json';d=json.load(open(p));d['model_name']='DirichletFM_d1280L14_fulltext8_b${B}';json.dump(d,open(p,'w'),indent=2)" 2>/dev/null || true
echo "### [$(ts)] recovery sweep"
python scripts/recovery_check.py --ckpt "$CKPT" \
    --alphas 0.1,0.3,0.5,0.7,1.0 --n 256 --steps 200 \
    --out "${R}/${ARM}/recovery.json" > "${LOG}/recovery_${ARM}.log" 2>&1

echo "### FULLTEXT8 COMPLETE $(ts) — batch=$B lr=$LR; metrics in ${R}/${ARM}/eval_all.json + recovery.json"
