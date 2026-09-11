#!/usr/bin/env bash
# Full-text8 DirichletFM CONTINUATION run (warm-start from the 96h-capped ckpt).
#
# Context: drive_fulltext8_converge.sh hit a 96 h wall-clock cap at epoch 10/30
# (val still improving, es=0/3 — it had NOT converged) and restored the best-val
# ep10 weights as .../DirichletFM/epoch_final.pt. That 96 h cap should not have
# been set. This driver continues training those weights.
#
# Binding constraint: the CLUSTER UPDATES in ~4 days (≈2026-06-24 09:16), so the
# whole thing — continued training + eval — must finish before then. At
# ~9.8 h/epoch that is only ~8-9 more epochs, NOT the ~20 needed to reach 30.
# So we size --max-hours to the deadline (minus an eval reserve) and take as many
# epochs as fit; early stopping + best-val restore keep the best checkpoint.
#
# Sequence:
#   1. WAIT for the prior pipeline (recovery sweep / driver) to free the GPU.
#   2. SNAPSHOT the ep10 checkpoint (insurance; we also write to a separate dir).
#   3. TRAIN warm-started from epoch_final.pt (fresh optimizer + full cosine over
#      --epochs) on the FULL split, batch 48, lr sqrt-scaled, early-stop, into
#      a SEPARATE arm dir (DirichletFM_continue) so the ep10 artifacts are kept.
#   4. EVAL: generation (KL/H_ratio/BPC) + recovery sweep, mirroring the original
#      driver, for apples-to-apples comparison against the ep10 run.
set -uo pipefail
export PYTHONPATH="/layers/paketo-buildpacks_pip-install/packages/lib/python3.11/site-packages:/layers/paketo-buildpacks_pip/pip/lib/python3.11/site-packages:/layers/paketo-buildpacks_cpython/cpython:src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/renku/work/aitchinson-flow

SCALE=a100_20g_L256_d1280L14_full
SRC_ARM=DirichletFM                 # the ep10 (capped) run
ARM=DirichletFM_continue            # the continuation (own dir; ep10 not clobbered)
R="runs/sflm_bench_${SCALE}"
LOG="${R}/_driver"; mkdir -p "$LOG"
SRC_CKPT="${R}/${SRC_ARM}/epoch_final.pt"
CKPT="${R}/${ARM}/epoch_final.pt"
B=48                                # the probe-selected batch from the ep10 run
LR=$(python -c "import math;print(f'{3e-4*math.sqrt($B/16):.6g}')")
EPOCHS=10                           # cosine-anneal horizon; the deadline cap binds first

# Hard deadline: cluster update ~2026-06-24 09:16. Stop with margin and reserve
# time for the post-train eval so the whole run finishes before the update.
DEADLINE="2026-06-24 06:00:00"
EVAL_RESERVE_H=6
ts(){ date +%Y-%m-%d_%H:%M:%S; }

echo "### CONTINUE START $(ts) — waiting for the prior pipeline to free the GPU"
# Proceed only once no GPU job from the old pipeline is alive (recovery sweep,
# generation eval, the trainer, or the original driver itself).
while pgrep -f 'drive_fulltext8_converge.sh|recovery_check.py|eval_all.py|train_for_sflm_bench' >/dev/null 2>&1; do
  sleep 120
done
echo "### [$(ts)] GPU free — prior pipeline finished"
if [ ! -f "$SRC_CKPT" ]; then
  echo "### [$(ts)] source checkpoint $SRC_CKPT missing — ABORT"; exit 1
fi

# --- 2. snapshot the ep10 checkpoint (insurance) --------------------------
cp -n "$SRC_CKPT" "${R}/${SRC_ARM}/epoch_final_ep10_capped.pt" \
  && echo "### [$(ts)] snapshot -> ${R}/${SRC_ARM}/epoch_final_ep10_capped.pt"

# --- 3. compute the deadline-derived wall-clock cap -----------------------
DEAD_S=$(date -d "$DEADLINE" +%s)
NOW_S=$(date +%s)
MAXH=$(python -c "print(f'{max(1.0,($DEAD_S-$NOW_S)/3600.0-$EVAL_RESERVE_H):.2f}')")
echo "### [$(ts)] deadline=$DEADLINE  reserve=${EVAL_RESERVE_H}h  -> --max-hours=$MAXH"
echo "### [$(ts)] warm-start train $ARM @ $SCALE — FULL split, batch $B, lr $LR, epochs $EPOCHS, early-stop, cap ${MAXH}h"

python scripts/train_for_sflm_bench.py --scale "$SCALE" --only "$ARM" --force \
    --full-split --batch "$B" --lr "$LR" \
    --resume-weights "$SRC_CKPT" \
    --epochs "$EPOCHS" --val-eval --early-stop-patience 3 --es-min-delta 0.002 \
    --max-hours "$MAXH" \
    > "${LOG}/train_${ARM}.log" 2>&1
echo "### [$(ts)] train exit=$?  meta=$(tr -d '\n ' < "${R}/${ARM}/train_meta.json" 2>/dev/null)"

if [ ! -f "$CKPT" ]; then
  echo "### [$(ts)] no checkpoint at $CKPT — ABORT eval (see ${LOG}/train_${ARM}.log)"; exit 1
fi

# --- 4. eval (generation + recovery), mirroring the original driver -------
echo "### [$(ts)] generation eval"
python scripts/eval_all.py --ckpt "$CKPT" --model-kind DirichletFM \
    --split test --n 256 --steps 200 --bpc-mc 8 \
    --out "${R}/${ARM}/eval_all.json" > "${LOG}/eval_${ARM}.log" 2>&1
python -c "import json;p='${R}/${ARM}/eval_all.json';d=json.load(open(p));d['model_name']='DirichletFM_d1280L14_fulltext8_continue_b${B}';json.dump(d,open(p,'w'),indent=2)" 2>/dev/null || true
echo "### [$(ts)] recovery sweep"
python scripts/recovery_check.py --ckpt "$CKPT" \
    --alphas 0.1,0.3,0.5,0.7,1.0 --n 256 --steps 200 \
    --out "${R}/${ARM}/recovery.json" > "${LOG}/recovery_${ARM}.log" 2>&1

echo "### CONTINUE COMPLETE $(ts) — batch=$B lr=$LR cap=${MAXH}h; metrics in ${R}/${ARM}/eval_all.json + recovery.json"
