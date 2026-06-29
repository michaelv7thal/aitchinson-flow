#!/usr/bin/env bash
# Full-text8 DirichletFM convergence — TAIL of the 8-epoch anneal, no early-stop.
#
# Context. drive_fulltext8_converge2.sh ran the gentle anneal (warm from ep5 best
# 0.8554; B16, peak 1.5e-4 → 0 over 8 ep, warmup 0). It completed 4 epochs cleanly
# (train 0.865→0.858→0.855→0.851, monotone down) but its val readout is now
# dominated by eval_step noise (random t + fresh Dirichlet draw each eval, unseeded):
# val read 0.818, 0.800, 0.876, 0.845 — bouncing while train kept descending. With
# val an artifact at this stage, early-stopping (and its best-val ckpt RESTORE at
# the end) would select on noise instead of the fully-annealed model. So we finish
# the schedule WITHOUT early stopping and keep the LAST epoch's weights.
#
# This driver warm-starts from the 4-epoch snapshot (epoch4_anneal.pt, the model
# trained through lr 1.04e-4; global_step 87892) and runs the remaining 4 epochs at
# the schedule's tail LRs — peak 7.5e-5 cosine→0 (7.5e-5, 6.4e-5, 3.75e-5, 1.1e-5),
# warmup 0, NO --early-stop-patience. With early-stop OFF the runner does NOT do the
# best-val restore, so epoch_final.pt = the fully-annealed epoch-8-equivalent model.
# Then eval (generation + recovery) on that final model.
set -uo pipefail
export PYTHONPATH="/layers/paketo-buildpacks_pip-install/packages/lib/python3.11/site-packages:/layers/paketo-buildpacks_pip/pip/lib/python3.11/site-packages:/layers/paketo-buildpacks_cpython/cpython:src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/renku/work/aitchinson-flow
PY=/home/renku/work/.venv/bin/python

SCALE=a100_20g_L256_d1280L14_full
ARM=DirichletFM_converge
R="runs/sflm_bench_${SCALE}"
LOG="${R}/_driver"; mkdir -p "$LOG"
WARM="${R}/${ARM}/epoch4_anneal.pt"     # the 4-epoch annealed snapshot
CKPT="${R}/${ARM}/epoch_final.pt"
B=16
LR=7.5e-5                                # the schedule's e4 LR (continues the anneal)
EPOCHS=4                                 # epochs 5..8 of the original 8-epoch plan
ts(){ date +%Y-%m-%d_%H:%M:%S; }

echo "### CONVERGE2-TAIL START $(ts)"
if pgrep -f 'train_for_sflm_bench' >/dev/null 2>&1; then
  echo "### [$(ts)] a train_for_sflm_bench process is already running — ABORT"; exit 1
fi
if [ ! -f "$WARM" ]; then
  echo "### [$(ts)] warm-start snapshot $WARM missing — ABORT"; exit 1
fi

echo "### [$(ts)] TRAIN tail $ARM — FULL split, B=$B, lr=$LR, warmup=0, cosine→0/${EPOCHS}ep, NO early-stop, NO cap"
echo "### [$(ts)] warm from: $WARM (4-epoch annealed, gstep 87892)"
$PY scripts/train_for_sflm_bench.py --scale "$SCALE" --only "$ARM" --force \
    --full-split --batch "$B" --lr "$LR" --warmup-epochs 0 \
    --resume-weights "$WARM" \
    --epochs "$EPOCHS" --val-eval \
    > "${LOG}/train_${ARM}_tail.log" 2>&1
echo "### [$(ts)] train exit=$?  meta=$(tr -d '\n ' < "${R}/${ARM}/train_meta.json" 2>/dev/null)"

if [ ! -f "$CKPT" ]; then
  echo "### [$(ts)] no checkpoint at $CKPT — ABORT eval (see ${LOG}/train_${ARM}_tail.log)"; exit 1
fi

echo "### [$(ts)] generation eval"
$PY scripts/eval_all.py --ckpt "$CKPT" --model-kind DirichletFM \
    --split test --n 256 --steps 200 --bpc-mc 8 \
    --out "${R}/${ARM}/eval_all.json" > "${LOG}/eval_${ARM}.log" 2>&1
$PY -c "import json;p='${R}/${ARM}/eval_all.json';d=json.load(open(p));d['model_name']='DirichletFM_d1280L14_fulltext8_converge_b${B}_8ep';json.dump(d,open(p,'w'),indent=2)" 2>/dev/null || true
echo "### [$(ts)] recovery sweep"
$PY scripts/recovery_check.py --ckpt "$CKPT" \
    --alphas 0.1,0.3,0.5,0.7,1.0 --n 256 --steps 200 \
    --out "${R}/${ARM}/recovery.json" > "${LOG}/recovery_${ARM}.log" 2>&1

echo "### CONVERGE2-TAIL COMPLETE $(ts) — full 8-epoch anneal done (final = annealed weights); metrics in ${R}/${ARM}/eval_all.json + recovery.json"
