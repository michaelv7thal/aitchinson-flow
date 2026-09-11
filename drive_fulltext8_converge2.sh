#!/usr/bin/env bash
# Full-text8 DirichletFM CONVERGENCE run #2 — warm-start tuned to settle WITHOUT
# the val "bump" the first continuation showed.
#
# Context. drive_fulltext8_converge.sh trained ep1-10 (96h cap) → best-val ckpt
# at ep10 (val 0.8571). drive_fulltext8_continue.sh warm-started it over a fresh
# 10-epoch cosine that RE-PEAKED the LR to 5.2e-4 (above the ~4e-4 where the loss
# starts to oscillate), so val bumped UP (0.857→0.899) and only oscillated back
# to 0.8554 by ep5 before being killed mid-ep6. The bump was high-LR oscillation
# on already-good weights, partly masked by genuinely-noisy val (eval_step draws
# a random t and a fresh Dirichlet sample each eval).
#
# This run prevents the bump:
#   * warm-start from the BEST checkpoint so far (DirichletFM_continue ep5, val
#     0.8554) — or the best of a partial run of THIS arm, so re-launches resume;
#   * warmup OFF (--warmup-epochs 0): the cosine starts at the peak immediately,
#     so no re-ramp AND no ~10 h epoch wasted at near-zero LR;
#   * LOW peak LR 1.5e-4 (≈2.7× below the 4e-4 oscillation onset; larger-batch
#     gradient noise is also lower) → monotone descent, no bump;
#   * cosine → 0 over 8 epochs so the LR fully anneals and the model SETTLES;
#   * batch 16, NO grad-checkpointing: throughput is ~flat in windows/s across
#     batch (probe 9.63→9.97), so B16 does ~3× the gradient updates of B48 in the
#     same ~10 h/epoch → fastest-to-converge, and 8.35 GB peak = no OOM risk
#     (B48 fit the original run but is needlessly tight here);
#   * early stop (patience 3) restores the best-val ckpt as epoch_final.pt and
#     stops once val plateaus; NO wall-clock cap (the 2026-06-24 cluster-update
#     deadline that motivated the earlier caps has passed) — train to convergence.
#
# Sequence: (0 refuse to double-launch) → snapshot insurance → TRAIN warm-started
# into its OWN arm dir → EVAL (generation + recovery), mirroring the prior driver.
set -uo pipefail
export PYTHONPATH="/layers/paketo-buildpacks_pip-install/packages/lib/python3.11/site-packages:/layers/paketo-buildpacks_pip/pip/lib/python3.11/site-packages:/layers/paketo-buildpacks_cpython/cpython:src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/renku/work/aitchinson-flow
PY=/home/renku/work/.venv/bin/python

SCALE=a100_20g_L256_d1280L14_full
SRC_ARM=DirichletFM_continue        # the ep5 best (val 0.8554) — warm-start source
ARM=DirichletFM_converge            # this run (own dir; prior artifacts kept)
R="runs/sflm_bench_${SCALE}"
LOG="${R}/_driver"; mkdir -p "$LOG"
SRC_CKPT="${R}/${SRC_ARM}/epoch_best.pt"
CKPT="${R}/${ARM}/epoch_final.pt"
B=16
LR=1.5e-4
EPOCHS=8
ts(){ date +%Y-%m-%d_%H:%M:%S; }

echo "### CONVERGE2 START $(ts)"
# 0. refuse to start if a trainer is already alive (avoid two jobs on one MIG).
if pgrep -f 'train_for_sflm_bench' >/dev/null 2>&1; then
  echo "### [$(ts)] a train_for_sflm_bench process is already running — ABORT"; exit 1
fi
# Resume-aware warm-start: prefer the best of a partial run of THIS arm (so an
# interrupted launch picks up its own progress); else the ep5 source.
WARM="$SRC_CKPT"
if [ -f "${R}/${ARM}/epoch_best.pt" ]; then
  WARM="${R}/${ARM}/epoch_best.pt"
  echo "### [$(ts)] found a partial ${ARM} run — warm-starting from its best: $WARM"
fi
if [ ! -f "$WARM" ]; then
  echo "### [$(ts)] warm-start checkpoint $WARM missing — ABORT"; exit 1
fi

# snapshot the warm-start source (insurance; this run writes to a different dir).
cp -n "$SRC_CKPT" "${R}/${SRC_ARM}/epoch_best_ep5_val0p8554.pt" 2>/dev/null \
  && echo "### [$(ts)] snapshot -> ${R}/${SRC_ARM}/epoch_best_ep5_val0p8554.pt" || true

echo "### [$(ts)] TRAIN $ARM @ $SCALE — FULL split, B=$B, lr=$LR, warmup=0, cosine→0/${EPOCHS}ep, early-stop p3, NO cap"
echo "### [$(ts)] warm from: $WARM"
$PY scripts/train_for_sflm_bench.py --scale "$SCALE" --only "$ARM" --force \
    --full-split --batch "$B" --lr "$LR" --warmup-epochs 0 \
    --resume-weights "$WARM" \
    --epochs "$EPOCHS" --val-eval --early-stop-patience 3 --es-min-delta 0.002 \
    > "${LOG}/train_${ARM}.log" 2>&1
echo "### [$(ts)] train exit=$?  meta=$(tr -d '\n ' < "${R}/${ARM}/train_meta.json" 2>/dev/null)"

if [ ! -f "$CKPT" ]; then
  echo "### [$(ts)] no checkpoint at $CKPT — ABORT eval (see ${LOG}/train_${ARM}.log)"; exit 1
fi

# EVAL (generation + recovery), mirroring the prior drivers for apples-to-apples.
echo "### [$(ts)] generation eval"
$PY scripts/eval_all.py --ckpt "$CKPT" --model-kind DirichletFM \
    --split test --n 256 --steps 200 --bpc-mc 8 \
    --out "${R}/${ARM}/eval_all.json" > "${LOG}/eval_${ARM}.log" 2>&1
$PY -c "import json;p='${R}/${ARM}/eval_all.json';d=json.load(open(p));d['model_name']='DirichletFM_d1280L14_fulltext8_converge_b${B}';json.dump(d,open(p,'w'),indent=2)" 2>/dev/null || true
echo "### [$(ts)] recovery sweep"
$PY scripts/recovery_check.py --ckpt "$CKPT" \
    --alphas 0.1,0.3,0.5,0.7,1.0 --n 256 --steps 200 \
    --out "${R}/${ARM}/recovery.json" > "${LOG}/recovery_${ARM}.log" 2>&1

echo "### CONVERGE2 COMPLETE $(ts) — B=$B lr=$LR warmup=0 epochs=$EPOCHS (early-stop, no cap); metrics in ${R}/${ARM}/eval_all.json + recovery.json"
