#!/usr/bin/env bash
# Phase K — UQ on HaluEval-QA. Idempotent: skips steps whose outputs exist.
# Usage:
#   bash scripts/run_phaseK.sh                  # default GPT-2, max-n=10000
#   LM=gpt2 MAX_N=10000 bash scripts/run_phaseK.sh
set -euo pipefail
cd "$(dirname "$0")/.."

LM="${LM:-gpt2}"
MAX_N="${MAX_N:-10000}"
L="${L:-160}"
LM_BS="${LM_BS:-16}"

CACHE="data/hallueval_cache_${LM/\//_}.pt"

# 1. Cache (skip if present).
if [ ! -f "$CACHE" ]; then
  echo "[phaseK] caching → $CACHE  (lm=$LM, max_n=$MAX_N, L=$L)"
  mkdir -p data
  python scripts/cache_hallueval.py \
    --lm "$LM" --max-n "$MAX_N" --L "$L" \
    --out "$CACHE" --lm-batch-size "$LM_BS"
else
  echo "[phaseK] cache present: $CACHE  ($(du -h "$CACHE" | cut -f1))"
fi

# 2. UQ eval — both pool methods.
for POOL in meanpool lasttoken; do
  RUN="hal_${LM/\//_}_qa_${POOL}"
  OUT="runs/$RUN/uq_eval.json"
  if [ ! -f "$OUT" ]; then
    echo "[phaseK] UQ eval → $OUT  (pool=$POOL)"
    mkdir -p "runs/$RUN"
    python scripts/eval_uq.py \
      --cache "$CACHE" --train-frac 0.8 --pool "$POOL" \
      --out "$OUT"
    python scripts/plot_uq_calibration.py \
      --json "$OUT" --out "runs/$RUN/uq_calibration.png"
  else
    echo "[phaseK] UQ eval present: $OUT"
  fi
done

echo "[phaseK] done."
