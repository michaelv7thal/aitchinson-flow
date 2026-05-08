#!/usr/bin/env bash
# Phase 17 — healing/recovery test on every EqM-style checkpoint we have.
# Idempotent per checkpoint (skips if runs/<name>/healing.json exists).
set -euo pipefail
cd "$(dirname "$0")/.."

CKPTS=(
  "runs/eqm_data50k_ep5_mse_v3/epoch_final.pt"
  "runs/eqm_data50k_ep5_hilbert_soft/epoch_final.pt"
  "runs/eqm_data50k_ep5_hilbert_softmax/epoch_final.pt"
)

for ckpt in "${CKPTS[@]}"; do
  name=$(basename "$(dirname "$ckpt")")
  out="runs/$name/healing.json"
  if [ -f "$out" ]; then
    echo "[phase17] skip $name (healing.json exists)"
    continue
  fi
  if [ ! -f "$ckpt" ]; then
    echo "[phase17] skip $name (ckpt missing: $ckpt)"
    continue
  fi
  echo "[phase17] healing eval for $name"
  python scripts/eval_healing.py \
    --ckpt "$ckpt" --n 256 --steps 200 \
    --out "$out"
done

echo "[phase17] done. Outputs:"
for ckpt in "${CKPTS[@]}"; do
  name=$(basename "$(dirname "$ckpt")")
  echo "  runs/$name/healing.{json,png}"
done
