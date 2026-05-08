#!/usr/bin/env bash
# Phase 14 — Hilbert-loss EqM head-to-head at the data_50k_ep5 platform.
# Idempotent: skips cells whose runs/<name>/eval.json already exists.
#
# Usage:
#   bash scripts/run_phase14.sh                  # both cells, no W&B
#   ONLY=eqm_data50k_ep5_hilbert_soft bash scripts/run_phase14.sh
set -euo pipefail
cd "$(dirname "$0")/.."

EXTRA=()
if [ -n "${ONLY:-}" ]; then
  EXTRA+=("--only" "$ONLY")
fi

python scripts/run_sweep.py \
  --sweep sweeps/phase14_hilbert.yaml \
  --n 256 --steps 200 \
  "${EXTRA[@]}"

echo "[phase14] done. Results in:"
echo "  runs/eqm_data50k_ep5_mse_v3/eval.json"
echo "  runs/eqm_data50k_ep5_hilbert_soft/eval.json"
echo "  runs/sweep_results.jsonl  (last two rows)"
