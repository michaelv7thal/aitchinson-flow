#!/usr/bin/env bash
# Phase 15 — softmax+Hilbert variant. Run AFTER Phase 14's hilbert_soft
# cell completes so we can compare apples-to-apples. Idempotent: skipped
# if runs/eqm_data50k_ep5_hilbert_softmax/eval.json exists.
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/run_sweep.py \
  --sweep sweeps/phase15_softmax_hilbert.yaml \
  --n 256 --steps 200

echo "[phase15] done. Result: runs/eqm_data50k_ep5_hilbert_softmax/eval.json"
