#!/bin/bash
# Auto-train (10 epochs) + OOD bench + generation eval for a100_20g.
set -eu
cd /home/renku/work/aitchinson-flow
export PYTHONPATH="src:.:$PYTHONPATH"
PY=/home/renku/work/.venv/bin/python

echo "=== $(date) STAGE 1/2: OOD bench (auto-trains 8 arms @ 10 ep) ==="
$PY -u scripts/bench_sflm_ebm.py --scale a100_20g --epochs 10
echo "=== $(date) STAGE 2/2: Generation eval (no auto-train; reuses ckpts) ==="
$PY -u scripts/eval_generation.py --scale a100_20g --no-auto-train
echo "=== $(date) DONE ==="
