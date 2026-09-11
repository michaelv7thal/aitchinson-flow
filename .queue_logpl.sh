#!/bin/bash
# Wait for the fit512 word run to release the GPU, then run the two word-level LOG_pl arms.
cd "$HOME/projects/aitchinson-flow"
while pgrep -f "ood_plausible_swap.py" >/dev/null; do sleep 60; done
echo "### GPU free at $(date); starting LOG_pl arms (falseinfo, plausible)"
exec .venv/bin/python scripts/run_bench_heal.py \
  --only logreg_pl_falseinfo,logreg_pl_plausible
