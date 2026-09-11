#!/usr/bin/env bash
# Re-run the plausible-vs-random swap experiment (tab:ood-plausible) with the
# canonical arguments of bench_ood_final/plausible/plausible_swap.json, plus the
# new training-free VARIANCE head row ("Var") and a dump of the swapped windows
# (*.tokens.pt) so later readouts can be added by replay. Writes to a NEW json;
# the canonical file is never touched. Queued behind whatever holds the GPU.
set -uo pipefail
cd /home/michael/projects/aitchinson-flow
CKPT="runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM_converge/epoch_final.pt"
OUT=bench_ood_final/plausible/plausible_swap_var.json
LOG=${OUT%.json}.log
WAIT_PID=${1:-}
ts(){ date +%Y-%m-%d_%H:%M:%S; }
echo "### queued $(ts) (waiting for pid ${WAIT_PID:-none} and a free card)" > "$LOG"
if [ -n "$WAIT_PID" ]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  echo "### pid $WAIT_PID gone $(ts)" >> "$LOG"
fi
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1500 ]; do sleep 60; done
echo "### gpu free $(ts)" >> "$LOG"
PYTHONUNBUFFERED=1 .venv/bin/python scripts/ood_plausible_swap.py \
  --ckpt "$CKPT" --split test --n 64 --fit-seqs 512 --rate 0.15 \
  --t-eval 7.5 --t-nll 3.0 --blr-t-eval 4.5 --lin-t-eval 4.5 --n-cands 48 \
  --ref-lm gpt2 --swap-mode min_nll --seed 42 \
  --var-t-eval 7.5 --var-ridge 0.1 \
  --out "$OUT" >> "$LOG" 2>&1
echo "exit $?" >> "$LOG"; echo "### done $(ts)" >> "$LOG"; touch "$OUT.done"
