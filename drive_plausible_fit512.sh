#!/usr/bin/env bash
# Re-run the plausible-swap experiment (tab:ood-plausible) with the PLAUSIBLE
# specialist and the unified heads fitted on all 512 windows instead of 128.
#
# In the published run --fit-plaus-seqs defaulted to 128, because each swap slot
# costs n_cands forward passes, so LOG_pl saw a quarter of the windows LOG_fi saw
# and the plausible negatives were a minority of every unified head's training
# pool. This arm removes that asymmetry; nothing else changes.
#
# Only LOG_pl and the three unified rows can move: bgmm/blr/var/log_fi and both
# GPT-2 arms are all built BEFORE plaus_fit in ood_plausible_swap.py (lines
# 450-486), and the evaluation corruptions are built earlier still. Those rows
# reproducing exactly is the consistency check on this run.
#
# Writes a NEW json; the canonical plausible_swap_var.json is never touched.
set -uo pipefail
cd /home/michael/projects/aitchinson-flow
CKPT="runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM_converge/epoch_final.pt"
OUT=bench_ood_final/plausible/plausible_swap_fit512.json
LOG=${OUT%.json}.log
ts(){ date +%Y-%m-%d_%H:%M:%S; }
echo "### [$(ts)] queued behind the rate-matched control"
until [ -f bench_ood_final/logreg_fi_r15/bayes_linear_logreg_fi_r15_sweep.json ]; do sleep 30; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1500 ]; do sleep 60; done
echo "### [$(ts)] start (fit-plaus-seqs 512)"
PYTHONUNBUFFERED=1 .venv/bin/python scripts/ood_plausible_swap.py \
  --ckpt "$CKPT" --split test --n 64 --fit-seqs 512 --rate 0.15 \
  --t-eval 7.5 --t-nll 3.0 --blr-t-eval 4.5 --lin-t-eval 4.5 --n-cands 48 \
  --ref-lm gpt2 --swap-mode min_nll --seed 42 \
  --var-t-eval 7.5 --var-ridge 0.1 \
  --fit-plaus-seqs 512 \
  --out "$OUT" > "$LOG" 2>&1
echo "### [$(ts)] exit $?"
