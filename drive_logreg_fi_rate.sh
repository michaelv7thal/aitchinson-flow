#!/usr/bin/env bash
# Control for the LOG_fi discrepancy: the bench-protocol false-info logistic head
# reads 0.761 word-max at rate 0.15, while the transfer track's false-info logistic
# head reads 0.912 on text8. The two differ in THREE ways: training-negative rate
# (0.5 bench default vs 0.15 matched to eval), fit split (test-block vs train-split
# windows), and eval block (256 bench windows vs the first 128 test windows).
# This arm changes ONE of them — --train-rate 0.15 — holding the bench protocol
# otherwise fixed, so the rate's contribution is measured rather than assumed.
set -uo pipefail
cd /home/michael/projects/aitchinson-flow
CKPT="runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM_converge/epoch_final.pt"
ts(){ date +%Y-%m-%d_%H:%M:%S; }
# Marker file, not pgrep (see drive_logreg_heal.sh for why).
echo "### [$(ts)] queued behind the heal family"
until [ -f bench_heal_final/.logreg_heal.done ]; do sleep 30; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1500 ]; do sleep 60; done
mkdir -p bench_ood_final/logreg_fi_r15
OUT=bench_ood_final/logreg_fi_r15/bayes_linear_logreg_fi_r15_sweep.json
echo "### [$(ts)] logreg_fi_r15  --train-schemes falseinfo --train-rate 0.15"
PYTHONUNBUFFERED=1 .venv/bin/python scripts/ood_bayes_linear.py \
  --ckpt "$CKPT" --split test --n 256 --fit-seqs 512 --seed 42 \
  --rates 0.05,0.1,0.15,0.2,0.25,0.3,0.5,0.7,1.0 --schemes replace,shuffle,falseinfo \
  --head logistic --pca-dim 0 --no-plot \
  --train-schemes falseinfo --train-rate 0.15 --t-eval 4.5 \
  --out "$OUT" > bench_ood_final/logreg_fi_r15/logreg_fi_r15.log 2>&1
echo "### [$(ts)] logreg_fi_r15 exit $?"
