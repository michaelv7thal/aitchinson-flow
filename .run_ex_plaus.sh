#!/bin/bash
cd "$HOME/projects/aitchinson-flow"
exec .venv/bin/python -u scripts/heal_dirichlet.py \
  --ckpt runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM_converge/epoch_final.pt \
  --split test --n-demo 64 --n-seeds 1 --corrupt-rate 0.15 --nfe 100 \
  --target-fprs 0.02,0.05,0.10 --seed 42 \
  --localizer logistic_fi --t-eval 4.5 --corrupt-scheme plausible \
  --n-examples 64 \
  --out bench_heal_final/examples/logreg_fi_plausible_examples.json
