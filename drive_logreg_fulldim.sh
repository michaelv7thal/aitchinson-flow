#!/usr/bin/env bash
# Full-dimension LOGISTIC probe as a bench detector (the "LOG" family), scored on
# the SAME protocol as every other row of tab:ood-word / tab:ood-seq: test split,
# fit 512 windows, eval the next n=256 held out, the nine-rate ladder, seed 42.
#
# The probe already existed in the paper only as (a) an in-sample latent-split
# ceiling (tab:latent-ladder "full") and (b) the transfer track's false-info head
# (tab:ood-transfer). Neither is a bench row. These three arms mirror the LinE
# family one-for-one, differing from it in the objective alone (logistic loss in
# place of the bounded-energy hinge), so LOG/LOG_all/LOG_fi drop in beside
# LinE/LinE_all/LinE_fi:
#   logreg      <- blr      : --train-schemes replace                    , t_eval 4.5
#   logreg_adv  <- blr_adv  : --train-schemes replace,shuffle,falseinfo,both, t_eval 7.5
#   logreg_fi   <- blr_fi   : --train-schemes falseinfo                  , t_eval 4.5
# Writes to NEW directories; no existing artifact is touched.
set -uo pipefail
cd /home/michael/projects/aitchinson-flow
CKPT="runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM_converge/epoch_final.pt"
RATES="0.05,0.1,0.15,0.2,0.25,0.3,0.5,0.7,1.0"
SCHEMES="replace,shuffle,falseinfo"
COMMON=(--ckpt "$CKPT" --split test --n 256 --fit-seqs 512 --rates "$RATES"
        --seed 42 --schemes "$SCHEMES" --head logistic --pca-dim 0 --no-plot)
ts(){ date +%Y-%m-%d_%H:%M:%S; }

run_arm() {  # $1=dir  $2=json  shift 2 -> extra args
  local dir=$1 json=$2; shift 2
  mkdir -p "bench_ood_final/$dir"
  local out="bench_ood_final/$dir/$json" log="bench_ood_final/$dir/$dir.log"
  echo "### [$(ts)] $dir  $*" | tee -a "$log"
  PYTHONUNBUFFERED=1 .venv/bin/python scripts/ood_bayes_linear.py \
    "${COMMON[@]}" "$@" --out "$out" >> "$log" 2>&1
  echo "### [$(ts)] $dir exit $?" | tee -a "$log"
}

until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1500 ]; do sleep 60; done
run_arm logreg    bayes_linear_logreg_sweep.json     --train-schemes replace                       --t-eval 4.5
run_arm logreg_fi bayes_linear_logreg_fi_sweep.json  --train-schemes falseinfo                     --t-eval 4.5
run_arm logreg_adv bayes_linear_logreg_adv_sweep.json --train-schemes replace,shuffle,falseinfo,both --t-eval 7.5
echo "### [$(ts)] LOGREG FAMILY COMPLETE"
