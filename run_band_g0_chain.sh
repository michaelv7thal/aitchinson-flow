#!/usr/bin/env bash
# Queue the gamma_lo=0 pair behind whatever is currently on the GPU.
#   1. band_L256_ep10_d10k_g0   gamma_lo=0 at the dev budget (A/B vs the finished run)
#   2. band_L256_ep30_d30k_g0   gamma_lo=0 at 30ep x 30k, budget-matched to
#                               DirichletFM_ep30_d30k (KL_bi 0.208, Delta@0.5 +0.193)
# Waits for the GPU to free so it does not contend with a running job.
set -uo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ts(){ date +%Y-%m-%d_%H:%M:%S; }

echo "### [$(ts)] waiting for the GPU to free"
while pgrep -f 'scripts/(recovery_check|run_sweep|band_ood_score|train_for_sflm_bench)\.py' >/dev/null 2>&1; do
  sleep 60
done
echo "### [$(ts)] GPU free — starting the chain"
CELLS="band_L256_ep10_d10k_g0 band_L256_ep30_d30k_g0" exec ./drive_band_L256.sh
