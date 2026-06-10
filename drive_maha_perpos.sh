#!/usr/bin/env bash
# PerPosMaha — per-position MAHALANOBIS-distance OOD detector on DirichletFM
# (DirichletFlowMatching; NOT the Discrete-FM 'DFM' arm). Concentration-robust
# complement to PerPosVarGP. DISTINCT detector + outputs entirely
# (runs/ood_maha_perpos_dirichletfm/, maha_perpos_sweep.json).
#
# Runs AFTER PerPosVarGP (drive_vargp_perpos.sh) to avoid GPU contention.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/renku/work/aitchinson-flow
SC=a100_20g_L256; R="runs/sflm_bench_${SC}"
OUT="runs/ood_maha_perpos_dirichletfm"; LOG="$OUT/_driver"; mkdir -p "$LOG"
ts(){ date +%Y-%m-%d_%H:%M:%S; }
VG="runs/ood_vargp_perpos_dirichletfm/_driver/vargp_perpos.out"

echo "### MAHA-PERPOS START $(ts) — waiting for VARGP-PERPOS COMPLETE"
while ! grep -q "VARGP-PERPOS COMPLETE" "$VG" 2>/dev/null; do sleep 60; done
echo "### [$(ts)] PerPosVarGP done — starting PerPosMaha runs"

run () {  # $1=tag  $2=ckpt  $3=pca-dim
  local tag="$1" ck="$2" pca="$3"
  local od="$OUT/${tag}_pca${pca}"; mkdir -p "$od"
  if [ ! -f "$ck" ]; then echo "### [$(ts)] MISSING $ck — skip $tag"; return; fi
  echo "### [$(ts)] PerPosMaha $tag pca=$pca"
  python scripts/ood_mahalanobis_perpos.py --ckpt "$ck" --pca-dim "$pca" \
      --fit-seqs 512 --n 256 --shrinkage 0.1 \
      --out "$od/maha_perpos_sweep.json" \
      > "$LOG/maha_${tag}_pca${pca}.log" 2>&1 \
    && echo "    OK -> $od/maha_perpos_sweep.json" \
    || echo "    FAILED (see $LOG/maha_${tag}_pca${pca}.log)"
}

# primary: bigger DirichletFM, full-d_model Mahalanobis
run bigger_ep30_d30k   "$R/DirichletFM_ep30_d30k/epoch_final.pt" 0
# diagnostic: top-128 PC Mahalanobis (cleaner Sigma)
run bigger_ep30_d30k   "$R/DirichletFM_ep30_d30k/epoch_final.pt" 128
# control: baseline DirichletFM, full-d
run baseline_ep10_d10k "$R/DirichletFM/epoch_final.pt"            0
echo "### MAHA-PERPOS COMPLETE $(ts) — see $OUT/*/maha_perpos_sweep.json"
