#!/usr/bin/env bash
# PerPosVarGP — per-position one-class VARIANCE GP OOD detector on DirichletFM
# (DirichletFlowMatching, Stark 2024; NOT the Discrete-FM 'DFM' arm). This is a
# DISTINCT detector from the energy-hinge SVGP (HingeSVGP): score = predictive
# variance (one-class), not trained mean/prob. Separate outputs entirely
# (runs/ood_vargp_perpos_dirichletfm/, vargp_perpos_sweep.json).
#
# Runs AFTER the hinge 2x2 (drive_ood_svgp.sh) so the two never contend for GPU.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/renku/work/aitchinson-flow
SC=a100_20g_L256; R="runs/sflm_bench_${SC}"
OUT="runs/ood_vargp_perpos_dirichletfm"; LOG="$OUT/_driver"; mkdir -p "$LOG"
ts(){ date +%Y-%m-%d_%H:%M:%S; }
HINGE="runs/ood_svgp_dirichletfm/_driver/ood_svgp.out"

echo "### VARGP-PERPOS START $(ts) — waiting for OOD-SVGP COMPLETE (hinge 2x2 done)"
while ! grep -q "OOD-SVGP COMPLETE" "$HINGE" 2>/dev/null; do sleep 60; done
echo "### [$(ts)] hinge 2x2 done — starting PerPosVarGP runs"

run () {  # $1=tag  $2=ckpt  $3=pca-dim
  local tag="$1" ck="$2" pca="$3"
  local od="$OUT/${tag}_pca${pca}"; mkdir -p "$od"
  if [ ! -f "$ck" ]; then echo "### [$(ts)] MISSING $ck — skip $tag"; return; fi
  echo "### [$(ts)] PerPosVarGP $tag pca=$pca"
  python scripts/ood_variance_perpos.py --ckpt "$ck" --pca-dim "$pca" \
      --fit-seqs 512 --n 256 --inducing 256 --fit-steps 500 \
      --out "$od/vargp_perpos_sweep.json" \
      > "$LOG/vargp_${tag}_pca${pca}.log" 2>&1 \
    && echo "    OK -> $od/vargp_perpos_sweep.json" \
    || echo "    FAILED (see $LOG/vargp_${tag}_pca${pca}.log)"
}

# primary: bigger DirichletFM, raw d_model + ARD (the chosen 'per-position + ARD')
run bigger_ep30_d30k   "$R/DirichletFM_ep30_d30k/epoch_final.pt" 0
# diagnostic: does mild unsupervised PCA reduction un-saturate the variance?
run bigger_ep30_d30k   "$R/DirichletFM_ep30_d30k/epoch_final.pt" 64
# control: baseline DirichletFM, raw + ARD
run baseline_ep10_d10k "$R/DirichletFM/epoch_final.pt"            0
echo "### VARGP-PERPOS COMPLETE $(ts) — see $OUT/*/vargp_perpos_sweep.json"
