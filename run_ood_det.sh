#!/usr/bin/env bash
# Portable standalone OOD-detector runner for a DirichletFM checkpoint
# (DirichletFlowMatching / "Dirichlet FM" — NOT the Discrete-FM 'DFM' arm; the
# scripts guard against it). Runs all three detectors with the deterministic
# feature-extraction fix (already in the committed code). No waiting, no
# box-specific paths — drop this on any cluster, from the repo root.
#
# Usage:
#   CKPT=runs/.../DirichletFM/epoch_final.pt OUT=ood_out bash run_ood_det.sh
# Optional env: POOL=mean|attention (hinge), N=256 (eval seqs),
#               FITSEQS=512 (one-class fit seqs), PCA=0 (0=raw+ARD), PY=python
set -uo pipefail
: "${CKPT:?set CKPT=path/to/DirichletFM/epoch_final.pt}"
OUT="${OUT:-ood_out}"; POOL="${POOL:-mean}"; N="${N:-256}"
FITSEQS="${FITSEQS:-512}"; PCA="${PCA:-0}"; PY="${PY:-python}"
mkdir -p "$OUT"
echo "CKPT=$CKPT  OUT=$OUT  POOL=$POOL  N=$N  FITSEQS=$FITSEQS  PCA=$PCA"

#echo "=== [1/4] HingeSVGP — energy-hinge, score = trained mean/prob ==="
#$PY scripts/fit_dfm_svgp_hinge.py --ckpt "$CKPT" --pooling "$POOL" \
#    --out-dir "$OUT/hinge_${POOL}" --n-epochs 5 --eval-n "$N"
#$PY scripts/sweep_dfm_svgp_corruption.py \
#    --ckpt "$OUT/hinge_${POOL}/model_with_svgp_hinge.pt" \
#    --out "$OUT/hinge_${POOL}/svgp_corruption_sweep.json" --n "$N"

#echo "=== [2/4] PerPosMaha — per-position Mahalanobis distance ==="
#$PY scripts/ood_mahalanobis_perpos.py --ckpt "$CKPT" --pca-dim "$PCA" \
#    --fit-seqs "$FITSEQS" --n "$N" --shrinkage 0.1 \
#    --out "$OUT/maha_pca${PCA}/maha_perpos_sweep.json"

#echo "=== [3/4] PerPosVarGP — per-position one-class GP variance ==="
#$PY scripts/ood_variance_perpos.py --ckpt "$CKPT" --pca-dim "$PCA" \
#    --fit-seqs "$FITSEQS" --n "$N" --inducing 256 --fit-steps 500 \
#    --out "$OUT/vargp_pca${PCA}/vargp_perpos_sweep.json"

echo "=== [4/5] BayesLinHead — linear energy head + Laplace uncertainty (per-token + sequence) ==="
$PY scripts/ood_bayes_linear.py --ckpt "$CKPT" --pca-dim "$PCA" \
    --fit-seqs "$FITSEQS" --n "$N" \
    --out "$OUT/bayeslin_pca${PCA}/bayes_linear_sweep.json"

# RECOMMENDED detector: training-free per-token denoiser NLL (seq AUROC ~1.0 for
# replace AND shuffle; per-token localization ~0.85-0.91, beating the supervised
# energy head). Carries the Bayesian-linear VARIANCE baseline at t_var=7.5 — the
# only path-time where Var is not inverted (it collapses->core and inverts at the
# t=4.5 the energy head uses). See scripts/legacy/diag_var_tsweep.py for the
# t-sweep that fixes t_nll/t_var.
echo "=== [5/5] DenoiserNLL — per-token model surprise + variance@t_var (RECOMMENDED) ==="
$PY scripts/ood_denoiser_nll.py --ckpt "$CKPT" \
    --fit-seqs "$FITSEQS" --n "$N" --t-nll 3.0 --t-var 7.5 \
    --out "$OUT/nll/denoiser_nll_sweep.json"

echo "=== DONE -> $OUT (bayes_linear_sweep / denoiser_nll_sweep .json) ==="
