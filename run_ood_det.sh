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

echo "=== [1/4] HingeSVGP — energy-hinge, score = trained mean/prob ==="
$PY scripts/fit_dfm_svgp_hinge.py --ckpt "$CKPT" --pooling "$POOL" \
    --out-dir "$OUT/hinge_${POOL}" --n-epochs 5 --eval-n "$N"
$PY scripts/sweep_dfm_svgp_corruption.py \
    --ckpt "$OUT/hinge_${POOL}/model_with_svgp_hinge.pt" \
    --out "$OUT/hinge_${POOL}/svgp_corruption_sweep.json" --n "$N"

echo "=== [2/4] PerPosMaha — per-position Mahalanobis distance ==="
$PY scripts/ood_mahalanobis_perpos.py --ckpt "$CKPT" --pca-dim "$PCA" \
    --fit-seqs "$FITSEQS" --n "$N" --shrinkage 0.1 \
    --out "$OUT/maha_pca${PCA}/maha_perpos_sweep.json"

echo "=== [3/4] PerPosVarGP — per-position one-class GP variance ==="
$PY scripts/ood_variance_perpos.py --ckpt "$CKPT" --pca-dim "$PCA" \
    --fit-seqs "$FITSEQS" --n "$N" --inducing 256 --fit-steps 500 \
    --out "$OUT/vargp_pca${PCA}/vargp_perpos_sweep.json"

echo "=== [4/4] BayesLinHead — linear energy head + Laplace uncertainty (per-token + sequence) ==="
$PY scripts/ood_bayes_linear.py --ckpt "$CKPT" --pca-dim "$PCA" \
    --fit-seqs "$FITSEQS" --n "$N" \
    --out "$OUT/bayeslin_pca${PCA}/bayes_linear_sweep.json"

echo "=== DONE -> $OUT (svgp_corruption_sweep / maha_perpos_sweep / vargp_perpos_sweep / bayes_linear_sweep .json) ==="
