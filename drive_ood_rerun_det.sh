#!/usr/bin/env bash
# Corrected OOD re-run with DETERMINISTIC feature extraction (the _sample_xt
# sampling bug fix: feed the Dirichlet MEAN through the backbone, no sampling).
# Writes to *_det dirs to preserve the buggy sampled-feature results for the
# before/after comparison. All 3 detectors on DirichletFM (NOT Discrete DFM).
# Runs AFTER the capacity test frees the GPU.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/renku/work/aitchinson-flow
SC=a100_20g_L256; R="runs/sflm_bench_${SC}"
BIG="$R/DirichletFM_ep30_d30k/epoch_final.pt"
BASE="$R/DirichletFM/epoch_final.pt"
CAP="runs/sflm_bench_a100_20g_L256_d1280L14/_driver/capacity_test.out"
LOG="runs/ood_det_rerun/_driver"; mkdir -p "$LOG"
ts(){ date +%Y-%m-%d_%H:%M:%S; }

echo "### OOD-DET-RERUN START $(ts) — waiting for CAPACITY-TEST COMPLETE"
while ! grep -q "CAPACITY-TEST COMPLETE" "$CAP" 2>/dev/null; do sleep 120; done
echo "### [$(ts)] capacity done — re-running OOD with DETERMINISTIC features (-> *_det)"

# 1) HingeSVGP 2x2 (pool_features now deterministic by default)
for pool in mean attention; do
  for pair in "bigger_ep30_d30k $BIG" "baseline_ep10_d10k $BASE"; do
    set -- $pair; tag=$1; ck=$2
    od="runs/ood_svgp_dirichletfm_det/${tag}_${pool}"; mkdir -p "$od"
    [ -f "$ck" ] || { echo "  skip $tag/$pool (no ckpt)"; continue; }
    echo "### [$(ts)] HINGE $tag/$pool"
    if python scripts/fit_dfm_svgp_hinge.py --ckpt "$ck" --pooling "$pool" \
         --out-dir "$od" --n-epochs 5 --eval-n 500 > "$LOG/hinge_${tag}_${pool}.log" 2>&1; then
      python scripts/sweep_dfm_svgp_corruption.py --ckpt "$od/model_with_svgp_hinge.pt" \
          --out "$od/svgp_corruption_sweep.json" --n 500 >> "$LOG/hinge_${tag}_${pool}.log" 2>&1 \
        && echo "    OK -> $od" || echo "    sweep FAILED"
    else echo "    fit FAILED (see $LOG/hinge_${tag}_${pool}.log)"; fi
  done
done

# 2) PerPosMaha (deterministic)
for spec in "bigger_ep30_d30k $BIG 0" "bigger_ep30_d30k $BIG 128" "baseline_ep10_d10k $BASE 0"; do
  set -- $spec; tag=$1; ck=$2; pca=$3
  od="runs/ood_maha_perpos_dirichletfm_det/${tag}_pca${pca}"; mkdir -p "$od"
  [ -f "$ck" ] || continue
  echo "### [$(ts)] MAHA $tag pca=$pca"
  python scripts/ood_mahalanobis_perpos.py --ckpt "$ck" --pca-dim "$pca" \
      --fit-seqs 512 --n 256 --shrinkage 0.1 --out "$od/maha_perpos_sweep.json" \
      > "$LOG/maha_${tag}_pca${pca}.log" 2>&1 && echo "    OK" || echo "    FAILED"
done

# 3) PerPosVarGP (deterministic)
for spec in "bigger_ep30_d30k $BIG 0" "bigger_ep30_d30k $BIG 64" "baseline_ep10_d10k $BASE 0"; do
  set -- $spec; tag=$1; ck=$2; pca=$3
  od="runs/ood_vargp_perpos_dirichletfm_det/${tag}_pca${pca}"; mkdir -p "$od"
  [ -f "$ck" ] || continue
  echo "### [$(ts)] VARGP $tag pca=$pca"
  python scripts/ood_variance_perpos.py --ckpt "$ck" --pca-dim "$pca" \
      --fit-seqs 512 --n 256 --inducing 256 --fit-steps 500 --out "$od/vargp_perpos_sweep.json" \
      > "$LOG/vargp_${tag}_pca${pca}.log" 2>&1 && echo "    OK" || echo "    FAILED"
done

# 4) BayesLinHead — linear energy head + Laplace uncertainty (per-token + sequence)
for pair in "bigger_ep30_d30k $BIG" "baseline_ep10_d10k $BASE"; do
  set -- $pair; tag=$1; ck=$2
  od="runs/ood_bayeslin_dirichletfm_det/${tag}"; mkdir -p "$od"
  [ -f "$ck" ] || continue
  echo "### [$(ts)] BAYESLIN $tag"
  python scripts/ood_bayes_linear.py --ckpt "$ck" --pca-dim 0 \
      --fit-seqs 512 --n 256 --out "$od/bayes_linear_sweep.json" \
      > "$LOG/bayeslin_${tag}.log" 2>&1 && echo "    OK" || echo "    FAILED"
done
echo "### OOD-DET-RERUN COMPLETE $(ts) — corrected results in runs/ood_*_det/"
