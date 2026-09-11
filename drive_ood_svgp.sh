#!/usr/bin/env bash
# SVGP OOD detector study on DirichletFM (Dirichlet FM, Stark et al. 2024 —
# DirichletFlowMatching). This is NOT the Discrete-FM "DFM" arm; the fit script
# guards against wrapping a DiscreteFlowMatching checkpoint.
#
# 2x2 controlled design: {bigger 30ep×30k, baseline 10ep×10k} × {mean, attention
# pooling}. Each cell: fit_dfm_svgp_hinge.py (Stage-2 hinge SVGP on FROZEN
# DirichletFM features) -> sweep_dfm_svgp_corruption.py (replace/shuffle/both ×
# rate -> AUROC; headline the shuffle axis). Isolates model-size AND pooling
# effects on OOD separability at L256.
#
# Runs only after the extended DirichletFM finishes (GPU free + ckpt exists).
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/renku/work/aitchinson-flow
SC=a100_20g_L256
R="runs/sflm_bench_${SC}"
OOD="runs/ood_svgp_dirichletfm"; LOG="$OOD/_driver"; mkdir -p "$LOG"
ts(){ date +%Y-%m-%d_%H:%M:%S; }
DE="$R/_driver/dfm_extended.out"

echo "### OOD-SVGP START $(ts) — waiting for DFM-EXT COMPLETE (extended DirichletFM done)"
while ! grep -q "DFM-EXT COMPLETE" "$DE" 2>/dev/null; do sleep 60; done
echo "### [$(ts)] extended DirichletFM done — starting 2x2 SVGP OOD study"

# Both checkpoints are DirichletFM (DirichletFlowMatching) — NOT the DFM/Discrete arm.
run_cell () {  # $1=tag  $2=ckpt  $3=pooling
  local tag="$1" ck="$2" pool="$3"
  local od="$OOD/${tag}_${pool}"; mkdir -p "$od"
  if [ ! -f "$ck" ]; then echo "### [$(ts)] MISSING ckpt for $tag ($ck) — skip"; return; fi
  echo "### [$(ts)] FIT   $tag / $pool"
  python scripts/fit_dfm_svgp_hinge.py --ckpt "$ck" --pooling "$pool" \
      --out-dir "$od" --n-epochs 5 --eval-n 500 \
      > "$LOG/fit_${tag}_${pool}.log" 2>&1 \
    && echo "    fit OK" \
    || { echo "    fit FAILED (see $LOG/fit_${tag}_${pool}.log)"; return; }
  echo "### [$(ts)] SWEEP $tag / $pool"
  python scripts/sweep_dfm_svgp_corruption.py \
      --ckpt "$od/model_with_svgp_hinge.pt" \
      --out "$od/svgp_corruption_sweep.json" --n 500 \
      > "$LOG/sweep_${tag}_${pool}.log" 2>&1 \
    && echo "    sweep OK -> $od/svgp_corruption_sweep.json" \
    || echo "    sweep FAILED (see $LOG/sweep_${tag}_${pool}.log)"
}

for pool in mean attention; do
  run_cell bigger_ep30_d30k   "$R/DirichletFM_ep30_d30k/epoch_final.pt" "$pool"
  run_cell baseline_ep10_d10k "$R/DirichletFM/epoch_final.pt"            "$pool"
done
echo "### OOD-SVGP COMPLETE $(ts) — see $OOD/*/svgp_corruption_sweep.json"
