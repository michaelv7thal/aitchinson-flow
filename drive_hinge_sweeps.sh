#!/usr/bin/env bash
# HingeSVGP corruption sweeps on the already-FITTED models (fit succeeded; only
# the sweep was skipped by a now-fixed detach bug). Cheap (no re-fit). Gated
# after the 50ep run to avoid GPU contention.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/renku/work/aitchinson-flow
B50="runs/sflm_bench_a100_20g_L256_d1280L14_b16/_driver/bigdfm_50ep.out"
LOG="runs/ood_det_rerun/_driver"; ts(){ date +%Y-%m-%d_%H:%M:%S; }
echo "### HINGE-SWEEPS START $(ts) — waiting for BIGDFM-50EP COMPLETE"
while ! grep -q "BIGDFM-50EP COMPLETE" "$B50" 2>/dev/null; do sleep 120; done
echo "### [$(ts)] 50ep done — HingeSVGP sweeps on saved (energy_head) models"
for c in bigger_ep30_d30k_mean bigger_ep30_d30k_attention baseline_ep10_d10k_mean baseline_ep10_d10k_attention; do
  od="runs/ood_svgp_dirichletfm_det/$c"
  [ -f "$od/model_with_svgp_hinge.pt" ] || { echo "  skip $c"; continue; }
  echo "### [$(ts)] sweep $c"
  python scripts/sweep_dfm_svgp_corruption.py --ckpt "$od/model_with_svgp_hinge.pt" \
      --out "$od/svgp_corruption_sweep.json" --n 500 \
      > "$LOG/hingesweep_${c}.log" 2>&1 && echo "    OK" || echo "    FAILED"
done
echo "### HINGE-SWEEPS COMPLETE $(ts)"
