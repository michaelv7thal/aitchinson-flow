#!/usr/bin/env bash
# W3 OOD eval scorecard — runs eval_ood.py over the three phase10 checkpoints.
# Generated for the cluster training plan (CLUSTER_TRAINING_PLAN.md §W3).
set -euo pipefail
for run in eqm_data50k_ep5_v2 dfm_data50k_ep5_v2 fmclr_data50k_ep5_v2; do
    ckpt="runs/$run/epoch_final.pt"
    if [[ ! -f "$ckpt" ]]; then
        echo "[W3] skipping $run — no checkpoint" >&2
        continue
    fi
    python scripts/eval_ood.py \
        --ckpt "$ckpt" \
        --n 256 \
        --out "runs/$run/ood_eval.json"
done
echo "[W3] DONE"
