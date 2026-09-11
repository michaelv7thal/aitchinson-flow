#!/usr/bin/env bash
# W1 post-hoc Euler sampler sweep on the eqm_data50k_ep5_v2 checkpoint.
# Generated for the cluster training plan (CLUSTER_TRAINING_PLAN.md §W1).
#
# Reads:  runs/eqm_data50k_ep5_v2/epoch_final.pt
# Writes: runs/eqm_data50k_ep5_v2/eval_euler_nfe<NFE>.json
#         runs/eqm_data50k_ep5_v2/eval_euler_sigma<SIGMA>.json
#         runs/eqm_data50k_ep5_v2/eval_euler_usegrad.json
set -euo pipefail
CKPT="runs/eqm_data50k_ep5_v2/epoch_final.pt"
if [[ ! -f "$CKPT" ]]; then
    echo "[W1] missing $CKPT — phase10 hasn't finished" >&2
    exit 1
fi

echo "[W1] NFE sweep over Euler sampler"
for nfe in 32 64 128 200; do
    python scripts/eval_full.py \
        --ckpt "$CKPT" \
        --override eqm.sampler=euler \
        --override eqm.euler_nfe="$nfe" \
        --steps "$nfe" \
        --out "runs/eqm_data50k_ep5_v2/eval_euler_nfe${nfe}.json"
done

echo "[W1] sigma_init sweep at NFE=128"
for sigma in 0.05 0.1 0.3; do
    python scripts/eval_full.py \
        --ckpt "$CKPT" \
        --override eqm.sampler=euler \
        --override eqm.euler_nfe=128 \
        --override eqm.sample_sigma_init="$sigma" \
        --steps 128 \
        --out "runs/eqm_data50k_ep5_v2/eval_euler_sigma${sigma}.json"
done

echo "[W1] conservative-grad ablation (Euler integrator, ∇⟨x,f⟩ instead of raw f)"
python scripts/eval_full.py \
    --ckpt "$CKPT" \
    --override eqm.sampler=euler \
    --override eqm.euler_use_grad=true \
    --override eqm.euler_nfe=128 \
    --steps 128 \
    --out "runs/eqm_data50k_ep5_v2/eval_euler_usegrad.json"

echo "[W1] DONE"
