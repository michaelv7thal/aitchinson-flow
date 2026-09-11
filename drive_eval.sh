#!/usr/bin/env bash
# Generation-benchmark EVAL driver (Stage C generation + Stage C2 recovery), one tier.
# Env: SC (scale); optional N (default 256), STEPS (200), BPC_MC (8).
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/renku/work/aitchinson-flow
: "${SC:?set SC}"; N="${N:-256}"; STEPS="${STEPS:-200}"; BPC_MC="${BPC_MC:-8}"
R="runs/sflm_bench_${SC}"; LOG="${R}/_driver"; mkdir -p "$LOG"
ts(){ date +%Y-%m-%d_%H:%M:%S; }
ARMS="EqM_OneHot EqM EqMAE DFM DirichletFM SFLM FMonCLR"
echo "### EVAL DRIVER START $(ts)  SC=$SC N=$N STEPS=$STEPS BPC_MC=$BPC_MC"

echo "### [$(ts)] STAGE C: generation eval (KL_uni/bi/tri, H_ratio, bpc[DFM])"
for ARM in $ARMS; do
  [ -f "$R/$ARM/epoch_final.pt" ] || { echo "  --- $ARM: no ckpt, skip ---"; continue; }
  echo "  --- eval $ARM @ $(ts) ---"
  python scripts/eval_all.py --ckpt $R/$ARM/epoch_final.pt \
      --model-kind $ARM --split test --n $N --steps $STEPS --bpc-mc $BPC_MC \
      --out $R/$ARM/eval_all.json > "${LOG}/eval_${ARM}.log" 2>&1 \
    && echo "    OK -> $R/$ARM/eval_all.json" || echo "    FAILED exit=$? (see ${LOG}/eval_${ARM}.log)"
done

echo "### [$(ts)] STAGE C2: recovery sweep (Δ@α, α=0.1,0.3,0.5,0.7,1.0)"
for ARM in $ARMS; do
  [ -f "$R/$ARM/epoch_final.pt" ] || { echo "  --- $ARM: no ckpt, skip ---"; continue; }
  echo "  --- recovery $ARM @ $(ts) ---"
  python scripts/recovery_check.py --ckpt $R/$ARM/epoch_final.pt \
      --alphas 0.1,0.3,0.5,0.7,1.0 --n $N --steps $STEPS \
      --out $R/$ARM/recovery.json > "${LOG}/recovery_${ARM}.log" 2>&1 \
    && echo "    OK -> $R/$ARM/recovery.json" || echo "    FAILED exit=$? (see ${LOG}/recovery_${ARM}.log)"
done
echo "### EVAL DRIVER COMPLETE $(ts)"
