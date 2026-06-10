#!/usr/bin/env bash
# Crash-safe backfill: after the post chain completes, re-run any eval_all.json
# / recovery.json that is MISSING or OLDER than its checkpoint. This catches the
# 3 L256 EqM generation evals that failed pre-chunking-fix (their ckpts are from
# today's training, so newer than any stale eval) and any other gap, across both
# the L256 and L40 tiers. Idempotent: good (post-fix) artifacts are left alone.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/renku/work/aitchinson-flow
ts(){ date +%Y-%m-%d_%H:%M:%S; }
P=runs/sflm_bench_a100_20g_L256/_driver/post_chain.out
echo "### BACKFILL START $(ts) — waiting for POST CHAIN COMPLETE"
while ! grep -q "POST CHAIN COMPLETE" "$P" 2>/dev/null; do sleep 60; done
echo "### [$(ts)] post chain complete — scanning for missing/stale eval artifacts"

ARMS="EqM_OneHot EqM EqMAE DFM DirichletFM SFLM FMonCLR"
for SC in a100_20g_L256 local; do
  R="runs/sflm_bench_${SC}"; LOG="${R}/_driver"; mkdir -p "$LOG"
  for ARM in $ARMS; do
    ck="$R/$ARM/epoch_final.pt"
    [ -f "$ck" ] || { echo "  skip $SC/$ARM (no ckpt)"; continue; }
    if [ "$ck" -nt "$R/$ARM/eval_all.json" ]; then
      echo "### [$(ts)] backfill GEN $SC/$ARM"
      python scripts/eval_all.py --ckpt "$ck" --model-kind "$ARM" \
          --split test --n 256 --steps 200 --bpc-mc 8 \
          --out "$R/$ARM/eval_all.json" > "${LOG}/backfill_eval_${ARM}.log" 2>&1 \
        && echo "    OK -> $R/$ARM/eval_all.json" \
        || echo "    FAILED (see ${LOG}/backfill_eval_${ARM}.log)"
    fi
    if [ "$ck" -nt "$R/$ARM/recovery.json" ]; then
      echo "### [$(ts)] backfill RECOVERY $SC/$ARM"
      python scripts/recovery_check.py --ckpt "$ck" \
          --alphas 0.1,0.3,0.5,0.7,1.0 --n 256 --steps 200 \
          --out "$R/$ARM/recovery.json" > "${LOG}/backfill_recovery_${ARM}.log" 2>&1 \
        && echo "    OK -> $R/$ARM/recovery.json" \
        || echo "    FAILED (see ${LOG}/backfill_recovery_${ARM}.log)"
    fi
  done
done
echo "### BACKFILL COMPLETE $(ts)"
