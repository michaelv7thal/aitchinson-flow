#!/usr/bin/env bash
# Wait for the SFLM+SFM rerun (task b565twiuo) to finish training, then run the
# canonical generation + recovery eval (scripts/eval_generation.py) on whichever
# of the two arms trained successfully. eval_generation.py is the only harness
# that correctly supports SFM (sample()->ids + Fisher-sphere partial-path
# recovery); it also handles SFLM. --no-auto-train so it never retrains.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/renku/work/aitchinson-flow
R=runs/sflm_bench_a100_20g_L256
LOG=$R/_driver
ts(){ date +%Y-%m-%d_%H:%M:%S; }

echo "### EVAL CHAIN START $(ts) — waiting for SFLM+SFM training"
# Poll until each arm has produced either train_meta.json (done) or FAILED.json.
arm_settled(){ [ -f "$R/$1/train_meta.json" ] || [ -f "$R/$1/FAILED.json" ]; }
while ! { arm_settled SFLM && arm_settled SFM; }; do sleep 60; done
echo "### [$(ts)] training settled"

ONLY=""
for A in SFLM SFM; do
  if [ -f "$R/$A/epoch_final.pt" ] && [ -f "$R/$A/train_meta.json" ]; then
    echo "  $A: OK (epoch_final.pt + train_meta.json present)"
    ONLY="${ONLY:+$ONLY,}$A"
  else
    echo "  $A: NOT trained (see $R/$A/FAILED.json) — excluded from eval"
  fi
done
if [ -z "$ONLY" ]; then echo "### no arms to eval — abort"; exit 1; fi

echo "### [$(ts)] STAGE EVAL+RECOVERY: eval_generation.py --only $ONLY"
python scripts/eval_generation.py --scale a100_20g_L256 --no-auto-train \
    --only "$ONLY" --n 256 --steps 200 \
    --out "$R/generation_eval_sflm_sfm.json" \
    > "$LOG/eval_generation_sflm_sfm.log" 2>&1
echo "### EVAL CHAIN COMPLETE exit=$? $(ts) -> $R/generation_eval_sflm_sfm.json"
