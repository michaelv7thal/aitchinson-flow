#!/usr/bin/env bash
# Relaunch of the 50k DirichletFM eval sweep (interrupted right after the
# bigdfm driver echoed "generation eval"; training itself finished, epoch_final
# .pt exists). Mirrors drive_bigdfm_50ep.sh lines 33-42 exactly so the outputs
# are byte-identical to what the original pipeline would have produced.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# torch lives in the buildpack site-packages (NOT the venv); the ambient
# PYTHONPATH surfaces it. A detached non-interactive shell does not inherit the
# profile, so replicate the fulltext8 driver's PYTHONPATH verbatim (+ src).
export PYTHONPATH="/layers/paketo-buildpacks_pip-install/packages/lib/python3.11/site-packages:/layers/paketo-buildpacks_pip/pip/lib/python3.11/site-packages:/layers/paketo-buildpacks_cpython/cpython:src"
cd /home/renku/work/aitchinson-flow

R=runs/sflm_bench_a100_20g_L256_d1280L14_b16
ARM=DirichletFM_ep50_d50k
LOG="$R/_driver"
CKPT="$R/$ARM/epoch_final.pt"
ts(){ date +%Y-%m-%d_%H:%M:%S; }

echo "### EVAL-RELAUNCH START $(ts)  ckpt=$CKPT"

echo "### [$(ts)] generation eval"
python scripts/eval_all.py --ckpt "$CKPT" --model-kind DirichletFM \
    --split test --n 256 --steps 200 --bpc-mc 8 \
    --out "$R/$ARM/eval_all.json" > "$LOG/eval_${ARM}.log" 2>&1
echo "### [$(ts)] eval_all exit=$?"
python -c "import json;p='$R/$ARM/eval_all.json';d=json.load(open(p));d['model_name']='DirichletFM_d1280L14_ep50_d50k_b16';json.dump(d,open(p,'w'),indent=2)" 2>/dev/null || true

echo "### [$(ts)] recovery sweep"
python scripts/recovery_check.py --ckpt "$CKPT" \
    --alphas 0.1,0.3,0.5,0.7,1.0 --n 256 --steps 200 \
    --out "$R/$ARM/recovery.json" > "$LOG/recovery_${ARM}.log" 2>&1
echo "### [$(ts)] recovery exit=$?"

# Honest completion marker: only if BOTH eval artifacts were actually produced.
# This unblocks the fulltext8 driver's gate fallback (hinge driver not running
# AND this marker present) without re-running the skipped hinge sweeps.
if [ -s "$R/$ARM/eval_all.json" ] && [ -s "$R/$ARM/recovery.json" ]; then
  echo "### BIGDFM-50EP COMPLETE $(ts) — eval re-run after session interruption" \
      >> "$LOG/bigdfm_50ep.out"
  echo "### EVAL-RELAUNCH COMPLETE $(ts) — metrics in $R/$ARM/eval_all.json + recovery.json"
else
  echo "### EVAL-RELAUNCH FAILED $(ts) — missing eval_all.json/recovery.json; see $LOG/eval_${ARM}.log + recovery_${ARM}.log"
fi
