#!/usr/bin/env bash
# Run after cell B (L=128) training finishes:
#   1. Reduced-n recovery_check on cell A (cell A field_probe.json already exists).
#   2. Field_probe on cell B.
#   3. Reduced-n recovery_check on cell B.
#
# Idempotent — skips any artifact that already exists. Uses n=128 / steps=100
# to keep runtime manageable for the 11-alpha sweep (n=256 / steps=200 would
# be ~2 h per cell).

set -e
cd "$(dirname "$0")/.."

unset PYTHONPATH
unset VIRTUAL_ENV
export WANDB_MODE=disabled
PY=.venv/bin/python

CELL_A=latent_cluster_d256_data20k_ep15
CELL_B=latent_cluster_d256_L128_data20k_ep10

ALPHAS=0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,1.0
N=128
STEPS=100

echo "[diag] waiting for runs/${CELL_B}/eval.json"
until [ -f "runs/${CELL_B}/eval.json" ]; do
  sleep 30
done
echo "[diag] ${CELL_B} eval.json appeared"

# Cell A — only recovery (field_probe already exists)
CKPT_A=runs/${CELL_A}/epoch_final.pt
if [ -f "${CKPT_A}" ] && [ ! -f "runs/${CELL_A}/recovery.json" ]; then
  echo "[diag] recovery_check on ${CELL_A} (n=$N, steps=$STEPS)"
  $PY scripts/recovery_check.py \
    --ckpt "${CKPT_A}" --n "$N" --steps "$STEPS" \
    --alphas "$ALPHAS" \
    --out "runs/${CELL_A}/recovery.json" \
    2>&1 | tee "logs/${CELL_A}_recovery.log"
fi

# Cell B — both
CKPT_B=runs/${CELL_B}/epoch_final.pt
if [ -f "${CKPT_B}" ]; then
  if [ ! -f "runs/${CELL_B}/field_probe.json" ]; then
    echo "[diag] field_probe on ${CELL_B}"
    $PY scripts/field_probe.py \
      --ckpt "${CKPT_B}" --n 64 \
      --out "runs/${CELL_B}/field_probe.json" \
      2>&1 | tee "logs/${CELL_B}_field_probe.log"
  fi
  if [ ! -f "runs/${CELL_B}/recovery.json" ]; then
    echo "[diag] recovery_check on ${CELL_B} (n=$N, steps=$STEPS)"
    $PY scripts/recovery_check.py \
      --ckpt "${CKPT_B}" --n "$N" --steps "$STEPS" \
      --alphas "$ALPHAS" \
      --out "runs/${CELL_B}/recovery.json" \
      2>&1 | tee "logs/${CELL_B}_recovery.log"
  fi
fi

echo "[diag] DONE"
