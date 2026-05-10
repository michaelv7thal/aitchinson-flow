#!/usr/bin/env bash
# Chain: wait for cell A's eval.json, run diagnostics on it, train cell B,
# run diagnostics on B. Idempotent — re-running picks up where it left off.
#
# Run from repo root: bash scripts/babysit_cluster_chain.sh

set -e
cd "$(dirname "$0")/.."

unset PYTHONPATH
unset VIRTUAL_ENV
export WANDB_MODE=disabled
PY=.venv/bin/python

CELL_A=latent_cluster_d256_data20k_ep15
CELL_B=latent_cluster_d256_L128_data20k_ep10

echo "[babysit] waiting for runs/${CELL_A}/eval.json"
until [ -f "runs/${CELL_A}/eval.json" ]; do
  sleep 30
done
echo "[babysit] ${CELL_A} eval.json appeared"

CKPT_A=runs/${CELL_A}/epoch_final.pt
if [ -f "${CKPT_A}" ]; then
  if [ ! -f "runs/${CELL_A}/field_probe.json" ]; then
    echo "[babysit] field_probe on ${CELL_A}"
    $PY scripts/field_probe.py \
      --ckpt "${CKPT_A}" --n 64 \
      --out "runs/${CELL_A}/field_probe.json" \
      2>&1 | tee "logs/${CELL_A}_field_probe.log"
  else
    echo "[babysit] field_probe.json already exists for ${CELL_A}"
  fi
  if [ ! -f "runs/${CELL_A}/recovery.json" ]; then
    echo "[babysit] recovery_check on ${CELL_A}"
    $PY scripts/recovery_check.py \
      --ckpt "${CKPT_A}" --n 256 --steps 200 \
      --alphas 0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,1.0 \
      --out "runs/${CELL_A}/recovery.json" \
      2>&1 | tee "logs/${CELL_A}_recovery.log"
  else
    echo "[babysit] recovery.json already exists for ${CELL_A}"
  fi
else
  echo "[babysit] WARNING: ${CKPT_A} missing — skipping diagnostics for ${CELL_A}"
fi

echo "[babysit] training ${CELL_B}"
$PY -u scripts/run_sweep.py \
  --sweep sweeps/latent_cluster.yaml \
  --only "${CELL_B}" \
  --runs-root runs --n 256 --steps 200 \
  2>&1 | tee "logs/${CELL_B}.log"

CKPT_B=runs/${CELL_B}/epoch_final.pt
if [ -f "${CKPT_B}" ]; then
  if [ ! -f "runs/${CELL_B}/field_probe.json" ]; then
    echo "[babysit] field_probe on ${CELL_B}"
    $PY scripts/field_probe.py \
      --ckpt "${CKPT_B}" --n 64 \
      --out "runs/${CELL_B}/field_probe.json" \
      2>&1 | tee "logs/${CELL_B}_field_probe.log"
  fi
  if [ ! -f "runs/${CELL_B}/recovery.json" ]; then
    echo "[babysit] recovery_check on ${CELL_B}"
    $PY scripts/recovery_check.py \
      --ckpt "${CKPT_B}" --n 256 --steps 200 \
      --alphas 0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,1.0 \
      --out "runs/${CELL_B}/recovery.json" \
      2>&1 | tee "logs/${CELL_B}_recovery.log"
  fi
fi

echo "[babysit] DONE"
