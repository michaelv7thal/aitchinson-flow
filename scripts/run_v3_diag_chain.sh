#!/usr/bin/env bash
# Wait for the in-flight v3 orchestrator to finish, then relaunch
# scripts/run_ae_scaling.py on the same sweep. The second pass skips
# every cell whose eval.json already exists (z128, vae, z256) and runs
# only the appended diag cell (ae_d1024_l8_z128_v3_diag), which exercises
# the new r<.33 / r<.66 / r<1 echo-trap diagnostic added to eqm.py.
#
# Launch:
#   nohup bash scripts/run_v3_diag_chain.sh >/dev/null 2>&1 < /dev/null &

set -u
cd /home/renku/work/aitchinson-flow

PARENT_PID=${PARENT_PID:-257714}
LOG=runs/_logs/ae_long_large_v3.log
CHAIN_LOG=runs/_logs/ae_long_large_v3_diag_chain.log
PY=.venv/bin/python

mkdir -p runs/_logs
{
  echo "[chain] $(date '+%F %T') waiting for parent pid=$PARENT_PID"
} >> "$CHAIN_LOG"

while kill -0 "$PARENT_PID" 2>/dev/null; do sleep 60; done

{
  echo "[chain] $(date '+%F %T') parent exited, second pass: diag cell only"
} >> "$CHAIN_LOG"

echo "[chain-relaunch] $(date '+%F %T') second pass for diag cell" >> "$LOG"

env -u PYTHONPATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" scripts/run_ae_scaling.py \
  --sweep sweeps/ae_long_large_v3.yaml \
  --runs-root runs \
  >> "$LOG" 2>&1

{
  echo "[chain] $(date '+%F %T') second pass exit=$?"
} >> "$CHAIN_LOG"
