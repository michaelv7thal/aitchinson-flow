#!/usr/bin/env bash
# Retrain SFLM with the schedule fix (uniform, alpha_hi=1.0) after the current
# generation evals free the GPU, then re-eval. force=re-run despite manifest.
cd /home/renku/work/aitchinson-flow
LOG=runs/_l256_logs; W=/tmp/l256.sh; STATUS=$LOG/retrain_status.txt
stamp(){ date '+%F %T'; }; note(){ echo "[$(stamp)] $*" | tee -a "$STATUS"; }
note "retrain_sflm: waiting for current evals (p0_main 1096391 + finish_dfm 1202210)"
for p in 1096391 1202210; do while kill -0 "$p" 2>/dev/null; do sleep 60; done; done
note "GPU free; retraining SFLM (uniform schedule, alpha_hi=1.0)"
note "BEGIN SFLM:train"; $W scripts/run_experiment.py E1:SFLM:train --force >> "$LOG/retrain.log" 2>&1; note "END SFLM:train rc=$?"
note "BEGIN SFLM:eval";  $W scripts/run_experiment.py E1:SFLM:eval  --force >> "$LOG/retrain.log" 2>&1; note "END SFLM:eval rc=$?"
note "retrain_sflm DONE"
