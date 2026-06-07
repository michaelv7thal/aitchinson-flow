#!/usr/bin/env bash
# Fire when SFLMEBM epoch 15 finishes (epoch 16 appears in the log => epoch_final.pt
# saved at epoch 15) OR if the SFLMEBM process exits unexpectedly first.
cd /home/renku/work/aitchinson-flow
SFLM_PID="$(cat runs/_l256_logs/sflmebm.pid 2>/dev/null)"
LOG=runs/_l256_logs/SFLMEBM.log
while true; do
  if grep -aq "epoch: 16:" "$LOG" 2>/dev/null; then
    echo "RESULT=EPOCH15_DONE  ($(date '+%F %T'))  epoch16 started => epoch_final.pt = epoch15"; break
  fi
  if [ -n "$SFLM_PID" ] && ! kill -0 "$SFLM_PID" 2>/dev/null; then
    echo "RESULT=SFLMEBM_EXITED_EARLY ($(date '+%F %T'))  -- check rc / orchestrator retry"; break
  fi
  sleep 30
done
echo "--- epoch_final.pt now ---"
ls -l --time-style=+%H:%M:%S runs/sflm_bench_a100_20g_L256/SFLMEBM/epoch_final.pt 2>/dev/null
