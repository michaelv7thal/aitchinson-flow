#!/usr/bin/env bash
# Smart pipeline monitor for the L256 workstream. Exits (which re-invokes the
# Claude session via the harness) on EITHER:
#   - normal completion  (orchestrator pid gone)            -> exit 0
#   - DFM crash          (DFM pid gone, no epoch_final.pt)  -> exit 1
#   - hard error in any arm log (CUDA OOM / Traceback)      -> exit 2
# Prints a clear reason marker + diagnostics so the next session turn knows
# what happened and whether to fix-and-relaunch or proceed to evals.
cd /home/renku/work/aitchinson-flow
LOG=runs/_l256_logs
ORCH_PID="$(cat $LOG/orch.pid 2>/dev/null)"
DFM_PID="$(cat $LOG/DFM.pid 2>/dev/null)"
CKPT_DFM=runs/sflm_bench_a100_20g_L256/DFM/epoch_final.pt
ARM_LOGS="$LOG/DFM.log $LOG/SFLMEBM.log $LOG/SFLMEBM_FM.log $LOG/DFM_SVGP_SWEEP.log"

inventory() {
  echo "---- STATUS.txt ----"; cat $LOG/STATUS.txt 2>/dev/null
  echo "---- checkpoints ----"
  ls -la runs/sflm_bench_a100_20g_L256/*/epoch_final.pt runs/dfm_svgp_L256/epoch_final.pt 2>/dev/null
}

while true; do
  # 1) normal completion
  if [ -n "$ORCH_PID" ] && ! kill -0 "$ORCH_PID" 2>/dev/null; then
    echo "==== RESULT=COMPLETE $(date '+%F %T') ===="
    inventory
    exit 0
  fi
  # 2) DFM exited without producing a checkpoint -> crash (fit() writes the
  #    ckpt before the process exits, so pid-gone + no-ckpt == failure)
  if [ -n "$DFM_PID" ] && ! kill -0 "$DFM_PID" 2>/dev/null && [ ! -f "$CKPT_DFM" ]; then
    echo "==== RESULT=DFM_CRASH $(date '+%F %T') ===="
    tail -30 $LOG/DFM.log 2>/dev/null
    exit 1
  fi
  # 3) hard crash signature in any arm log
  hit="$(grep -lE 'CUDA out of memory|OutOfMemoryError|Traceback \(most recent call last\)' $ARM_LOGS 2>/dev/null)"
  if [ -n "$hit" ]; then
    echo "==== RESULT=ERROR_SIG $(date '+%F %T') :: $hit ===="
    for f in $hit; do echo "---- tail $f ----"; tail -20 "$f"; done
    inventory
    exit 2
  fi
  sleep 60
done
