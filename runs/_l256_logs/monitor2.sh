#!/usr/bin/env bash
# Race-free pipeline monitor (v2). Re-invokes the Claude session on EITHER:
#   - FATAL marker file present (an arm exhausted its retries)  -> exit 2
#   - orchestrator pid gone (all arms attempted, pipeline done) -> exit 0
# No log-grepping, so it never false-fires on a transient failure that the
# orchestrator is about to retry.
cd /home/renku/work/aitchinson-flow
LOG=runs/_l256_logs
ORCH_PID="$(cat $LOG/orch.pid 2>/dev/null)"
FATAL=$LOG/FATAL
inventory(){
  echo "---- STATUS (tail) ----"; tail -25 $LOG/STATUS.txt 2>/dev/null
  echo "---- checkpoints ----"
  ls -la runs/sflm_bench_a100_20g_L256/*/epoch_final.pt runs/dfm_svgp_L256/epoch_final.pt 2>/dev/null
}
while true; do
  if [ -f "$FATAL" ]; then
    echo "==== RESULT=FATAL $(date '+%F %T') ===="; cat "$FATAL"; inventory; exit 2
  fi
  if [ -n "$ORCH_PID" ] && ! kill -0 "$ORCH_PID" 2>/dev/null; then
    echo "==== RESULT=COMPLETE $(date '+%F %T') ===="; inventory; exit 0
  fi
  sleep 60
done
