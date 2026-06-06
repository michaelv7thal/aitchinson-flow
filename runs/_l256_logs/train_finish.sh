#!/usr/bin/env bash
# Finish-up orchestrator: the previous orchestrator (sweep + SFLMEBM_FM) is
# still running and already moved past the FATAL'd SFLMEBM, so re-run SFLMEBM
# here with the val-eval-disabled fix (eval_every>epochs), after the current
# orchestrator finishes (so there's no GPU contention). Same retry/FATAL safety.
set -uo pipefail
cd /home/renku/work/aitchinson-flow
LOG=runs/_l256_logs
W=/tmp/l256.sh
STATUS=$LOG/STATUS.txt
FATAL=$LOG/FATAL
ROOT=runs/sflm_bench_a100_20g_L256
PREV_PID="${1:-}"

stamp(){ date '+%F %T'; }
note(){ echo "[$(stamp)] $*" | tee -a "$STATUS"; }

note "===== FINISH orchestrator: waiting on prev orchestrator pid=$PREV_PID (sweep+SFLMEBM_FM) ====="
if [ -n "$PREV_PID" ]; then
  while kill -0 "$PREV_PID" 2>/dev/null; do sleep 60; done
fi
note "prev orchestrator done; re-running SFLMEBM (val eval disabled)"

train_arm(){
  local name="$1"; shift
  local max=2 rc=1 a
  rm -f "$ROOT/$name/epoch_final.pt"
  for a in $(seq 1 $max); do
    note "BEGIN $name (attempt $a/$max) :: $*"
    "$@" > "$LOG/$name.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then note "END $name OK rc=0 (attempt $a)"; return 0; fi
    cp "$LOG/$name.log" "$LOG/$name.fail$a.log" 2>/dev/null
    note "FAIL $name attempt $a rc=$rc"
    sleep 15
  done
  note "FATAL $name exhausted $max attempts (rc=$rc)"
  echo "FATAL $name rc=$rc $(stamp)" >> "$FATAL"
  return $rc
}

train_arm SFLMEBM $W scripts/train_for_sflm_bench.py --scale a100_20g_L256 --only SFLMEBM --epochs 20

note "===== FINISH orchestrator DONE ====="
