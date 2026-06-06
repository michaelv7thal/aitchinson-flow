#!/usr/bin/env bash
# Resume orchestrator after the SFLMEBM NVML-allocator crash (DFM already done).
# Runs SFLMEBM -> SVGP sweep -> SFLMEBM_FM, each with up to 2 attempts. On a
# clean success it moves on; if an arm exhausts its retries it writes an
# explicit FATAL marker (which the monitor watches) and continues to the next
# arm so independent work still completes. Uses the fixed wrapper (no
# expandable_segments -> no MIG NVML assert).
set -uo pipefail
cd /home/renku/work/aitchinson-flow
LOG=runs/_l256_logs
W=/tmp/l256.sh
STATUS=$LOG/STATUS.txt
FATAL=$LOG/FATAL
ROOT=runs/sflm_bench_a100_20g_L256

stamp(){ date '+%F %T'; }
note(){ echo "[$(stamp)] $*" | tee -a "$STATUS"; }

rm -f "$FATAL"
note "===== RESUME orchestrator start (allocator fixed; DFM complete) ====="
if [ -f "$ROOT/DFM/epoch_final.pt" ]; then note "DFM ckpt present OK"; else note "WARNING: DFM ckpt missing"; fi

# Training arm, success := rc==0 (train_for_sflm_bench has no post-fit eval).
train_arm(){
  local name="$1"; shift
  local max=2 rc=1 a
  rm -f "$ROOT/$name/epoch_final.pt"   # drop any stale partial
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

# SVGP sweep, success := epoch_final.pt exists (hinge needs only the generator
# weights; a trailing sample-eval crash is non-fatal for our purpose).
sweep_arm(){
  local max=2 rc=1 a
  for a in $(seq 1 $max); do
    note "BEGIN DFM_SVGP_SWEEP (attempt $a/$max)"
    rm -f runs/dfm_svgp_L256/eval.json   # defeat idempotent skip on a retrain
    "$@" > "$LOG/DFM_SVGP_SWEEP.log" 2>&1
    rc=$?
    if [ -f runs/dfm_svgp_L256/epoch_final.pt ]; then
      note "END DFM_SVGP_SWEEP OK ckpt present (attempt $a, rc=$rc)"; return 0
    fi
    cp "$LOG/DFM_SVGP_SWEEP.log" "$LOG/DFM_SVGP_SWEEP.fail$a.log" 2>/dev/null
    note "FAIL DFM_SVGP_SWEEP attempt $a rc=$rc (no ckpt)"
    sleep 15
  done
  note "FATAL DFM_SVGP_SWEEP exhausted $max attempts (rc=$rc)"
  echo "FATAL DFM_SVGP_SWEEP rc=$rc $(stamp)" >> "$FATAL"
  return $rc
}

train_arm SFLMEBM    $W scripts/train_for_sflm_bench.py --scale a100_20g_L256 --only SFLMEBM    --epochs 20
sweep_arm            $W scripts/run_sweep.py --sweep sweeps/_dfm_svgp_L256.yaml --runs-root runs --n 128
train_arm SFLMEBM_FM $W scripts/train_for_sflm_bench.py --scale a100_20g_L256 --only SFLMEBM_FM --epochs 20

note "===== RESUME orchestrator DONE ====="
