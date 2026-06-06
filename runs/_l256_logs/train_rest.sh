#!/usr/bin/env bash
# Serial training orchestrator for the L256 workstream (arms 2..5).
# Waits for the already-running DFM job to finish (no GPU contention),
# then trains each remaining arm continue-on-error, logging per-arm and
# appending to STATUS.txt. Launch in background; monitor STATUS.txt.
set -uo pipefail
cd /home/renku/work/aitchinson-flow
LOG=runs/_l256_logs
W=/tmp/l256.sh
STATUS=$LOG/STATUS.txt
DFM_PID="${1:-}"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
note()  { echo "[$(stamp)] $*" | tee -a "$STATUS"; }

note "orchestrator start (waiting on DFM pid=$DFM_PID)"
# Wait for the DFM job to exit before using the GPU.
if [ -n "$DFM_PID" ]; then
  while kill -0 "$DFM_PID" 2>/dev/null; do sleep 60; done
fi
note "DFM job no longer running; checking checkpoint"
if [ -f runs/sflm_bench_a100_20g_L256/DFM/epoch_final.pt ]; then
  note "DFM epoch_final.pt present OK"
else
  note "WARNING: DFM epoch_final.pt MISSING (continuing anyway)"
fi

run_step() {
  local name="$1"; shift
  note "BEGIN $name :: $*"
  "$@" > "$LOG/$name.log" 2>&1
  local rc=$?
  note "END   $name rc=$rc"
}

# Arm order: SFLMEBM (energy-OOD headline) -> DFM_SVGP sweep (hinge base,
# lean 10k) -> SFLMEBM_FM. All at the 127M (d1024/10L/16H) shared scale.
# DirichletFM (optional generator) dropped to keep the total near ~2 days;
# the bench/gen scripts handle its absence gracefully (skipped arm).
run_step SFLMEBM    $W scripts/train_for_sflm_bench.py --scale a100_20g_L256 --only SFLMEBM    --epochs 20
run_step DFM_SVGP_SWEEP $W scripts/run_sweep.py --sweep sweeps/_dfm_svgp_L256.yaml --runs-root runs --n 128
run_step SFLMEBM_FM $W scripts/train_for_sflm_bench.py --scale a100_20g_L256 --only SFLMEBM_FM --epochs 20

note "orchestrator DONE"
