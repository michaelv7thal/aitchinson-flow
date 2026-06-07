#!/usr/bin/env bash
# P0 front-loaded eval driver: run evals on existing seed-42 L256 checkpoints
# via the harness (idempotent skip + manifest records + one internal retry).
# Continue-on-failure; failures are recorded to results/manifest.jsonl.
cd /home/renku/work/aitchinson-flow
LOG=runs/_l256_logs; W=/tmp/l256.sh; STATUS=$LOG/p0_status.txt
stamp(){ date '+%F %T'; }; note(){ echo "[$(stamp)] $*" | tee -a "$STATUS"; }
note "===== P0 front-loaded eval driver start ====="
for e in "E1:DFM:eval" "E4a:SVGP" "E4g:gpt2_bench" "E4c:DFM" "E4d:DFM" \
         "E4b:SFLMEBM" "E4f:hinge_vs_fm" "E1:SFLMEBM:eval" "E1:SFLMEBM_FM:eval"; do
  note "BEGIN $e"
  $W scripts/run_experiment.py "$e" >> "$LOG/p0_evals.log" 2>&1
  note "END $e rc=$?"
done
note "===== P0 front-loaded eval driver DONE ====="
