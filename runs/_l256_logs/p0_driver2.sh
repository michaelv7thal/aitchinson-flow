#!/usr/bin/env bash
# Restructured P0 driver: existing-ckpt evals -> 3 arms @5ep -> their evals.
# Bench (E4g) intentionally EXCLUDED (it auto-trains @50ep -> runaway); run watched.
cd /home/renku/work/aitchinson-flow
LOG=runs/_l256_logs; W=/tmp/l256.sh; STATUS=$LOG/p0_status2.txt
stamp(){ date '+%F %T'; }; note(){ echo "[$(stamp)] $*" | tee -a "$STATUS"; }
note "===== P0 driver-2 start (evals on existing ckpts -> train 3 arms @5ep -> eval) ====="
for e in \
  "E4a:sweep" "E4c:DFM" "E4d:DFM" "E4b:SFLMEBM" "E4f:hinge_vs_fm" \
  "E1:EqM:eval" "E1:EqM_OneHot:eval" "E1:DirichletFM:svgpbase_eval" \
  "E1:SFLMEBM:eval" "E1:SFLMEBM_FM:eval" \
  "E1:EqMLatent:train" "E1:SFLM:train" "E1:DirichletFM:train" \
  "E1:EqMLatent:eval" "E1:SFLM:eval" "E1:DirichletFM:eval"; do
  note "BEGIN $e"
  $W scripts/run_experiment.py "$e" >> "$LOG/p0_evals2.log" 2>&1
  note "END $e rc=$?"
done
note "===== P0 driver-2 DONE (bench E4g still to run, watched) ====="
