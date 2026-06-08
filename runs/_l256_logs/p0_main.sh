#!/usr/bin/env bash
cd /home/renku/work/aitchinson-flow
LOG=runs/_l256_logs; W=/tmp/l256.sh; STATUS=$LOG/p0_main_status.txt
stamp(){ date '+%F %T'; }; note(){ echo "[$(stamp)] $*" | tee -a "$STATUS"; }
run(){ note "BEGIN $1 ${2:-}"; $W scripts/run_experiment.py "$1" ${2:-} >> "$LOG/p0_main.log" 2>&1; note "END $1 rc=$?"; }
note "===== P0 main driver start ====="
run "E4c:DFM" "--force"; run "E4b:SFLMEBM" "--force"
run "E1:EqMLatent:train"; run "E1:SFLM:train"; run "E1:DirichletFM:train"
run "E1:EqMLatent:eval"; run "E1:SFLM:eval"; run "E1:DirichletFM:eval"
run "E1:EqM:eval"; run "E1:EqM_OneHot:eval"; run "E1:DirichletFM:svgpbase_eval"
run "E1:SFLMEBM:eval"; run "E1:SFLMEBM_FM:eval"
note "===== P0 main driver DONE (bench E4g next, watched) ====="
