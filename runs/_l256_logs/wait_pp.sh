#!/usr/bin/env bash
cd /home/renku/work/aitchinson-flow
PID="$(cat runs/_l256_logs/p0_pp.pid 2>/dev/null)"
while [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; do sleep 90; done
echo "==== PER-POSITION OOD DONE ($(date '+%F %T')) ===="; cat runs/_l256_logs/p0_pp_status.txt 2>/dev/null
echo "---- artifacts ----"; ls -la runs/sflm_bench_a100_20g_L256/{DFM,SFLMEBM}/ood_eval.json runs/sflm_bench_a100_20g_L256/DFM/ood_baselines.json 2>/dev/null
