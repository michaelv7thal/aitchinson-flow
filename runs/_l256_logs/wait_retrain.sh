#!/usr/bin/env bash
cd /home/renku/work/aitchinson-flow
P="$(cat runs/_l256_logs/retrain_sflm.pid 2>/dev/null)"
while [ -n "$P" ] && kill -0 "$P" 2>/dev/null; do sleep 90; done
echo "==== SFLM RETRAIN DONE ($(date '+%F %T')) ===="; cat runs/_l256_logs/retrain_status.txt 2>/dev/null
echo "---- SFLM KL_bi before(1.33) vs after ----"
python -c "import json;d=json.load(open('runs/sflm_bench_a100_20g_L256/SFLM/eval_all.json'));print('  SFLM KL_bi =',round(d.get('KL_bi'),3),' collapsed =',d.get('collapsed'))" 2>/dev/null
