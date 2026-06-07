#!/usr/bin/env bash
cd /home/renku/work/aitchinson-flow
PID="$(cat runs/_l256_logs/p0_driver2.pid 2>/dev/null)"
while [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; do sleep 120; done
echo "==== P0 DRIVER-2 DONE ($(date '+%F %T')) ===="
echo "---- status ----"; cat runs/_l256_logs/p0_status2.txt 2>/dev/null
echo "---- manifest ----"
python -c "import json;[print(' ',r['exp_id'],r['status'],f\"{r['wall_time_s']:.0f}s\") for r in (json.loads(l) for l in open('results/manifest.jsonl'))]" 2>/dev/null
echo "---- still TODO: E4g bench (watched) ----"
