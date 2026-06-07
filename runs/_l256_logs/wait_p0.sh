#!/usr/bin/env bash
cd /home/renku/work/aitchinson-flow
PID="$(cat runs/_l256_logs/p0_driver.pid 2>/dev/null)"
while [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; do sleep 120; done
echo "==== P0 EVAL BATCH DONE ($(date '+%F %T')) ===="
echo "---- p0_status ----"; cat runs/_l256_logs/p0_status.txt 2>/dev/null
echo "---- manifest records ----"
python -c "import json;[print(r['exp_id'], r['status'], f\"{r['wall_time_s']:.0f}s\") for r in (json.loads(l) for l in open('results/manifest.jsonl'))]" 2>/dev/null || echo "(no manifest yet)"
echo "---- artifacts written ----"
ls -1 runs/sflm_bench_a100_20g_L256/*/eval_all.json runs/sflm_bench_a100_20g_L256/*/ood_eval.json runs/sflm_bench_a100_20g_L256/DFM_SVGP/*.json runs/sflm_bench_a100_20g_L256/*.json 2>/dev/null
