#!/usr/bin/env bash
cd /home/renku/work/aitchinson-flow
PID="$(cat runs/_l256_logs/p0_main.pid 2>/dev/null)"
while [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; do sleep 120; done
echo "==== P0 MAIN DRIVER DONE ($(date '+%F %T')) ===="; cat runs/_l256_logs/p0_main_status.txt 2>/dev/null
echo "---- new arm checkpoints ----"; for d in EqMLatent SFLM DirichletFM; do f=runs/sflm_bench_a100_20g_L256/$d/epoch_final.pt; [ -f "$f" ] && echo "  $d ✓" || echo "  $d ✗"; done
echo "---- manifest done count ----"; python -c "import json;rs=[json.loads(l) for l in open('results/manifest.jsonl')];print(' done:',sum(r['status']=='done' for r in rs),'/ total',len(rs))" 2>/dev/null
echo "---- NEXT: bench E4g (watched) ----"
