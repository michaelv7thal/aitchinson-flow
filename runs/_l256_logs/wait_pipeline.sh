#!/usr/bin/env bash
cd /home/renku/work/aitchinson-flow
ORCH_PID="${1:-6095}"
while kill -0 "$ORCH_PID" 2>/dev/null; do sleep 120; done
echo "==== TRAINING PIPELINE COMPLETE ($(date '+%F %T')) ===="
echo "---- STATUS.txt ----"; cat runs/_l256_logs/STATUS.txt 2>/dev/null
echo "---- checkpoints ----"
ls -la runs/sflm_bench_a100_20g_L256/*/epoch_final.pt 2>/dev/null
ls -la runs/dfm_svgp_L256/epoch_final.pt 2>/dev/null
