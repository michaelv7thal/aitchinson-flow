#!/usr/bin/env bash
cd /home/renku/work/aitchinson-flow
P="$(cat runs/_l256_logs/p0_main.pid 2>/dev/null)"
while [ -n "$P" ] && kill -0 "$P" 2>/dev/null; do sleep 60; done
echo "[finish_dfm] p0_main done; re-running DirichletFM:eval (n_mc fix)"
/tmp/l256.sh scripts/run_experiment.py E1:DirichletFM:eval >> runs/_l256_logs/p0_main.log 2>&1
echo "[finish_dfm] DirichletFM:eval rc=$?"
