#!/usr/bin/env bash
# Per-position OOD (the L-robust path), hardened+forced re-run (prev silent-empty).
cd /home/renku/work/aitchinson-flow
LOG=runs/_l256_logs; W=/tmp/l256.sh; STATUS=$LOG/p0_pp_status.txt
stamp(){ date '+%F %T'; }; note(){ echo "[$(stamp)] $*" | tee -a "$STATUS"; }
note "===== per-position OOD driver start ====="
for e in "E4c:DFM" "E4b:SFLMEBM" "E4d:DFM"; do
  note "BEGIN $e"; $W scripts/run_experiment.py "$e" --force >> "$LOG/p0_pp.log" 2>&1; note "END $e rc=$?"
done
note "===== per-position OOD driver DONE ====="
