#!/bin/bash
cd "$HOME/projects/aitchinson-flow"
exec .venv/bin/python scripts/run_bench_heal.py --only logreg_pl_falseinfo,logreg_pl_plausible
