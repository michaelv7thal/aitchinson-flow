#!/usr/bin/env bash
# Overnight run for the FisherFM benchmark arm (Fisher-Flow, Davis et al. 2024).
# Prepared 2026-08-04. Runs train -> generation scorecard -> recovery ladder,
# then prints the numbers the capstone paper needs. Safe to run unattended.
#
#   bash run_fisher_fm.sh
#
# ~4 h end to end on the RTX PRO 1000 (train ~2 h, evals ~1.5 h). Idempotent:
# the trainer skips FisherFM if epoch_final.pt already exists (add --force to
# retrain). Logs land in $R/FisherFM/.
set -euo pipefail
cd "$(dirname "$0")"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# /workspace/.venv inside the devcontainer; fall back to the repo-local venv so
# the script also works from a shell that is already in the project directory.
PY=/workspace/.venv/bin/python
[ -x "$PY" ] || PY="$PWD/.venv/bin/python"
if [ ! -x "$PY" ]; then
    echo "no usable python: build the venv (uv sync) or run inside the devcontainer" >&2
    exit 1
fi
R=runs/sflm_bench_a100_20g_L256
D=$R/FisherFM
mkdir -p "$D"

echo "=================================================================="
echo " FisherFM overnight run  |  $(date)"
echo " matched budget: a100_20g_L256 (d=1024, 10L, 16H, batch 8, L=256)"
echo "=================================================================="

# 1. train (~2 h) — one seed, matched budget, fp32 (no TF32/bf16/autocast).
echo "[1/3] train …"
$PY scripts/train_for_sflm_bench.py \
    --scale a100_20g_L256 --only FisherFM --epochs 10 --seed 42 \
    2>&1 | tee "$D/train.log"

# 2. generation scorecard -> $D/eval_all.json
echo "[2/3] generation scorecard …"
$PY scripts/eval_all.py \
    --ckpt "$D/epoch_final.pt" --split test --n 256 --steps 200 \
    2>&1 | tee "$D/eval_all.log"

# 3. recovery ladder -> $D/recovery.json  (--out REQUIRED or nothing is written)
echo "[3/3] recovery ladder …"
$PY scripts/recovery_check.py \
    --ckpt "$D/epoch_final.pt" --alphas 0.1,0.3,0.5,0.7,1.0 \
    --n 256 --steps 200 --out "$D/recovery.json" \
    2>&1 | tee "$D/recovery.log"

# 4. print the paper numbers (plan §9)
echo "=================================================================="
echo " FisherFM results for chapters/results.tex"
echo "=================================================================="
$PY - "$D" <<'PYEOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
ev = json.loads((d / "eval_all.json").read_text())
print("--- generation (eval_all.json) ---")
for k in ("KL_uni", "KL_bi", "KL_tri", "H_ratio", "bpc",
          "generation_metric_valid", "collapsed"):
    if k in ev:
        print(f"  {k:24s} = {ev[k]}")
rec = json.loads((d / "recovery.json").read_text())
print("--- recovery (recovery.json) ---")
rows = rec if isinstance(rec, list) else rec.get("rows", rec.get("ladder", []))
for row in (rows if isinstance(rows, list) else []):
    a = row.get("alpha")
    if a is not None and abs(float(a) - 0.5) < 1e-9:
        print(f"  Δ@.50 = {row.get('delta')}   (full row: {row})")
print("\nNOTE: bpc is NOT a real bound (no exact Fisher-Flow likelihood), paper prints an em-rule.")
print("Matched-budget reference is DirichletFM: KL_bi 0.959 / Delta@.50 +0.090.")
print("If FisherFM beats that, STOP and report: it would invalidate a selection claim")
print("the paper already makes. Prior no-smoothing run: KL_bi 1.578 / Delta@.50 -0.016.")
PYEOF
echo "done: $(date)"
