#!/usr/bin/env bash
#
# Recovery evaluation for the matched MSE-vs-Hilbert ablation
# (sweeps/comp_uniform_gamma.yaml, all four cells at eqm.gamma_power=1.0).
#
# The four cells are already TRAINED and evaluated for generation; this only
# adds the recovery numbers that chapters/results.tex tab:hilbert-null reports
# (acc@.10 / acc@.30 / acc@.50 and Delta@.50).
#
# RUN IT WITH:
#     cd ~/projects/aitchinson-flow && bash scripts/run_matched_recovery.sh
#
# Notes
#   * Needs the GPU free. Nothing else should be training.
#   * Resumable: a cell with runs/<cell>/recovery.json is skipped, so you can
#     Ctrl-C and rerun without losing finished work.
#   * Per-cell stdout goes to runs/<cell>/recovery_check.log.
#   * At the end it writes runs/compu_matched_summary.txt (human readable) and
#     runs/compu_matched_summary.json (for pasting back into the paper work).
#   * Deliberately contains NO `pgrep` wait loop. An earlier version waited on
#     `pgrep -f "run_sweep.py --sweep sweeps/comp_uniform_gamma"`, which
#     self-matched a monitor process holding that same string in its own
#     command line and deadlocked for hours. Do not reintroduce one.
#
set -u

cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"
echo "[recovery] repo: $REPO"
echo "[recovery] started $(date '+%Y-%m-%d %H:%M:%S')"

ALPHAS="0.05,0.1,0.2,0.3,0.4,0.5,1.0"   # same grid as the published recovery.json
CELLS="compu_mse_thick compu_hilbert_thick compu_mse_det compu_hilbert_det"

for cell in $CELLS; do
  ck="runs/$cell/epoch_final.pt"
  if [ ! -f "$ck" ]; then
    echo "[recovery] SKIP $cell — no checkpoint at $ck"
    continue
  fi
  if [ -f "runs/$cell/recovery.json" ]; then
    echo "[recovery] SKIP $cell — recovery.json already present"
    continue
  fi
  echo "[recovery] START $cell  $(date '+%H:%M:%S')"
  PYTHONUNBUFFERED=1 uv run python scripts/recovery_check.py \
      --ckpt "$ck" \
      --alphas "$ALPHAS" \
      --n 256 \
      --steps 200 \
      --seed 42 \
      --out "runs/$cell/recovery.json" \
      > "runs/$cell/recovery_check.log" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "[recovery] DONE  $cell  $(date '+%H:%M:%S')"
  else
    echo "[recovery] FAILED $cell (exit $rc). Last 20 log lines:"
    tail -20 "runs/$cell/recovery_check.log"
  fi
done

echo "[recovery] building summary"

uv run python - <<'PY'
import json, os

CELLS = [
    ("compu_mse_thick",     "thickened",     "MSE"),
    ("compu_hilbert_thick", "thickened",     "Hilbert"),
    ("compu_mse_det",       "deterministic", "MSE"),
    ("compu_hilbert_det",   "deterministic", "Hilbert"),
]
# published (confounded: MSE at gamma_power 1.5, Hilbert at 1.0), for reference
PUBLISHED = [
    ("comp_mse_seed42",      "thickened",     "MSE"),
    ("comp_hilbert_seed42",  "thickened",     "Hilbert"),
    ("comp_ref_det_mse",     "deterministic", "MSE"),
    ("comp_ref_det_hilbert", "deterministic", "Hilbert"),
]

def row(cell):
    out = {"cell": cell}
    ev = f"runs/{cell}/eval.json"
    if os.path.exists(ev):
        e = json.load(open(ev))
        out["KL_uni"] = e.get("unigram_kl")
        out["KL_bi"] = e.get("bigram_kl")
        out["H_ratio"] = e.get("H_ratio")
        out["grad_at_gen"] = e.get("grad_at_gen")
        out["sample0"] = (e.get("samples") or [""])[0][:40]
    rc = f"runs/{cell}/recovery.json"
    if os.path.exists(rc):
        rows = json.load(open(rc)).get("rows", [])
        for r in rows:
            if r.get("mode") != "recovery":
                continue
            a = r.get("alpha")
            out[f"acc@{a}"] = r.get("token_acc")
            out[f"accpert@{a}"] = r.get("token_acc_perturbed")
            if a == 0.5:
                ta, tp = r.get("token_acc"), r.get("token_acc_perturbed")
                if ta is not None and tp is not None:
                    out["delta@.50"] = ta - tp
    return out

def fmt(rows, title):
    L = [title, "-" * len(title)]
    L.append(f"{'cell':22}{'recipe':15}{'loss':9}{'KL_uni':>9}{'KL_bi':>9}"
             f"{'acc@.10':>9}{'acc@.30':>9}{'acc@.50':>9}{'D@.50':>9}")
    for cell, recipe, loss in rows:
        r = row(cell)
        def g(k, w=9, p=4):
            v = r.get(k)
            return f"{v:>{w}.{p}f}" if isinstance(v, float) else f"{'-':>{w}}"
        L.append(f"{cell:22}{recipe:15}{loss:9}"
                 f"{g('KL_uni')}{g('KL_bi')}"
                 f"{g('acc@0.1',9,3)}{g('acc@0.3',9,3)}{g('acc@0.5',9,3)}{g('delta@.50',9,4)}")
    return "\n".join(L)

text = (fmt(CELLS, "MATCHED (new, uniform gamma, seed 42) — use these for tab:hilbert-null")
        + "\n\n"
        + fmt(PUBLISHED, "PUBLISHED (old, confounded: MSE gp=1.5 vs Hilbert gp=1.0)"))

print()
print(text)
open("runs/compu_matched_summary.txt", "w").write(text + "\n")
json.dump({c: row(c) for c, _, _ in CELLS + PUBLISHED},
          open("runs/compu_matched_summary.json", "w"), indent=2)
print("\nwrote runs/compu_matched_summary.txt and runs/compu_matched_summary.json")
PY

echo "[recovery] ALL DONE $(date '+%Y-%m-%d %H:%M:%S')"