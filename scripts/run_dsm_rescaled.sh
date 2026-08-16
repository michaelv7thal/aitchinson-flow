#!/usr/bin/env bash
#
# Retrain the two denoising-score-matching cells with a sigma ladder matched to
# the CLR feature scale (sweeps/dsm_sigma_rescaled.yaml), then evaluate
# generation and recovery.
#
# The published dsm_clr_{det,dir} cells used dsm.sigma_max = 1.0 against a CLR
# per-position feature norm of 12.27, which under-scales BOTH the recovery test
# (alpha=0.5 perturbs at sigma 6.14, 6x above anything seen in training) and
# generation (annealed Langevin inits inside the data shell). This rerun sets
# sigma_max = 12.3 and n_sigma = 24 and changes nothing else.
#
# RUN IT WITH:
#     cd ~/projects/aitchinson-flow && bash scripts/run_dsm_rescaled.sh
#
#   * Needs the GPU free. Run scripts/run_matched_recovery.sh first if that has
#     not finished; do not run the two at the same time (8 GB card).
#   * Resumable: cells with eval.json are skipped by the sweep runner, and
#     cells with recovery.json are skipped below.
#   * Writes runs/dsmx_summary.txt and runs/dsmx_summary.json at the end.
#   * Contains no `pgrep` wait loop, deliberately: an earlier version of the
#     companion script waited on a pattern that self-matched a monitor process
#     holding the same string, and deadlocked. Do not add one.
#
set -u

cd "$(dirname "$0")/.." || exit 1
echo "[dsmx] repo: $(pwd)"
echo "[dsmx] started $(date '+%Y-%m-%d %H:%M:%S')"

echo "[dsmx] === stage 1/2: train + evaluate generation ==="
uv run python scripts/run_sweep.py \
    --sweep sweeps/dsm_sigma_rescaled.yaml \
    --n 256 --steps 200
echo "[dsmx] sweep exit=$?"

echo "[dsmx] === stage 2/2: recovery ==="
ALPHAS="0.05,0.1,0.2,0.3,0.4,0.5,1.0"
for cell in dsmx_clr_det dsmx_clr_dir; do
  ck="runs/$cell/epoch_final.pt"
  if [ ! -f "$ck" ]; then
    echo "[dsmx] SKIP $cell — no checkpoint"
    continue
  fi
  if [ -f "runs/$cell/recovery.json" ]; then
    echo "[dsmx] SKIP $cell — recovery.json present"
    continue
  fi
  echo "[dsmx] START recovery $cell  $(date '+%H:%M:%S')"
  PYTHONUNBUFFERED=1 uv run python scripts/recovery_check.py \
      --ckpt "$ck" --alphas "$ALPHAS" --n 256 --steps 200 --seed 42 \
      --out "runs/$cell/recovery.json" \
      > "runs/$cell/recovery_check.log" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "[dsmx] DONE recovery $cell  $(date '+%H:%M:%S')"
  else
    echo "[dsmx] FAILED recovery $cell (exit $rc). Last 20 lines:"
    tail -20 "runs/$cell/recovery_check.log"
  fi
done

echo "[dsmx] building summary"
uv run python - <<'PY'
import json, os, math

ROWS = [
    ("dsmx_clr_det",  "deterministic", "sigma_max=12.3 (NEW)"),
    ("dsmx_clr_dir",  "thickened",     "sigma_max=12.3 (NEW)"),
    ("dsm_clr_ablation/dsm_clr_det", "deterministic", "sigma_max=1.0 (published)"),
    ("dsm_clr_ablation/dsm_clr_dir", "thickened",     "sigma_max=1.0 (published)"),
]

def read(cell):
    out = {"cell": cell}
    ev = f"runs/{cell}/eval.json"
    if os.path.exists(ev):
        e = json.load(open(ev))
        out["KL_uni"] = e.get("unigram_kl")
        out["KL_bi"] = e.get("bigram_kl")
        out["H_gen"] = e.get("H_gen")
        out["H_ratio"] = e.get("H_ratio")
        out["sample0"] = (e.get("samples") or [""])[0][:40]
    rc = f"runs/{cell}/recovery.json"
    if os.path.exists(rc):
        for r in json.load(open(rc)).get("rows", []):
            if r.get("mode") != "recovery":
                continue
            a, ta, tp = r.get("alpha"), r.get("token_acc"), r.get("token_acc_perturbed")
            if a is not None:
                out[f"acc@{a}"] = ta
            if a == 0.5 and ta is not None and tp is not None:
                out["delta@.50"] = ta - tp
    return out

L = []
L.append("DSM sigma-ladder rescaling — uniform ln(27) = %.4f is pure noise" % math.log(27))
L.append("")
L.append(f"{'cell':34}{'recipe':15}{'ladder':26}{'KL_uni':>9}{'KL_bi':>9}{'H_gen':>8}{'acc@.50':>9}{'D@.50':>9}")
data = {}
for cell, recipe, ladder in ROWS:
    r = read(cell); data[cell] = r
    def g(k, w=9, p=4):
        v = r.get(k)
        return f"{v:>{w}.{p}f}" if isinstance(v, float) else f"{'-':>{w}}"
    L.append(f"{cell:34}{recipe:15}{ladder:26}{g('KL_uni')}{g('KL_bi')}{g('H_gen',8,4)}"
             f"{g('acc@0.5',9,3)}{g('delta@.50')}")
L.append("")
L.append("samples:")
for cell, _, ladder in ROWS:
    L.append(f"  {cell:34} {ladder:26} {data[cell].get('sample0','-')}")
L.append("")
L.append("READ IT LIKE THIS:")
L.append("  H_gen near 3.296 means the output is still indistinguishable from uniform noise,")
L.append("  i.e. the rescaled ladder did NOT buy generation and the paper's claim stands.")
L.append("  A clearly lower H_gen with a lower KL_bi means the rescaled DSM cells DO generate,")
L.append("  which would need conclusion.tex:70 and the pointwise-vs-distributional cut revisited.")
L.append("  Delta@.50 is now measured inside the trained noise range, so it is informative")
L.append("  either way and the caveat sentence in results.tex can go.")

text = "\n".join(L)
print()
print(text)
open("runs/dsmx_summary.txt", "w").write(text + "\n")
json.dump(data, open("runs/dsmx_summary.json", "w"), indent=2)
print("\nwrote runs/dsmx_summary.txt and runs/dsmx_summary.json")
PY

echo "[dsmx] ALL DONE $(date '+%Y-%m-%d %H:%M:%S')"
