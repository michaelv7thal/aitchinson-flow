#!/usr/bin/env bash
# ============================================================================
#  Band-EqM at FULL scale: can it beat the full-corpus Dirichlet FM?
# ============================================================================
#
#  Same recipe as drive_band_L256.sh, scaled to the model it is chasing:
#  the full-text8 Dirichlet FM at d1280/14L, L=256, whole afmck/text8 split
#  (runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM_converge, ~28
#  epochs across resumed legs).
#
#  Cell: band_L256_ep30_full_d1280L14 in sweeps/band_L256.yaml
#        d1280/14L/16H · B=8 · full split (~351.5k windows/epoch) · 30 epochs
#
#  What we are trying to beat, on KL_bi (lower is better):
#        Dirichlet FM, matched budget          0.959
#        EqM det-CLR, whole path, L=256        1.582
#        this band recipe at d1024/10L, 10k×10ep   0.265   <- the dev result
#
#  Writes to its OWN folder ($ROOT, default runs_band_cloud/) so nothing
#  collides with the laptop runs under runs/.
#
#  ---------------------------------------------------------------------------
#  USAGE
#    ./drive_band_cloud.sh              # the full 30-epoch run
#    ./drive_band_cloud.sh summary      # re-print the summary from the JSONs
#    SKIP=recovery,ood ./drive_band_cloud.sh     # stage 1 only
#
#  Knobs: ROOT CELL B LR GRAD_CKPT N STEPS ALPHAS SKIP
#    B=16 (default) or 32 if the card has the headroom; LR defaults to
#    3e-4*sqrt(B/8), sqrt-scaled from the dev recipe. GRAD_CKPT=0 drops
#    activation checkpointing (~30% faster) when memory allows.
#    Batch is baked into the run directory name (..._b16), so runs at
#    different batches never collide or reuse each other's checkpoints.
#
#  WALL CLOCK. The full split is ~351,500 windows at L=256, so one epoch is
#  ~22,000 optimizer steps at B=16 (~11,000 at B=32), and 30 epochs is ~660k
#  (~330k) steps. EqM's conservative gradient is second-order, so a step costs
#  more than the first-order Dirichlet FM this is chasing. Still budget days,
#  and expect to run it in legs the way the Dirichlet FM run itself was.
#
#  NO RESUME. run_sweep.py forces checkpoint_every=10^9 (only epoch_final.pt is
#  kept, and it carries no optimizer state), so an interrupted run restarts from
#  zero. Worse, run_sweep SKIPS training when epoch_final.pt exists -- and that
#  file is rewritten every epoch as crash insurance -- so re-running after a
#  crash would silently evaluate a PARTIAL model. If this run is interrupted,
#  delete or move $ROOT/$CELL before relaunching. Ask for real resume support if
#  the run needs to survive interruption; it is a small change to run_sweep.py.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# Cluster environment (mirrors drive_fulltext8_converge.sh). Harmless locally:
# the PYTHONPATH entries simply do not exist off the cluster.
export PYTHONPATH="/layers/paketo-buildpacks_pip-install/packages/lib/python3.11/site-packages:/layers/paketo-buildpacks_pip/pip/lib/python3.11/site-packages:/layers/paketo-buildpacks_cpython/cpython:src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset WANDB_PROJECT WANDB_ENTITY WANDB_NAME WANDB_RUN_GROUP
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true

MODE="${1:-run}"
ROOT="${ROOT:-runs_band_cloud}"
CELL="${CELL:-band_L256_ep30_full_d1280L14}"
N="${N:-256}"
STEPS="${STEPS:-400}"
# Batch. The dev cell used B=8 because the repo's scale table sizes the 20 GB
# MIG for second-order arms; on a bigger card 16 (or 32) fits. Bigger batch =
# proportionally fewer optimizer steps, which is the main wall-clock lever here.
B="${B:-16}"
# LR, sqrt-scaled from the dev recipe (3e-4 @ B=8) for the fewer updates. This
# is the repo's own convention -- drive_fulltext8_converge.sh scales the same
# way (3e-4*sqrt(B/16) off its b16 base). Set LR explicitly to override.
LR="${LR:-auto}"
# Activation checkpointing: needed at d1280/14L on a 20 GB slice, ~30% slower.
# With real headroom, GRAD_CKPT=0 buys that back.
GRAD_CKPT="${GRAD_CKPT:-1}"
ALPHAS="${ALPHAS:-0.3 0.5 0.6 0.7 0.8 1.0}"
SKIP="${SKIP:-}"
BASE_SWEEP=sweeps/band_L256.yaml
PY="uv run python"

if [ "$LR" = "auto" ]; then
  LR=$($PY -c "import math;print(f'{3e-4*math.sqrt($B/8):.6g}')")
fi
# Batch is part of the run's identity, so it is part of the directory name --
# a b16 run must never silently reuse a b8 run's checkpoint.
CELL_RUN="${CELL}_b${B}"
RUN_DIR="$ROOT/$CELL_RUN"
SWEEP="$ROOT/_sweep_${CELL_RUN}.yaml"

skipped () { case ",$SKIP," in *",$1,"*) return 0;; *) return 1;; esac; }
have ()    { [ -s "$1" ]; }
ts ()      { date +%Y-%m-%d_%H:%M:%S; }

# Rescale factors, copied verbatim from drive_band_L256.sh: g(α) = |x_γ|/|z_α|
# with |x_γ|=0.539 and |z_α|=12.27·sqrt(1+Kα²), K=27. Per position, so they are
# independent of L AND of model size -- the same table the dev run used. Stage 2
# prints embed_norm_mean; if it is not ~12.27 these are stale.
g_for_alpha () {
  case "$1" in
    0.1) echo 0.0390 ;;
    0.3) echo 0.0237 ;;
    0.5) echo 0.0160 ;;
    0.6) echo 0.01342 ;;
    0.7) echo 0.01165 ;;
    0.8) echo 0.01027 ;;
    1.0) echo 0.0083 ;;
    *)   echo "no rescale factor tabulated for alpha=$1" >&2; return 1 ;;
  esac
}
tag_for_alpha () { echo "${1/./}"; }

summary () {
$PY - "$RUN_DIR" <<'PYEOF'
import json, sys
from pathlib import Path
band = Path(sys.argv[1])
print("\n" + "=" * 78)
print("1. GENERATION  (lower is better)")
print("=" * 78)
ev = band / "eval.json"
if ev.is_file():
    d = json.loads(ev.read_text())
    print(f"{'THIS RUN  ' + band.name:<44} "
          f"{d['unigram_kl']:>6.3f} {d['bigram_kl']:>6.3f} {d['trigram_kl']:>6.3f}"
          f"   H_ratio {d.get('H_ratio', float('nan')):.3f}   ep {d.get('epoch')}")
    for s in (d.get("samples") or [])[:3]:
        print(f"    {s[:96]!r}")
else:
    print(f"  (no {ev} yet)")
print(f"{'':<44} {'KL_uni':>6} {'KL_bi':>6} {'KL_tri':>6}")
for name, u, b, t in (
    ("band d1024/10L, 10k x 10ep (the dev result)", 0.010, 0.265, 1.744),
    ("tab:gen  Dirichlet FM, matched budget      ", 0.009, 0.959, 4.088),
    ("tab:gen  EqM det-CLR, whole path, L=256    ", 0.018, 1.582, 5.042),
    ("tab:gen  Discrete FM (collapsed flag)      ", 0.001, 1.466, 5.141),
):
    print(f"{name:<44} {u:>6.3f} {b:>6.3f} {t:>6.3f}")

print("\n" + "=" * 78)
print("2. RECOVERY, input rescaled into the band  (Delta vs the unrescaled argmax)")
print("=" * 78)
A = ("0.3", "0.5", "0.6", "0.7", "0.8", "1.0")
TAG = {a: a.replace(".", "") for a in A}
print(f"{'':<38} " + " ".join(f"{'a=' + a:>9}" for a in A))
def row(label, vals):
    print(f"{label:<38} " + " ".join(
        (f"{vals[a]:>+9.3f}" if vals.get(a) is not None else f"{'—':>9}") for a in A))
meas = {}
for a in A:
    f = band / f"recovery_bandmap_a{TAG[a]}.json"
    if f.is_file():
        meas[a] = json.loads(f.read_text())["rows"][0]["delta"]
row(f"THIS RUN  {band.name}", meas)
for label, vals in (
    ("band d1024/10L 10k x 10ep (dev)", {"0.3": 0.040, "0.5": 0.122, "0.6": 0.123,
                                         "0.7": 0.117, "0.8": 0.109, "1.0": 0.098}),
    ("tab:gen   Dirichlet FM, matched", {"0.3": 0.051, "0.5": 0.090, "0.7": 0.101, "0.8": 0.094}),
    ("tab:gen   Dirichlet FM, full corpus", {"0.3": 0.139, "0.5": 0.282, "0.7": 0.289, "0.8": 0.184}),
    ("tab:gen   EqM det-CLR, no rescale", {"0.3": -0.000, "0.5": -0.094, "0.7": -0.158, "0.8": -0.125}),
):
    row(label, vals)

print("\n" + "=" * 78)
print("3. OOD  (word-level AUROC of per-position ||grad E||, max-pooled)")
print("=" * 78)
f = band / "band_ood.json"
if not f.is_file():
    print("  (no band_ood.json yet)")
else:
    d = json.loads(f.read_text())
    rows = [r for r in d["rows"] if r["auroc_word_max_upos"] == r["auroc_word_max_upos"]]
    print(f"  band_trained={d['band_trained']}  n={d.get('n')}")
    for scheme in sorted({r["scheme"] for r in rows}):
        at15 = [r for r in rows if r["scheme"] == scheme and abs(r["rate"] - 0.15) < 1e-9]
        if at15:
            best = max(at15, key=lambda r: r["auroc_word_max_upos"])
            print(f"    {scheme:>8} @0.15: best {best['auroc_word_max_upos']:.3f} "
                  f"at gamma={best['gamma']:g}")
print("  reference @0.15: Dirichlet FM denoiser NLL 0.981 word-level, "
      "GPT-2 <= 0.75, chance 0.5")
print("=" * 78)
PYEOF
}

if [ "$MODE" = "summary" ]; then summary; exit 0; fi

mkdir -p "$ROOT"
echo "### BAND-CLOUD START $(ts)"

# Derive a one-cell sweep from the base cell with this batch/LR baked in, so
# run_sweep (which has no CLI override) stays untouched.
$PY - "$BASE_SWEEP" "$SWEEP" "$CELL" "$CELL_RUN" "$B" "$LR" "$GRAD_CKPT" <<'PYGEN'
import sys, yaml
base, out, cell, cell_run, B, lr, gc = sys.argv[1:8]
cells = {c["name"]: c for c in yaml.safe_load(open(base))}
if cell not in cells:
    sys.exit(f"cell {cell!r} not in {base}")
c = dict(cells[cell]); o = dict(c["overrides"])
o["training.B"] = int(B)
o["loader_settings.batch_size"] = int(B)
o["text8_dataset.batch_size"] = int(B)
o["training.lr"] = float(lr)
o["transformer.grad_checkpointing"] = gc not in ("0", "false", "False", "")
c["name"] = cell_run; c["overrides"] = o
yaml.safe_dump([c], open(out, "w"), sort_keys=False)
print(f"    wrote {out}: B={B} lr={lr} grad_ckpt={o['transformer.grad_checkpointing']} "
      f"epochs={o['training.epochs']} d{o['transformer.d_model']}/{o['transformer.num_layers']}L")
PYGEN

STEPS_PER_EPOCH=$($PY -c "print(f'{351562//$B:,}')")
echo "### cell=$CELL_RUN  root=$ROOT  n=$N  steps=$STEPS"
echo "### full split at B=$B is ~$STEPS_PER_EPOCH steps/epoch, x30 epochs."

# ---------------------------------------------------------------------------
echo "### [$(ts)] stage 1/3  train + generation eval: $CELL_RUN"
# ---------------------------------------------------------------------------
if skipped train; then
  echo "  skipped (SKIP=$SKIP)"
elif have "$RUN_DIR/eval.json"; then
  echo "  $RUN_DIR/eval.json exists, reusing"
else
  if have "$RUN_DIR/epoch_final.pt"; then
    echo "!!! $RUN_DIR/epoch_final.pt exists without eval.json -- that is the"
    echo "!!! per-epoch crash-insurance file from an INTERRUPTED run, and"
    echo "!!! run_sweep would skip training and evaluate it as if complete."
    echo "!!! Move or delete $RUN_DIR and relaunch."
    exit 1
  fi
  $PY scripts/run_sweep.py --sweep "$SWEEP" --runs-root "$ROOT" \
      --only "$CELL_RUN" --n "$N" --steps "$STEPS"
fi

# run_sweep catches per-cell exceptions and exits 0, so `set -e` will not fire.
# No eval.json means no trained model and everything downstream would be junk.
if ! skipped train && ! have "$RUN_DIR/eval.json"; then
  echo
  echo "!!! [$(ts)] stage 1 produced no $RUN_DIR/eval.json -- training/eval FAILED."
  [ -s "$RUN_DIR/error.txt" ] && { echo "!!! error.txt:"; sed 's/^/!!!   /' "$RUN_DIR/error.txt"; }
  echo "!!! Refusing to run stages 2-3 against a partial checkpoint."
  exit 1
fi

# ---------------------------------------------------------------------------
echo "### [$(ts)] stage 2/3  recovery ladder with the input rescaled into the band"
# ---------------------------------------------------------------------------
if skipped recovery; then
  echo "  skipped (SKIP=$SKIP)"
else
  for a in $ALPHAS; do
    g="$(g_for_alpha "$a")"
    out="$RUN_DIR/recovery_bandmap_a$(tag_for_alpha "$a").json"
    if have "$out"; then echo "  $out exists, reusing"; continue; fi
    echo "  [$(ts)] alpha=$a  rescale g=$g"
    $PY scripts/recovery_check.py --ckpt "$RUN_DIR/epoch_final.pt" \
        --alphas "$a" --band-gamma "$g" --band-mode scale \
        --n "$N" --steps "$STEPS" --skip-uncond --out "$out"
  done
fi

# ---------------------------------------------------------------------------
echo "### [$(ts)] stage 3/3  OOD scorecard, energy read inside the band"
# ---------------------------------------------------------------------------
if skipped ood; then
  echo "  skipped (SKIP=$SKIP)"
elif have "$RUN_DIR/band_ood.json"; then
  echo "  $RUN_DIR/band_ood.json exists, reusing"
else
  $PY scripts/band_ood_score.py --ckpt "$RUN_DIR/epoch_final.pt" \
      --n "$N" --out "$RUN_DIR/band_ood.json"
fi

summary
echo
echo "### BAND-CLOUD COMPLETE $(ts) — everything landed in $RUN_DIR"
echo "Re-print this summary any time with:  ./drive_band_cloud.sh summary"
