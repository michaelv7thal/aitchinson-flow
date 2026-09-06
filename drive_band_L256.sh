#!/usr/bin/env bash
# ============================================================================
#  Band-EqM at benchmark scale (L=256): is it competitive, and does its energy
#  give the per-token OOD score the equilibrium route promised?
# ============================================================================
#
#  Three questions, in order:
#
#   1. GENERATION.  Train the deterministic-CLR EqM arm at L=256 with γ drawn
#      only from the band [0.005, 0.03] and the equilibrium moved to γ*=0.03,
#      then score it with the paper's generation eval. Compare against tab:gen:
#         EqM deterministic CLR (whole path)   KL_uni 0.018  KL_bi 1.582  KL_tri 5.042
#         Dirichlet FM (the selected arm)      KL_uni 0.009  KL_bi 0.959  KL_tri 4.088
#         Discrete FM (flagged collapsed)                    KL_bi 1.466
#      At L=40 the band took EqM from 1.657 to 0.906, and to 0.287 at 5× data,
#      against a Discrete FM control of 0.148 (tab:eqm-band).
#
#   2. RECOVERY.  Run the α ladder with the perturbed input rescaled into the
#      band, the protocol of tab:eqm-band-recovery. The rescale is one positive
#      factor, so it moves no argmax and Δ_α is still measured against the
#      unrescaled input. At L=40 the 50k band cell gave
#         Δ@.30 +0.046   Δ@.50 +0.161   Δ@.70 +0.155   Δ@1.0 +0.122
#      against +0.000 everywhere without the rescale, and the whole-path field
#      gave −0.014 / +0.048 / +0.074 / +0.071 with it. The benchmark Dirichlet FM
#      gains +0.090 at α=0.5 (tab:gen), the full-corpus one +0.282.
#      The ladder runs α ∈ {0.3, 0.5, 0.6, 0.7, 0.8, 1.0}. Two of those earn
#      their wall clock beyond filling the curve: α=0.6 is where every transport
#      budget peaks (fig:recovery-curve), so it is where a competing arm should
#      look best, and α=0.8 is the matched-damage column tab:gen reads the two
#      Fisher–Rao arms at, where Dirichlet FM gains +0.094. Neither was measured
#      at L=40, so those two columns have no dev-scale reference to compare to.
#
#   3. OOD.  Read the band field's per-position ‖∇E‖ and sequence energy AFTER
#      rescaling the candidate window into the band, and score corrupted vs clean
#      characters, words, and sequences with the same helpers the published
#      detectors use. γ=1.0 in that sweep is the data-scale control, i.e. the
#      reading the paper already has ("a minimum at every vertex"). Reference at
#      a 15% replace rate: the frozen Dirichlet FM denoiser NLL reaches a
#      word-level AUROC of 0.981, the GPT-2 baselines at most 0.75, chance 0.5.
#
#  The whole-path benchmark arm is run through 2 and 3 as the control, from its
#  existing checkpoint, so no extra training is needed for it.
#
#  ---------------------------------------------------------------------------
#  USAGE
#
#    ./drive_band_L256.sh smoke          # ~10 min: wiring only, junk numbers
#    ./drive_band_L256.sh                # the matched benchmark budget (~9-14 h)
#    CELL=band_L256_ep5_d50k ./drive_band_L256.sh    # 5× data (~15-21 h)
#    ./drive_band_L256.sh summary        # re-print the summary from the JSONs
#
#  Knobs (environment variables):
#    CELL     which cell of sweeps/band_L256.yaml to train (default the 10ep/10k one)
#    N        eval samples / windows per stage (default 256; 128 halves stages 2-3)
#    STEPS    descent steps for the band sampler (default 400, the eval_fixed protocol)
#    ALPHAS   recovery ladder (default "0.3 0.5 0.6 0.7 0.8 1.0")
#    SKIP     comma list of stages to skip, e.g. SKIP=train,control
#    CONTROL  whole-path checkpoint used as the control (default the bench arm)
#
#  Every stage is idempotent: run_sweep skips a cell whose eval.json exists, and
#  each stage below skips its own output file if it is already there. Safe to
#  re-run after an interruption; delete the JSON you want recomputed.
#
#  Wall clock and memory (laptop RTX PRO 1000, 8 GB, measured): training is
#  ~1.18 s/step at B=8 / L=256 / d1024-10L, peak ~5.3 GB, so stage 1 is ~4.1 h
#  for ep10_d10k and ~10.2 h for ep5_d50k. Stages 2 and 4 are the other slow
#  ones: each ladder point is n=256 windows through 400 NAG steps with
#  second-order autograd, and there are six of them per field, so budget a few
#  hours for stage 2 and somewhat less for the control at its 200 steps. Stage 3
#  is forward+backward only and takes minutes. Levers if it is too slow or too
#  tight: N=128 halves stages 2-4, ALPHAS="0.5 0.8" keeps only the two columns
#  the paper reads, --chunk in stage 3, and transformer.grad_checkpointing in
#  the yaml.
#
#  MEMORY, the 8 GB card (settled 2026-08-27, after two OOM'd attempts).
#  Both died at epoch 5, and neither cause was training: training peaks ~5.0 GB
#  and is fine. The culprit is the VALIDATION pass. cfg.training.eval_every=5
#  means it first fires at epoch 5, and training/loops.py:evaluate() calls
#  eval_step() with NO torch.no_grad(), so EqM's second-order conservative-
#  gradient graph is built on top of the live params+grads+Adam footprint:
#  measured 7.29 GB of a 7.57 GB card, reproduced exactly outside the run.
#  sweeps/band_L256.yaml therefore pins, for both full-scale cells:
#      training.eval_every: 11     # > epochs, so validation never fires
#      training.sample_eval_n: 16  # the n=64 probe batch was also near the edge
#  Cost: no val_loss in history. Nothing reads it here -- early stopping is off
#  and the three questions above use generation/recovery/OOD only.
#  Also worth exporting: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#
#  Do NOT reach for transformer.grad_checkpointing for this: it costs ~30%
#  wall clock to shrink the one thing that already fits.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# No Weights & Biases. Every cell of the sweep sets wandb.enabled: false, which
# short-circuits before wandb is even imported; build_wandb_logger ORs
# WANDB_PROJECT from the environment into that gate, so clear it here too.
unset WANDB_PROJECT WANDB_ENTITY WANDB_NAME WANDB_RUN_GROUP
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true

MODE="${1:-run}"
# CELLS runs several cells back-to-back through the whole pipeline (stages 1-3
# per cell; the control in stage 4 is shared and its outputs are reused). CELL
# stays supported for a single cell.
CELLS="${CELLS:-}"
CELL="${CELL:-band_L256_ep10_d10k}"
N="${N:-256}"
STEPS="${STEPS:-400}"
ALPHAS="${ALPHAS:-0.3 0.5 0.6 0.7 0.8 1.0}"
SKIP="${SKIP:-}"
CONTROL="${CONTROL:-runs/sflm_bench_a100_20g_L256/EqM_OneHot_gp1p0/epoch_final.pt}"
CONTROL_STEPS="${CONTROL_STEPS:-200}"   # the whole-path arm's own published protocol
SWEEP=sweeps/band_L256.yaml
PY="uv run python"

if [ "$MODE" = "smoke" ]; then
  CELL=band_L256_smoke; N=32; STEPS=50; ALPHAS="0.5"
  echo "### SMOKE MODE: 1 epoch on 500 windows, n=$N, $STEPS steps. Numbers are junk."
fi

if [ -n "$CELLS" ]; then CELL="${CELLS%% *}"; fi
RUN_DIR="runs/$CELL"
skipped () { case ",$SKIP," in *",$1,"*) return 0;; *) return 1;; esac; }
have ()    { [ -s "$1" ]; }

# The rescale factor per α. Renormalizes the perturbed state to the radius of a
# band interpolant at γ*≈0.016:  g(α) = |x_γ| / |z_α| with |x_γ| = 0.539 and
# |z_α| = 12.27 · sqrt(1 + K α²), K = 27. Per position, so it is independent of
# L: the same factors that produced tab:eqm-band-recovery at L=40. At α=0.5 this
# value also matches the band's own source noise exactly, (1-g)·σ = g·α·12.27.
# Stage 2 prints embed_norm_mean; if it is not ~12.27 these factors are stale.
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
tag_for_alpha () { echo "${1/./}"; }   # 0.3 -> 03, 1.0 -> 10

summary () {
$PY - "$RUN_DIR" "$(dirname "$CONTROL")" <<'PYEOF'
import json, sys
from pathlib import Path

band, ctrl = Path(sys.argv[1]), Path(sys.argv[2])
PAPER_GEN = [
    ("tab:gen  EqM det-CLR, whole path, L=256", 0.018, 1.582, 5.042),
    ("tab:gen  Dirichlet FM, matched budget  ", 0.009, 0.959, 4.088),
    ("tab:gen  Discrete FM (collapsed flag)  ", 0.001, 1.466, 5.141),
    ("L=40     band 5ep/50k                  ", 0.018, 0.287, 1.781),
    ("L=40     Discrete FM control 5ep/50k   ", 0.007, 0.148, 1.402),
]
print("\n" + "=" * 78)
print("1. GENERATION  (lower is better)")
print("=" * 78)
ev = band / "eval.json"
if ev.is_file():
    d = json.loads(ev.read_text())
    print(f"{'THIS RUN  ' + band.name:<41} "
          f"{d['unigram_kl']:>6.3f} {d['bigram_kl']:>6.3f} {d['trigram_kl']:>6.3f}"
          f"   H_ratio {d.get('H_ratio', float('nan')):.3f}")
    for s in (d.get("samples") or [])[:3]:
        print(f"    {s[:96]!r}")
else:
    print(f"  (no {ev} yet)")
print(f"{'':<41} {'KL_uni':>6} {'KL_bi':>6} {'KL_tri':>6}")
for name, u, b, t in PAPER_GEN:
    print(f"{name:<41} {u:>6.3f} {b:>6.3f} {t:>6.3f}")

print("\n" + "=" * 78)
print("2. RECOVERY, input rescaled into the band  (Delta vs the unrescaled argmax)")
print("=" * 78)
ALPHAS_SHOWN = ("0.3", "0.5", "0.6", "0.7", "0.8", "1.0")
TAG = {a: a.replace(".", "") for a in ALPHAS_SHOWN}
# Reference ladders, all published. A missing alpha was never measured there and
# prints as a dash rather than being filled in from a neighbour.
REFS = (
    ("L=40      band 5ep/50k, rescaled",
     {"0.3": 0.046, "0.5": 0.161, "0.7": 0.155, "1.0": 0.122}),
    ("L=40      whole path, rescaled",
     {"0.3": -0.014, "0.5": 0.048, "0.7": 0.074, "1.0": 0.071}),
    ("tab:gen   EqM det-CLR, no rescale",
     {"0.3": -0.000, "0.5": -0.094, "0.7": -0.158, "0.8": -0.125}),
    ("tab:gen   Dirichlet FM, matched",
     {"0.3": 0.051, "0.5": 0.090, "0.7": 0.101, "0.8": 0.094}),
    ("tab:gen   Dirichlet FM, full corpus",
     {"0.3": 0.139, "0.5": 0.282, "0.7": 0.289, "0.8": 0.184}),
)
print(f"{'':<34} " + " ".join(f"{'a=' + a:>9}" for a in ALPHAS_SHOWN))
def row(label: str, vals: dict):
    cells = [(f"{vals[a]:>+9.3f}" if vals.get(a) is not None else f"{'—':>9}")
             for a in ALPHAS_SHOWN]
    print(f"{label:<34} " + " ".join(cells))
def measured(where: Path) -> dict:
    out = {}
    for a in ALPHAS_SHOWN:
        f = where / f"recovery_bandmap_a{TAG[a]}.json"
        if f.is_file():
            out[a] = json.loads(f.read_text())["rows"][0]["delta"]
    return out
row(f"THIS RUN  {band.name}", measured(band))
row(f"CONTROL   {ctrl.name}", measured(ctrl))
for label, vals in REFS:
    row(label, vals)
print("  a=0.6 is where every transport budget peaks; a=0.8 is the matched-damage")
print("  column tab:gen reads the Fisher-Rao arms at. Neither has an L=40 reference.")

print("\n" + "=" * 78)
print("3. OOD  (word-level AUROC of per-position ||grad E||, max-pooled)")
print("=" * 78)
for where, label in ((band, "THIS RUN"), (ctrl, "CONTROL ")):
    f = where / "band_ood.json"
    if not f.is_file():
        print(f"{label} {where.name}: (no band_ood.json yet)")
        continue
    d = json.loads(f.read_text())
    rows = [r for r in d["rows"]
            if r["auroc_word_max_upos"] == r["auroc_word_max_upos"]]
    print(f"{label} {where.name}   band_trained={d['band_trained']}")
    for scheme in sorted({r["scheme"] for r in rows}):
        at15 = [r for r in rows if r["scheme"] == scheme and abs(r["rate"] - 0.15) < 1e-9]
        if at15:
            best = max(at15, key=lambda r: r["auroc_word_max_upos"])
            ctl = [r for r in at15 if r["gamma"] == 1.0]
            ctl_s = (f", data scale {ctl[0]['auroc_word_max_upos']:.3f}" if ctl else "")
            print(f"    {scheme:>8} @0.15: best {best['auroc_word_max_upos']:.3f} "
                  f"at gamma={best['gamma']:g}{ctl_s}")
    over = max(rows, key=lambda r: r["auroc_word_max_upos"])
    print(f"    best overall {over['auroc_word_max_upos']:.3f} "
          f"(gamma={over['gamma']:g}, {over['scheme']} {over['rate']:.2f})")
print("  reference @0.15: Dirichlet FM denoiser NLL 0.981 word-level, "
      "GPT-2 <= 0.75, chance 0.5")
print("=" * 78)
PYEOF
}

if [ "$MODE" = "summary" ]; then
  if [ -n "$CELLS" ]; then
    for c in $CELLS; do CELL="$c"; RUN_DIR="runs/$c"; echo; echo "########## $c"; summary; done
  else
    summary
  fi
  exit 0
fi

# ---------------------------------------------------------------------------
echo "### stage 1/4  train + generation eval: $CELL"
# ---------------------------------------------------------------------------
if skipped train; then
  echo "  skipped (SKIP=$SKIP)"
elif have "$RUN_DIR/eval.json"; then
  echo "  $RUN_DIR/eval.json exists, reusing"
else
  $PY scripts/run_sweep.py --sweep "$SWEEP" --only "$CELL" --n "$N" --steps "$STEPS"
fi

# run_sweep.py CATCHES a per-cell exception, writes error.txt and exits 0, so
# `set -e` does not fire. Twice (2026-08-26/27) that let stage 1 die of a CUDA
# OOM at epoch 5 while stages 2-4 marched on and spent hours building a ladder
# against the HALF-TRAINED epoch_final.pt the runner leaves as crash insurance
# -- results that then look exactly like real ones on disk. Stop instead: no
# eval.json means no trained model, and everything downstream would be junk.
if ! skipped train && ! have "$RUN_DIR/eval.json"; then
  echo
  echo "!!! stage 1 produced no $RUN_DIR/eval.json -- training or eval FAILED."
  [ -s "$RUN_DIR/error.txt" ] && { echo "!!! $RUN_DIR/error.txt says:"; sed 's/^/!!!   /' "$RUN_DIR/error.txt"; }
  if [ -s "$RUN_DIR/epoch_final.pt" ]; then
    echo "!!! NOTE: $RUN_DIR/epoch_final.pt exists but is a PARTIAL checkpoint"
    echo "!!!       (saved every epoch as crash insurance). Do not score it."
  fi
  echo "!!! Refusing to run stages 2-4 against it. Fix stage 1 and re-run;"
  echo "!!! every finished stage is reused, so nothing already done is lost."
  exit 1
fi

# ---------------------------------------------------------------------------
echo "### stage 2/4  recovery ladder with the input rescaled into the band"
# ---------------------------------------------------------------------------
if skipped recovery; then
  echo "  skipped (SKIP=$SKIP)"
else
  for a in $ALPHAS; do
    g="$(g_for_alpha "$a")"
    out="$RUN_DIR/recovery_bandmap_a$(tag_for_alpha "$a").json"
    if have "$out"; then echo "  $out exists, reusing"; continue; fi
    echo "  alpha=$a  rescale g=$g"
    $PY scripts/recovery_check.py --ckpt "$RUN_DIR/epoch_final.pt" \
        --alphas "$a" --band-gamma "$g" --band-mode scale \
        --n "$N" --steps "$STEPS" --skip-uncond --out "$out"
  done
fi

# ---------------------------------------------------------------------------
echo "### stage 3/4  OOD scorecard, energy read inside the band"
# ---------------------------------------------------------------------------
if skipped ood; then
  echo "  skipped (SKIP=$SKIP)"
elif have "$RUN_DIR/band_ood.json"; then
  echo "  $RUN_DIR/band_ood.json exists, reusing"
else
  $PY scripts/band_ood_score.py --ckpt "$RUN_DIR/epoch_final.pt" \
      --n "$N" --out "$RUN_DIR/band_ood.json"
fi

# ---------------------------------------------------------------------------
echo "### stage 4/4  the whole-path benchmark arm, same two protocols (control)"
# ---------------------------------------------------------------------------
if skipped control; then
  echo "  skipped (SKIP=$SKIP)"
elif [ ! -s "$CONTROL" ]; then
  echo "  no control checkpoint at $CONTROL, skipping"
else
  CTRL_DIR="$(dirname "$CONTROL")"
  for a in $ALPHAS; do
    g="$(g_for_alpha "$a")"
    out="$CTRL_DIR/recovery_bandmap_a$(tag_for_alpha "$a").json"
    if have "$out"; then echo "  $out exists, reusing"; continue; fi
    echo "  control alpha=$a  rescale g=$g"
    $PY scripts/recovery_check.py --ckpt "$CONTROL" \
        --alphas "$a" --band-gamma "$g" --band-mode scale \
        --n "$N" --steps "$CONTROL_STEPS" --skip-uncond --out "$out"
  done
  if have "$CTRL_DIR/band_ood.json"; then
    echo "  $CTRL_DIR/band_ood.json exists, reusing"
  else
    $PY scripts/band_ood_score.py --ckpt "$CONTROL" \
        --n "$N" --out "$CTRL_DIR/band_ood.json"
  fi
fi

summary
echo
echo "Done. Re-print this summary any time with:  ./drive_band_L256.sh summary"
echo "Everything landed in $RUN_DIR (and the control's own directory)."

# Chain: if CELLS listed more than the one we just finished, run the next one.
# Re-exec rather than loop so each cell gets a clean process (the CUDA allocator
# state from a 30-epoch train does not carry into the next cell's sampler).
if [ -n "$CELLS" ]; then
  rest=""; seen=0
  for c in $CELLS; do
    if [ "$seen" = "1" ]; then rest="$rest $c"; fi
    if [ "$c" = "$CELL" ]; then seen=1; fi
  done
  rest="$(echo "$rest" | sed 's/^ *//')"
  if [ -n "$rest" ]; then
    next="${rest%% *}"
    echo
    echo "### CHAIN $(date +%H:%M:%S) — next cell: $next   (remaining: $rest)"
    CELLS="$rest" CELL="$next" exec "$0"
  fi
  echo "### CHAIN COMPLETE — all cells done."
fi
