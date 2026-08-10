#!/usr/bin/env bash
# Re-run OOD detection, healing, and the latent readouts on the FINAL model.
#
# WHY. The published bench_ood/ + bench_heal/ + heal_poc_insulin/ trees (and the
# latent JSONs under bench_ood/latent_*) all read
#     runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM/epoch_best.pt
# which — despite living in the DirichletFM/ directory — is the CONTINUATION leg's
# epoch-5 best (its embedded cfg.training.checkpoint_dir is DirichletFM_continue;
# B=24 after two OOM fallbacks from 48; val CE 0.8554). Generation and transport
# recovery instead read the fully annealed model,
#     runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM_converge/epoch_final.pt
# so the paper currently reports two different sets of weights. This driver puts
# detection, healing, and the latent readouts on the same final model as generation.
#
# WHAT IT WRITES. New trees, so nothing the paper currently cites is overwritten:
#     bench_ood_final/  bench_heal_final/  heal_poc_insulin_final/
# The published trees stay in place as the comparison baseline, and the last stage
# diffs new against old so it is immediately visible whether any number moved.
#
# The two GPT-2 arms (gpt2_se, gpt2_nll) do not read our checkpoint at all, so
# their numbers MUST come out identical. The compare stage checks exactly that and
# warns if they differ: that would mean something other than the checkpoint changed.
#
# COST on one GPU, sequential (the 8 GB card holds one job at a time):
#     1  bench_ood detector sweep, 9 arms                        ~4-5 h
#     2  plausible frequency-matched control                     ~20 min
#     3  insulin transfer head (+ GPT-2 baseline)                ~15 min
#     4  aggregate + heatmaps                                    minutes
#     5  bench_heal, 6 text8 arms + 3 insulin arms               ~1-2 h
#     6  latent rate-0.3 grid + 9-rate ladder + permutation null ~2 h
#     7  latent plausible, rate 0.3 then 0.15                    ~12 h
#     8  compare against the published trees                     seconds
# Stage 7 is the whole tail: 48 model-scored candidates per swapped word slot,
# ~2.0 s/slot. Skip it with --skip-plausible and add it later; --resume picks up.
#
# USAGE
#     nohup ./drive_bench_final.sh > bench_final.log 2>&1 &
#     tail -f bench_final.log
#
#     --dry-run         print every command without running anything (validate first)
#     --quick           only nll/blr/bgmm + aggregate + compare (~2 h, verdict only)
#     --resume          skip arms already recorded done in the new manifests
#     --skip-latent     stages 6 and 7 off
#     --skip-plausible  stage 7 off (the 12 h tail)
#     --with-recovery   also top up the transport-recovery ladder for this arm
#                       (drive_recovery_fine.sh already reads epoch_final.pt, so
#                        this is a no-op unless an alpha file is missing)
#
# Env overrides: ARM_DIR (which run leg), SUFFIX (output-dir suffix),
# OLD_OOD / OLD_HEAL (comparison baselines), FREQ_T_EVAL, PLAUSIBLE_RATES.
#
# NOT DONE HERE, deliberately. This driver only produces numbers; it rewrites no
# paper input. If the new tree is adopted, five figure scripts hardcode the old
# bench roots and each needs its root pointed at the new tree before re-rendering:
#     scripts/paper_figs/fig_ood_word_auroc.py    fig_ood_falseinfo.py
#     scripts/paper_figs/fig_latent_split.py      fig_latent_ladder.py   (bench_ood)
#     scripts/paper_figs/fig_heal_compare.py                             (bench_heal)
# fig_latent_split.py additionally expects the coordinate dumps in
# latent_split_all/ and latent_split_plausible/, which stage 6a and 7a write.
#
# Every stage is resumable and a failing stage is logged and stepped over rather
# than killing the run, so an overnight launch survives one bad arm.
set -uo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------------------------------------------------------------- configuration
ARM_DIR="${ARM_DIR:-runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM_converge}"
ARM_DIR="${ARM_DIR%/}"
CKPT="$ARM_DIR/epoch_final.pt"
SUFFIX="${SUFFIX:-_final}"
OOD="bench_ood${SUFFIX}"
HEAL="bench_heal${SUFFIX}"
INSULIN="heal_poc_insulin${SUFFIX}"
OLD_OOD="${OLD_OOD:-bench_ood}"
OLD_HEAL="${OLD_HEAL:-bench_heal}"
LOG="$OOD/_driver"
CMP="$OOD/_compare"
PY="uv run python"

# The published frequency-matched control ran at t_eval 6.0 (the script default)
# while its min-NLL counterpart ran at 7.5 via run_bench_ood.py. Reproduced as-is
# so the new control is comparable to the old one; set FREQ_T_EVAL=7.5 to make the
# control strictly comparable to the arm instead.
FREQ_T_EVAL="${FREQ_T_EVAL:-6.0}"
# Published plausible ladder covers 0.15 (here) and 0.3 (stage 7a, linked in).
# Rate 0.5 was cancelled at ~10.5 h and is not part of the appendix.
PLAUSIBLE_RATES="${PLAUSIBLE_RATES:-0.15}"

RESUME=""; DRY=""; QUICK=""; SKIP_LATENT=""; SKIP_PLAUSIBLE=""; WITH_RECOVERY=""
for a in "$@"; do
  case "$a" in
    --resume)         RESUME="--resume" ;;
    --dry-run)        DRY=1 ;;
    --quick)          QUICK=1 ;;
    --skip-latent)    SKIP_LATENT=1 ;;
    --skip-plausible) SKIP_PLAUSIBLE=1 ;;
    --with-recovery)  WITH_RECOVERY=1 ;;
    *) echo "unknown argument: $a"; exit 2 ;;
  esac
done
[ -n "$QUICK" ] && { SKIP_LATENT=1; SKIP_PLAUSIBLE=1; }

ts(){ date +%Y-%m-%d_%H:%M:%S; }

# run NAME LOGFILE CMD...   — log, keep going on failure, honour --dry-run
run(){
  local name="$1" log="$2"; shift 2
  if [ -n "$DRY" ]; then
    echo "### [dry-run] $name"; echo "      $*"; return 0
  fi
  echo "### [$(ts)] START $name  -> $log"
  mkdir -p "$(dirname "$log")"
  if "$@" > "$log" 2>&1; then
    echo "### [$(ts)] OK    $name"
  else
    echo "### [$(ts)] FAIL  $name (exit $?, see $log)"
  fi
}

# ------------------------------------------------------------------- preflight
echo "######## BENCH-FINAL START $(ts) ########"
echo "  arm dir : $ARM_DIR"
echo "  ckpt    : $CKPT"
echo "  outputs : $OOD  $HEAL  $INSULIN"
echo "  baseline: $OLD_OOD  $OLD_HEAL"

if [ ! -f "$CKPT" ]; then
  echo "### no checkpoint at $CKPT — ABORT"; exit 1
fi
if [ "$(basename "$ARM_DIR")" != "DirichletFM_converge" ]; then
  echo "### NOTE: ARM_DIR is not DirichletFM_converge, so these are NOT the final"
  echo "###       annealed weights the generation results use. Proceeding anyway."
fi

echo "  md5     : $(md5sum "$CKPT" | cut -d' ' -f1)"
if [ -z "$DRY" ]; then
  # Record WHICH leg the weights come from and check it against the directory they
  # sit in. A file whose embedded checkpoint_dir names a different leg is exactly
  # how DirichletFM/epoch_best.pt came to be read as if it were this run's own
  # best checkpoint, so the mismatch is worth shouting about rather than trusting
  # the path.
  $PY - "$CKPT" "$ARM_DIR" <<'PROV'
import os, sys, torch
d = torch.load(sys.argv[1], map_location="cpu", mmap=True, weights_only=False)
cfg = d.get("cfg") if isinstance(d.get("cfg"), dict) else {}
tr = cfg.get("training", {}) if isinstance(cfg, dict) else {}
leg = tr.get("checkpoint_dir")
print(f"  leg     : {leg}")
print(f"  epoch   : {d.get('epoch')}  global_step: {d.get('global_step')}"
      f"  lr: {tr.get('lr')}  epochs: {tr.get('epochs')}")
if leg and os.path.basename(str(leg).rstrip("/")) != os.path.basename(sys.argv[2]):
    print(f"  WARNING : these weights were trained in {leg}, not in {sys.argv[2]}."
          f" The directory does not tell you which leg they are from.")
PROV
fi

mkdir -p "$LOG" "$CMP"

if [ -z "$DRY" ]; then
  while pgrep -f 'train_for_sflm_bench|recovery_check\.py|run_bench_ood\.py|run_bench_heal\.py|plot_latent_split\.py|ood_plausible_swap\.py' > /dev/null 2>&1; do
    echo "### [$(ts)] another GPU job is running — waiting"
    sleep 300
  done
fi

# --------------------------------------------------- 1. detector sweep (9 arms)
ONLY=""
[ -n "$QUICK" ] && ONLY="--only nll,blr,bgmm"
echo "######## [$(ts)] 1/8  DETECTOR SWEEP -> $OOD ${QUICK:+(quick: nll,blr,bgmm)} ########"
run "bench_ood" "$LOG/run_bench_ood.log" \
  $PY scripts/run_bench_ood.py --ckpt "$CKPT" --out-dir "$OOD" $ONLY $RESUME

# ------------------------------------- 2. plausible frequency-matched control
if [ -z "$QUICK" ]; then
  echo "######## [$(ts)] 2/8  PLAUSIBLE FREQ-MATCHED CONTROL ########"
  D="$OOD/plausible_freqmatched"
  if [ -s "$D/plausible_swap.json" ]; then
    echo "### [$(ts)] freq-matched control already done — skip"
  else
    mkdir -p "$D"
    run "plausible_freqmatched" "$D/run.log" \
      $PY -u scripts/ood_plausible_swap.py --ckpt "$CKPT" --split test \
        --fit-seqs 512 --n 64 --rate 0.15 --swap-mode freq_matched \
        --t-eval "$FREQ_T_EVAL" --t-nll 3.0 --blr-t-eval 4.5 --n-cands 48 \
        --bgmm-max-iter 1000 --seed 42 --out "$D/plausible_swap.json"
  fi
fi

# ------------------------------------------------- 3. out-of-domain transfer
if [ -z "$QUICK" ]; then
  echo "######## [$(ts)] 3/8  INSULIN TRANSFER HEAD (+ GPT-2 baseline) ########"
  D="$OOD/transfer"
  if [ -s "$D/insulin_transfer_gpt2.json" ]; then
    echo "### [$(ts)] transfer already done — skip"
  else
    mkdir -p "$D"
    # --ref-lm gpt2 carries the baseline columns, so this one file is a superset of
    # the archived insulin_transfer.json (which predates the GPT-2 arm).
    run "transfer" "$D/run_gpt2.log" \
      $PY -u scripts/eval_detector_transfer.py --ckpt "$CKPT" \
        --article-json heal_poc_insulin/insulin_article_extract.json \
        --t-eval 4.5 --fit-seqs 512 --n 128 --rate 0.15 --ref-lm gpt2 --seed 42 \
        --out "$D/insulin_transfer_gpt2.json"
  fi
fi

# ------------------------------------------------------ 4. aggregate + figures
echo "######## [$(ts)] 4/8  AGGREGATE -> $OOD/RESULTS.md + figs ########"
run "aggregate_ood" "$LOG/bench_aggregate.log" \
  $PY scripts/bench_aggregate.py --bench-dir "$OOD"
run "heatmaps" "$LOG/plot_heatmaps_all.log" \
  $PY scripts/plot_heatmaps_all.py --bench-dir "$OOD"

# ------------------------------------------------------------- 5. healing bench
if [ -z "$QUICK" ]; then
  echo "######## [$(ts)] 5/8  HEALING -> $HEAL + $INSULIN ########"
  run "bench_heal" "$LOG/run_bench_heal.log" \
    $PY scripts/run_bench_heal.py --ckpt "$CKPT" --out-dir "$HEAL" \
      --insulin-out-dir "$INSULIN" \
      --insulin-article heal_poc_insulin/insulin_article_extract.json $RESUME
fi

# --------------------------------------------- 6. latent readouts (cheap schemes)
CHEAP="replace,shuffle,falseinfo"
COMMON=(--ckpt "$CKPT" --split test --n 256 --fit-seqs 192 --seed 42 --no-tsne)
if [ -z "$SKIP_LATENT" ]; then
  echo "######## [$(ts)] 6/8  LATENT: rate-0.3 grid, 9-rate ladder, perm null ########"

  # 6a. the rate-0.3 grid the main-text figures are drawn from (+ coordinate dumps)
  A="$OOD/latent_split_all"
  if [ -s "$A/latent_split.json" ]; then
    echo "### [$(ts)] latent stage A already done — skip"
  else
    run "latent/A_rate0.3" "$LOG/latent_all.log" \
      $PY scripts/plot_latent_split.py "${COMMON[@]}" --rate 0.3 \
        --schemes "$CHEAP" --token-scheme "$CHEAP" --dump-coords --out-dir "$A"
  fi

  # 6b. the same three schemes across the corruption ladder (appendix)
  for R in 0.05 0.1 0.15 0.2 0.25 0.3 0.5 0.7 1.0; do
    D="$OOD/latent_ladder/rate_${R}"
    if [ -s "$D/latent_split.json" ]; then
      echo "### [$(ts)] latent ladder rate=$R already done — skip"; continue
    fi
    run "latent/ladder_$R" "$LOG/latent_ladder_${R}.log" \
      $PY scripts/plot_latent_split.py "${COMMON[@]}" --rate "$R" \
        --schemes "$CHEAP" --token-scheme "$CHEAP" --out-dir "$D"
  done

  # 6c. label-permutation null for the two in-sample probes (both are fit on the
  # points they score, so the reported AUROC only means something against this).
  for R in 0.3 0.05; do
    D="$OOD/latent_null/rate_${R}"
    if [ -s "$D/latent_split.json" ]; then
      echo "### [$(ts)] latent null rate=$R already done — skip"; continue
    fi
    run "latent/null_$R" "$LOG/latent_null_${R}.log" \
      $PY scripts/plot_latent_split.py "${COMMON[@]}" --rate "$R" \
        --schemes "$CHEAP" --token-scheme "$CHEAP" --perm-null 5 --out-dir "$D"
  done
fi

# ------------------------------------------------- 7. latent plausible (the tail)
if [ -z "$SKIP_LATENT" ] && [ -z "$SKIP_PLAUSIBLE" ]; then
  echo "######## [$(ts)] 7/8  LATENT PLAUSIBLE: rate 0.3, then $PLAUSIBLE_RATES ########"

  # 7a. rate 0.3 with coordinate dumps: the tab:latent plausible row
  C="$OOD/latent_split_plausible"
  if [ -s "$C/latent_split.json" ]; then
    echo "### [$(ts)] latent plausible 0.3 already done — skip"
  else
    run "latent/plausible_0.3" "$LOG/latent_plausible.log" \
      $PY -u scripts/plot_latent_split.py "${COMMON[@]}" --rate 0.3 \
        --schemes plausible --token-scheme plausible --t-nll 3.0 --n-cands 48 \
        --dump-coords --out-dir "$C"
  fi

  # 7b. the ladder rates, metrics only; rate 0.3 is linked, never recomputed
  LADDER="$OOD/latent_ladder_plausible"
  mkdir -p "$LADDER"
  if [ -s "$C/latent_split.json" ] && [ ! -e "$LADDER/rate_0.3" ]; then
    ln -s ../latent_split_plausible "$LADDER/rate_0.3"
    echo "### [$(ts)] linked rate_0.3 -> latent_split_plausible"
  fi
  for R in $PLAUSIBLE_RATES; do
    D="$LADDER/rate_${R}"
    if [ -s "$D/latent_split.json" ]; then
      echo "### [$(ts)] latent plausible rate=$R already done — skip"; continue
    fi
    run "latent/plausible_$R" "$LOG/latent_plausible_${R}.log" \
      $PY -u scripts/plot_latent_split.py "${COMMON[@]}" --rate "$R" \
        --schemes plausible --token-scheme plausible --t-nll 3.0 --n-cands 48 \
        --out-dir "$D"
  done
fi

# --------------------------------------------------------- 7c. recovery top-up
if [ -n "$WITH_RECOVERY" ]; then
  echo "######## [$(ts)] 7c/8  RECOVERY LADDER (already epoch_final.pt) ########"
  run "recovery_fine" "$LOG/recovery_fine.log" \
    bash runs/sflm_bench_a100_20g_L256_d1280L14_full/_driver/drive_recovery_fine.sh "$ARM_DIR"
fi

# ------------------------------------------------------------------ 8. compare
echo "######## [$(ts)] 8/8  COMPARE against the published trees ########"
if [ -n "$DRY" ]; then
  echo "### [dry-run] would diff $OLD_OOD vs $OOD and $OLD_HEAL vs $HEAL into $CMP/"
else
  for pair in "$OLD_OOD:$OOD:ood" "$OLD_HEAL:$HEAL:heal"; do
    IFS=: read -r o n tag <<< "$pair"
    if [ -s "$o/RESULTS.md" ] && [ -s "$n/RESULTS.md" ]; then
      diff -u "$o/RESULTS.md" "$n/RESULTS.md" > "$CMP/${tag}_RESULTS.diff"
      echo "### [$(ts)] $tag RESULTS.md diff -> $CMP/${tag}_RESULTS.diff ($(wc -l < "$CMP/${tag}_RESULTS.diff") lines)"
    else
      echo "### [$(ts)] $tag: no RESULTS.md pair to diff ($o / $n)"
    fi
  done

  OLD_OOD="$OLD_OOD" NEW_OOD="$OOD" OLD_HEAL="$OLD_HEAL" NEW_HEAL="$HEAL" \
  OLD_INSULIN="heal_poc_insulin" NEW_INSULIN="$INSULIN" \
  $PY - <<'CMP_PY' | tee "$CMP/numeric_summary.txt"
import json, math, os

SKIP = ("example", "samples", "clean", "corrupted", "healed", "truec", "flag")

def leaves(o, pre=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, f"{pre}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, f"{pre}[{i}]")
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        if not math.isnan(float(o)):
            yield pre, float(o)

def numeric(path):
    with open(path) as fh:
        d = json.load(fh)
    return {k: v for k, v in leaves(d) if not any(s in k.lower() for s in SKIP)}

def compare(old_root, new_root, label):
    print(f"\n=== {label}: {old_root} -> {new_root}")
    if not os.path.isdir(new_root):
        print("    new tree missing"); return
    for dirpath, _, files in sorted(os.walk(new_root)):
        for f in sorted(files):
            if not f.endswith(".json") or f == "manifest.json":
                continue
            np_ = os.path.join(dirpath, f)
            rel = os.path.relpath(np_, new_root)
            op = os.path.join(old_root, rel)
            if not os.path.exists(op):
                print(f"    {rel:52s} NEW ONLY"); continue
            try:
                o, n = numeric(op), numeric(np_)
            except Exception as e:
                print(f"    {rel:52s} unreadable ({e})"); continue
            shared = [k for k in n if k in o]
            if not shared:
                print(f"    {rel:52s} no comparable numeric fields"); continue
            moved = sorted(((abs(n[k] - o[k]), k) for k in shared), reverse=True)
            moved = [(d, k) for d, k in moved if d > 1e-6]
            if not moved:
                print(f"    {rel:52s} {len(shared):5d} fields, IDENTICAL"); continue
            warn = ("  <-- WARNING: this arm never reads our checkpoint, "
                    "it should be identical") if rel.startswith("gpt2") else ""
            print(f"    {rel:52s} {len(shared):5d} fields, {len(moved):5d} moved, "
                  f"max|d|={moved[0][0]:.4f}{warn}")
            for _, k in moved[:3]:
                print(f"        {k}: {o[k]:.4f} -> {n[k]:.4f}  ({n[k] - o[k]:+.4f})")

compare(os.environ["OLD_OOD"], os.environ["NEW_OOD"], "detection + latent")
compare(os.environ["OLD_HEAL"], os.environ["NEW_HEAL"], "healing")
compare(os.environ["OLD_INSULIN"], os.environ["NEW_INSULIN"], "insulin article")
CMP_PY
fi

echo "######## BENCH-FINAL DONE $(ts) ########"
echo "  new trees : $OOD  $HEAL  $INSULIN"
echo "  provenance: $OOD/manifest.json (ckpt + md5), $HEAL/manifest.json"
echo "  comparison: $CMP/ood_RESULTS.diff  $CMP/heal_RESULTS.diff  $CMP/numeric_summary.txt"
echo "  NOTE the paper still cites $OLD_OOD / $OLD_HEAL; nothing here rewrites the prose."
