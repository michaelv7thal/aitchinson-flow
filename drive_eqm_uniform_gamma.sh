#!/usr/bin/env bash
# Train the three benchmark EqM arms with UNIFORM γ (gamma_power=1.0) at the
# published L=256 scale, then run the same generation + recovery evals as the
# published bench, so the new rows read directly against tab:gen.
#
# WHY. The published arms train with γ = √u (gamma_power=0.5), chosen when the
# failure was still read as "too little supervision near the data end". The
# failure analysis (paper sec:negative / app:training) implies the opposite:
# on the deterministic-CLR path the target is determined by its own input
# above γ≈0.03, so upweighting γ≈1 concentrates training on the trivial
# regime. Prediction: uniform γ changes nothing at benchmark scale, as it
# changed nothing at dev scale (tab:ablation2x2). These runs put a number on
# that and close the "the reshaped schedule did it" objection.
#
# WHAT IT WRITES — new sibling dirs, the canonical paper runs are never
# touched (the _gp suffix comes from train_for_sflm_bench.py --gamma-power):
#     runs/sflm_bench_a100_20g_L256/EqM_OneHot_gp1p0/
#     runs/sflm_bench_a100_20g_L256/EqM_gp1p0/
#     runs/sflm_bench_a100_20g_L256/EqMAE_gp1p0/
# each with epoch_final.pt, train_meta.json (records gamma_power), and the
# eval_all.json + recovery.json produced by the same commands as the published
# rows (eval_all --n 256 --steps 200; recovery_check --alphas 0.1..1.0).
#
# Everything else matches the published recipe exactly: d1024/10L/16H, B=8,
# L=256, 10 ep x 10k windows, seed 42, lr 3e-4; EqMAE reuses the SAME frozen
# VAE (runs/vae_a100_20g_L256/epoch_final.pt), so the γ draw is the only
# changed knob.
#
# COST on the 8 GB laptop GPU (measured 2026-08-22: ladder stage 0, peak
# ~5.3 GiB, ~1.2 s/step): ~4.1 h training per arm + eval, ~13-15 h total,
# sequential. Idempotent: a finished arm is skipped on re-run (no --force),
# so an interrupted chain can simply be restarted.
#
# USAGE (separate terminal):
#     cd ~/projects/aitchinson-flow
#     nohup ./drive_eqm_uniform_gamma.sh > eqm_uniform_gamma.log 2>&1 &
#     tail -f eqm_uniform_gamma.log
set -euo pipefail
cd "$(dirname "$0")"

GP=1.0
SUF=_gp1p0            # keep in sync with GP (the driver builds it from GP)
SCALE=a100_20g_L256
EPOCHS=10
R="runs/sflm_bench_${SCALE}"
AE_CKPT="runs/vae_${SCALE}/epoch_final.pt"
ts() { date +%F\ %H:%M:%S; }

[ -f "$AE_CKPT" ] || { echo "missing frozen VAE: $AE_CKPT"; exit 1; }

train() {  # train <arm> [extra driver args...]
  local arm="$1"; shift
  echo "### [$(ts)] TRAIN ${arm} (gamma_power=${GP})"
  uv run python scripts/train_for_sflm_bench.py \
      --scale "$SCALE" --only "$arm" --epochs "$EPOCHS" --seed 42 \
      --gamma-power "$GP" "$@"
  # The driver records FAILED.json instead of raising on a non-OOM failure;
  # stop the chain so a broken arm doesn't burn hours on the next ones.
  [ -f "$R/${arm}${SUF}/epoch_final.pt" ] \
    || { echo "!!! ${arm}: no checkpoint (see $R/${arm}${SUF}/FAILED.json)"; exit 1; }
}

# Evals are NON-FATAL: training is the 4 h/arm cost, an eval is minutes and is
# idempotently re-runnable, so a failed readout must not cancel the remaining
# arms' training (that is what happened on the 2026-08-22 first attempt). Every
# failure is recorded here and re-listed at the end so nothing fails silently.
EVAL_FAILURES=""

evals() {  # evals <arm>  — same commands as drive_backfill.sh / the published rows
  # NOTE: two statements on purpose. bash expands ALL words of `local` BEFORE
  # performing any of its assignments, so the one-liner
  # `local arm="$1" d="$R/${arm}${SUF}"` reads a still-unset ${arm} and dies
  # under `set -u` ("arm: unbound variable"). This killed the first run after
  # EqM_OneHot finished training. Do not re-merge these two lines.
  local arm="$1"
  local d="$R/${arm}${SUF}"
  if [ -f "$d/eval_all.json" ] && [ "$d/eval_all.json" -nt "$d/epoch_final.pt" ]; then
    echo "### [$(ts)] EVAL ${arm} — up to date, skipping"
  else
    echo "### [$(ts)] EVAL ${arm} (generation, n=256, 200 steps)"
    uv run python scripts/eval_all.py --ckpt "$d/epoch_final.pt" --model-kind "$arm" \
        --split test --n 256 --steps 200 --bpc-mc 8 --out "$d/eval_all.json" \
      || { echo "!!! [$(ts)] EVAL ${arm} FAILED (rc=$?) — continuing the chain"
           EVAL_FAILURES="${EVAL_FAILURES} ${arm}:eval_all"; }
  fi
  if [ -f "$d/recovery.json" ] && [ "$d/recovery.json" -nt "$d/epoch_final.pt" ]; then
    echo "### [$(ts)] RECOVERY ${arm} — up to date, skipping"
  else
    echo "### [$(ts)] RECOVERY ${arm} (alphas 0.1,0.3,0.5,0.7,1.0)"
    uv run python scripts/recovery_check.py --ckpt "$d/epoch_final.pt" \
        --alphas 0.1,0.3,0.5,0.7,1.0 --n 256 --steps 200 --out "$d/recovery.json" \
      || { echo "!!! [$(ts)] RECOVERY ${arm} FAILED (rc=$?) — continuing the chain"
           EVAL_FAILURES="${EVAL_FAILURES} ${arm}:recovery"; }
  fi
}

train EqM_OneHot
evals EqM_OneHot
train EqM
evals EqM
train EqMAE --ae-ckpt "$AE_CKPT"
evals EqMAE

echo "### [$(ts)] ALL DONE — uniform-γ rows vs the published √u rows:"
[ -z "${EVAL_FAILURES// /}" ] || echo "!!! FAILED EVALS (re-runnable):${EVAL_FAILURES}"
uv run python - "$R" "$SUF" <<'PY'
import json, sys
root, suf = sys.argv[1], sys.argv[2]
cols = ["KL_uni", "KL_bi", "KL_tri", "H_ratio"]
print(f"{'arm':<12} {'':<9}" + "".join(f"{c:>9}" for c in cols))
for arm in ["EqM_OneHot", "EqM", "EqMAE"]:
    for label, d in [("sqrt(u)", f"{root}/{arm}"), ("uniform", f"{root}/{arm}{suf}")]:
        try:
            e = json.load(open(f"{d}/eval_all.json"))
            print(f"{arm:<12} {label:<9}" + "".join(f"{e[c]:>9.3f}" for c in cols))
        except FileNotFoundError:
            print(f"{arm:<12} {label:<9}  (missing {d}/eval_all.json)")
PY
