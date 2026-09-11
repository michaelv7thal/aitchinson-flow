#!/usr/bin/env bash
# Latent-split geometry for the paper: the rate-0.3 grid (sec:latent, tab:latent,
# fig:latent-seq, fig:latent-token) and the corruption-rate ladder behind it
# (app:latent-ladder).
#
# Everything runs on the SAME ckpt / split / n=256 / fit-seqs=192 / seed as
# bench_ood, so the cells re-derive the published ones as a check. Rate 0.3 is
# computed twice (stage A with coordinate dumps for the figures, stage B inside
# the ladder) — the duplicate is a free reproduction check, not an oversight.
#
# Stage order is by cost, not by importance: A and B are bounded, C is not.
#   A  rate 0.3, replace/shuffle/falseinfo, both granularities, +coords   ~20 min
#   B  nine-rate ladder, same three schemes, metrics only                 ~1-2 h
#   C  plausible at rate 0.3, both granularities, +coords                 hours
#      (48 model-scored candidate words per swapped slot; the per-call cost
#       is logged so the plausible ladder can be budgeted from it)
# Each stage skips its own finished outputs, so the driver is resumable.
#
# Waits for a running recovery sweep first: the 8 GB card cannot hold both.
set -uo pipefail
cd /home/michael/projects/aitchinson-flow

CKPT="runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM_converge/epoch_final.pt"
CHEAP="replace,shuffle,falseinfo"
RATES="0.05 0.1 0.15 0.2 0.25 0.3 0.5 0.7 1.0"
COMMON=(--ckpt "$CKPT" --split test --n 256 --fit-seqs 192 --seed 42 --no-tsne)
LOG=bench_ood/_driver
mkdir -p "$LOG"
ts(){ date +%Y-%m-%d_%H:%M:%S; }

if [ ! -f "$CKPT" ]; then
  echo "### no checkpoint at $CKPT — ABORT"; exit 1
fi

while pgrep -f "drive_recovery_fine.sh" > /dev/null; do
  echo "### [$(ts)] recovery sweep still on the GPU — waiting"
  sleep 300
done

echo "### LATENT-LADDER START $(ts)  ckpt=$CKPT"

# ---- A: the rate-0.3 grid the main-text figures are drawn from ------------
A=bench_ood_final/latent_split_all
if [ -s "$A/latent_split.json" ]; then
  echo "### [$(ts)] stage A already done — skip"
else
  echo "### [$(ts)] stage A: $CHEAP at rate 0.3, both granularities"
  uv run python scripts/plot_latent_split.py "${COMMON[@]}" --rate 0.3 \
      --schemes "$CHEAP" --token-scheme "$CHEAP" --dump-coords --out-dir "$A" \
      > "$LOG/latent_all.log" 2>&1
  echo "### [$(ts)] stage A exit=$?"
  grep -E "^\[(seq|token):" "$LOG/latent_all.log"
fi

# ---- B: the same three schemes across the corruption ladder ---------------
for R in $RATES; do
  D="bench_ood_final/latent_ladder/rate_${R}"
  if [ -s "$D/latent_split.json" ]; then
    echo "### [$(ts)] ladder rate=$R already done — skip"; continue
  fi
  echo "### [$(ts)] ladder rate=$R start"
  uv run python scripts/plot_latent_split.py "${COMMON[@]}" --rate "$R" \
      --schemes "$CHEAP" --token-scheme "$CHEAP" --out-dir "$D" \
      > "$LOG/latent_ladder_${R}.log" 2>&1
  rc=$?
  echo "### [$(ts)] ladder rate=$R exit=$rc"
  grep -E "^\[(seq|token):" "$LOG/latent_ladder_${R}.log"
done

# ---- N: label-permutation null for the two in-sample probes ---------------
# Both probes are fit on the points they score. Per sequence that is 512 points
# in 1280 dimensions, where ANY labelling is linearly separable, so the reported
# AUROC has to be read against what the same estimator returns on shuffled
# labels. Measured at rate 0.3 (the tab:latent row) and at 0.05 (the one
# app:latent-ladder row whose per-token sample falls short of 4000).
for R in 0.3 0.05; do
  D="bench_ood_final/latent_null/rate_${R}"
  if [ -s "$D/latent_split.json" ]; then
    echo "### [$(ts)] null rate=$R already done — skip"; continue
  fi
  echo "### [$(ts)] null rate=$R start"
  uv run python scripts/plot_latent_split.py "${COMMON[@]}" --rate "$R" \
      --schemes "$CHEAP" --token-scheme "$CHEAP" --perm-null 5 --out-dir "$D" \
      > "$LOG/latent_null_${R}.log" 2>&1
  rc=$?
  echo "### [$(ts)] null rate=$R exit=$rc"
  grep -E "^\[(seq|token):" "$LOG/latent_null_${R}.log"
done

# ---- C: plausible at rate 0.3 (the expensive cells of tab:latent) ---------
C=bench_ood_final/latent_split_plausible
if [ -s "$C/latent_split.json" ]; then
  echo "### [$(ts)] stage C already done — skip"
else
  echo "### [$(ts)] stage C: plausible at rate 0.3, both granularities"
  uv run python scripts/plot_latent_split.py "${COMMON[@]}" --rate 0.3 \
      --schemes plausible --token-scheme plausible --t-nll 3.0 --n-cands 48 \
      --dump-coords --out-dir "$C" \
      > "$LOG/latent_plausible.log" 2>&1
  echo "### [$(ts)] stage C exit=$?"
  grep -E "^\[(seq|token|plausible)" "$LOG/latent_plausible.log"
fi

echo "### LATENT-LADDER COMPLETE $(ts)"
