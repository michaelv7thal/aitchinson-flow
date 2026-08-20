#!/usr/bin/env bash
# The plausible column of app:latent-ladder at the two rates flanking the main
# operating point: 0.15 and 0.5. Rate 0.3 is not run here — it is stage C of
# drive_latent_all.sh (bench_ood_final/latent_split_plausible), and this script links
# it into the ladder tree so the appendix row and tab:latent come from one JSON.
#
# Cost is set by the swap, not by the readout: every swapped word slot costs one
# forward over 48 model-scored candidates. Per rate that is 704 sequences at the
# ladder rate (192 fit + 256 eval per sequence, 256 eval per token) plus a fixed
# 192 at rate 0.5, the rate the token hinge head is trained at. So relative to
# the ~13.4k forwards of the 0.3 run: 0.15 ~ 0.66x, 0.5 ~ 1.45x.
#
# Same ckpt / split / n / fit-seqs / seed as every other latent run, so the rows
# are comparable with the three cheap schemes cell by cell. Metrics only: no
# --dump-coords (the appendix draws curves, not scatters) and no t-SNE.
# Resumable — a rate whose latent_split.json exists is skipped.
set -uo pipefail
cd /home/michael/projects/aitchinson-flow

CKPT="runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM_converge/epoch_final.pt"
RATES="0.15 0.5"                       # cheapest first
COMMON=(--ckpt "$CKPT" --split test --n 256 --fit-seqs 192 --seed 42 --no-tsne)
LADDER=bench_ood_final/latent_ladder_plausible
LOG=bench_ood/_driver
mkdir -p "$LOG" "$LADDER"
ts(){ date +%Y-%m-%d_%H:%M:%S; }

if [ ! -f "$CKPT" ]; then
  echo "### no checkpoint at $CKPT — ABORT"; exit 1
fi

# the 8 GB card holds one of these at a time
while pgrep -f "drive_latent_all.sh" > /dev/null; do
  echo "### [$(ts)] rate-0.3 plausible still on the GPU — waiting"
  sleep 300
done

echo "### PLAUSIBLE-LADDER START $(ts)"

# rate 0.3 comes from stage C; link rather than recompute (hours) or copy
if [ -s bench_ood_final/latent_split_plausible/latent_split.json ] \
   && [ ! -e "$LADDER/rate_0.3" ]; then
  ln -s ../latent_split_plausible "$LADDER/rate_0.3"
  echo "### [$(ts)] linked rate_0.3 -> latent_split_plausible"
fi

for R in $RATES; do
  D="$LADDER/rate_${R}"
  if [ -s "$D/latent_split.json" ]; then
    echo "### [$(ts)] plausible rate=$R already done — skip"; continue
  fi
  echo "### [$(ts)] plausible rate=$R start"
  # -u so the per-call [plausible] slot/second lines land in the log as they
  # happen; block-buffered stdout hid all progress on the 0.3 run.
  uv run python -u scripts/plot_latent_split.py "${COMMON[@]}" --rate "$R" \
      --schemes plausible --token-scheme plausible --t-nll 3.0 --n-cands 48 \
      --out-dir "$D" \
      > "$LOG/latent_plausible_${R}.log" 2>&1
  rc=$?
  echo "### [$(ts)] plausible rate=$R exit=$rc"
  grep -E "^\[(seq|token|plausible)" "$LOG/latent_plausible_${R}.log"
done

echo "### PLAUSIBLE-LADDER COMPLETE $(ts)"
