#!/usr/bin/env bash
# Queued post-v3 work: regenerate weights for every runs/latent_* cell
# whose epoch_final.pt is missing (trained locally, transferred without
# the .pt files), then run recovery_check on each to populate the
# unconditional KL and recovery curve.
#
# Phases:
#   0. wait for the v3 sweep launcher (PID 631554) to exit
#   1. regenerate missing pretrained-embedding files in data/
#      - data/skipgram_d27.pt   (used by 4 latent_fixed_skipgram_d27_* cells)
#      - data/ppmi_svd_d27.pt   (used by 2 latent_fixed_ppmi_d27_* cells)
#      - data/ppmi_svd_d16.pt   (used by latent_fixed_ppmi_d16_tied)
#      Default hyperparams from the generator scripts; will not be
#      byte-identical to the originals (seeds unknown) but the *kind* of
#      embedding (skipgram/PPMI at the given dim) matches.
#   2. retrain each latent_* cell from its saved config.json
#   3. recovery_check on the new checkpoint → recovery.json
#      (covers both unconditional generation eval and recovery curve)
#   4. rebuild scripts/build_final_summary.py
#
# Idempotent: each step skips if its target artefact already exists.
#
# Launch:
#   nohup bash scripts/run_latent_retrain_queue.sh \
#       >/dev/null 2>&1 < /dev/null &

set -u
cd /home/renku/work/aitchinson-flow
unset PYTHONPATH
mkdir -p runs/_logs data
LOG=runs/_logs/latent_retrain_queue.log
PARENT_PID=${PARENT_PID:-631554}
ALPHAS="0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,0.70,0.80,1.00"
PY=.venv/bin/python

echo "[wait] $(date '+%F %T') waiting for v3 launcher pid=$PARENT_PID" >>"$LOG"
while kill -0 "$PARENT_PID" 2>/dev/null; do sleep 60; done
echo "[wait] $(date '+%F %T') v3 launcher exited, beginning latent retrain queue" >>"$LOG"

# Pre-flight: confirm the project imports cleanly. A broken module (e.g.,
# an uncommitted syntax error) would otherwise cause every step below to
# fail in ~2s with the same error — abort loudly here instead.
if ! $PY -c "import sys; sys.path.insert(0,'src'); import aitchinson_flow.training, aitchinson_flow.models" \
     >> "$LOG" 2>&1; then
  echo "[abort] $(date '+%F %T') project does not import — fix before retrying" >>"$LOG"
  exit 1
fi

# ─── Phase 1: regenerate missing embedding files ────────────────────
declare -A EMB_SPEC=(
  [data/skipgram_d27.pt]="learn_skipgram_embeddings --d 27 --epochs 3"
  [data/ppmi_svd_d27.pt]="learn_ppmi_svd_embeddings --d 27"
  [data/ppmi_svd_d16.pt]="learn_ppmi_svd_embeddings --d 16"
)
for path in "${!EMB_SPEC[@]}"; do
  spec=${EMB_SPEC[$path]}
  script=${spec%% *}
  rest=${spec#* }
  if [ -f "$path" ]; then
    echo "[skip-emb] $path exists" >>"$LOG"
    continue
  fi
  echo "[emb] $script → $path $(date '+%F %T')" >>"$LOG"
  $PY scripts/${script}.py --out "$path" $rest \
    >> "runs/_logs/$(basename ${path%.*}).log" 2>&1
  echo "[emb-done] $path rc=$? $(date '+%F %T')" >>"$LOG"
done

# ─── Phase 2 & 3: retrain + recovery for each missing-weight cell ───
for d in runs/latent_*; do
  cell=$(basename "$d")
  ckpt="$d/epoch_final.pt"
  cfg="$d/config.json"
  rec="$d/recovery.json"

  if [ ! -f "$cfg" ]; then
    echo "[skip] $cell: no config.json" >>"$LOG"
    continue
  fi

  # ─── Phase 2: retrain ─────────────────────────────────────────────
  if [ -f "$ckpt" ]; then
    echo "[skip-train] $cell: ckpt exists" >>"$LOG"
  else
    echo "[train] $cell start=$(date '+%F %T')" >>"$LOG"
    $PY scripts/retrain_latent_from_config.py "$d" \
      >> "runs/_logs/${cell}_retrain.log" 2>&1
    rc=$?
    echo "[train-done] $cell rc=$rc $(date '+%F %T')" >>"$LOG"
    if [ ! -f "$ckpt" ]; then
      echo "[skip-recovery] $cell: train failed, no ckpt" >>"$LOG"
      continue
    fi
  fi

  # ─── Phase 3: recovery_check (overwrites stale recovery.json) ────
  echo "[rec] $cell start=$(date '+%F %T')" >>"$LOG"
  $PY scripts/recovery_check.py \
    --ckpt "$ckpt" --alphas "$ALPHAS" \
    --n 256 --steps 200 --out "$rec" \
    >> "runs/_logs/${cell}_recovery.log" 2>&1
  echo "[rec-done] $cell rc=$? $(date '+%F %T')" >>"$LOG"
done

# ─── Phase 4: refresh AE_EQM_FINAL_REPORT.md ────────────────────────
echo "[final-aggregator] $(date '+%F %T')" >>"$LOG"
$PY scripts/build_final_summary.py >>"$LOG" 2>&1
echo "[final-aggregator-done] rc=$? $(date '+%F %T')" >>"$LOG"

echo "[end] $(date '+%F %T') latent retrain queue complete" >>"$LOG"
