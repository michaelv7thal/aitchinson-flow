#!/usr/bin/env bash
# Fine α-ladder for a DirichletFM arm (fig:recovery-curve).
#
# Usage: drive_recovery_fine.sh [ARM_DIR]
#   ARM_DIR defaults to the full-corpus arm. Per-α JSONs land in
#   ARM_DIR/recovery_fine/, logs in the sibling <scale>/_driver/.
#
# The published recovery.json for these arms covers α ∈ {0.1,0.3,0.5,0.7,1.0}
# (drive_fulltext8_converge2_tail.sh for the full arm). This driver fills the
# ladder to 0.1 increments, α = 0.1 … 1.0, on the SAME checkpoint / n / nfe so
# every point is protocol-identical.
#
# Each α is a separate recovery_check.py invocation writing its own JSON. That
# is exactly equivalent to one 10-α run — recovery_check seeds per α
# (torch.manual_seed(seed + int(1000·α))) and reads the same val_ids[:n] — and
# it makes the multi-hour sweep durable and resumable: an α whose file already
# exists is skipped. --skip-uncond drops the unconditional pass (one full nfe
# sweep) since these arms' generation numbers are already in recovery.json /
# eval_all.json.
set -uo pipefail
cd /home/michael/projects/aitchinson-flow

R="${1:-runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM_converge}"
R="${R%/}"
CKPT="$R/epoch_final.pt"
OUT="$R/recovery_fine"
LOG="$(dirname "$R")/_driver"
ARM="$(basename "$R")"
mkdir -p "$OUT" "$LOG"
ts(){ date +%Y-%m-%d_%H:%M:%S; }

if [ ! -f "$CKPT" ]; then
  echo "### no checkpoint at $CKPT — ABORT"; exit 1
fi

echo "### RECOVERY-FINE START $(ts)  arm=$ARM  ckpt=$CKPT"
for A in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
  F="$OUT/alpha_${A}.json"
  if [ -s "$F" ]; then echo "### [$(ts)] α=$A already done — skip"; continue; fi
  echo "### [$(ts)] α=$A start"
  uv run python scripts/recovery_check.py --ckpt "$CKPT" \
      --alphas "$A" --n 256 --steps 200 --skip-uncond \
      --out "$F" > "${LOG}/recovery_fine_${ARM}_alpha_${A}.log" 2>&1
  echo "### [$(ts)] α=$A exit=$? -> $(python3 -c "
import json,sys
try:
    r=json.load(open('$F'))['rows'][0]
    print(f\"delta={r['delta']:.4f} acc={r['token_acc']:.4f} acc_pt={r['token_acc_perturbed']:.4f}\")
except Exception as e:
    print('NO RESULT', e)
")"
done
echo "### RECOVERY-FINE COMPLETE $(ts)  arm=$ARM"