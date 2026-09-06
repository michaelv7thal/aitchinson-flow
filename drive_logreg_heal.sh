#!/usr/bin/env bash
# Text repair (inpainting) with the full-dimension LOGISTIC probe as the per-token
# localizer — the healing counterpart of drive_logreg_fulldim.sh.
#
# Same pipeline, splits, calibration and operating points as bench_heal_final's
# blr_* arms (heal_dirichlet.py: localize -> clean-calibrated threshold -> inpaint
# -> score), so logreg_replace / logreg_falseinfo sit directly beside blr_replace /
# blr_falseinfo in tab:heal-synth. The localizer differs from the LinE hinge head
# in the objective alone (scripts/heal_dirichlet.py:train_logistic_localizer).
#
# Queued behind the detector family; no existing artifact is touched.
set -uo pipefail
cd /home/michael/projects/aitchinson-flow
CKPT="runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM_converge/epoch_final.pt"
COMMON=(--ckpt "$CKPT" --split test --n-demo 64 --n-seeds 3 --corrupt-rate 0.15
        --nfe 100 --target-fprs 0.02,0.05,0.10 --seed 42 --localizer logistic)
ts(){ date +%Y-%m-%d_%H:%M:%S; }

# Wait on the family's OUTPUT, not on a pgrep of the driver name: the launching
# shell embeds this script's text, so `pgrep -f <name>.sh` matches the waiter
# itself and never clears.
echo "### [$(ts)] waiting for the detector family"
until [ -f bench_ood_final/logreg_adv/bayes_linear_logreg_adv_sweep.json ]; do sleep 30; done
echo "### [$(ts)] detector family done"
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1500 ]; do sleep 60; done

for scheme in replace falseinfo; do
  out="bench_heal_final/logreg_${scheme}.json"
  log="bench_heal_final/logreg_${scheme}.log"
  echo "### [$(ts)] heal logistic / $scheme"
  PYTHONUNBUFFERED=1 .venv/bin/python scripts/heal_dirichlet.py \
    "${COMMON[@]}" --corrupt-scheme "$scheme" --out "$out" > "$log" 2>&1
  echo "### [$(ts)] heal logistic / $scheme exit $?"
done
echo "### [$(ts)] LOGREG HEAL COMPLETE"
touch bench_heal_final/.logreg_heal.done
