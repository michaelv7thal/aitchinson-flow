#!/usr/bin/env bash
# Train all 4 models on text8 character-level language modelling.
#
# Environment variables:
#   WANDB_PROJECT   — if set, enables W&B logging to that project
#   SAVE_DIR        — results root (default: ./results)
#   LOG_INTERVAL    — console/wandb log frequency in epochs (default: 1000)
#
# Usage:
#   bash scripts/run_text8.sh
#   WANDB_PROJECT=aitchison-flow bash scripts/run_text8.sh
#   docker run --gpus all -e WANDB_API_KEY=$WANDB_API_KEY \
#              -e WANDB_PROJECT=aitchison-flow \
#              -v $(pwd)/data:/workspace/data \
#              -v $(pwd)/results:/workspace/results \
#              my-image bash scripts/run_text8.sh

set -euo pipefail

SAVE_DIR="${SAVE_DIR:-results}"
LOG_INTERVAL="${LOG_INTERVAL:-1000}"
RUN_PREFIX="${RUN_PREFIX:-text8}"        # stable prefix; set by runai_submit.sh

SHARED=(
    --dataset      text8
    --preset       cluster
    --loss         hilbert
    --log-interval "$LOG_INTERVAL"
    --save-dir     "$SAVE_DIR"
)

WANDB_FLAGS=()
if [[ -n "${WANDB_PROJECT:-}" ]]; then
    WANDB_FLAGS=(--wandb --wandb-project "$WANDB_PROJECT")
fi

echo "======================================================================"
echo "  text8 benchmark  |  preset: cluster  |  prefix: $RUN_PREFIX"
echo "  device: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")')"
echo "======================================================================"

#python run.py "${SHARED[@]}" "${WANDB_FLAGS[@]}" --model flow    --run-name "${RUN_PREFIX}_flow"
python run.py "${SHARED[@]}" "${WANDB_FLAGS[@]}" --model eqm     --run-name "${RUN_PREFIX}_eqm"
#python run.py "${SHARED[@]}" "${WANDB_FLAGS[@]}" --model bayes   --run-name "${RUN_PREFIX}_bayes"
#python run.py "${SHARED[@]}" "${WANDB_FLAGS[@]}" --model auditor --run-name "${RUN_PREFIX}_auditor"

echo ""
echo "======================================================================"
echo "  Comparison pass ..."
echo "======================================================================"
#python run.py "${SHARED[@]}" "${WANDB_FLAGS[@]}" \
#    --model flow eqm bayes auditor --compare
