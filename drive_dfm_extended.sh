#!/usr/bin/env bash
# Extended-budget DirichletFM as a NEW arm (DirichletFM_ep30_d30k): 30 epochs ×
# 30k windows @ L256. Does NOT overwrite the matched-budget DirichletFM results
# (separate run dir). Runs ONLY after the current pipeline (post chain +
# backfill) finishes, so it never contends with it for the 20 GB MIG slice.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/renku/work/aitchinson-flow
SC=a100_20g_L256; ARM=DirichletFM_ep30_d30k
R="runs/sflm_bench_${SC}"; LOG="${R}/_driver"; mkdir -p "$LOG"
ts(){ date +%Y-%m-%d_%H:%M:%S; }
BF="${LOG}/backfill.out"

echo "### DFM-EXT START $(ts) — waiting for BACKFILL COMPLETE (current pipeline done)"
while ! grep -q "BACKFILL COMPLETE" "$BF" 2>/dev/null; do sleep 60; done
echo "### [$(ts)] pipeline complete — training $ARM (30 ep × 30k windows, L256)"

python scripts/train_for_sflm_bench.py --scale ${SC} --force --epochs 30 \
    --max-train-windows 30000 --only ${ARM} \
    > "${LOG}/train_${ARM}.log" 2>&1
echo "### [$(ts)] train exit=$?  meta=$(cat $R/$ARM/train_meta.json 2>/dev/null | tr -d '\n ')"

if [ ! -f "$R/$ARM/epoch_final.pt" ]; then
  echo "### [$(ts)] no checkpoint — aborting eval (see ${LOG}/train_${ARM}.log)"; exit 1
fi

echo "### [$(ts)] generation eval $ARM"
python scripts/eval_all.py --ckpt $R/$ARM/epoch_final.pt \
    --model-kind DirichletFM --split test --n 256 --steps 200 --bpc-mc 8 \
    --out $R/$ARM/eval_all.json > "${LOG}/eval_${ARM}.log" 2>&1
echo "### [$(ts)] gen eval exit=$?"
# Relabel model_name to the variant so the aggregate table shows a distinct row.
python -c "import json; p='$R/$ARM/eval_all.json'; d=json.load(open(p)); d['model_name']='$ARM'; json.dump(d,open(p,'w'),indent=2)" 2>/dev/null \
  && echo "    relabeled model_name -> $ARM" || echo "    (relabel skipped — eval_all.json missing)"

echo "### [$(ts)] recovery sweep $ARM"
python scripts/recovery_check.py --ckpt $R/$ARM/epoch_final.pt \
    --alphas 0.1,0.3,0.5,0.7,1.0 --n 256 --steps 200 \
    --out $R/$ARM/recovery.json > "${LOG}/recovery_${ARM}.log" 2>&1
echo "### [$(ts)] recovery exit=$?"
echo "### DFM-EXT COMPLETE $(ts) — $R/$ARM/{eval_all.json,recovery.json}"
