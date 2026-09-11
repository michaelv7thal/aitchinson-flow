#!/usr/bin/env bash
# CAPACITY-LEVER TEST: DirichletFM (Dirichlet FM) at a BIGGER backbone
# d1280/14L/16H (~277M, 2.18x) at the SAME 30ep x 30k as DirichletFM_ep30_d30k
# (d1024/10L, ~127M). Only capacity changes -> isolates whether capacity (not
# data/epochs) lifts generation. New scale dir, no overwrite.
#
# Runs AFTER the OOD chain (drive_maha_perpos.sh) frees the GPU.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/renku/work/aitchinson-flow
BIG=a100_20g_L256_d1280L14
R="runs/sflm_bench_${BIG}"; LOG="${R}/_driver"; mkdir -p "$LOG"
ARM=DirichletFM_ep30_d30k
ts(){ date +%Y-%m-%d_%H:%M:%S; }
MAHA="runs/ood_maha_perpos_dirichletfm/_driver/maha_perpos.out"

echo "### CAPACITY-TEST START $(ts) — waiting for MAHA-PERPOS COMPLETE (OOD chain done)"
while ! grep -q "MAHA-PERPOS COMPLETE" "$MAHA" 2>/dev/null; do sleep 60; done
echo "### [$(ts)] OOD chain done — training $ARM @ $BIG (d1280/14L ~277M, 30ep x 30k)"

python scripts/train_for_sflm_bench.py --scale ${BIG} --force --epochs 30 \
    --max-train-windows 30000 --only ${ARM} \
    > "${LOG}/train_${ARM}.log" 2>&1
echo "### [$(ts)] train exit=$?  meta=$(cat $R/$ARM/train_meta.json 2>/dev/null | tr -d '\n ')"

if [ ! -f "$R/$ARM/epoch_final.pt" ]; then
  echo "### [$(ts)] no checkpoint — aborting eval (see ${LOG}/train_${ARM}.log)"; exit 1
fi

echo "### [$(ts)] generation eval"
python scripts/eval_all.py --ckpt $R/$ARM/epoch_final.pt \
    --model-kind DirichletFM --split test --n 256 --steps 200 --bpc-mc 8 \
    --out $R/$ARM/eval_all.json > "${LOG}/eval_${ARM}.log" 2>&1
echo "### [$(ts)] gen eval exit=$?"
python -c "import json; p='$R/$ARM/eval_all.json'; d=json.load(open(p)); d['model_name']='DirichletFM_d1280L14_ep30_d30k'; json.dump(d,open(p,'w'),indent=2)" 2>/dev/null \
  && echo "    relabeled -> DirichletFM_d1280L14_ep30_d30k" || echo "    (relabel skipped)"

echo "### [$(ts)] recovery sweep"
python scripts/recovery_check.py --ckpt $R/$ARM/epoch_final.pt \
    --alphas 0.1,0.3,0.5,0.7,1.0 --n 256 --steps 200 \
    --out $R/$ARM/recovery.json > "${LOG}/recovery_${ARM}.log" 2>&1
echo "### [$(ts)] recovery exit=$?"
echo "### CAPACITY-TEST COMPLETE $(ts) — compare $R/$ARM/eval_all.json (d1280/14L) vs d1024/10L KL_bi=0.208"
