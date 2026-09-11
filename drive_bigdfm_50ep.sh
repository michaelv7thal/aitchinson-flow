#!/usr/bin/env bash
# Longer DirichletFM push (UNCONDITIONAL): train d1280/14L for 50 ep × 50k windows
# at BATCH 16 (DirichletFM is first-order, so the 20 GB MIG has headroom the
# batch-8 EqM-sized preset leaves idle). New scale+arm -> no overwrite.
# Runs after the OOD re-run frees the GPU (sequential: capacity -> OOD -> this).
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/renku/work/aitchinson-flow
BIGSCALE=a100_20g_L256_d1280L14_b16; ARM=DirichletFM_ep50_d50k
R="runs/sflm_bench_${BIGSCALE}"; LOG="${R}/_driver"; mkdir -p "$LOG"
CAPEVAL="runs/sflm_bench_a100_20g_L256_d1280L14/DirichletFM_ep30_d30k/eval_all.json"
RERUN="runs/ood_det_rerun/_driver/ood_rerun.out"
BASELINE_KLBI=0.208
ts(){ date +%Y-%m-%d_%H:%M:%S; }

echo "### BIGDFM-50EP START $(ts) — waiting for OOD-DET-RERUN COMPLETE"
while ! grep -q "OOD-DET-RERUN COMPLETE" "$RERUN" 2>/dev/null; do sleep 120; done
echo "### [$(ts)] OOD re-run done"

# Informational only (NOT gating): capacity d1280/14L 30ep KL_bi vs d1024 0.208.
KLBI=$(python -c "import json;print(json.load(open('$CAPEVAL'))['KL_bi'])" 2>/dev/null || echo "NA")
echo "### [$(ts)] (info) capacity d1280/14L 30ep KL_bi=$KLBI  vs d1024 baseline=$BASELINE_KLBI"

echo "### [$(ts)] training $ARM @ $BIGSCALE (d1280/14L, 50ep × 50k, batch 16) — unconditional"
python scripts/train_for_sflm_bench.py --scale ${BIGSCALE} --force --epochs 50 \
    --max-train-windows 50000 --only ${ARM} \
    > "${LOG}/train_${ARM}.log" 2>&1
echo "### [$(ts)] train exit=$?  meta=$(cat $R/$ARM/train_meta.json 2>/dev/null | tr -d '\n ')"

if [ ! -f "$R/$ARM/epoch_final.pt" ]; then
  echo "### [$(ts)] no checkpoint — abort eval (see ${LOG}/train_${ARM}.log)"; exit 1
fi
echo "### [$(ts)] generation eval"
python scripts/eval_all.py --ckpt $R/$ARM/epoch_final.pt \
    --model-kind DirichletFM --split test --n 256 --steps 200 --bpc-mc 8 \
    --out $R/$ARM/eval_all.json > "${LOG}/eval_${ARM}.log" 2>&1
python -c "import json; p='$R/$ARM/eval_all.json'; d=json.load(open(p)); d['model_name']='DirichletFM_d1280L14_ep50_d50k_b16'; json.dump(d,open(p,'w'),indent=2)" 2>/dev/null || true
echo "### [$(ts)] recovery sweep"
python scripts/recovery_check.py --ckpt $R/$ARM/epoch_final.pt \
    --alphas 0.1,0.3,0.5,0.7,1.0 --n 256 --steps 200 \
    --out $R/$ARM/recovery.json > "${LOG}/recovery_${ARM}.log" 2>&1
echo "### BIGDFM-50EP COMPLETE $(ts) — compare $R/$ARM/eval_all.json vs d1280/14L 30ep (KL_bi=$KLBI)"
