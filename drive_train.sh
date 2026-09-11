#!/usr/bin/env bash
# Generation-benchmark TRAIN driver (Stage A VAE + Stage B all 7 arms), one tier.
# Plain `python` — `uv run` is broken in this env (pulls a broken buildpack torch).
# Env: SC (scale), EP (epochs), MTW (max-train-windows), VAE_L (VAE seq-len == arm L).
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/renku/work/aitchinson-flow
: "${SC:?set SC}"; : "${EP:?set EP}"; : "${MTW:?set MTW}"; : "${VAE_L:?set VAE_L}"
R="runs/sflm_bench_${SC}"; LOG="${R}/_driver"; mkdir -p "$LOG"
ts(){ date +%Y-%m-%d_%H:%M:%S; }
echo "### TRAIN DRIVER START $(ts)  SC=$SC EP=$EP MTW=$MTW VAE_L=$VAE_L"

echo "### [$(ts)] STAGE A: VAE for EqMAE (seq-len ${VAE_L})"
python scripts/train_autoencoder.py --mode vae --out runs/vae_${SC} \
    --seq-len ${VAE_L} --epochs ${EP} --windows ${MTW} \
    --d-latent 64 --d-model 256 --num-layers 2 --nhead 4 \
    > "${LOG}/stageA_vae.log" 2>&1
echo "### [$(ts)] STAGE A exit=$?  -> $(ls -la runs/vae_${SC}/epoch_final.pt 2>/dev/null || echo MISSING)"

echo "### [$(ts)] STAGE B1: 6 single-stage arms (EqM_OneHot,EqM,DFM,DirichletFM,SFLM,FMonCLR)"
python scripts/train_for_sflm_bench.py --scale ${SC} --force --epochs ${EP} \
    --max-train-windows ${MTW} \
    --only EqM_OneHot,EqM,DFM,DirichletFM,SFLM,FMonCLR \
    > "${LOG}/stageB1_six.log" 2>&1
echo "### [$(ts)] STAGE B1 exit=$?"

echo "### [$(ts)] STAGE B2: EqMAE (VAE+EqM, --ae-ckpt)"
python scripts/train_for_sflm_bench.py --scale ${SC} --force --epochs ${EP} \
    --max-train-windows ${MTW} --only EqMAE \
    --ae-ckpt runs/vae_${SC}/epoch_final.pt \
    > "${LOG}/stageB2_eqmae.log" 2>&1
echo "### [$(ts)] STAGE B2 exit=$?"

echo "### TRAIN DRIVER COMPLETE $(ts) — checkpoint census:"
for A in EqM_OneHot EqM EqMAE DFM DirichletFM SFLM FMonCLR; do
  f="$R/$A/epoch_final.pt"
  if [ -f "$f" ]; then echo "  OK       $A  ($(stat -c%s "$f") bytes)";
  else echo "  MISSING  $A  $([ -f "$R/$A/FAILED.json" ] && echo '(FAILED.json present)')"; fi
done
