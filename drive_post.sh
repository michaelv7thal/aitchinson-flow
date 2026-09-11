#!/usr/bin/env bash
# Post-processing chain (crash-safe, fully autonomous):
#   0. wait for the running L256 train driver to finish
#   1. retrain FMonCLR @ L256 with the NEW Lipman code (--force)  [B1 used old code]
#   2. L256 eval + recovery for all 7 (drive_eval.sh)
#   3. L40 train all 7 with the now-fixed code (drive_train.sh, SC=local)
#   4. L40 eval + recovery for all 7 (drive_eval.sh)
# EqMAE is NOT retrained @ L256: it launched after the on-disk edits, so it
# already used the fixed VAE-load path. L40 trains everything fresh on fixed code.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/renku/work/aitchinson-flow
SCL=a100_20g_L256
R="runs/sflm_bench_${SCL}"; LOG="${R}/_driver"; mkdir -p "$LOG"
ts(){ date +%Y-%m-%d_%H:%M:%S; }
echo "### POST CHAIN START $(ts)"

# --- 0. wait for the L256 train driver to complete -------------------------
echo "### [$(ts)] waiting for L256 train driver (TRAIN DRIVER COMPLETE marker)"
while ! grep -q "TRAIN DRIVER COMPLETE" "${LOG}/driver_train.out" 2>/dev/null; do
  sleep 30
done
echo "### [$(ts)] L256 train driver finished"
[ -f "$R/EqMAE/epoch_final.pt" ] && echo "  EqMAE ckpt OK ($(stat -c%s "$R/EqMAE/epoch_final.pt") bytes)" \
                                 || echo "  WARNING: EqMAE ckpt MISSING"

# --- 1. retrain FMonCLR @ L256 (new code) ----------------------------------
echo "### [$(ts)] STEP 1: retrain FMonCLR @ L256 (new Lipman code, --force)"
python scripts/train_for_sflm_bench.py --scale ${SCL} --force --epochs 10 \
    --max-train-windows 10000 --only FMonCLR \
    > "${LOG}/retrain_FMonCLR.log" 2>&1
echo "### [$(ts)] FMonCLR retrain exit=$?  meta=$(cat $R/FMonCLR/train_meta.json 2>/dev/null | tr -d '\n ')"

# --- 2. L256 eval + recovery (all 7) ---------------------------------------
echo "### [$(ts)] STEP 2: L256 eval + recovery (all 7)"
SC=${SCL} bash drive_eval.sh > "${LOG}/driver_eval.out" 2>&1
echo "### [$(ts)] L256 eval driver exit=$?"

# --- 3. L40 train (all 7, fixed code) --------------------------------------
echo "### [$(ts)] STEP 3: L40 train (SC=local, fixed code)"
mkdir -p runs/sflm_bench_local/_driver
SC=local EP=10 MTW=10000 VAE_L=40 bash drive_train.sh \
    > "runs/sflm_bench_local/_driver/driver_train.out" 2>&1
echo "### [$(ts)] L40 train driver exit=$?"

# --- 4. L40 eval + recovery (all 7) ----------------------------------------
echo "### [$(ts)] STEP 4: L40 eval + recovery (all 7)"
SC=local bash drive_eval.sh > "runs/sflm_bench_local/_driver/driver_eval.out" 2>&1
echo "### [$(ts)] L40 eval driver exit=$?"

echo "### POST CHAIN COMPLETE $(ts)"
