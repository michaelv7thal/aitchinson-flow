#!/usr/bin/env bash
# Sequentially execute the three "unconditional generation remedies" from
# NOTE_WHY_UNCONDITIONAL_FAILS.md after the AE-scaling sweep finishes.
#
# Each step is idempotent: re-running this script after an interruption
# resumes where it left off (skip cells whose target artefact exists).
#
# Run order (queued):
#   1. SDE re-eval each AE-EqM cell      → runs/<cell>/eqm/eval_sde.json
#   2. Best-of-8 eval each AE-EqM cell   → runs/<cell>/eqm/eval_bestof8.json
#   3. Train VAE-AE + EqMVAE             → runs/vae_d256_l2_z64/{ae,eqm}/
#      and run recovery on the EqMVAE    → runs/vae_d256_l2_z64/eqm/recovery.json
#
# Compute is unmetered, so we run sequentially (each step uses ~5–12 GiB
# of the 20 GiB MIG slice; sequential is safe + reproducible).

set -u
cd /home/renku/work/aitchinson-flow
unset PYTHONPATH
LOG=runs/_logs/unconditional_remedies.log
mkdir -p runs/_logs
PY=.venv/bin/python

CELLS="ae_d256_l2_z64 ae_d512_l4_z128 ae_d1024_l6_z256 ae_d1024_l8_z512"

echo "[remedies] $(date '+%H:%M:%S') waiting for post-sweep dense rerun pid=130941" >> "$LOG"
while kill -0 130941 2>/dev/null; do sleep 60; done
echo "[remedies] $(date '+%H:%M:%S') previous chain finished, starting remedies" >> "$LOG"

# ─── Step 1: SDE re-eval ──────────────────────────────────────────────
for cell in $CELLS; do
    ckpt=runs/$cell/eqm/epoch_final.pt
    out=runs/$cell/eqm/eval_sde.json
    if [[ ! -f "$ckpt" ]]; then
        echo "[skip-sde] $cell (no ckpt)" >> "$LOG"
        continue
    fi
    if [[ -f "$out" ]]; then
        echo "[skip-sde] $cell (eval_sde.json exists)" >> "$LOG"
        continue
    fi
    echo "[sde] $cell start=$(date '+%H:%M:%S')" >> "$LOG"
    $PY scripts/sample_best_of_n.py \
        --ckpt "$ckpt" \
        --n-samples 256 --best-of 1 --steps 200 \
        --sampler sde --sde-alpha 0.1 \
        --out "$out" >> "runs/_logs/${cell}_sde.log" 2>&1
    rc=$?
    echo "[sde-done] $cell rc=$rc at $(date '+%H:%M:%S')" >> "$LOG"
done

# ─── Step 2: Best-of-8 NAG eval ───────────────────────────────────────
for cell in $CELLS; do
    ckpt=runs/$cell/eqm/epoch_final.pt
    out=runs/$cell/eqm/eval_bestof8.json
    if [[ ! -f "$ckpt" ]]; then
        echo "[skip-bestof] $cell (no ckpt)" >> "$LOG"
        continue
    fi
    if [[ -f "$out" ]]; then
        echo "[skip-bestof] $cell (eval_bestof8.json exists)" >> "$LOG"
        continue
    fi
    echo "[bestof] $cell start=$(date '+%H:%M:%S')" >> "$LOG"
    $PY scripts/sample_best_of_n.py \
        --ckpt "$ckpt" \
        --n-samples 256 --best-of 8 --steps 200 \
        --sampler nag \
        --out "$out" >> "runs/_logs/${cell}_bestof.log" 2>&1
    rc=$?
    echo "[bestof-done] $cell rc=$rc at $(date '+%H:%M:%S')" >> "$LOG"
done

# ─── Step 3: VAE-AE + EqMVAE baseline ────────────────────────────────
VAE_CELL=vae_d256_l2_z64
VAE_DIR=runs/$VAE_CELL
mkdir -p "$VAE_DIR"

# 3a. Train the VAE
if [[ ! -f $VAE_DIR/ae/epoch_final.pt ]]; then
    echo "[vae-train] start=$(date '+%H:%M:%S')" >> "$LOG"
    $PY scripts/train_autoencoder.py \
        --out $VAE_DIR/ae \
        --d-model 256 --num-layers 2 --nhead 4 --d-latent 64 \
        --denoising-schedule relative_uniform --denoising-sigma 0.5 \
        --latent-l2 1e-3 \
        --mode vae --vae-beta 0.1 --vae-beta-warmup-epochs 1 \
        --epochs 5 --windows 10000 --batch-size 64 --lr 3e-4 --seed 42 \
        >> "runs/_logs/${VAE_CELL}_ae.log" 2>&1
    echo "[vae-train-done] rc=$? at $(date '+%H:%M:%S')" >> "$LOG"
fi

# 3b. Train EqMVAE on the VAE latents (uses the same EqMAE model class —
#    encode_sample is automatically stochastic in VAE mode).
if [[ ! -f $VAE_DIR/eqm/epoch_final.pt && -f $VAE_DIR/ae/epoch_final.pt ]]; then
    YAML=$VAE_DIR/_eqm_sweep.yaml
    cat > "$YAML" <<YAML
- name: eqm
  overrides:
    training.model_name: EqMAE
    training.epochs: 5
    training.seed: 42
    text8_dataset.max_train_windows: 10000
    autoencoder.d_model: 256
    autoencoder.num_layers: 2
    autoencoder.nhead: 4
    autoencoder.d_latent: 64
    autoencoder.denoising_schedule: relative_uniform
    autoencoder.denoising_sigma: 0.5
    autoencoder.latent_l2: 0.001
    autoencoder.activation: gelu
    autoencoder.tie_embeddings: true
    autoencoder.mode: vae
    autoencoder.vae_beta: 0.1
    autoencoder.vae_beta_warmup_epochs: 1
    eqm_ae.ae_ckpt_path: $VAE_DIR/ae/epoch_final.pt
    eqm.lambda_ce: 0.5
    eqm.gamma_power: 0.5
    loss.mode: mse
YAML
    # Per-cell source_sigma — measure first.
    SOURCE_SIGMA=$($PY -c "
import sys
sys.path.insert(0, 'src')
import torch, json
from dataclasses import replace
from aitchinson_flow.config import Config
from aitchinson_flow.models import build_model
from aitchinson_flow.training import build_training_datamodule
cfg = Config()
cfg.training = replace(cfg.training, model_name='TextAE')
cfg.autoencoder = replace(cfg.autoencoder, mode='vae', vae_beta=0.1,
                          d_model=256, num_layers=2, nhead=4, d_latent=64,
                          denoising_schedule='relative_uniform', denoising_sigma=0.5,
                          latent_l2=1e-3, activation='gelu', tie_embeddings=True)
m = build_model(cfg).to(cfg.training.device)
state = torch.load('$VAE_DIR/ae/epoch_final.pt', map_location=cfg.training.device, weights_only=False)
m.load_state_dict(state['model_state_dict'])
m.eval()
dm, _ = build_training_datamodule(cfg)
val = dm.splits.val.long()[:512].to(cfg.training.device)
with torch.no_grad():
    z = m.encode(val)  # μ for VAE
print(z.std().item())
" 2>>"runs/_logs/${VAE_CELL}_sigma.log")
    echo "[vae-eqm] source_sigma=$SOURCE_SIGMA, training EqMVAE" >> "$LOG"
    # Patch the YAML with eqm.source_sigma override
    $PY -c "
import yaml
spec = yaml.safe_load(open('$YAML').read())
spec[0]['overrides']['eqm.source_sigma'] = float('$SOURCE_SIGMA')
open('$YAML', 'w').write(yaml.safe_dump(spec, sort_keys=False))
"
    $PY scripts/run_sweep.py --sweep "$YAML" --runs-root "$VAE_DIR" \
        >> "runs/_logs/${VAE_CELL}_eqm.log" 2>&1
    echo "[vae-eqm-done] rc=$? at $(date '+%H:%M:%S')" >> "$LOG"
fi

# 3c. Recovery on the EqMVAE
if [[ ! -f $VAE_DIR/eqm/recovery.json && -f $VAE_DIR/eqm/epoch_final.pt ]]; then
    echo "[vae-recovery] start=$(date '+%H:%M:%S')" >> "$LOG"
    $PY scripts/recovery_check.py \
        --ckpt $VAE_DIR/eqm/epoch_final.pt \
        --alphas 0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,0.70,0.80,1.00 \
        --n 256 --steps 200 \
        --out $VAE_DIR/eqm/recovery.json \
        >> "runs/_logs/${VAE_CELL}_recovery.log" 2>&1
    echo "[vae-recovery-done] rc=$? at $(date '+%H:%M:%S')" >> "$LOG"
fi

# 3d. Optional: SDE eval on the VAE (the headline use case — Gaussian latent
#     prior + EBM refinement should give the strongest unconditional samples).
if [[ ! -f $VAE_DIR/eqm/eval_sde.json && -f $VAE_DIR/eqm/epoch_final.pt ]]; then
    echo "[vae-sde-eval] start=$(date '+%H:%M:%S')" >> "$LOG"
    $PY scripts/sample_best_of_n.py \
        --ckpt $VAE_DIR/eqm/epoch_final.pt \
        --n-samples 256 --best-of 1 --steps 200 \
        --sampler sde --sde-alpha 0.1 \
        --out $VAE_DIR/eqm/eval_sde.json \
        >> "runs/_logs/${VAE_CELL}_sde.log" 2>&1
    echo "[vae-sde-eval-done] rc=$? at $(date '+%H:%M:%S')" >> "$LOG"
fi

echo "[remedies] $(date '+%H:%M:%S') all steps done" >> "$LOG"
