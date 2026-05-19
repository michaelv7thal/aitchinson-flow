# DFM + SVGP for OOD detection — findings

Two-stage pipeline: Stage 1 trains a Dirichlet Flow Matching denoiser
(Stark et al. 2024) on text8; Stage 2 freezes it and fits a Sparse
Variational GP head on pooled hidden states via a contrastive energy
hinge using `token_ids_invalid` (CorruptingCollate) as negatives. The
SVGP latent mean (read as a Bernoulli probability) is the OOD score.

Canonical run dir: `runs/dfm_svgp_pure50_lr3e4/`.

---

## Headline results

**Stage 1 — pure DFM** (1024 d_model / 8 layers, 50 ep, lr 3e-4):

| metric | value |
|---|---|
| KL_uni | **0.0074** |
| KL_bi | 0.4545 |
| KL_tri | 2.541 |
| H_ratio | 1.022 |

Best unconditional generation in the project; samples are recognisably
text-like (`"and ancmnau and is is read caw or yeria"`).

**Stage 2 — hinge SVGP** (frozen DFM, d_embed=256, kernel Matérn-5/2,
ℓ init = √d_embed = 16, margin = 2.0, 5 epochs):

| OOD split | mean(prob) | **AUROC(prob, OOD=1)** |
|---|---|---|
| positive (val) | 0.610 | — |
| scrambled (100% shuffle) | 0.749 | **0.914** |
| random simplex | 0.856 | **0.9995** |

`prob` is the OOD score (higher = more OOD). The hinge trains
`E_invalid > E_clean`, so `prob = sigmoid(E)` is high for OOD. AUROC on
`prob` ≡ AUROC on `E` (sigmoid is monotone).

**Corruption-severity sweep** (`svgp_corruption_sweep.{json,png}`),
trained only on r=0.15 random-replace negatives:

| rate r | replace | shuffle | both |
|---|---|---|---|
| 0.10 | 0.695 | 0.615 | 0.737 |
| 0.30 | 0.901 | 0.785 | 0.938 |
| 0.50 | 0.962 | 0.855 | 0.984 |
| 0.70 | 0.991 | 0.914 | 0.992 |
| 1.00 | **0.997** | **0.929** | **0.997** |

- Detector generalises far beyond its 15%-replace training distribution.
- Random replacement is far easier to detect than shuffle (shuffle
  preserves unigram statistics, the DFM-CE objective's strongest cue;
  ceiling ≈ 0.93).
- Combined ≈ replace (replacement signal dominates).

---

## Known limitation: SVGP predictive variance is uninformative

`std ≈ 1.59` constant for positive **and** OOD inputs (AUROC(std)=0.50).
Cause: concentration of measure. With standardised d_embed=256 features,
‖x − z_inducing‖ ≈ √(2·256) ≈ 22.6 ± O(1) for *every* query, so `K_xZ`
is the same magnitude for all inputs and the variance term saturates.
The OOD signal lives entirely in the trained **mean** (via `prob`), not
the epistemic variance.

Mitigations attempted and outcome:

| change | result |
|---|---|
| d_embed 256 → 27 (ℓ tied to √d) | no change — r/ℓ ratio invariant to d |
| d_embed=27 + ℓ=1 fixed-small | worse (kernel collapse; AUROC 0.91→0.81) |
| scale DFM to 1280/10L | data-starved, worse (KL_uni 0.0074→0.013) |
| larger lr 3e-4 → 1e-3 | worse (KL_uni 0.0074→0.129) |

Recovering an input-dependent variance would need ARD lengthscales or a
contrastive pooler-pretraining step — deferred as future work.

---

## Reproduce

All commands prefixed with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
and run from repo root with the project venv.

```bash
# Stage 1 — pure DFM (50 ep, lr 3e-4). ~13 min on the 8 GB GPU.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/run_sweep.py \
    --sweep sweeps/_dfm_svgp_pure50_lr3e4.yaml --runs-root runs

# Stage 2 — hinge-trained SVGP on the frozen DFM. ~30 s.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/fit_dfm_svgp_hinge.py \
    --ckpt runs/dfm_svgp_pure50_lr3e4/epoch_final.pt \
    --n-epochs 5 --lr 1e-3 --eval-n 500

# Qualitative — GT / unconditional / OOD valid+invalid with scores.
python scripts/eval_dfm_svgp_qualitative.py \
    --ckpt runs/dfm_svgp_pure50_lr3e4/model_with_svgp_hinge.pt --n 10

# Corruption-severity sweep — 10 rates × {replace, shuffle, both}.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/sweep_dfm_svgp_corruption.py \
    --ckpt runs/dfm_svgp_pure50_lr3e4/model_with_svgp_hinge.pt --n 500
```

### Artefacts

| file | content |
|---|---|
| `runs/dfm_svgp_pure50_lr3e4/epoch_final.pt` | Stage 1 DFM checkpoint |
| `runs/dfm_svgp_pure50_lr3e4/eval.json` | Stage 1 unconditional KLs |
| `runs/dfm_svgp_pure50_lr3e4/model_with_svgp_hinge.pt` | Stage 2 checkpoint |
| `runs/dfm_svgp_pure50_lr3e4/svgp_hinge_eval.json` | OOD AUROC (scrambled, random_simplex) |
| `runs/dfm_svgp_pure50_lr3e4/svgp_qualitative.json` | text + score report |
| `runs/dfm_svgp_pure50_lr3e4/svgp_corruption_sweep.{json,png}` | severity sweep table + plot |
| `runs/dfm_svgp_pure50_lr3e4/svgp_hinge_d27_l1.log` | failed ℓ=1 variant (kept for honest scope) |

### Config knobs (`cfg.dfm_svgp`)

`pooling=mean`, `d_embed=256`, `n_inducing=128`, `kernel=matern52`,
`t_eval=4.5`, `margin_energy=2.0`. Stage 1 sweep YAML:
`sweeps/_dfm_svgp_pure50_lr3e4.yaml`. Model class:
`aitchinson_flow.models.dirichlet_fm_svgp.DirichletFMSvgp`.
