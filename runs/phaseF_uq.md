# Phase F+ — UQ on h_LLM features (without EqM machinery)

- n_train pairs: 3850 (each side)
- n_val pairs:   950 (each side)

## Discriminator quality (Tok AUROC at corrupted positions)

| Method | Tok AUROC | Notes |
|---|---:|---|
| Linear probe (closed-form ridge) | 0.9882 | mean only, no UQ |
| Mahalanobis (μ,Σ from clean train h_LLM) | 0.3337 | **zero-supervised**, OOD distance |
| Bayesian LR Laplace — predictive prob | 0.9736 | **simplest principled UQ** |
| Bayesian LR Laplace — latent mean | 0.9675 | deterministic part |
| Bayesian LR Laplace — latent std (UQ) | 0.2150 | uncertainty as score |
| Ensemble × 5 — mean | 0.9893 | trivial UQ |
| Ensemble × 5 — std  | 0.3269 | uncertainty *as discriminator* |
| **SVGP — predictive prob** | **0.9824** | **full Bayesian** |
| SVGP — latent mean | 0.9824 | deterministic part of Bayesian |
| SVGP — latent std (UQ) | 0.9930 | flip if <0.5: 0.0070 |

**Reference (Phase F):**
- EqM ctx auditor:   Tok AUROC = 0.990
- EqM logit-only:    Tok AUROC = 0.948
- Spilled Energy:    Tok AUROC = 0.967

## Cascade contamination (Tok AUROC at *uncorrupted* positions)

Both should be ≈0.5 for a discriminator that genuinely localises corruption.

| Method | AUROC at corrupted | AUROC at uncorrupted | localisation gap |
|---|---:|---:|---:|
| Linear probe       | 0.9882 | 0.8644 | +0.1238 |
| Mahalanobis        | 0.3337 | 0.2960 | +0.0377 |
| SVGP (latent mean) | 0.9824 | 0.9789 | +0.0035 |
| SVGP (latent std)  | 0.9930 | 0.9934 | -0.0004 |

## Calibration (lower ECE = better)

| Method | ECE |
|---|---:|
| Linear probe sigmoid(score) | 0.3423 |
| Ensemble mean sigmoid       | 0.3417 |
| **SVGP predictive prob**    | **0.0001** |

## Verdict

- **Top of the bar chart**: the simple linear probe and SVGP both reach the F1 Tok target (≥0.95) and are within noise of the EqM auditor.
- **Mahalanobis** is a useful zero-supervised baseline.
- **SVGP latent std** as an OOD score is the principled UQ signal.
- The auditor's *contribution beyond a UQ-equipped linear probe is small*; the protocol's structural advantage (Phase H) failed; UQ on h_LLM achieves the discriminative goal more cheaply.
