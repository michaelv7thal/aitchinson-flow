# Healing (inpainting) benchmark — consolidated results

- **git**: `12c9cebfe8ed`  **gpu**: NVIDIA RTX PRO 1000 Blackwell Generation Laptop GPU  **created**: 2026-08-10T22:30:59+00:00
- **config**: split=test n_demo=64 n_seeds=3 corrupt_rate=0.15 nfe=100 target_fprs=0.02,0.05,0.10
- localizers: nll (denoiser surprise t=3) · blr (linear hinge t≈4.5) · bgmm (DP density t=7.5). GPT2-SE excluded (no inpainting).

Operating point = **least-damaging** (max net/corrupt over the FPR sweep). `loc_*` are the *localization* metrics — how well the detector finds the corrupt tokens; `fix`/`damage`/`net` are what the *inpainter* then does with them.

## Synthetic text8 — heal @ selected operating point

| localizer | scheme | fpr | loc_P | loc_R | loc_F1 | fix | damage | net/corrupt |
|---|---|---|---|---|---|---|---|---|
| nll | replace | 0.02 | +0.811 | +0.764 | +0.787 | +0.595 | +0.029 | +0.432 |
| nll | falseinfo | 0.02 | +0.341 | +0.090 | +0.142 | +0.018 | +0.020 | -0.146 |
| blr | replace | 0.02 | +0.669 | +0.731 | +0.699 | +0.510 | +0.046 | +0.252 |
| blr | falseinfo | 0.02 | +0.267 | +0.068 | +0.109 | +0.012 | +0.017 | -0.122 |
| bgmm | replace | 0.02 | +0.581 | +0.446 | +0.504 | +0.337 | +0.026 | +0.191 |
| bgmm | falseinfo | 0.02 | +0.185 | +0.037 | +0.061 | +0.005 | +0.009 | -0.070 |

## Real biomedical article (insulin) — Track A (char noise) / Track C (false info)

| localizer | track | fpr | loc_P | loc_R | loc_F1 | fix | damage | net/corrupt |
|---|---|---|---|---|---|---|---|---|
| nll | A | 0.02 | +0.771 | +0.785 | +0.778 | +0.587 | +0.037 | +0.377 |
| nll | C | 0.02 | +0.556 | +0.197 | +0.291 | +0.034 | +0.018 | -0.106 |
| gmm | A | 0.02 | +0.198 | +0.015 | +0.029 | +0.014 | +0.001 | +0.010 |
| gmm | C | 0.02 | +0.085 | +0.011 | +0.019 | +0.000 | +0.001 | -0.005 |
| bgmm | A | 0.02 | +0.582 | +0.358 | +0.443 | +0.261 | +0.026 | +0.117 |
| bgmm | C | 0.02 | +0.426 | +0.068 | +0.117 | +0.007 | +0.007 | -0.047 |

**Takeaway:** healing recovers *geometric* corruption (replace/char-noise) but not *contextual/false-info* — for false info the localizers flag few words and cannot restore the true token (net ≤ 0). Detection ≠ healing.

## Figure
- `figs/heal_compare.png`
