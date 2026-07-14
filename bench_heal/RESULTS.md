# Healing (inpainting) benchmark — consolidated results

- **git**: `ebe9fb38c4c4`  **gpu**: NVIDIA RTX PRO 1000 Blackwell Generation Laptop GPU  **created**: 2026-07-13T22:41:21+00:00
- **config**: split=test n_demo=64 n_seeds=3 corrupt_rate=0.15 nfe=100 target_fprs=0.02,0.05,0.10
- localizers: nll (denoiser surprise t=3) · blr (linear hinge t≈4.5) · bgmm (DP density t=7.5). GPT2-SE excluded (no inpainting).

Operating point = **least-damaging** (max net/corrupt over the FPR sweep). `loc_*` are the *localization* metrics — how well the detector finds the corrupt tokens; `fix`/`damage`/`net` are what the *inpainter* then does with them.

## Synthetic text8 — heal @ selected operating point

| localizer | scheme | fpr | loc_P | loc_R | loc_F1 | fix | damage | net/corrupt |
|---|---|---|---|---|---|---|---|---|
| nll | replace | 0.02 | +0.825 | +0.757 | +0.790 | +0.575 | +0.028 | +0.420 |
| nll | falseinfo | 0.02 | +0.347 | +0.089 | +0.142 | +0.021 | +0.020 | -0.140 |
| blr | replace | 0.02 | +0.666 | +0.691 | +0.679 | +0.461 | +0.046 | +0.202 |
| blr | falseinfo | 0.02 | +0.261 | +0.058 | +0.095 | +0.010 | +0.015 | -0.113 |
| bgmm | replace | 0.02 | +0.599 | +0.318 | +0.416 | +0.242 | +0.017 | +0.144 |
| bgmm | falseinfo | 0.02 | +0.185 | +0.032 | +0.055 | +0.003 | +0.008 | -0.059 |

## Real biomedical article (insulin) — Track A (char noise) / Track C (false info)

| localizer | track | fpr | loc_P | loc_R | loc_F1 | fix | damage | net/corrupt |
|---|---|---|---|---|---|---|---|---|
| nll | A | 0.02 | +0.785 | +0.787 | +0.786 | +0.581 | +0.037 | +0.374 |
| nll | C | 0.02 | +0.513 | +0.196 | +0.283 | +0.040 | +0.023 | -0.142 |
| gmm | A | 0.02 | +0.132 | +0.012 | +0.021 | +0.011 | +0.001 | +0.007 |
| gmm | C | 0.02 | +0.066 | +0.012 | +0.020 | +0.000 | +0.001 | -0.004 |
| bgmm | A | 0.02 | +0.589 | +0.329 | +0.422 | +0.241 | +0.021 | +0.122 |
| bgmm | C | 0.02 | +0.246 | +0.059 | +0.096 | +0.003 | +0.010 | -0.075 |

**Takeaway:** healing recovers *geometric* corruption (replace/char-noise) but not *contextual/false-info* — for false info the localizers flag few words and cannot restore the true token (net ≤ 0). Detection ≠ healing.

## Figure
- `figs/heal_compare.png`
