# Healing (inpainting) benchmark — consolidated results

- **git**: `0a059199070c`  **gpu**: NVIDIA RTX PRO 1000 Blackwell Generation Laptop GPU  **created**: 2026-09-05T16:31:52+00:00
- **config**: split=test n_demo=64 n_seeds=3 corrupt_rate=0.15 nfe=100 target_fprs=0.02,0.05,0.10
- localizers: nll (`--localizer nll --t-nll 3.0`) · blr (`--localizer linear`) · bgmm (`--localizer bgmm --t-eval 7.5 --bgmm-covariance-type full --bgmm-pca-dim 64 --bgmm-max-iter 1000`) · blr_adv (`--localizer linear_all --t-eval 7.5`) · blr_fi (`--localizer linear_fi --t-eval 4.5`) · logreg_fi (`--localizer logistic_fi --t-eval 4.5`) · var (`--localizer var --t-eval 7.5 --var-ridge 0.1`) · logreg_pl (`--localizer logistic_pl --t-eval 4.5 --plaus-fit-seqs 128`). GPT2-SE and GPT2-NLL excluded (external scorers, no inpainting path).

Operating point = **least-damaging** (max net/corrupt over the FPR sweep). `loc_*` are the *localization* metrics — how well the detector finds the corrupt tokens; `fix`/`damage`/`net` are what the *inpainter* then does with them.

## Synthetic text8 — heal @ selected operating point

| localizer | scheme | fpr | loc_P | loc_R | loc_F1 | fix | damage | net/corrupt |
|---|---|---|---|---|---|---|---|---|
| nll | replace | 0.02 | +0.811 | +0.764 | +0.787 | +0.595 | +0.029 | +0.432 |
| nll | shuffle | 0.02 | +0.715 | +0.664 | +0.689 | +0.529 | +0.039 | +0.279 |
| nll | falseinfo | 0.02 | +0.341 | +0.090 | +0.142 | +0.018 | +0.020 | -0.146 |
| nll | plausible | 0.02 | +0.002 | +0.000 | +0.001 | +0.000 | +0.026 | -0.211 |
| blr | replace | 0.02 | +0.669 | +0.731 | +0.699 | +0.510 | +0.046 | +0.252 |
| blr | shuffle | 0.02 | +0.505 | +0.568 | +0.535 | +0.372 | +0.064 | -0.039 |
| blr | falseinfo | 0.02 | +0.267 | +0.068 | +0.109 | +0.012 | +0.017 | -0.122 |
| blr | plausible | 0.02 | +0.002 | +0.001 | +0.001 | +0.000 | +0.024 | -0.193 |
| bgmm | replace | 0.02 | +0.581 | +0.446 | +0.504 | +0.337 | +0.026 | +0.191 |
| bgmm | shuffle | 0.02 | +0.520 | +0.432 | +0.472 | +0.325 | +0.031 | +0.124 |
| bgmm | falseinfo | 0.02 | +0.185 | +0.037 | +0.061 | +0.005 | +0.009 | -0.070 |
| bgmm | plausible | 0.02 | +0.041 | +0.008 | +0.013 | +0.000 | +0.012 | -0.095 |
| blr_adv | replace | 0.02 | +0.652 | +0.579 | +0.613 | +0.426 | +0.028 | +0.269 |
| blr_adv | shuffle | 0.02 | +0.514 | +0.417 | +0.460 | +0.315 | +0.027 | +0.143 |
| blr_adv | falseinfo | 0.02 | +0.106 | +0.049 | +0.067 | +0.009 | +0.012 | -0.087 |
| blr_adv | plausible | 0.02 | +0.000 | +0.000 | +0.000 | +0.000 | +0.010 | -0.080 |
| blr_fi | replace | 0.02 | +0.383 | +0.428 | +0.404 | +0.139 | +0.104 | -0.446 |
| blr_fi | shuffle | 0.02 | +0.310 | +0.312 | +0.311 | +0.101 | +0.092 | -0.483 |
| blr_fi | falseinfo | 0.02 | +0.434 | +0.199 | +0.273 | +0.052 | +0.027 | -0.167 |
| blr_fi | plausible | 0.02 | +0.003 | +0.001 | +0.001 | +0.000 | +0.018 | -0.147 |
| logreg_fi | replace | 0.02 | +0.300 | +0.433 | +0.355 | +0.108 | +0.149 | -0.730 |
| logreg_fi | shuffle | 0.02 | +0.256 | +0.341 | +0.293 | +0.105 | +0.126 | -0.697 |
| logreg_fi | falseinfo | 0.02 | +0.436 | +0.213 | +0.286 | +0.043 | +0.024 | -0.154 |
| logreg_fi | plausible | 0.02 | +0.004 | +0.001 | +0.001 | +0.000 | +0.013 | -0.106 |
| var | replace | 0.02 | +0.398 | +0.122 | +0.187 | +0.091 | +0.011 | +0.026 |
| var | shuffle | 0.02 | +0.292 | +0.091 | +0.138 | +0.064 | +0.013 | -0.017 |
| var | falseinfo | 0.02 | +0.149 | +0.038 | +0.060 | +0.003 | +0.007 | -0.053 |
| var | plausible | 0.02 | +0.098 | +0.025 | +0.040 | +0.000 | +0.009 | -0.068 |
| logreg_pl | falseinfo | 0.02 | +0.056 | +0.012 | +0.020 | +0.000 | +0.008 | -0.066 |
| logreg_pl | plausible | 0.02 | +0.519 | +0.435 | +0.473 | +0.027 | +0.019 | -0.125 |

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
