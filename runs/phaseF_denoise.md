# Phase F denoising test — aud_gpt2_ctx

- ckpt: `runs/aud_gpt2_ctx/epoch_final.pt`
- ctx mode: product_concat
- n_eval: 60, n_corrupt: 950, n_uncorrupt: 2890
- K (descent steps): 30, η: 0.05, γ: 1.0

## Distance to clean simplex (mean per-position L2²)

|   region   | before | after | Δ% mean | % positions where d ↓ |
|------------|-------:|------:|--------:|----------------------:|
| **corrupted** | 22.858 | 22.334 | +0.5% | 62.3% |
| uncorrupted | 11.054 | 10.597 | +479729504.0% | 67.4% |

## Argmax-slot flips during descent

|   region   | flips toward clean | flips away from clean |
|------------|-------------------:|----------------------:|
| **corrupted** (950) | 0 | 3 |
| uncorrupted (2890) | 0 | 3 |

## Verdict

**The auditor's −∇E direction does NOT point toward clean** at corrupted positions. The energy field acts as a discriminator (small E at clean, large E at invalid) but its gradient does not give a useful denoising signal. The auditor's structural advantage over a linear probe is therefore *not* in the energy gradient itself.
