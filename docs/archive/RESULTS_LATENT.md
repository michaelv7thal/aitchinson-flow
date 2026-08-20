# Latent-EqM (learned-embedding EqM) — initial results


> **HISTORICAL (2026-05-09). Superseded; retained as the only record of the fixed-embedding recipe statistics (§"Fixed-embedding variant").**
>
> This documents the `EqMLatent` arm — Equilibrium Matching on a learnable `nn.Embedding`, tied decoder, no VAE. **It is not the paper's "EqM, VAE latent" row** (`tab:gen`), which is a genuine VAE at d_latent=64, β=0.1 (`tab:config`; frozen VAE config preserved at `runs/vae_a100_20g_L256/`). Do not read the numbers here as that arm's.
>
> The paper's verdict differs on which lever matters: the metric is a null, as this document says, but the cause is the regression TARGET, not the representation — the VAE-latent arm "changed the numbers without changing the outcome" (paper §The Negative Result). One live item survives: the `ce_min_gamma=0.0` result below (§"The trivial-CE-shortcut and the structural fix") is the configuration the paper lists as future work, already run here on this arm.
>
> Dead pointers: `scripts/sampler_artefact_check.py`, `scripts/hilbert_counterexample.py` (output survives at `runs/hilbert_counter/results.json`), `scripts/eqm_mnist_sanity.py`.

Generated 2026-05-09. The simplex-EqM Phase 0-3 results in
`RESULTS_DIRICHLET.md` motivate this pivot: the simplex *representation*
(one-hot → CLR) is the bottleneck, not the metric on it.

## What changed

`models/eqm_latent.py` introduces `EquilibriumFlowMatchingLatent`:

* Data lives in `R^{d_embed}` (default `d=32`), not the K-simplex.
* `nn.Embedding(K, d_embed)` is a *trainable* parameter — flow matching and
  the CE auxiliary share the same embedding matrix (tied decoder
  `z @ embed.weight.T`).
* Source noise is centred Gaussian in `R^d` (no V_d projection).
* Flow regression is plain MSE — geometry is whatever the embedding learns.
* `decode_to_logprobs(z) = log_softmax(decode_to_logits(z))` — no CLR shortcut.
* Aux CE on the implied-x1 reconstruction maps `pred_x1 ∈ R^d` →
  K-dim logits via tied weight, then `log_softmax` + NLL. This is what pulls
  embeddings apart so the flow loss can't trivially collapse them.
* `embed_pairwise_min` and `embed_norm_mean` are logged each step as the
  collapse-canary diagnostics.

What stays:

* Conservative-gradient parameterisation `f` ↦ `∇⟨x, f⟩`.
* NAG-GD and Euler samplers.
* γ importance sampling, decay strategies, ce_min_gamma gating.
* All EBM machinery (`energy`, `position_uncertainty`).

## Phase 2 — sanity (1 epoch, d=32, default everything else)

Training-end metrics:

```
flow_loss=0.20  ce=0.92                    embed.norm_mean=1.04
g<.33=0.10      g<.66=0.21    g<1=0.39      embed.pairwise_min=0.84
                                            embed.pairwise_mean=1.47
```

Embeddings did *not* collapse — `pairwise_min=0.84` and `norm_mean=1.04`
(grew 5.8× from the `1/√32 ≈ 0.18` init), confirming CE pulled them apart
while the flow loss didn't trivialise them.

## The breakthrough — `sample_return_best` is the entire artefact

The same checkpoint, evaluated under two NAG-GD modes (`scripts/sampler_artefact_check.py`):

| sampler                      | KL_uni | KL_bi   | H_gen vs H_gt | sample[0]                                  |
|------------------------------|-------:|--------:|--------------:|--------------------------------------------|
| `return_best=True` (default) | 0.596  | **4.98** | 3.22 (over-spread) | `miagghxwfhzumwq gegxansj poteswwaodgra v` |
| `return_best=False`          | 0.381  | **1.86** | 2.57 (under-spread)| `' ilngsetehcemse gegyyest pgtescenomnrl c'` |

The `return_best=True` flag (introduced in `cfg.eqm.sample_return_best`,
default True since Phase R, `models/eqm.py:436-450`) returns the iterate
with the smallest mean ‖∇E‖. For an undertrained EBM the lowest-grad iterate
is the *noise init* — there's a basin at γ≈0 because the model has only
seen Dirichlet-style data targets and never learned that noise should be
high-energy. NAG starts there, leaves briefly, but the trajectory minimum
reverts to step 0.

This explains the **bit-identical** Phase 3 / Phase 4 / Phase 5 KL values
in the simplex experiments (RESULTS_DIRICHLET.md): same seed → same
`torch.randn` after deterministic training → same noise init → same
"best" iterate → same decoded samples regardless of model.

**With `return_best=False`, latent EqM at 1 epoch already delivers
KL_bi=1.86 — beating baseline_5ep's 1.99 and approaching the unigram
baseline's 1.69.** And `H_gen=2.57 < H_gt=2.85` means the generation is
actually *less* entropic than the corpus (no longer the uniform-noise
output). Sample[0] has spaces, repeated patterns, and 2-3-letter
substrings that look like English fragments.

## Comparison vs simplex baselines

| run                          | KL_uni | KL_bi  | KL_tri | H_ratio |
|------------------------------|-------:|-------:|-------:|--------:|
| unigram baseline (corpus iid)| 0.0009 | 1.6917 | 7.158  | 0.995   |
| baseline_5ep (det CLR + MSE) | 0.0513 | 1.9874 | 7.217  | 0.955   |
| dphase3 dir (5ep, return_best=True)         | 0.6414 | 5.806  | 13.479 | 1.157   |
| dphase4 λ=3 dir (5ep, return_best=True)     | 0.6414 | 5.806  | 13.479 | 1.157   |
| dphase5 hilbert dir (5ep, return_best=True) | 0.6414 | 5.806  | 13.479 | 1.157   |
| **latent d=32 (1ep, return_best=False)**    | 0.381  | **1.86** | 12.515 | 0.901   |

The 1-epoch latent run **beats every simplex result** on KL_bi, including
the well-trained baseline_5ep. The headline 5-epoch run is in flight; if
the trajectory continues, KL_bi well below 1.69 (the unigram floor) is the
expectation.

## The trivial-CE-shortcut and the structural fix

The 5-epoch `latent_d32_seed42` headline (now discarded) showed KL_bi
**monotonically regressing** epoch-by-epoch even as flow_loss and CE both
dropped:

| epoch | flow_loss | ce    | embed.norm | embed.pwise_min | KL_uni | KL_bi |
|-------|----------:|------:|-----------:|----------------:|-------:|------:|
| 1     | 0.20      | 0.92  | 1.04       | 0.84            | 0.36   | 1.93  |
| 2     | 0.15      | 0.18  | 1.18       | 1.01            | 0.22   | 2.07  |
| 3     | 0.12      | 0.13  | 1.28       | 1.12            | 0.30   | 2.88  |
| 4     | 0.10      | 0.11  | 1.34       | 1.18            | 0.18   | 3.30  |
| 5     | 0.10      | 0.10  | 1.37       | 1.21            | 0.19   | **3.46** |

Cause (user-flagged): with **`tie_decoder=True` + `ce_min_gamma=0.5`**, the
CE term is satisfied trivially. At γ=1, `pred_x1 ≈ embed(t)` and the tied
decoder gives `argmax(embed(t) @ embed.weight^T) = t` whenever embedding
rows are distinct. CE drops without the model's grad_g learning to
navigate from noise; flow_loss drops because Adam fits any local target
configuration. NAG sampling starts from noise, drifts to whatever
low-energy basin the model happens to have (which is *not* a data
embedding because the model never had to learn that), and decodes
randomly.

The structural fix — `tie_decoder=False` + `ce_min_gamma=0.0` — forces:

1. The decoder is a separate `nn.Linear(d, K)`, so the embedding can't
   satisfy CE via self-similarity. CE has to push `decode_to_logits(pred_x1)`
   to the right region of `R^K`.
2. CE is applied at every γ ∈ [0,1], so at γ≈0 the model has to deliver
   `grad_g` such that `pred_x1 = x_0 - λ·grad_g` decodes to the right token
   — i.e., the model genuinely navigates from noise to data.

`latent_d32_untied_ce0` (`embedding.tie_decoder=false`,
`eqm.ce_min_gamma=0.0`, `eqm.lambda_ce=1.0`, 5 epochs):

| epoch | flow_loss | ce   | embed.norm | KL_uni | KL_bi |
|-------|----------:|-----:|-----------:|-------:|------:|
| 1     | 0.34      | 2.39 | 0.99       | 0.30   | 2.32  |
| 2     | 0.44      | 1.43 | 1.07       | 0.10   | 2.00  |
| 3     | 0.38      | 1.16 | 1.17       | 0.10   | 1.86  |
| 4     | 0.33      | 1.03 | 1.24       | 0.08   | **1.667** ← below unigram |
| 5     | 0.31      | 0.97 | 1.27       | 0.04   | **1.647** |

Monotonic improvement, KL_bi crosses **below the unigram baseline (1.69)**
at epoch 4 and finishes at 1.647 — the first run in the entire
capstone (simplex baseline_5ep, all dirichlet experiments, even
deterministic Phase 3 parity) that delivers meaningful joint structure
modeling at this compute budget. Trajectory still showing nice reduction
at epoch 5 — extended-training cell `latent_d32_untied_ce0_ep8` queued.

## Fixed-embedding variant (eliminates moving target by construction)

Even with the structural CE fix, jointly training embeddings has the
fundamental friction that `x_1 = embed(token)` is a moving target — the
flow regression is fitting a target that's still adapting. An alternative
is to **pre-compute fixed embeddings** so `x_1` is constant throughout
training. New artefacts:

* `scripts/learn_ppmi_svd_embeddings.py` — PPMI-SVD recipe (Levy &
  Goldberg). For K=27 the PPMI matrix is 27×27 so d_embed ≤ 27 is the
  intrinsic-rank ceiling; we save `d=27` (full rank) and `d=16`
  (truncated).
* `scripts/learn_skipgram_embeddings.py` — word2vec skip-gram trained on
  text8 char pairs in a window of 5. d=27 and d=32 both well-defined.
* `EmbeddingConfig.fixed_path` — when set, EqMLatent loads the
  pre-computed embeddings and `embed.weight.requires_grad_(False)`.
  Sweep cells: `latent_fixed_ppmi_d27_*`, `latent_fixed_skipgram_d27_*`.

Embedding statistics:

| source            | shape   | norm_mean | pairwise_min |
|-------------------|---------|----------:|-------------:|
| PPMI-SVD d=27     | (27,27) | 0.832     | 0.550        |
| PPMI-SVD d=16     | (27,16) | 0.764     | 0.334        |
| skip-gram d=27    | (27,27) | 1.589     | 0.924        |
| skip-gram d=32    | (27,32) | 1.582     | 0.790        |
| trainable d=32 (1ep, ce0+untied) | (27,32) | 0.99 | 0.79 |

Skip-gram has the largest spread (`pairwise_min=0.92`) — naturally
discriminative. PPMI-SVD has smaller spread but is purely closed-form.
The trainable d=32 embedding ends up at `pairwise_min=0.79` after 1
epoch — comparable to skip-gram's static value, suggesting that fixed
skip-gram embeddings could match what trainable embeddings reach without
the moving-target instability.

Both variants are queued in `sweeps/latent_eqm.yaml`. With fixed
embeddings + `tie_decoder=True`, the readout is `z @ embed.weight^T`
where `embed.weight` is also fixed, so CE has *only* the model parameters
to update — directly forcing `pred_x1 = x_γ - λ·grad_g` to be in the
right region of `R^d`.

## Headline numbers so far

| run                         | epochs | KL_uni | KL_bi  | KL_tri | H_ratio | grad_gen | grad_gt |
|-----------------------------|-------:|-------:|-------:|-------:|--------:|---------:|--------:|
| unigram baseline            | —      | 0.001  | 1.692  | 7.158  | 0.995   | —        | —       |
| baseline_5ep (simplex)      | 5      | 0.051  | 1.987  | 7.217  | 0.955   | 0.23     | 0.18    |
| dphase3 simplex (broken-ce) | 5      | 0.641  | 5.806  | 13.479 | 1.157   | 7.63     | 25.33   |
| latent untied_ce0           | 5      | 0.047  | **1.597** | 6.394 | 0.931 | 3.43     | 3.49    |
| latent tied_ce0             | 5      | 0.080  | **1.539** | 7.219 | 0.966 | 3.24     | 2.41    |
| latent untied_ce0_ep8       | 8      | 0.026  | **1.458** | 6.663 | 0.971 | —        | —       |
| latent untied_ce0_lam5      | 5      | 0.069  | 1.610  | 7.599  | 0.984   | —        | —       |

ep8 (8 epochs) gives ~9 % improvement on KL_bi over the 5-epoch headline
(1.597 → 1.458), and the val/train flow_loss never plateaued during
training. The cosine schedule decays lr from 3e-4 to ~3e-5 inside 5
epochs, which truncates training before convergence.

## Long-training variants (queued)

20-epoch cells with `cosine_eta_min=1e-4` so the late-training lr stays
meaningful:

* `latent_d32_untied_ce0_ep20` — extended-training trainable embeddings.
* `latent_fixed_ppmi_d27_tied_ep20` — extended PPMI fixed.
* `latent_fixed_skipgram_d27_tied_ep20` — extended skip-gram fixed.

Each ~70 min, queued behind the PPMI + skip-gram 5-epoch sweeps.

## Sanity checks

### Hilbert counter-example — interior-of-simplex compositional data

`scripts/hilbert_counterexample.py` trains FM on a synthetic 3-Dirichlet
mixture in `S_5` (interior, ratio-structured, where Hilbert geometry was
designed to apply). Two losses × 3 seeds × 4000 steps, MLP velocity field:

| loss          | forward_kl mean ± std | hilbert_match_mean |
|---------------|----------------------:|-------------------:|
| MSE           | 4.989 ± 0.175         | 0.445              |
| Hilbert_soft  | **4.843 ± 0.120**     | 0.463              |

Hilbert beats MSE by ~3 %, *within 1 σ*. Even on the canonical "Hilbert
should help" setup (interior, compositional, ratio-meaningful), the gain
is modest. This **strengthens** the main capstone claim: Hilbert
geometry isn't a major lever, period — not just on one-hot text.

The headline argument becomes: "the choice of representation (one-hot
simplex → learned embedding) dominates the choice of metric (Euclidean
→ Hilbert) by an order of magnitude." Latent-EqM untied_ce0 over the
simplex Phase 3 dirichlet improves KL_bi by 3.6× (5.806 → 1.597);
Hilbert over MSE on the favourable Dirichlet-mixture setting improves
KL by 0.03×. Different scales of effect.

### MNIST EqM — continuous-pixel sanity (queued)

`scripts/eqm_mnist_sanity.py` runs standard EqM on flattened MNIST (D=784,
[-1, 1] normalisation, MLP velocity, no simplex/CLR/embedding). Confirms
the framework works on continuous high-dim data without the simplex
machinery. Queued to run after the long-training cells finish.

## Open calibration knobs (Phase 3)

The user flagged that **CE must do real work** because `x1 = embed(token)`
is a moving target — the flow loss alone could collapse embeddings. With
`lambda_ce=0.5` and `ce_min_gamma=0.5` the CE contribution is currently
~95 % of total loss; embeddings stayed differentiated this run, but
calibration should sweep:

* `lambda_ce ∈ {0.5, 1.0, 2.0, 5.0}` — ensure CE dominates
* `ce_min_gamma ∈ {0.0, 0.25, 0.5}` — apply CE everywhere vs only at high γ
* `source_sigma ∈ {0.1, 0.5, 1.0, 2.0}` — match σ to the embedding scale
* `embedding.init_std_factor ∈ {0.5, 1.0, 2.0}` — initial spread

Track `embed.pairwise_min` as the collapse canary on every cell.

## Files added in this session (latent-EqM half)

| path                                      | role                                  |
|-------------------------------------------|---------------------------------------|
| `src/aitchinson_flow/models/eqm_latent.py` | the learned-embedding EqM             |
| `src/aitchinson_flow/config.py:EmbeddingConfig` | `enabled`, `d_embed`, `init_std_factor`, `freeze_steps`, `tie_decoder`, … |
| `scripts/sampler_artefact_check.py`       | `return_best=True` vs `False` post-hoc |
| `sweeps/latent_eqm.yaml`                  | sanity + headline + Phase-5 ablations |
| `runs/latent_phase2_sanity_d32/`          | 1-epoch sanity result                 |
