# DirichletFM Auditor — HaluEval-QA UQ via closed-form EBM

End-to-end test of the proposal: a DirichletFM denoiser trained on the
clean half of HaluEval-QA induces a closed-form mixture-of-Dirichlets
density `p_t(x | h)` on the LM's top-K simplex; we score `−log p_t` at
each position's actual LM top-K distribution and aggregate over the
answer span.

## Setup

- Cache: `data/hallueval_topk_gpt2.pt` (4000 rows, K=32, L=160) +
  `data/hallueval_cache_gpt2.pt` (h_LLM 768-dim per position).
- 80/20 pair-level split: 3200 train / 800 val rows (400 clean +
  400 hallucinated, balanced).
- Model: 4-layer Transformer, d=192, ~1.9M params.
- Training: only the **clean** half is used (denoiser learns the LM's
  clean top-K geometry conditioned on h_LLM).
- Energy eval at t=4.0 (Dirichlet path, t∈[1, t_max=8]).

## Headline (n=800 rows)

| signal | AUROC |
|---|---:|
| top-K entropy (zero-train) | 0.54 |
| 1 − top-1 prob (zero-train) | 0.55 |
| **paper ΔE seq sum** (training-free, Phase Q baseline) | **0.71** |
| Hilbert-FM trajectory UQ best (Phase Q) | 0.58 |
| **DirichletFM EBM, no h_LLM context** (this work) | **0.77** |
| **DirichletFM EBM + h_LLM context** (this work) | **0.78** |
| Phase K supervised SVGP on h_LLM (ceiling) | 0.996 |

## Interpretation

- **+6 AUROC over paper ΔE** at parity compute (training-free vs trained-
  on-clean-only). The EBM contains strictly more information than ΔE: ΔE
  is a per-position chain-rule self-consistency identity that uses only
  the LM's logits at the chunk's actual token; the DFM EBM additionally
  uses (i) a trained denoiser's posterior over which slot is correct
  given the simplex point, (ii) the closed-form Dirichlet density at
  that simplex point under the chosen path time t.
- **+19 AUROC over top-K entropy.** The result is not just measuring
  peakedness of the LM's distribution — entropy-only is at chance on
  HaluEval-QA.
- **+19 AUROC over Phase Q's best Hilbert-FM trajectory signal.** The
  closed-form per-timestep density is materially better than trajectory-
  based UQ on the same task and similar model size.
- **Context conditioning (h_LLM) contributes nothing detectable** here:
  with-ctx 0.78, no-ctx 0.77. The CE loss on the denoiser drops faster
  with context (1.96→1.50 vs 2.03→1.92 over 5 epochs), but the AUROC at
  the EBM is identical. Implication: the discriminative signal lives in
  the *shape of the LM's top-K simplex point* under the trained Dirichlet
  prior, not in correlations with `h_LLM`. This is interpretable: the
  denoiser learns *where the LM's mass typically goes at this position*
  (purely from position + the perturbed simplex point), and clean vs
  hallucinated rows differ in how well their actual LM distribution
  matches that learned target.
- Below the supervised h_LLM ceiling (SVGP 0.996, BLR 0.987). The DFM
  EBM is the strongest *unsupervised* signal in the family and beats the
  strongest training-free signal by a clear margin, but does not
  threaten the supervised regime — consistent with the Phase F+
  diagnosis that h_LLM linearly separates the two classes.

## Files

- `src/aitchinson_flow/models/dirichlet_fm_auditor.py` — model + EBM.
- `src/aitchinson_flow/data/hallueval_dfm.py` — paired-cache datamodule.
- `scripts/run_dirichlet_fm_auditor.py` — driver.
- `runs/dfm_auditor_v1/` — with-context checkpoint + summary.json.
- `runs/dfm_auditor_v1_nctx/` — no-context ablation.

## Where this is novel vs Stark et al. (arXiv:2402.05841)

The DFM paper introduces the conditional path family but evaluates only
generation quality (KL, MSE, FBD on DNA design). The closed-form
mixture-of-Dirichlets density `−log p_t(x | h)` falls out of their setup
as a corollary but is not operationalised. We use it as a calibrated
per-timestep UQ surface and benchmark on hallucination detection — a
distinct application that the original paper does not target.
