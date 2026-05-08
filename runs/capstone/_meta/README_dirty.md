# Pre-existing repo state (snapshot at start of this protocol)

Date: 2026-05-08

## Pre-flight gaps vs. CAPSTONE_EXPERIMENTS.md §2

The protocol assumes these files exist; none were present at start:

- `runs/eqm_data50k_ep5_v2/epoch_final.pt`     missing
- `runs/dfm_data50k_ep5_v2/epoch_final.pt`     missing
- `runs/fmclr_data50k_ep5_v2/epoch_final.pt`   missing
- `runs/lkflow_data50k_ep5/epoch_final.pt`     missing
- `runs/aud_gpt2_ctx/epoch_final.pt`           missing
- `runs/best_auditor.pt`                       missing
- `data/wiki_cache_gpt2.pt`                    missing

The associated `eval.json` files do exist and their numbers match the
protocol's reference numbers (DFM KL_bi=0.148, EqM 1.382, etc.). The
checkpoints were evidently pruned to save disk; only the eval artifacts
were kept.

## What we did

Re-trained the missing checkpoints at parity compute (5 ep × 50 k windows,
L=40, K=27, B=32 to fit 8 GiB GPU). New checkpoints land under
`runs/capstone/checkpoints/`. The original `runs/<run-name>/` dirs are
left as-is so DECISION_LOG.md history stays interpretable.

## Compute target

All trainings in this run target the protocol's parity compute: 50 k
windows × 5 epochs × B=64 × d=1024 (≈ 14 min/cell on a 20 GB A100 MIG).
Eval at protocol default (256 samples × 200 NAG / 128 NFE).

## Other notes

- GPU: NVIDIA RTX PRO 1000 Blackwell Generation Laptop GPU, 8 GiB.
  Per-cell wall-clock 1.7× the protocol's A100 budget — accounted for in
  the §10 budget.
- Working dir is `/workspace`, not `/workspace/aitchinson_flow` as the
  protocol states. CLAUDE.md is the truth.
- `Config.eqm` is named `EqM`, not `EqMConfig`; the protocol's
  `EqMConsGradConfig` is implemented as a method-level field on `EqM`
  rather than a sibling class, to minimize churn.
