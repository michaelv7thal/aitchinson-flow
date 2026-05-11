# Compositional EqM — recovery sweep summary

Sweep spec: [`sweeps/compositional_eqm_test.yaml`](../sweeps/compositional_eqm_test.yaml).
Runbook: [`RUNBOOK_COMPOSITIONAL_EQM_TEST.md`](../RUNBOOK_COMPOSITIONAL_EQM_TEST.md).
Proposal: [`PROPOSAL_COMPOSITIONAL_EQM.md`](../PROPOSAL_COMPOSITIONAL_EQM.md).

8 cells trained: smoke + 3 seeds × {Compositional MSE, Compositional Hilbert} + 2 deterministic-CLR references. After training, `scripts/recovery_check.py` was run on each `epoch_final.pt` with `α ∈ {0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 1.00}`, n=256, NAG steps=200. Results below.

The recovery diagnostic perturbs each held-out window's CLR features by `α · ‖x₁‖`, runs the EqM sampler, and reports both the perturbation's argmax accuracy ("perturbed") and the recovered sampler output's accuracy ("token_acc"). `Δ@α = token_acc − token_acc_perturbed` measures how much real work the sampler does at perturbation level α.

## Headline

| condition                     | KL_uni | KL_bi | H_ratio | acc@.10 | acc@.30 | acc@.50 | Δ@.50          |
|-------------------------------|--------|-------|---------|---------|---------|---------|----------------|
| Compositional MSE             |  0.666 | 5.994 |  1.157  |  1.000  |  0.903  |  0.583  | **+0.060 ±0.000** |
| Compositional Hilbert         |  0.666 | 5.994 |  1.157  |  1.000  |  0.909  |  0.577  | **+0.054 ±0.003** |
| Det. MSE (reference)          |  0.042 | 1.347 |  0.920  |  1.000  |  0.892  |  0.523  |  +0.000        |
| Det. Hilbert (reference)      |  0.021 | 1.513 |  0.986  |  1.000  |  0.893  |  0.523  |  +0.000        |

KL columns are unconditional-generation diagnostics from `eval.json`; `acc@α` is mean recovered token accuracy at perturbation magnitude α.

### Reading the headline

- **Compositional cells recover, deterministic cells don't.** Both compositional conditions sit at Δ@.50 ≈ +0.05–0.06; both deterministic references sit at Δ@.50 = 0.000. The deterministic field is a no-op on perturbed inputs (the failure mode `PROPOSAL_COMPOSITIONAL_EQM.md §1` targets); the compositional field provides real recovery in the α=0.3–0.5 regime.
- **MSE vs Hilbert is a wash.** Δ@.50 of +0.060 (MSE) vs +0.054 (Hilbert) is within the across-seed noise. The proposal §5.3 predicted Hilbert would be marginally ahead (~2–3%); we see MSE marginally ahead (~0.6%). No reliable Hilbert advantage at this scale.
- **Det. references look "better" on KL.** KL_uni drops from ~0.67 to ~0.04 because the deterministic models overfit to the unigram corpus distribution — but as the recovery columns show, their sampler does no real conditional work. This confirms the runbook's reading: KL alone is misleading; recovery is the headline metric for this recipe.

## Per-α recovery accuracy (mean across seeds)

| condition                     | α=0.05 | α=0.10 | α=0.20 | α=0.30 | α=0.40 | α=0.50 | α=1.00 |
|-------------------------------|--------|--------|--------|--------|--------|--------|--------|
| Compositional MSE             |  1.000 |  1.000 |  0.996 |  0.903 |  0.737 |  0.583 |  0.187 |
| Compositional Hilbert         |  1.000 |  1.000 |  0.996 |  0.909 |  0.739 |  0.577 |  0.187 |
| Det. MSE (reference)          |  1.000 |  1.000 |  0.995 |  0.892 |  0.695 |  0.523 |  0.187 |
| Det. Hilbert (reference)      |  1.000 |  1.000 |  0.997 |  0.893 |  0.695 |  0.523 |  0.187 |

## Per-α Δ (sampler − argmax(perturbed))

| condition                     | α=0.05 | α=0.10 | α=0.20 | α=0.30 | α=0.40 | α=0.50 | α=1.00 |
|-------------------------------|--------|--------|--------|--------|--------|--------|--------|
| Compositional MSE             | +0.000 | +0.000 | −0.001 | +0.010 | +0.042 | +0.060 | +0.000 |
| Compositional Hilbert         | +0.000 | +0.000 | −0.001 | +0.016 | +0.044 | +0.054 | +0.000 |
| Det. MSE (reference)          | +0.000 | +0.000 | −0.003 | −0.001 | +0.000 | +0.000 | +0.000 |
| Det. Hilbert (reference)      | +0.000 | +0.000 |  0.000 |  0.000 |  0.000 |  0.000 |  0.000 |

Δ kicks in around α=0.3 (where the perturbation starts knocking tokens off their argmax), peaks at α=0.5, and collapses by α=1.0 (perturbation ≈ ‖x₁‖ — the input is structurally noise and even a perfectly trained field can't recover the original sequence).

## Notes / caveats

- Compositional KL_uni and KL_bi are identical between MSE and Hilbert at 3-decimal precision (0.666 / 5.994). That is suspicious enough to flag — most likely an artefact of the eval being deterministic given the seed, not a real "they sample identically" claim. Worth verifying by seed-shuffling the unconditional eval if this becomes load-bearing.
- All 8 cells trained for 5 epochs on 10k windows with the default backbone (d_model=1024, 8 layers, K=27, L=40). Numbers are not directly comparable to scaled-up runs.
- The deterministic references were trained with the *same* compositional recipe minus the Dirichlet thickening (`transformation.dirichlet_sampling: false`). Everything else — backbone, optimiser, loss, seed — matches the corresponding compositional cell.
- Sample-level qualitative read: at α=0.50 the perturbed input typically has ~50% characters wrong, and the recovered sample stays in the "looks roughly like text" regime but accumulates errors at the boundary positions (see `runs/comp_mse_seed42/recovery.json`'s `rc_sample0`).

## Pointers

- Per-cell history: `runs/comp_*/history.jsonl`
- Per-cell unconditional eval: `runs/comp_*/eval.json`
- Per-cell recovery rows: `runs/comp_*/recovery.json`
- Aggregation snapshot (machine-readable): `runs/_logs/comp_aggregation.json`
- Recovery sweep run log: `runs/_logs/recovery_sweep.log` (start 15:30, finish 18:55)
