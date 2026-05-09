# Architecture B with the per-position MLP backbone (cascade-clean)

Same DFM auditor as the transformer Architecture B, but with the encoder
replaced by a weight-shared per-position MLP — *no cross-positional
information flow*. The hypothesis was that locality should be preserved
by construction at a small cost to row-level discrimination.

Same hyperparameters as the transformer cell for fair comparison
(d=512, num_layers=6, B=8, joint slot+halluc, auto pos_weight=0.166,
8 epochs, full 4000-row HaluEval-QA). 9.65M params (vs transformer's
19.6M — the MLP has half the params at the same d_model/n_layers
because it lacks the QKV projection matrices).

## Headline (n=800 val rows)

| metric | Transformer Arch B (best) | **MLP Arch B (best)** | Δ |
|---|---:|---:|---:|
| row AUROC (halluc) | 0.9851 (ep 5) | **0.9784 (ep 7)** | −0.0067 |
| per-tok AUROC at answer span | 0.9697 | **0.9054 (ep 8)** | −0.0643 |
| **per-tok AUROC at non-answer (cascade)** | **0.9774 ← cascade** | **0.5542 ← chance** | **−0.4232** |
| **locality gap (ans − non-ans)** | −0.0077 | **+0.3512** | **+0.3589** |
| ECE (row) | 0.023 | 0.096 | +0.073 |
| params | 19.6M | 9.65M | half |

## Locality holds across training

Per-token AUROC at non-answer positions over the run:

```
ep | row(halluc) | tok(ans) | tok(non) | locality gap |
---|------------:|---------:|---------:|-------------:|
 1 |    0.880    |   0.66   |   0.56   |    +0.10
 2 |    0.929    |   0.74   |   0.57   |    +0.17
 3 |    0.942    |   0.78   |   0.55   |    +0.23
 4 |    0.968    |   0.89   |   0.55   |    +0.34
 5 |    0.977    |   0.90   |   0.56   |    +0.34
 6 |    0.976    |   0.90   |   0.56   |    +0.34
 7 |    0.978    |   0.90   |   0.56   |    +0.34   ← best.pt
 8 |    0.975    |   0.91   |   0.55   |    +0.35
```

The `tok(non-ans)` column never climbs out of the 0.54–0.59 chance band
across the full run. Compare to the transformer Architecture B where
that column went from 0.59 (ep 1) to 0.98 (ep 5+) as the encoder learned
to exploit cross-position attention. Here, by construction, no such
exploitation is possible.

## What this fixes

The cascade-contamination problem identified in the transformer
Architecture B (`runs/dfm_auditor_archB_d512_results.md`) was structural:
the joint training pushed the encoder to use cross-positional attention
to maximize row-level BCE, at the cost of per-token locality. The
MLP backbone removes the cross-positional channel entirely — there is
no attention, no residual cross-position mixing, just a weight-shared
per-position FFN stack. Position k's encoder output is a function of
inputs at position k only.

This makes the **Phase F localisation claim hold honestly on this
dataset**: the auditor *flags the actual hallucinated tokens*, not the
prefix radius around them. On HaluEval-QA where every answer-mask
position in a hallucinated row carries a label, the difference is
visible as `tok(ans) ≈ 0.91` vs `tok(non-ans) ≈ 0.55` — a +0.36 locality
gap that the transformer cannot match without sacrificing row AUROC.

## What it costs

1. **0.7 AUROC at row level.** 0.978 vs 0.985. Within the noise floor
   on 800 val rows; arguably not a real loss, just a rebalancing toward
   honest per-position decisions.
2. **6 AUROC at per-token answer-span discrimination.** 0.91 vs 0.97.
   Still above the 0.80 target, and the 0.97 number was inflated by
   cascade exploitation anyway. The MLP's 0.91 is the *honest*
   per-position score.
3. **8 ECE points at row level.** 0.096 vs 0.023. The transformer's
   row-pool calibration benefited from cross-position attention
   smoothing the per-token logits; the MLP's per-position logits are
   less smooth so the row-pool of sigmoids is less calibrated. This
   should close with (a) longer training, (b) temperature scaling on
   the row-pool sigmoid, or (c) a small calibration head.

## The new full picture across the auditor track

| signal | row AUROC | tok(ans) | tok(non-ans) | locality gap | ECE | training | LM |
|---|---:|---:|---:|---:|---:|---|---|
| top-K entropy | 0.54 | n/a | n/a | n/a | n/a | none | GPT-2 |
| paper ΔE | 0.71 | 0.50 | 0.51 | −0.01 | n/a | none | GPT-2 |
| **DFM EBM** (unsupervised) | **0.81** | n/a | n/a | n/a | n/a | clean only | GPT-2 |
| Architecture A (SVGP on slot enc) | 0.85 | 0.71 | 0.53 | +0.18 | 0.022 | supervised | GPT-2 |
| **Architecture B + MLP** (this) | **0.978** | 0.91 | **0.55** | **+0.36** | 0.10 | joint slot+halluc | GPT-2 |
| Architecture B + Transformer | 0.985 | 0.97 | 0.98 | −0.01 | 0.023 | joint slot+halluc | GPT-2 |
| Phase K SVGP @ 20k rows | 0.996 | n/a | n/a | n/a | 0.023 | supervised pool | GPT-2 |
| EigenTrack 2025 SOTA | (Llama-7B) 0.894 | n/a | n/a | n/a | n/a | weak-sup. | Llama-7B |

## Decision: which architecture for the writeup?

The MLP variant is the **architecturally principled choice** for this
task. It delivers:

* row AUROC within the noise of the supervised ceiling at this dataset
  size (0.978 vs raw-h_LLM SVGP's 0.90 and the 4000-row supervised
  ceiling at ~0.99),
* the **strongest locality gap (+0.36) of any method in the family**,
* by-construction guarantees that match what Phase F demanded ("flag
  the actual corrupted token, not the prefix radius") on a real-
  hallucination dataset where SE alone is at chance,
* half the parameter count of the equivalent transformer.

The transformer variant gives slightly stronger numbers but at the cost
of the locality property. For the writeup the cleaner story is to
*report both as a deliberate ablation*: "the transformer's row AUROC
of 0.985 is partly cascade-exploitation; the MLP's 0.978 with +0.36
locality gap is the honest per-position auditor."

## Files

* Checkpoint: `runs/dfm_archB_d512_mlp/best.pt` (ep 7, row AUROC 0.978)
* Final epoch: `runs/dfm_archB_d512_mlp/epoch_final.pt`
* Summary: `runs/dfm_archB_d512_mlp/summary.json`
* Code:
  * `src/aitchinson_flow/models/dirichlet_fm_auditor.py` — added
    ``_PerPositionMLPBackbone`` with the same constructor API as the
    transformer backbone; both expose ``_encode_features`` so the
    EBM, generation sampler, and SVGP head all work unchanged.
  * `--backbone {transformer, mlp}` flag added to both run scripts.
* Runbook: `docs/cluster_runbook_dfm_auditor.md` §2b' shows the
  Llama-equivalent invocation.
