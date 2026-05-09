# DFM auditor — diagnostic plots

Generated from the trained GPT-2 checkpoints by
`scripts/plot_dfm_auditor_diagnostics.py`. All numbers are on the same
n=800 held-out HaluEval-QA val rows (400 clean / 400 hallucinated).

## fig1_auroc_locality.png

Four bar charts: row-level AUROC, per-token AUROC at answer-span,
per-token AUROC at non-answer (the *cascade contamination* test),
and the *locality gap* (ans − non-ans).

Headline numbers:

| signal | row | tok(ans) | tok(non) | gap |
|---|---:|---:|---:|---:|
| paper ΔE | 0.71 | 0.52 | 0.55 | −0.02 |
| DFM EBM (slot only) | 0.81 | 0.63 | 0.53 | +0.10 |
| Arch B Transformer | **0.985** | 0.97 | **0.98** | **−0.01** ← cascade |
| Arch B MLP | 0.979 | 0.90 | **0.56** | **+0.35** ← clean |

## fig2_score_distributions.png

Class-conditional histograms of the row-level signal for each method.
Per-method AUROC printed in the title. The transformer's distribution
is *dramatically bimodal* (clean → 0, halluc → 1) which is what the
0.985 AUROC reflects, but the calibration is poor (long tail of clean
rows misclassified high). The MLP variant has more graceful, less
collapsed distributions while still hitting 0.979 AUROC.

## fig3_position_trace.png ★ the cascade visualization

Mean halluc score by position, relative to the answer-span start
(offset = 0), separated by class. The gold band is the typical answer
span (μ ≈ 9 tokens for HaluEval-QA).

**Transformer (left panel):** the hallucinated curve sits at ~0.8 across
*the entire range* (-50 to +50), including positions *before* the
answer span starts. Cross-position attention has propagated the
hallucination signal both forward AND backward through the sequence.
The clean curve sits at ~0.0 throughout. There's no useful per-position
localization — the transformer is essentially producing a row-level
score replicated across positions.

**MLP (right panel):** the hallucinated curve and clean curve overlap
*outside* the gold band; the halluc score only spikes *inside* the
typical answer span. Outside the answer span, the MLP correctly
reports "no per-position evidence of hallucination at this position
because the answer-span tokens haven't been seen at this position."

This is the clearest direct visualization in the result set — the
shape of the curves shows what "localization" *means* and what
"cascade contamination" *looks like*. **The MLP curve outside the gold
band is what Phase F's stated goal demanded; the transformer curve
shows why high per-token AUROC numbers in the supervised-probe
literature can be misleading.**

## fig4_token_heatmaps.png ★ the "which tokens are faulty" view

Four held-out rows (1 clean + 3 hallucinated), each as a small panel:

* **Channel 1 (Reds):** transformer halluc probability per token
* **Channel 2 (Greens):** MLP halluc probability per token
* **Channel 3 (PuOr):** DFM EBM energy (z-scored)
* **Channel 4 (RdBu):** paper ΔE (z-scored)

Token strings appear as the x-tick labels. The gold vertical band
marks the answer span.

What the rows show:

* **Row 692 (CLEAN):** all four channels mostly low/neutral → all
  methods agree this row is clean.
* **Row 409, 215, 507 (HALLUC):** transformer (Reds) lights up red
  *across the whole sequence*, including prompt tokens; MLP (Greens)
  lights up only inside or near the gold answer span. ΔE (RdBu) shows
  scattered per-token signal that's noisier than either.

The contrast row-by-row gives a concrete reading of "which token does
each method think is the smoking gun?" — and shows that the
transformer's answer is *all of them, including unrelated prompt
tokens*, which is the cascade.

## How to regenerate

```bash
python scripts/plot_dfm_auditor_diagnostics.py \
    --ckpt-transformer-archB runs/dfm_auditor_archB_d512/best.pt \
    --ckpt-mlp-archB         runs/dfm_archB_d512_mlp/best.pt \
    --ckpt-slot-only         runs/dfm_auditor_d512_hidden/best.pt \
    --out runs/dfm_auditor_plots
```

Default checkpoint paths point to the GPT-2 d=512 best checkpoints from
this session. To re-run on Llama-1B / Llama-7B checkpoints from the
cluster, point `--ckpt-*` at the appropriate `runs/dfm_archB_llama*/`
directories — the script is backbone- and LM-agnostic.

## Files

* `fig1_auroc_locality.png` — bar-chart summary
* `fig2_score_distributions.png` — class-conditional histograms
* `fig3_position_trace.png` — cascade signature (the key visualization)
* `fig4_token_heatmaps.png` — per-token annotated heatmaps for example rows
* `summary.json` — raw AUROC numbers used in fig1
