# DFM auditor — per-sequence heatmaps (energy + SVGP variance)

10-sequence × 50-token heatmaps for each architecture, with × overlays
on the answer-span ("faulty") positions of hallucinated rows. Two
channels: DFM EBM energy (`-log p_t(x|h)` from the closed-form
mixture-of-Dirichlets density) and SVGP predictive variance from a
post-hoc SVGP fitted to that architecture's encoder features.

## Files

| file | architecture | content |
|---|---|---|
| `sequence_heatmaps_transformer.png` | Architecture B Transformer | 2×2 grid: {energy, variance} × {clean, halluc} |
| `sequence_heatmaps_mlp.png` | Architecture B MLP | same layout |
| `summary.json` | — | indices of selected rows + colorbar limits |

## How to read

* **y-axis**: 10 held-out validation sequences per panel
  (clean / hallucinated, balanced)
* **x-axis**: per-row aligned tokens — column index is the offset
  from each row's answer-span START. The vertical black line marks
  the answer-span boundary; tokens to the left of it are prompt
  context, tokens to the right include the answer span (variable
  length per row) and any post-answer context. Padding is rendered
  as neutral grey.
* **× overlays**: every position whose `answer_mask = True` in a
  hallucinated row is marked with a white ×. These are the "faulty"
  positions the auditor should flag (HaluEval-QA labels rows, not
  tokens — within a hallucinated row, the entire answer-span is the
  factually-wrong content).
* **Color scale (RdBu_r for energy, PuOr_r for variance)**: each cell
  is *centered against its row's prompt baseline* (mean over non-answer,
  non-padding tokens). Red = above baseline (more anomalous /
  more uncertain at this position than at this row's prompt tokens);
  blue = below baseline. Symmetric range, 95th percentile clipping.

## What you should see (and what you actually see)

The visualization is honest about a structural fact already
documented in Phase Q:

> *"HaluEval-QA hallucinations are locally plausible by construction —
> they are factually wrong but the LM's per-token top-K distribution on
> a hallucinated answer still looks natural."*

So the per-position EBM energy is **not a strong per-token signal** on
this dataset. Looking at the energy panels:

* **Hallucinated answer-span tokens (× cells)** are not systematically
  more red than the prompt-baseline. Some are red, some blue, often
  similar to what you see in the clean panel at the same relative
  positions. The closed-form mixture-of-Dirichlets density at any
  individual hallucinated token is *not* dramatically lower than at
  the surrounding prompt tokens.
* **Strong row-level discrimination (AUROC ≈ 0.81 unsupervised)
  comes from pooling**, not from individual smoking-gun tokens.

The SVGP variance panels show more per-position structure, especially
in the **MLP variant**:

* Clean rows have small, fairly uniform variance after row-centering.
* Hallucinated rows show some patches of elevated variance (orange)
  overlapping with × marks — the SVGP is more uncertain at OOD-ish
  positions because their encoder features are further from the
  inducing-point distribution.
* The signal is subtler in the **Transformer variant** because the
  encoder routes information across positions (cascade), so the SVGP
  features at each position become more uniform.

## Why "energy" looks weak per-token

Three reasons, none of which are a code bug:

1. **Locally plausible hallucinations.** ChatGPT-generated wrong
   answers are crafted to look fluent. GPT-2's top-K distribution
   on a hallucinated token is shape-similar to its top-K on a clean
   token — the EBM only sees the simplex shape, so the per-position
   density is similar.
2. **Row-level signal lives in the integral.** A small consistent
   bias of ~0.05 nats per token, summed over an 8-token answer
   span, gives ~0.4 nats per row — a reliable AUROC ≈ 0.81 separator
   when ranked across rows, but invisible at any single token.
3. **Cross-distribution self-consistency** (the strong pieces of
   paper ΔE) lives in the *chain rule* across timesteps, not in the
   per-position simplex shape. The EBM encodes the per-position part,
   not the cross-timestep part. The halluc-head we trained in
   Architecture B *can* exploit cross-position info via attention
   (the cascade) — that's how it gets to 0.97 per-token AUROC, but
   at the cost of localisation.

## Where the heatmap *would* light up

For comparison, on the **WikiText-2 span-corrupted setting (Phase F)**,
the per-position EBM signal *is* localised because the corruption is
literal token replacement (random vocab IDs). Phase F documented a
clean per-position SE signal with locality gap +0.33 on that dataset.
The same heatmap on a Llama-1B WikiText-2 cache would show clear red
× cells. This is a property of the corruption mechanism, not the
auditor.

## Re-running

```bash
python scripts/plot_dfm_auditor_sequence_heatmaps.py \
    --ckpt-transformer runs/dfm_auditor_archB_d512/best.pt \
    --ckpt-mlp         runs/dfm_archB_d512_mlp/best.pt \
    --window 50 --n-per-class 10
```

To regenerate on a Llama checkpoint cache (cluster), point `--ckpt-*`
at the appropriate `runs/dfm_archB_llama*/best.pt` paths and the
script will (1) re-fit the SVGP on that encoder's features and (2)
detect Llama's pad token via the `attn_mask` heuristic (currently the
script hard-codes GPT-2's 50256 pad token; for Llama swap to whatever
is in `tokenizer.pad_token_id` of the corresponding cache).
