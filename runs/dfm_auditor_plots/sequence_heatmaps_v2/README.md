# DFM auditor — per-sequence heatmaps (halluc-head probability + SVGP variance)

10-sequence × 50-token heatmaps for each Architecture B variant, with ×
overlays on the answer-span ("faulty") positions of hallucinated rows.
Two channels:

* **Halluc-head probability per position** (top row, Reds, [0, 1]).
  This is the supervised signal that gets the high per-token AUROC
  (transformer 0.97, MLP 0.91). It's what users typically mean by
  "the auditor's energy at this token". Sigmoid of the halluc head's
  raw logit, evaluated at the LM's actual top-K simplex distribution.
* **SVGP predictive variance** (bottom row, magma, raw). Epistemic
  uncertainty from a post-hoc SVGP fitted to the encoder features.
  Sequential colormap; values are the absolute predictive standard
  deviation (always non-negative — the previous version of this plot
  had a bug where I row-centred the variance, producing meaningless
  negative numbers).

## Files

| file | architecture | content |
|---|---|---|
| `sequence_heatmaps_transformer.png` | Architecture B Transformer | 2×2 grid: {halluc prob, SVGP var} × {clean, halluc} |
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
  positions the auditor should flag.

## What the figures show — and why this matches the AUROC numbers

### Transformer Architecture B (top file)

* **Clean rows (top-left)**: mostly white/very light pink → halluc
  probability ≈ 0 across all visible tokens. Clean rows look clean.
* **Hallucinated rows (top-right)**: ★ **cascade visible** ★ — the
  ENTIRE visible row is bright red, including positions LEFT of the
  vertical line (prompt tokens before the answer-span starts) and
  positions far past the × marks. The transformer halluc-head's
  cross-positional attention has propagated the hallucination signal
  across the whole row. The × marks are buried inside a red sea.
* **SVGP variance**: counter-intuitively *higher on clean rows* than
  on halluc rows. The supervised joint training has pushed the
  halluc encoder features for halluc rows into a tight cluster
  (low SVGP variance because they're close to inducing points trained
  on hallucinated examples), while clean rows fall into the lower-
  data region of feature space (higher variance).

### MLP Architecture B (bottom file)

* **Clean rows (top-left)**: mostly white. Clean rows look clean.
* **Hallucinated rows (top-right)**: ★ **localisation visible** ★ —
  red is concentrated *at and around* the × marks, with much weaker
  spillover to non-answer positions. The MLP backbone's lack of
  cross-positional attention prevents the cascade. You can see, row
  by row, where each row's answer span sits and the halluc-head
  fires there.
* **SVGP variance**: more uniform across clean and halluc, with some
  structure inside the answer-span band on halluc rows.

## Mapping back to the AUROC table

| metric | Transformer | MLP | what it looks like above | what it actually measures |
|---|---:|---:|---|---|
| row AUROC | 0.985 | 0.979 | both panels: halluc rows red, clean rows white | "is this row hallucinated?" (pool then score) |
| tok(ans) AUROC | 0.970 | 0.904 | red intensity at × marks | "is this position from a halluc row?" — *row* discrimination at per-token resolution |
| tok(non-ans) AUROC | 0.979 (cascade) | 0.558 (clean) | how far red bleeds outside × marks | also "is this position from a halluc row?" — measured at non-answer positions; transformer fires here too because of cross-positional attention |
| **within-row localisation AUROC** | **0.679** | **0.719** | inside a halluc row, can score rank × positions over non-× positions? | "is *this* the faulty token?" — the honest per-token metric |

**Important framing**: `tok(ans)` and `tok(non-ans)` AUROC are *not* per-token
faulty/non-faulty discriminators — they are row-discrimination metrics
evaluated at a particular slice of positions. The transformer's
`tok(non-ans) = 0.98` therefore means "even at non-answer positions, the
score reliably tells you the row is hallucinated", which is a direct
consequence of cascade, not a per-token failure of the model. The honest
"can the auditor *localise* the faulty token within a halluc row?" question
is the **within-row localisation AUROC**, where MLP (0.72) modestly
outperforms Transformer (0.68) — both are far below the row-discrimination
numbers, and that gap is the cascade quantified.

The difference between "transformer halluc rows are red everywhere" and
"MLP halluc rows are red mostly at × marks" remains real and visible above
— it is the visualisation of the cascade, even though both backbones
land at similar within-row AUROC.

## Notes on the previous version of this plot

The earlier `sequence_heatmaps/` directory used the closed-form
mixture-of-Dirichlets EBM `−log p_t(x|h)` as the "energy" channel.
That was the unsupervised signal (row-AUROC 0.81), which is structurally
weak per-token on HaluEval-QA — Phase Q diagnosed that hallucinations
are locally plausible, so the per-position EBM doesn't dramatically
distinguish faulty from non-faulty tokens. The previous heatmaps
correctly displayed that weak signal, which looked like noise and
contradicted the strong per-token AUROC numbers from the other plots.

This version (`sequence_heatmaps_v2/`) uses the **halluc-head
probability** — the supervised signal whose per-token AUROC is what
the headline numbers report. It also fixes the variance bug (no more
row-centering on a non-negative quantity).

## Re-running

```bash
python scripts/plot_dfm_auditor_sequence_heatmaps.py \
    --ckpt-transformer runs/dfm_auditor_archB_d512/best.pt \
    --ckpt-mlp         runs/dfm_archB_d512_mlp/best.pt \
    --window 50 --n-per-class 10 \
    --out-suffix _v2
```

Use `--out-suffix _vN` to keep multiple sweeps without overwriting.
