# DFM auditor — per-method sequence heatmaps

Four self-contained 2×2 heatmap figures, one per method, on the same
10 clean + 10 hallucinated held-out HaluEval-QA validation rows
(seed=0 selection — identical row sets across all four files for direct
comparison). Each row in each panel is per-row aligned to its own
answer-span start (vertical black line); padding is rendered as neutral
grey; `×` overlays mark the answer-span (faulty) positions in
hallucinated rows.

## Files

| file | method | top-row "score" channel | bottom-row "uncertainty" channel |
|---|---|---|---|
| `sequence_heatmaps_paper_dE.png` | paper ΔE (training-free) | `ΔE = E_logit − E_marg` per position | `E_marg` (LM marginal energy as proxy uncertainty) |
| `sequence_heatmaps_dfm_ebm.png` | DFM EBM (slot-only, unsupervised) | `−log p_t(x|h)` (closed-form mixture-of-Dirichlets) | SVGP variance fitted to slot-only encoder |
| `sequence_heatmaps_archB_transformer.png` | Arch B Transformer (joint supervised) | halluc-head probability | SVGP variance fitted to B-Transformer encoder |
| `sequence_heatmaps_archB_mlp.png` | Arch B MLP (joint supervised, cascade-clean) | halluc-head probability | SVGP variance fitted to B-MLP encoder |

## What the four files show — at a glance

| method | row AUROC | within-row loc. AUROC | what the **score top-right panel** looks like |
|---|---:|---:|---|
| **paper ΔE** | 0.713 | 0.537 | mottled reds with no clear answer-span signature; near-chance per-token localisation |
| **DFM EBM** | 0.811 | 0.632 | slightly more saturated blue in halluc panel than clean; weak per-token signal — row AUROC comes from pooling |
| **Arch B Transformer** | **0.985** | 0.679 | **entire visible row red**, cascading well past the × marks — cross-positional attention propagates the row-level signal globally; the high `tok(non-ans)`=0.98 in fig1 is *not* per-token faultiness, it is row-discrimination measured at non-answer positions |
| **Arch B MLP** | **0.979** | **0.719** | red **concentrated at and around the × marks**; much less spillover to non-answer positions — cascade-clean by construction, and best within-row localisation |

The four files together visualise the locality story the AUROC table in
`runs/dfm_auditor_plots/fig1_auroc_locality.png` summarises. **Read the
within-row column for the honest "which token is faulty?" comparison** —
the row AUROC is "is the row hallucinated?" and `tok(ans)` / `tok(non-ans)`
in fig1 are also row-discrimination at a per-position resolution, *not*
per-token faulty/non-faulty discriminators.

## Channel-by-channel reading

### paper ΔE (`sequence_heatmaps_paper_dE.png`)

* **Top row (ΔE)**: per-position `−logit_{i−1}[id(x_i)] − (−logsumexp(logits_i))`.
  This is the cached training-free spilled-energy score.
* **Bottom row (E_marg)**: `−logsumexp(logits_i)`. High values indicate
  the LM's distribution at this position is sharply peaked (low
  marginal entropy); low values indicate spread/uncertainty.
* **What you see**: ΔE is a per-position scalar that doesn't dramatically
  light up the × marks. The signal works at the row-pool level
  (AUROC 0.71) but isn't a strong per-token discriminator on
  HaluEval-QA — consistent with Phase Q's diagnosis.

### DFM EBM slot-only (`sequence_heatmaps_dfm_ebm.png`)

* **Top row (`−log p_t(x|h)`)**: the closed-form mixture-of-Dirichlets
  energy. Higher (darker blue) = more out-of-distribution under the
  trained denoiser's prior.
* **Bottom row (SVGP variance)**: epistemic uncertainty from a SVGP
  fitted post-hoc on the slot-only encoder's per-position features.
* **What you see**: subtle per-token signal — halluc rows show slightly
  higher (darker) energy values inside the answer span, and the SVGP
  variance has some structure that overlaps with × marks, especially
  near the answer-span boundary.

### Arch B Transformer (`sequence_heatmaps_archB_transformer.png`) — the cascade

* **Top row (halluc-head probability)**: sigmoid of the supervised
  halluc head at the LM's actual top-K simplex distribution. [0, 1].
* **Bottom row (SVGP variance)**: SVGP fitted to the B-Transformer
  encoder's features.
* **What you see**: the cascade is visually striking. Clean rows are
  nearly white (low halluc probability across all positions). Hallucinated
  rows are bright red across **the entire visible row**, including
  positions left of the answer-span boundary (prompt tokens) and far
  past the × marks. The transformer's cross-positional attention has
  routed the hallucination signal globally — every position learns to
  flag the row, not the actual faulty tokens. SVGP variance is
  *higher* on clean rows than hallucinated rows because the supervised
  joint training drove halluc encoder features into a tight cluster.

### Arch B MLP (`sequence_heatmaps_archB_mlp.png`) — the cascade fix

* **Top row**: same halluc-head probability as the Transformer panel.
* **What you see**: hallucinated rows show red **concentrated at and
  immediately around the × marks**. Outside the answer span, the
  halluc-head probability stays low (white/light pink). The MLP
  backbone's lack of cross-positional attention means each position's
  output depends only on its own input — so non-answer positions don't
  inherit the row-level hallucination signal.
* **What this is and isn't**: this *is* "the red is in the right place"
  — per-position scores no longer fire on prompt tokens. It is *not*
  pinpoint faulty-token detection — within-row localisation AUROC is
  0.72 (vs 0.68 for the Transformer), modestly above chance. The
  cascade-clean design recovers the locality property of the score
  *map* without dramatically improving the per-token decision quality.
* **SVGP variance**: more uniform than the Transformer counterpart,
  consistent with the MLP encoder producing per-position features that
  don't collapse into class-specific clusters.

## How to regenerate

```bash
python scripts/plot_dfm_auditor_sequence_heatmaps.py \
    --ckpt-transformer runs/dfm_auditor_archB_d512/best.pt \
    --ckpt-mlp         runs/dfm_archB_d512_mlp/best.pt \
    --ckpt-slot-only   runs/dfm_auditor_d512_hidden/best.pt \
    --window 50 --n-per-class 10 \
    --out-suffix _all_methods
```

To plot only a subset of methods (e.g. just the two Architecture B
variants for the headline comparison):

```bash
python scripts/plot_dfm_auditor_sequence_heatmaps.py \
    --methods archB_transformer archB_mlp \
    --out-suffix _ab_only
```

## Note on the "uncertainty" channel for paper ΔE

There is no native per-position variance for a training-free scalar like
ΔE. We use `E_marg = −logsumexp(logits)` as a *proxy uncertainty*
because it captures the LM's marginal entropy at that position
(low E_marg ⇒ peaked LM ⇒ confident; high E_marg ⇒ spread LM ⇒
uncertain). This is a stretch — strictly, a fair per-position SVGP
variance for paper ΔE would need an SVGP fitted to ΔE features, which
is essentially a 1-D feature and produces variance that's just a
function of `|ΔE − inducing_point|` (i.e., reproduces the score
itself). The E_marg proxy is at least an independent channel.
