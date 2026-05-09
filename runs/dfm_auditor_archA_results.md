# Architecture A — supervised SVGP on frozen DFM encoder features

Sanity check before the dual-head retrain. Take a trained DFM auditor
checkpoint (d=512, L=6), extract per-position encoder features
``z[r, k] = encoder(x_LM[r, k], t=4, h_LLM[r, k])``, fit a supervised
SVGP (128 inducing pts, RBF + Bernoulli likelihood, 400 iters) on the
per-position features with the row's hallucination label propagated to
every answer-mask position.

## Headline (n=800 val rows, 6611 answer-mask positions)

| feature source | row AUROC | per-tok AUROC (ans) | per-tok AUROC (non-ans) | per-tok ECE |
|---|---:|---:|---:|---:|
| **DFM encoder, `hidden_only` (this)** | 0.847 | **0.757** | 0.613 | 0.022 |
| **DFM encoder, `product_concat` (this)** | 0.845 | 0.706 | **0.526** ← good locality | 0.011 |
| **raw h_LLM, per-pos SVGP + row pool** | **0.896** | 0.510 | 0.501 | 0.004 |
| raw h_LLM, row-pool then SVGP | 0.625 | n/a | n/a | n/a |
| DFM EBM (unsupervised, hidden_only) reference | 0.811 | per-pos via E_t (uncomputed here) | — | n/a |
| Phase K supervised SVGP, 20000 rows ceiling | 0.996 | n/a | n/a | 0.023 |

## Diagnosis

1. **Architecture A passes the *very low* floor (beats EBM by ~3 AUROC) but
   does not reach the supervised ceiling.** Best DFM-encoder cell: 0.85
   row AUROC. The 0.92 floor I targeted for "encoder carries the signal"
   is not met.

2. **The encoder *trades* row-level signal for per-token locality.** Compare:

   * Raw h_LLM per-pos SVGP: per-tok AUROC = 0.51 at *both* answer and
     non-answer positions (chance everywhere). The row-level signal
     (0.90) is purely from averaging slightly-biased noise over many
     positions; individual positions carry no information.
   * DFM encoder per-pos SVGP: per-tok AUROC = 0.71–0.76 at answer
     positions; 0.53–0.61 at non-answer positions. *Genuine* per-token
     localisation. `product_concat` mode has near-chance (0.526) at non-
     answer positions, which is the cleanest locality result we have on
     HaluEval-QA — better than Phase F's `h_LLM` linear probe (0.97
     non-answer locality, cascade-contaminated).

3. **The supervised ceiling at this dataset size is ≈0.90, not 0.99.**
   The Phase K headline of 0.996 was on 20000 rows; we have 4000.
   Phase Q's pool-first approach lands at 0.625 on this subset because
   the SVGP only gets 1600 training row-features in 768-d. Per-position
   training works better in this small-cache regime because each row
   contributes ~30 training examples.

4. **The bottleneck is the slot-CE training objective.** It learns
   features useful for predicting which top-K vocab slot the next token
   is, not features that retain h_LLM's OOD-distance information. The
   ~5 AUROC gap between raw h_LLM (0.90) and the DFM encoder (0.85) is
   the "supervised signal that CE training discarded".

5. **Cascade locality is the unique value-add.** Phase F documented
   that h_LLM cascade-contaminates per-token AUROC on WikiText-2 (0.97
   at uncorrupted positions ≈ 0.99 at corrupted; the auditor flags the
   prefix radius, not the corrupted token). On HaluEval-QA the same
   thing happens to raw h_LLM (per-pos AUROC ≈ 0.50 everywhere because
   the row-mean signal dominates). DFM-encoder + SVGP gives the cleanest
   per-position localisation we've measured: 0.71 / 0.53 split
   between answer / non-answer. **No other method in the family produces
   this split.**

## Decision

Architecture A's row AUROC (0.85) ≥ EBM (0.81) but < raw h_LLM (0.90).
The encoder carries *some* but not *all* of the supervised signal. The
unique benefit is per-token locality at near-chance non-answer AUROC,
which neither raw h_LLM SVGP nor the DFM EBM alone delivers.

Proceeding to **Architecture B (dual-head retrain)** with realistic
targets revised against the actual data-regime ceiling (~0.90):

* row AUROC ≥ 0.92 (beat raw h_LLM per-pos pool baseline)
* per-tok AUROC at answer span ≥ 0.80
* per-tok AUROC at non-answer ≤ 0.55 (preserve cascade locality)
* per-tok ECE ≤ 0.05 (calibrated)

If B reaches all four, the writeup story is *DFM-as-feature-extractor
+ supervised SVGP head delivers locality + discriminative power that
neither raw h_LLM SVGP nor SE alone can match*.

## Files

* `scripts/run_dfm_auditor_archA.py` — DFM-encoder Architecture A driver.
* `scripts/run_dfm_auditor_archA_baseline.py` — raw h_LLM control.
* `runs/dfm_auditor_archA_hidden/`, `runs/dfm_auditor_archA_both/` —
  Architecture A summaries on the two encoder modes.
* `runs/dfm_auditor_archA_baseline_hLLM/summary.json` — raw h_LLM
  control numbers.
