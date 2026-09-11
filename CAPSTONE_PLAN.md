# Capstone implementation plan — UQ for LLM generations + continuous-FM validity


> **Superseded, 2026-05-08. This is a plan for a different capstone.**
> The project pivoted away from the two-model LLM-UQ pipeline described here. The finished paper is `../capstone-paper/`: a nine-arm text8 generation benchmark, a frozen-backbone per-token OOD detector, a localize-then-inpaint repair loop, and the Equilibrium Matching negative result. Nothing on the correctness-track side of this document (SVGP on GPT-2 hidden states, HaluEval, TruthfulQA, the combined predictor) appears in the paper, and the GPT-2/WikiText auditor line it builds on is retired (see CLAUDE.md "Known dead code").
>
> Two specific corrections, because both mislead:
> - **"The OOD prong is closed" is inverted.** OOD detection became the paper's main positive result (`chapters/results.tex` §Out-of-Distribution Detection).
> - **Priority 2a misattributes its own reference.** arXiv:2405.16441 and the 1.39 text8 BPC are Cheng et al., *Statistical Flow Matching* (the paper's "Statistical FM" arm, `models/sfm.py`), not Stark et al., whose Dirichlet Flow Matching is a different construction. Davis et al., *Fisher Flow Matching* (arXiv:2405.14664) is a third, separate arm — see `FISHER_FM_PLAN.md`. The paper trains both Fisher-Rao arms and never writes "the Fisher-Rao arm" in the singular.
> - Its Priority-2a success criterion ("KL_bi ≥ 1.0 confirms the continuous-on-simplex negative at the published-method level") is a conclusion the paper explicitly declines to draw.
>
> **Preserved unreported result:** Priority 1 ran and passed its own bar. SVGP on GPT-2 features reaches AUROC 0.996 at ECE 0.023 on HaluEval-QA (`runs/hal_gpt2_qa_{lasttoken,meanpool}/uq_eval.json`). This is not in the paper and this document plus those two JSONs are its only record.

*Strategic document for the next-phase capstone work. Companion to
[`TRAINING_PROTOCOL_v2.md`](TRAINING_PROTOCOL_v2.md) which gives the
operational/per-phase steps. Reads from
[`REPORT.md`](REPORT.md) (this session's findings) and
[`runs/DECISION_LOG.md`](runs/DECISION_LOG.md) (append-only diary). New
sessions should read this file once for the strategic framing, then
operate phase-by-phase from `TRAINING_PROTOCOL_v2.md`.*

---

## 1. Goals (unchanged from the original capstone)

1. **Uncertainty quantification of LLM generations.** Given an LLM and
   a (prompt, candidate completion) pair, produce a calibrated estimate
   of whether the completion is correct/valid, and ideally per-token
   attribution for *where* it goes wrong.
2. **Generation of valid text via continuous flow-matching models.**
   Establish whether continuous-on-simplex flow-matching models can
   generate competitive natural-language text — at character level, BPE
   level, or with the right geometry.

The original protocol bet that *one* EqM-style auditor could do both.
The 2026-05-07 session decisively falsified that unification (see
[`REPORT.md`](REPORT.md)). The new bet is **decomposition**: a strong
discrete generator for validity, a calibrated UQ classifier on cached
LM features for correctness, integrated into a two-model pipeline.

## 2. What the previous session established

For full numbers and traceability, see
[`REPORT.md`](REPORT.md) and the 2026-05-07 entries in the decision
log. Headline findings that constrain the next session's design:

| Finding | What it constrains |
|---|---|
| **DFM dominates continuous-on-simplex by ≈10× on text8 KL_bi** at K=27, across three independent continuous parameterisations (EqM, FMonCLR, LogitKLFlow). The bottleneck is regime-level (continuous vs discrete at small vocab), not parameterisation. | New continuous methods must demonstrate they overcome this regime issue *first*. Future continuous work should test SFM (√p reparameterisation, published BPC 1.39) or move to BPE-scale vocab where curse-of-dimensionality may flip the trade-off. |
| **DFM also dominates sequence-level OOD detection.** Spilled Energy (zero-train) saturates every text8 corruption contrast at AUROC ≥ 0.99 including `valid_perm`. EqM's per-position uncertainty has no remaining advantage on text8. | The OOD prong of the original protocol is closed. New OOD work must move to harder corruption types (real hallucinations, semantic shifts) or to per-token attribution where SE wins. |
| **EqM auditor on WikiText-2 with context conditioning hits Seq AUROC=1.0 / Tok AUROC=0.994** — but **a one-matmul linear probe on the same `h_LLM` features hits 0.988 within noise**. **An SVGP head on `h_LLM`** matches the AUROC and gives **3000× better calibration** (ECE = 1.2 × 10⁻⁴ vs 0.34 for the linear-probe sigmoid). | The discriminative architecture of choice is **SVGP/BLR-Laplace on cached LM hidden states**, *not* the trained EqM auditor. Build downstream UQ work on this baseline. |
| **The auditor's energy gradient is not a denoising direction.** Running `x ← x − η ∇E(x)` from invalid simplex points produces 0/950 argmax flips toward clean. **Phase H auditor-driven generation fails at all three protocol thresholds** (NLL = 8.89 vs ≤ 5.5 success / ≤ 7.0 partial; vs 9.03 random baseline). | The "auditor is also a generator" hypothesis is dead. Generation must come from a model trained explicitly on the generative objective — DFM, SFM, or AR. |
| **Per-token localisation on `h_LLM` is cascade-contaminated.** Tok AUROC at *uncorrupted* positions in invalid sequences is ≈ 0.97 — essentially the same as at corrupted positions — because GPT-2 is autoregressive and `h[k]` carries the trace of any upstream corruption. **Spilled Energy is the only metric with locality**: its uncorrupted-position AUROC drops to 0.66, and it cleanly recovers as the LM forgets the corruption (see `runs/phaseF_distance.png`). | For per-token attribution, use SE or a residualised local feature, *not* `h_LLM`-based discriminators. The two-model pipeline should pair sequence-level UQ (SVGP on `h_LLM`) with token-level localisation (SE). |
| **Bit-identical inputs give AUROC = 0.5** at positions before the first corruption (verified empirically — `max\|h_inv − h_clean\| = 0.0`). The cascade decay profile differs sharply between methods: SE recovers in ~5 positions, the linear probe partially recovers over 15, the trained EqM auditor doesn't recover within 15. | The recovery profile is reportable in its own right (`runs/phaseF_distance.png`). When designing UQ probes, choose a feature-locality level that matches the granularity of the claim you want to make. |

## 3. Strategic framing — decomposed architecture

The capstone is best presented as a **two-model system with a clean
separation of concerns**:

```
                ┌────────────────────────┐
                │   prompt + completion  │
                └────────┬───────────────┘
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
    ┌─────────────┐           ┌──────────────┐
    │ Validity    │           │  Correctness │
    │ generator   │           │  validator   │
    │  (DFM /     │           │  (SVGP on    │
    │  SFM / AR)  │           │   h_LLM)     │
    └────┬────────┘           └──────┬───────┘
         │                           │
         ▼                           ▼
  ┌──────────────┐           ┌────────────────┐
  │ p_valid(text)│           │ p_correct      │
  │ via NLL      │           │ + epistemic σ  │
  │ (likelihood) │           │ + per-token SE │
  └──────────────┘           └────────────────┘
```

* **Validity track** — given a candidate text, score `p_valid(text)`
  via the *generator's own* NLL. A strong generative model gives free
  validity scoring (the "good gen → good detection" implication).
  Discrete methods (DFM, SEDD) are the established choice at K=27;
  continuous methods are still under question (see Priority 2).

* **Correctness track** — given (prompt, completion) and an external
  LM's hidden states, score `p_correct` via SVGP/BLR-Laplace and
  per-token SE for attribution. This is well-understood now; the
  remaining question is whether it generalises from synthetic
  corruption (WikiText-2 span) to real hallucinations.

The two scores are independent diagnostics. Their combination
(e.g., a max-rule, or a learned 2-feature classifier) gives the
capstone's headline UQ predictor.

## 4. Three priorities, in order

The full operational details — cell names, configs, eval commands,
decision criteria — live in
[`TRAINING_PROTOCOL_v2.md`](TRAINING_PROTOCOL_v2.md).

### Priority 1 — Validate the UQ pipeline on real hallucinations

**Why:** The session's UQ headline numbers (SVGP at AUROC 0.99) come
from synthetic span-corruption. Real hallucinations are subtler. If
the pipeline only works on random-vocab corruption, the capstone's
practical claim is hollow.

**Datasets:**

* **HaluEval QA** (~10 k question-answer pairs with paired correct
  and hallucinated answers from ChatGPT). Available on HuggingFace:
  [`pminervini/HaluEval`](https://huggingface.co/datasets/pminervini/HaluEval).
  Primary benchmark.
* **TruthfulQA** (~800 questions with multiple correct/incorrect
  answer choices). Available on HuggingFace:
  [`truthful_qa`](https://huggingface.co/datasets/truthful_qa).
  Smaller, harder, more interesting failure modes. Secondary
  benchmark.

**What to build:**

1. New cache producer: `scripts/cache_hallueval.py`, modelled on the
   existing `scripts/cache_wiki.py`. Same schema (clean / invalid
   features per (q, a) pair) but adapted to (prompt, completion)
   format. Cache sequences are concatenated `[prompt; completion]`
   strings; the per-position labels mark which positions are inside
   the *answer* span (only those positions enter the UQ classifier).
2. Re-use `scripts/phaseF_uq.py` with minimal modification — the
   cache schema is compatible if `mask_corrupt` is replaced by
   `is_completion`. The classifier's positive class is "this
   completion was hallucinated"; negative is "this completion was
   correct".
3. Eval: per-question AUROC + ECE; calibration plot; optional
   per-token localisation using SE (the answer's per-token NLL under
   the LM).

**Success criteria:**

| Outcome | Verdict |
|---|---|
| SVGP AUROC ≥ 0.85 on HaluEval-QA held-out + ECE ≤ 0.10 | **Strong positive**: UQ pipeline generalises to real hallucinations. Capstone's UQ goal is met. |
| AUROC 0.65–0.85 + ECE ≤ 0.15 | **Acceptable**: real hallucinations are harder than synthetic corruption; results are reportable as a partial positive. |
| AUROC < 0.65 | **Documented negative**: the simple UQ pipeline doesn't transfer. Pivot to per-token localisation as the headline (SE shines there). |

**Estimated compute:** ~30 min cache (depends on LM), ~10 min train
(SVGP + BLR), ~5 min eval. Total: ~1 hour per dataset.

**Estimated dev time:** 3–5 days for the cache producer + integration
work.

### Priority 2 — Test continuous FM where it could plausibly win

**Why:** The session's continuous-on-simplex negative is *only* tested
at K=27 char-level. The published positive results (SFM at BPC 1.39
on text8; Plaid; SEDD) suggest geometry or vocab size is the missing
ingredient. Either confirm the regime-level claim (negative for the
broader continuous-FM hypothesis) or overturn it (positive for the
capstone's continuous-text goal).

**Two options, ordered by leverage:**

#### Priority 2a — SFM (√p reparameterisation, sphere geometry)

[Stark et al. 2024 — arXiv:2405.16441](https://arxiv.org/abs/2405.16441)
parameterises sequences as `q = √p` on the positive orthant of the
unit sphere `S^{K−1}`, making the Fisher–Rao metric Euclidean.
Reported: text8 BPC = 1.39 (vs SEDD 1.32, LinearFM 1.65). This is
*the* published positive continuous result on text8.

**What to build:**

1. New model `src/aitchinson_flow/models/sfm.py`. The conditional
   probability paths follow Eq. 13–17 in the paper; tangent-vector
   velocity head; Riemannian Euler ODE sampler.
2. New sweep cell `sfm_data50k_ep5` at the data_50k_ep5 platform
   (parity compute with EqM/DFM).
3. Optional: long-run cell `sfm_data50k_ep25` (matches the original
   paper's compute more closely).

**Success criteria:**

| Outcome | Verdict |
|---|---|
| SFM `KL_bi ≤ 0.30` at parity compute | **Strong positive for continuous FM**. Geometry was the lever. Major writeup contribution. |
| SFM `KL_bi ∈ [0.30, 0.80]` | **Partial positive**: SFM closes most of the EqM-DFM gap. Worth long-run extension to claim 1.39 BPC reproduction. |
| SFM `KL_bi ≥ 1.0` | **Continuous-on-simplex negative confirmed at the published-method level**. Strong writeup negative. |

**Estimated dev time:** 3–5 days (the model + sampler are subtle but
documented in the paper).

#### Priority 2b — BPE-scale continuous FM (optional alternative)

Train an EqM/LogitKLFlow on the top-K=64 BPE simplex (using the
existing `data/wiki_cache_gpt2.pt`). Tests whether the regime-level
negative survives at larger vocab.

**Why this is interesting:** at K=27 (char), the discrete denoiser
has only 27 categorical outputs to predict — easy. At K=64 BPE
(restricted to per-position top-K), the discrete denoiser still has
64 outputs — also easy. But the *continuous* method's task gets
*harder* because it has to model a richer simplex shape per position.
BPE-K=50,000 (full vocab) would be worse for continuous methods
still, but practically infeasible.

**Recommended only if Priority 2a fails.** Otherwise Priority 2a
gives a cleaner answer.

### Priority 3 — Two-model integration

**Pre-condition:** Priority 1 produced calibrated UQ on at least one
real hallucination dataset.

**What to build:**

1. New script `scripts/integrate_pipeline.py`. Produces:
   - **Generator scoring**: feed the candidate completion through
     the validity-track generator (DFM at parity, or SFM if 2a passed),
     compute per-token NLL or full-sequence likelihood.
   - **Validator scoring**: feed (prompt, completion) through the
     correctness-track UQ (SVGP on `h_LLM`), get calibrated
     `p_correct` + epistemic σ + per-token SE.
   - **Combined score**: max-rule, or a small learned 2-feature
     logistic regression.

2. Evaluate on:
   - HaluEval-QA held-out (the headline dataset).
   - TruthfulQA (harder, smaller, complementary).
   - Optionally: a self-generated dataset where the validity-track
     generator produces samples and the correctness-track validator
     scores them — closing the loop on the "generate + validate"
     pipeline.

3. Report:
   - Combined AUROC vs each track alone.
   - Calibration curves (reliability diagrams) for the combined
     score.
   - Decoded examples where the two tracks disagree (validator says
     correct, generator says low-likelihood — i.e. *unusual but true*
     vs *typical but wrong*).

**Success criteria:**

| Outcome | Verdict |
|---|---|
| Combined AUROC > each track alone, with calibration ECE ≤ 0.10 | **Strong positive integration**. The two-model design adds value over either component alone. Headline capstone result. |
| Combined ≈ best-track-alone, with the two tracks correlated | The components agree; the combination doesn't add separability but the pipeline is still defensible. Report as "complementary diagnostics, not orthogonal signal". |
| Combined < best-track-alone | The combination is hurting; report as a negative. The decomposition argument still stands theoretically but doesn't compound empirically. |

**Estimated dev time:** 3–5 days.

## 5. Components to (re)use vs build

### Reusable from the previous session

* `src/aitchinson_flow/data/wiki.py` — `WikiAuditorDataset`, span
  corruption, AR-shifted Spilled Energy. Generalises directly to
  HaluEval (different label semantics; same h_LLM caching logic).
* `src/aitchinson_flow/data/wiki_auditor_datamodule.py` — minimal
  data-only datamodule. Reusable as a template for HaluEval/TruthfulQA.
* `scripts/cache_wiki.py` — template for `scripts/cache_hallueval.py`.
* `scripts/phaseF_uq.py` — full UQ pipeline (linear probe, BLR-Laplace,
  ensemble, SVGP). Reusable with cache schema changes only.
* `scripts/eval_auditor_wiki.py` and `scripts/plot_phaseF*.py` —
  AUROC/ECE eval and plotting.
* `scripts/phaseF_distance_audit.py` — per-position AUROC by
  distance-to-corruption. Could be adapted to "per-position AUROC by
  distance-to-token-the-validator-flagged".
* `runs/best_auditor.pt` and `data/wiki_cache_gpt2.pt` — kept around
  but no longer the headline. Useful for back-comparison if needed.

### To build

| File | Purpose | Effort |
|---|---|---|
| `src/aitchinson_flow/data/hallueval.py` | HaluEval/TruthfulQA dataset wrapper | 0.5 day |
| `scripts/cache_hallueval.py` | Cache producer for `data/hallueval_cache_<lm>.pt` | 0.5 day |
| `scripts/eval_uq.py` | Generalised version of `eval_auditor_wiki.py` for any cache | 1 day |
| `src/aitchinson_flow/models/sfm.py` | SFM √p reparameterisation model + sampler | 2–3 days |
| `sweeps/phaseN_sfm.yaml` | SFM sweep spec | 0.5 day |
| `scripts/integrate_pipeline.py` | Two-model integration | 1–2 days |
| `scripts/plot_uq_calibration.py` | Reliability diagrams + per-method comparison | 0.5 day |
| `runs/DECISION_LOG.md` updates | Append-only diary | continuous |

Total: ~7–10 days of dev work plus ~3–5 days of training/eval.

## 6. Reading order for the next session

1. Open `runs/DECISION_LOG.md`, read the topmost `## NEXT SESSION`
   block (the 22:50 / 22:32 / 23:03 / 22:25 UTC blocks have the
   relevant context from the prior session).
2. Read `REPORT.md` once for the full context.
3. Read this file (`CAPSTONE_PLAN.md`) for the strategic framing.
4. Switch to `TRAINING_PROTOCOL_v2.md` and execute phase by phase.
5. After each cell, append a 5-line entry to `DECISION_LOG.md`
   following the existing format (timestamp / hypothesis / result /
   decision / next).

## 7. Termination criteria

Stop when **any** of the following is true:

1. **Headline result achieved**: Priority 1 lands at AUROC ≥ 0.85 on
   HaluEval-QA AND Priority 3 produces a combined system whose
   AUROC ≥ Priority-1-alone with calibration ECE ≤ 0.10. The
   capstone's UQ goal is met. *Plus*: Priority 2a produced any
   conclusive answer (positive or negative) on continuous-FM
   validity at parity compute. The continuous-text goal is settled.

2. **All priorities exhausted** (with positive or negative
   verdicts): final summary block in `DECISION_LOG.md`.

3. **Compute budget exhausted**: ~30 GPU-hours total across all
   priorities (rough estimate based on prior session's pace).
   Pause and write a `## NEXT SESSION` block describing what's
   pending.

4. **A retry budget is exceeded**: any single cell retried 3× with
   non-trivially-different configs and still failing. Escalate to
   a `## NEXT SESSION` block; do not loop.

## 8. What the writeup looks like at the end

The capstone's structure for a final writeup:

* **Section 1 — Background**: continuous flow matching for text;
  the unification hypothesis.
* **Section 2 — The unification hypothesis fails**: protocol's
  Phases B/C/E/F/H + sanity checks. The cleanest single-result
  contribution: discrimination ≠ generation, with multiple
  empirical demonstrations (AUROC parity with linear probes;
  energy-gradient-not-denoiser; auditor-driven-gen-fails).
* **Section 3 — UQ via cached LM features works**: SVGP on
  `h_LLM` matches any complex baseline on detection AUROC and
  is dramatically better-calibrated. Validated on HaluEval-QA
  and TruthfulQA (Priority 1).
* **Section 4 — AR cascade contamination**: `h_LLM` features carry
  long-range memory of upstream corruption; SE is the only
  cleanly-localising metric. Distance-to-corruption recovery profile
  (the `phaseF_distance.png` figure) is a nice empirical
  demonstration.
* **Section 5 — Continuous-FM validity at K=27** (Priority 2a):
  either SFM closes the gap (positive headline) or it doesn't
  (regime-level negative confirmed across all published methods).
  Either result is publishable.
* **Section 6 — Two-model pipeline** (Priority 3): the
  integration story. Calibrated p_correct from the validator;
  p_valid from the generator's NLL; combined predictor with
  reliability diagram.
* **Section 7 — Limitations and future work**: what's tested only
  with GPT-2 small; sensitivity to LM choice; open questions about
  much-larger-vocab continuous FM.

That arc carries either a positive or a clean-negative ending and
in either case ends the capstone with a defensible contribution.

---

*See [`TRAINING_PROTOCOL_v2.md`](TRAINING_PROTOCOL_v2.md) for the
operational steps. Begin every new session by reading
`runs/DECISION_LOG.md` first.*
