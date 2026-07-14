# Session findings — OOD detection & self-healing benchmark (2026-07-12)

Scope: built a **reproducible, comparable benchmark** for Objective 3 (OOD detection
+ localize-then-inpaint healing) on the frozen full-text8 L256 `DirichletFM`
checkpoint (`runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM/epoch_best.pt`,
md5 `f1d809e8a42d`). Polished write-up lives in **`RESULTS.md` §Objective 3**; this
file is the session's findings + decisions + fixes. Artifacts: `bench_ood/`
(detection), `bench_heal/` (healing) — each with a self-contained `manifest.json`
(git SHA + `git_dirty` + `repro.patch` + `new_scripts/` snapshot + per-arm CLI).

Config unless noted: split=**test** (in-distribution — text8 train/val/test are one
distribution), n=256, fit_seqs=512, seed=42, corruption-rate grid 0.05…1.0. Detectors
each at their own best path-time: NLL t=3, BLR t=4.5, BGMM t=7.5, GPT2-SE (external).

---

## 1. What we detect, and how well

Four corruptions, ordered by distance from the language distribution:
**replace** (random char) · **shuffle** (reorder) · **falseinfo** (random real
same-length word) · **plausible** (the word that *minimises the model's own NLL* —
fluent but wrong).

### Finding 1 — geometric corruption is easy; training-free NLL wins (FINAL, 2026-07-14)
- Sequence AUROC saturates to ~1.0 for every detector by rate 0.15.
- **At the common (word) unit, NLL wins decisively:** replace **0.981**, shuffle
  **0.963** — vs GPT-2's 0.738 / 0.746. A zero-training denoiser-surprise readout is
  the best localizer for char/order noise, and the margin holds at the *fair* unit, not
  just at our native one (char: 0.968 / 0.948).
- GPT-2 cannot localize char noise **in any unit**: 0.480 (ΔE) / 0.557 (NLL) at BPE ≈
  chance, because the corruption re-tokenizes the text (Finding 5d).
- The old comparators (0.847 / 0.817) are gone: they were a mislabelled NLL scored with
  uniform BPE→char spreading. See **Finding 5**.

### Finding 2 — false-info localizes to the WORD (revised; and GPT-2 beats us here)
- Per-**character** AUROC is **weak and rate-invariant (~0.73 flat)** — the swap is a
  valid near-manifold word, so no per-character signal exists.
- **Sequence** AUROC *aggregates* that weak signal, so it **rises with corruption
  rate** (~0.6 @15% → ~1.0 @100%).
- ⚠️ **Revised (2026-07-14): "sequence-triage ONLY" was too pessimistic — an artifact of
  measuring per-character.** False-info is a *word-level* corruption, and at the word
  level our NLL reaches **0.813** (0.731 per-char). It *does* localize, to the word.
  Two consequences:
  - The claim **"false-info needs a corruption-specific supervised head"** does **not**
    survive: `BLR_FI`, trained on false-info, gets **0.800** at word level — it does not
    beat the zero-training NLL (0.813). (Supervision *is* still required for **plausible**
    — see Finding 3.)
  - **GPT-2 beats us here**: GPT2_NLL word **0.853** vs our 0.813 (seq 0.893 vs 0.743).
    An honest loss, on the axis where a pretrained LM's world knowledge pays.
- Practical framing survives at the sequence level (**"this passage is off — human,
  double-check"**), but word-level flagging is now defensible too.

### Finding 3 — the plausible boundary (the session's most interesting result)
Same slots, min-NLL swap. Seq / token AUROC on **plausible**:

| detector | plausible (min-NLL) | plausible (freq-matched control) |
|---|---|---|
| NLL | 0.45 / **0.26** (inverts by construction) | 0.46 / 0.34 |
| BLR / BGMM / logistic-fi | ~chance | ~chance |
| **logistic head trained on plausible** | **0.81 / 0.90** | **0.66 / 0.79** |
| GPT2-SE (external LM, untrained) | 0.73 / 0.59 | 0.76 / 0.65 |

- Plausible errors are **invisible to the model's own unsupervised detectors** (NLL
  *inverts*; density at chance) — because the swap was chosen to be low-surprise *to
  those readouts*.
- **A corruption-specific supervised head extracts a real signal**: 0.90 token raw.
  The **frequency-matched control** (replacement drawn from the *original word's own
  frequency band*) drops it to **0.79** — so **~0.11 was a genericness artifact**
  (min-NLL inserts common words) and **the residual ~0.79 is genuine
  contextual-wrongness** that the model's representation encodes but its likelihood
  readout leaves on the table.
- **Cross-transfer: a strict diagonal, measured in BOTH directions** (the earlier
  write-up only tested one, and the `Logistic_plaus → false-info` cell was never
  reported — it is the `random` column of `plausible_swap.json`, since "random" there
  *is* `corrupt_false_info` on the same slots). Token AUROC / *seq*:

  | trained on ↓ &nbsp; eval → | **false-info** | **plausible** |
  |---|---|---|
  | **false-info** (`Logistic_fi`) | **0.783** / *0.737* ✅ | 0.473 / *0.342* ❌ |
  | **plausible** (`Logistic_plaus`) | 0.471 / *0.384* ❌ | **0.904** / *0.808* ✅ |

  The off-diagonals are not merely at chance — at sequence level they are **below** it
  (0.342, 0.384): the heads **anti-transfer**. The sign is the finding. The false-info
  head learns *"this word is out of place"*, and a plausible swap is engineered to be
  maximally in-place → it scores **cleaner than clean**. The plausible head learns *"this
  word is suspiciously fluent for its slot"*, and a random word fails that test in the
  opposite direction. They detect **orthogonal defects**, not one defect at two
  difficulties.

### Finding 3b — a UNIFIED head *is* possible; the geometry dictates its shape (2026-07-14)

⚠️ Supersedes the earlier conclusion *"no general wrongness head exists"* — that inferred a
property of the **representation** from a property of how the **specialists were trained**.
Two measurements separate them (standardised features, t=4.5):

```
cos(d_falseinfo, d_plausible) = −0.49   PAIRED mean shifts OPPOSE  → why specialists anti-fire
cos(w_falseinfo, w_plausible) = +0.40   LEARNED directions ALIGN   → a shared w must exist
```
*(the paired difference — same positions, clean vs corrupt — is essential: an unpaired mean
difference gives a spurious +0.85, because corrupted positions are always letters, never
spaces, so it measures "letter vs space" instead of corruption.)*

Clean sits **between** the corruption clusters ⇒ a *signed* head must pick a side; the
geometry asks for a **shell** (sign-agnostic *departure* from clean). Training ONE head on
the union (replace+shuffle+falseinfo+plausible), held-out token / seq AUROC:

| head | → false-info | → plausible |
|---|---|---|
| `Logistic_fi` (specialist) | **0.783** / 0.737 | 0.473 / 0.342 ❌ |
| `Logistic_plaus` (specialist) | 0.471 / 0.384 ❌ | **0.904** / 0.808 |
| `ALL_linear` (signed) | 0.740 / 0.706 | 0.597 / 0.611 ✅ |
| **`ALL_quad`** (rank-8, sign-agnostic) | 0.727 / **0.740** | 0.683 / **0.704** ✅ |
| `ALL_mlp` | 0.706 / 0.654 | **0.710** / 0.686 ✅ |

- **Unification works** — all three clear chance on both.
- **Capacity buys plausible monotonically** (0.597 → 0.683 → 0.710): the hyperplane was the
  wrong shape.
- **The quadratic is the sweet spot, and that is the tell.** A rank-8 sign-agnostic form
  captures most of the MLP's gain and posts the **best sequence AUROC of any head**, while
  an MLP with far more capacity adds only ~0.03 ⇒ the cross-corruption structure genuinely
  IS a shell around clean. The geometry is the functional form, not a metaphor.
- **A real trade-off survives**: none reaches the plausible specialist's 0.904 (ceiling
  ~0.71, a **−0.19 tax**). The MLP hit **0.99 train accuracy vs 0.710 held-out** ⇒ a
  **representation** limit, not a capacity one — the two defects compete along shared
  directions.
- **At a DEPLOYABLE threshold the quadratic wins outright** — by far more than AUROC shows.
  Per-token F1 @5% FPR on **plausible**: `ALL_quad` **0.229** vs `ALL_mlp` 0.065 vs
  `ALL_linear` 0.031 (**7× the linear, 3.5× the MLP**), and it also leads on false-info
  (0.267). Why: a *signed projection* has no natural scale, so its threshold is arbitrary;
  **distance-from-clean does**. On AUROC the MLP looked competitive (0.710 vs 0.683) — at
  the operating point it is not. This is the clearest case in the whole benchmark for
  reporting F1 next to AUROC.
- **Deployment:** know the failure mode → specialist; don't → the **quadratic** unified
  head. True factual verification still needs world knowledge / retrieval or an external LM.

**Mechanism (why each row lands where it does).**
- *NLL inverts (0.26).* NLL **is** the surprise the swap was chosen to minimise, so
  swapped tokens have *lower* NLL than clean ones; token-AUROC ("do corrupt tokens
  score higher?") drops *below* 0.5. The detector is fooled by construction, not weak.
- *Density / wrong-trained heads at chance.* A plausible swap is a valid fluent word →
  its feature sits **near the ID manifold** (high density), so a one-class density
  can't see it; and heads trained on *replace* / *random-falseinfo* learned a
  different corruption signature.
- *Plausible-trained head, 0.90 → 0.79.* It learns the systematic clean-vs-swapped
  feature difference. The min-NLL swap has a confound — it tends to insert
  **generic/high-frequency** words (common words are the "safe" low-surprise pick).
  The **frequency-matched control** (candidate from the *original word's own frequency
  band*) removes that: the **0.11 drop = genericness artifact**, the **residual 0.79 =
  genuine contextual-wrongness**.
- *Why the signal is in the features but not the NLL.* NLL is **one** (softmax)
  readout of the features at `t_nll`; the swap was optimised against *that* readout.
  The full 1280-d feature at `t_eval` carries **more** than that readout uses, so a
  supervised head reading the whole vector finds a separating direction the softmax
  head doesn't exploit — the representation *encodes* the wrongness the likelihood
  leaves on the table.
- *GPT-2 (0.73, untrained).* The swap is adversarial to *DirichletFM's* NLL, not
  GPT-2's; a larger subword/web-trained LM flags it with no training — so the swaps
  are char-model-adversarial, not intrinsically invisible.
- *No cross-transfer between SPECIALISTS* ⇒ the two corruptions occupy different feature
  signatures. **But this does not mean one head can't cover both** — see **Finding 3b**: a
  single head trained on the *union* does, provided it is **sign-agnostic** (rank-8
  quadratic), because clean sits *between* the corruption clusters and a signed hyperplane
  must pick a side. Cost: ~0.19 AUROC vs the plausible specialist.

### Finding 4 — transfer to real out-of-domain text
Full-dim logistic false-info head trained on text8, evaluated on the **insulin
Wikipedia article** (out-of-domain):

| domain | seq AUROC | token AUROC |
|---|---|---|
| text8 (in-distribution) | 0.853 | 0.869 |
| insulin (out-of-domain) | 0.801 | 0.801 |

**Discrimination transfers** (~0.07 drop); **threshold calibration does not** (clean
OOD scores shift). This reconciles the AUROC-vs-recall gap: ranking survives OOD,
calibrated recall (which healing needs) collapses.

**Mechanism.**
- *AUROC is threshold-free* — it measures *ranking* (corrupt vs clean token), so the
  small 0.87 → 0.80 drop means the detector's ordering ability survives the domain
  shift.
- *A deployed detector needs a threshold* τ (flag if score > τ), set on clean text8.
  On insulin the **clean scores shift up** — clean biomedical text is itself partly
  OOD to a text8 model (every domain word is mildly surprising) — so the text8-set τ
  lands in the wrong place and the operating point mis-fires.
- *This reconciles the earlier insulin healing failure.* Track-C healing reported
  loc-**recall 0.08–0.28** (measured at a fixed τ), which looked like total failure;
  the threshold-free AUROC (0.80) shows **discrimination was fine — calibration
  wasn't**. A *calibration* problem, not a *discrimination* problem.
- *General principle* (the "both would be OOD" regime, made precise): when the clean
  reference itself shifts OOD, **absolute thresholds break but relative ranking
  holds**. Deployment fix: per-passage score normalisation, rank-based (top-k%)
  flagging, or recalibrating τ on a small clean sample from the target domain.
- *Note on the absolute level.* This text8 number (0.869) exceeds the ~0.72 quoted
  elsewhere because the false-info training negatives were generated at the **eval
  rate (0.15)**, not 0.5 — matched-rate training lifts the held-out ceiling to ~0.87,
  closer to the in-sample 0.93.

### Finding 5 — the GPT-2 baseline was wrong twice, and fixing it exposed a real methodological result (2026-07-13)

Two independent bugs in the external-LM baseline, plus a finding that came out of
fixing them. Measured on the *final* corrected arms (`bench_ood/gpt2_se/`,
`bench_ood/gpt2_nll/`; n=256, test).

**(a) "Spilled energy" was never spilled energy.** All five sites computed the
SAME-STEP per-token NLL `logsumexp(logits[i-1]) − logits[i-1][x_i]` = `−log p(x_i|x_<i)`.
The real quantity (Minut, Dewidar & Masi, *Spilled Energy in LLMs*, ICLR 2026,
arXiv:2602.18671, Def 4.1 / Eq. 8) is CROSS-STEP:
`ΔE_i = logsumexp(logits[i]) − logits[i-1][x_i]` — logit energy read at step `i-1`,
marginal energy at step `i`. They differ by `logsumexp(logits[i]) − logsumexp(logits[i-1])`,
which is precisely the signal. Now validated to **0.0 error** against the authors'
own code (`OmnAI-Lab/spilled-energy`, `energy.py`) by `scripts/validate_spilled_energy.py`.
**Sign trap:** follow the code (`delta = -E_margin + E`), *not* the paper's Eq. 8 prose,
which carries a sign typo — a flip inverts AUROC.

**Zero-property, and why it looked broken.** The paper says ΔE→0 for correctly-modelled
text. Measured: **−0.30 on in-domain English** (✓ holds) but **+3.4 on clean text8**.
That is *not* a bug — GPT-2 genuinely finds lowercase/unpunctuated text8 out of
distribution. Check the property on in-domain English, never on text8.

**(b) ΔE is not a localizer — by construction.** It straddles two decoding steps, so it
cannot attribute to one. Sequence AUROC ~1.0; BPE-level AUROC ~0.48 (chance). The old
headline "our NLL beats GPT-2 spilled energy per-token" was really *our NLL vs GPT-2's
NLL*; beating ΔE at localization is a straw man. The bench now runs **two** GPT-2 arms:
`gpt2_se` (real ΔE) and **`gpt2_nll`** (same-step NLL — the fair localization comparator).

**(c) The evaluation protocol (the real result).** Our detector scores CHARACTERS, GPT-2
scores BPE TOKENS. Attributing a BPE score *down* onto characters is ill-posed — GPT-2
has no per-character opinion — and the old code did it by spreading each token's score
uniformly over its chars, which smeared localization. Pooling *up* is exact. So:

> **flow-matching → char (native) · GPT-2 → BPE (native) · both → WORD (the comparison)**

Words are exact on both sides (GPT-2's pre-tokenizer splits on whitespace, so a BPE token
never crosses a word boundary) and are the unit the corruptions actually use.
Sequence level is already comparable *provided* the total is normalised per **character**
(the bits-per-character convention); a mean over BPE tokens is **not** comparable — see (d).

**(d) Character corruption SHATTERS GPT-2's tokenization.** This is the finding.

| | n BPE tokens (256×256 chars) | BPE prevalence |
|---|---|---|
| clean / falseinfo@0.05 | ~13,455 | 0.050 |
| **replace@0.15** | **22,801 (+70%)** | **0.399** |
| replace@0.5 | 33,014 (+145%) | 0.771 |
| falseinfo@0.15 | 13,683 (+1.7%) | 0.158 |

Random characters re-tokenize the text into a mass of fragments: at 15% char corruption
**40% of BPE tokens touch a corrupted char** (not 15%), at 50% it is 77%. So for
char-level corruption the BPE unit is *itself a function of the corruption* — "which BPE
token is wrong" is not a stable question, and BPE-level AUROC there (SE 0.48, NLL 0.56)
is measuring a broken unit as much as a weak score. For **word-level** corruption the
opposite holds: `falseinfo` swaps real words in, tokenization is stable (+1.7%),
prevalence tracks the corruption rate (0.15→0.158), and BPE numbers become meaningful.

⇒ **The word level is not merely the *fair* unit; for char corruptions it is the only
*stable* unit GPT-2 has.** This is the methodological justification for the protocol.

**(e) Corrected GPT-2 numbers** (seq | BPE native | word max | word mean):

| corruption | GPT2_SE (ΔE) | GPT2_NLL |
|---|---|---|
| replace@0.15 | 1.000 \| 0.480 \| 0.727 \| 0.524 | 1.000 \| 0.557 \| 0.738 \| 0.596 |
| shuffle@0.15 | 1.000 \| 0.493 \| 0.724 \| 0.542 | 1.000 \| 0.581 \| 0.746 \| 0.623 |
| falseinfo@0.15 | 0.873 \| 0.621 \| 0.690 \| 0.672 | 0.893 \| **0.768** \| **0.853** \| 0.838 |

Two caveats we should state rather than hide:
- **GPT-2's perfect sequence AUROC on replace/shuffle (1.000) is cheap.** It is detecting
  "this is no longer English" via the tokenization blow-up, not "this token is wrong."
- **GPT-2_NLL is genuinely strong on false-info** (word 0.853) — plausibly *better* than
  our detectors (~0.72 per-char). This is the one axis where the external LM may beat us,
  and it should be reported as a real competitive result, not a walkover. Pending the FM
  word-level rerun for the head-to-head.

---

## 2. Healing recovers geometry, never meaning

`bench_heal/` — localize (NLL/BLR/BGMM) → mask → Dirichlet-FM inpaint → score.
Reported at the **least-damaging (max net/corrupt)** operating point:

| localizer | replace | falseinfo | insulin char-noise / false-info |
|---|---|---|---|
| NLL | **+0.42** | −0.13 | **+0.37** / −0.14 |
| BLR | +0.21 | −0.09 | — |
| BGMM | +0.15 | −0.06 | −0.30 / −0.50 |

- Char/geometric healing is **net-positive and transfers to the real article**
  (NLL +0.37 on insulin Track A).
- False-info healing is **net-negative at every operating point** — a fluent wrong
  word is information-theoretically unrecoverable from context, regardless of
  localizer. **Detection ≠ healing.**
- Healing is **damage-averse**: net/corrupt increases monotonically as FPR drops
  (fewer, higher-confidence edits) — heal only what you're confident about.

---

## 3. Methodological corrections made this session

1. **In-sample vs held-out.** The latent-split full-dim linear probe reported **0.93**
   per-token on false-info, but `_probe_auroc` **fits and scores the same tokens**.
   Deployed **held-out** it is **~0.75–0.87** (the range is training-rate sensitivity:
   train-rate 0.5 → ~0.75; matched to eval rate 0.15 → ~0.87). Report held-out.
2. **SVGP does not help** (confirmed against prior repo experience): the t-SNE shows
   no nonlinear structure beyond the linear axis (information limit, not capacity), so
   a nonlinear kernel saturates at the same ~0.72–0.87; and its predictive variance
   **collapses to a constant** in 1280-d (concentration of measure) — which is why the
   repo already retired `DirichletFMSvgp` for the closed-form Laplace variance.
3. **Detection is contrast against a *normal background*.** OOD *words* on an
   in-distribution background are easy; an OOD *background* (domain shift) collapses
   the reference (the "both would be OOD" regime) — the reason insulin (out-of-domain)
   is harder than text8 false-info.
4. **Supervised beats unsupervised; SPECIALISTS are corruption-specific, but one head can
   still span them.** A full-dim head trained on a corruption reads signal the model's own
   NLL/density miss (labeled-negatives cost). Specialists do **not** cross-transfer — they
   actively anti-fire on each other, because the paired mean shifts oppose (cos = −0.49) and
   a *signed* head must pick a side. The fix is the **functional form**, not more data or
   more capacity: a **sign-agnostic** rank-8 quadratic ("distance from clean") covers both
   (Finding 3b), and an MLP adds nothing beyond it. Residual −0.19 AUROC vs the plausible
   specialist is a **representation** limit (MLP: 0.99 train acc, 0.710 held-out).

## 4. Bugs found & fixed

1. **BLR CUDA OOM on 8 GB** — the linear head materialised *all* per-token features on
   the GPU (multi-GB at fit_seqs=512, ×5 for the adversarial mix). Fixed in
   `ood_bayes_linear.py` and `heal_dirichlet.py:train_localizer`: features stay
   **CPU-resident**, the head trains on **16k minibatches**, and the Laplace
   covariance/cholesky runs **chunked on CPU** (matching NLL/BGMM). Arms were run
   sequentially (not in parallel) — the orchestrator blocks per arm.
2. **Healing selected the *worst* operating point.** `heal_dirichlet` chose the FPR by
   **max cal-F1**, which over-weights recall — but healing is damage-averse, so the
   high-recall point does net harm (bgmm_replace −0.09 when +0.15 was available).
   Fixed: deployable selection = cal-set **F0.5** (precision-weighted); the benchmark
   reports the **least-damaging (max net/corrupt)** point.
3. **Latent-split ran on val, not test** (no `--split` arg). Added `--split`
   (default test) for consistency with the detectors; clarified that clean text8 is
   in-distribution (only corruption is OOD).

## 5. Reproducibility

Every arm's exact CLI is in the manifest. Because the working tree is dirty (new
benchmark scripts are untracked), each `manifest.json` also carries `git_dirty`, a
`repro.patch` (tracked diffs), and a `new_scripts/` snapshot (untracked scripts).
Reproduce any arm:

```
git checkout <manifest.git_sha> && git apply bench_ood/repro.patch \
  && cp bench_ood/new_scripts/* scripts/ && <arm cmd from manifest.arms[]>
```

New/edited scripts: `_bench_common.py` (provenance, `DETECTOR_KEYS`,
`heal_style_examples`, `make_adversarial_negatives`, `finalize_manifest`),
`run_bench_ood.py`, `run_bench_heal.py`, `bench_aggregate.py`, `plot_heatmaps_all.py`,
`ood_plausible_swap.py` (+`--swap-mode freq_matched`), `eval_detector_transfer.py`,
`ood_bgmm_perpos.py`; detector edits to `ood_bayes_linear.py` (`--head logistic`,
`--train-schemes`), `ood_denoiser_nll.py`, `ood_gpt2_spilled_energy.py`
(+falseinfo/examples), `plot_latent_split.py`.

## 6. Open threads / next steps

- The held-out full-dim head at **matched training rate reaches ~0.87** on false-info
  — closer to the in-sample ceiling than the earlier ~0.72 suggested. Worth a clean
  train-rate sweep to pin the deployable ceiling.
- **Escalation cascade**: char-model flags distributional anomalies cheaply → route
  the fluent/factual tail to GPT-2 (0.73–0.76 on plausible) or retrieval. Only real
  path for genuine factual verification.
- Insulin **detection** transfers (AUROC) but **calibration** doesn't — an OOD-robust
  threshold (e.g. per-passage normalisation) is the missing piece for deployment.
