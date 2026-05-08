# Capstone follow-up experiments — protocol

**Version:** 1.0  •  **Drafted:** 2026-05-08  •  **Supersedes:** none (new)

This protocol covers three focused follow-up tracks identified after the
Phase A–Q work documented in `runs/DECISION_LOG.md`. It is a *closing*
protocol: each phase is scoped to either lock a writeup contribution or
falsify it cleanly. No exploratory branching.

---

## §1  Goals and framing

The capstone writeup arc, after Phase Q, has three pillars:

1. **The regime-level triangulation.**  Three independent
   continuous-on-simplex methods (EqM, FMonCLR, LogitKLFlow) cluster
   at KL_bi ≈ 1.4 vs DFM 0.148 at parity compute.  The current
   framing is "structural, regime-level"; the goal of **Protocol A**
   is to identify *what specifically* bounds them.  The leading
   hypothesis is sampler determinism — continuous ODEs encode all
   sample diversity in the initial noise, while discrete denoisers
   draw a fresh categorical sample at every step.

2. **The W1 ablation.**  Euler on `f` underperforms Euler on
   `∇⟨x,f⟩` by ~22 % KL_bi on the same trained checkpoint.  The
   conclusion is that EqM stores its data-pulling structure in the
   conservative gradient, not in the velocity itself — a side
   effect of FM regression.  **Protocol B** tests whether training
   directly on the conservative gradient closes that gap.

3. **The cascade contamination caveat.**  Per-token AUROC at
   uncorrupted positions in invalid sequences is 0.97 for the
   trained EqM auditor and 0.97 for a vanilla linear probe on
   `h_LLM`, vs 0.66 for Spilled Energy.  **Protocol C** turns this
   into a methodological audit applicable to a non-trivial chunk of
   the recent hallucination-detection literature.

Each protocol terminates on either (a) a positive constructive
finding that lands as a writeup contribution, or (b) a clean
negative that bounds future work.  Both outcomes are publishable;
we do not retry on no-result.

---

## §2  Pre-flight checks (start of every session)

Run these in order.  If any fail, fix before launching any phase.

```bash
# Repository state
cd /workspace/aitchinson_flow
git status              # expect clean working tree
git log --oneline -1    # note commit SHA

# Existing checkpoints needed
ls runs/eqm_data50k_ep5_v2/epoch_final.pt
ls runs/dfm_data50k_ep5_v2/epoch_final.pt
ls runs/fmclr_data50k_ep5_v2/epoch_final.pt
ls runs/lkflow_data50k_ep5/epoch_final.pt
ls runs/aud_gpt2_ctx/epoch_final.pt
ls runs/best_auditor.pt   # symlink check
ls data/wiki_cache_gpt2.pt
ls data/hallueval_cache_gpt2.pt

# Python / GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "from src.aitchinson_flow.config import EqMConfig; print('config OK')"

# W&B (optional — group auto-derives from sweep stem)
[ -f ~/.netrc ] && grep -q api.wandb.ai ~/.netrc && echo "wandb authed"
```

Numbers below assume the same compute platform as the v1/v2 protocols
(20 GB A100 MIG ≈ 14 min per 5 ep × 50 k window training cell at
d=1024/8L, or the 8 GB Blackwell at B=32 ≈ 1.7× longer).  Re-time the
first cell of each phase before scaling.

---

## §3  Common conventions

- **Run names:** `{family}_{platform}_{key}` — e.g. `eqm_sde_T0p1`,
  `eqm_K2_data50k_ep5`.  No spaces.
- **Sweep specs:** YAML in `sweeps/`, one per phase.  Idempotent via
  the eval.json existence check — **never** delete a sweep cell's
  run-dir to "force" a re-run; that is the smoke-test escape hatch
  documented in the v1 NEXT-SESSION block.
- **Eval probe:** `_unigram_kl_probe` in `runner.py` continues to
  emit `unigram_kl`, `bigram_kl`, `trigram_kl`.  Decision criteria
  in this protocol all use the **post-training** `eval.json`
  numbers (256 samples × 200 NAG steps unless overridden), not the
  in-training probe.
- **Decision-log discipline:** every cell appends one entry to
  `runs/DECISION_LOG.md` with the format `## [<UTC>] <run>` →
  `Hypothesis / Result / Decision / Next`.
- **Termination rule:** a phase exits successfully on either PASS
  or FAIL.  PARTIAL means continue to the next sub-cell *within*
  the phase; PARTIAL never escalates to an unplanned sweep.
- **Compute discipline:** every phase has a hard wall-clock cap
  in §4–§6.  If a cell exceeds 1.3× its cap, kill it and log the
  partial result as a NEGATIVE.  No "let it run a bit longer".

---

# Protocol A — Bound identification

**Goal:** identify what specifically bounds continuous-on-simplex
methods at K=27 to KL_bi ≈ 1.4.  Leading hypothesis: sampler
stochasticity.  Two phases, both clean falsifications of distinct
sub-hypotheses.

## §4  Phase R — SDE sampling on continuous methods (eval-only)

**Hypothesis (R-H1):** the EqM/FMonCLR/LogitKLFlow gap to DFM is
caused by deterministic-ODE sampling, not by the trained field.
Adding Langevin-style noise injection at every integration step
should close the gap.

**Sub-hypothesis (R-H2):** the σ_init=0.3 partial recovery in the
W1 sweep (KL_bi 1.682 → 1.384 on the EqM checkpoint with raw-f
Euler) is a preview of this — entry noise alone partially substitutes
for in-trajectory stochasticity, but full Langevin should do better.

### Pre-conditions
- `runs/{eqm,fmclr,lkflow}_data50k_ep5_v2/epoch_final.pt` exist
  (Phase 10 / W1 / W4 checkpoints).
- The decision criterion (KL_bi ≤ 0.40) is calibrated against DFM's
  0.148 at parity compute.  PASS = "within 3× of DFM at parity
  compute via sampler change alone".

### Code changes
1. **`src/aitchinson_flow/sampling/sde.py`** (new).  Single function:

   ```python
   @torch.no_grad()
   def sde_flow_sample(
       model, x0, *, n_steps, gamma_schedule="linear",
       use_grad=False, alpha=0.0, jitter_seed=None,
   ):
       """
       Stochastic-flow Euler-γ sampler.
       
       Update rule per step (with γ stepping 0→1):
           v = ∇⟨x,f(x;γ)⟩ if use_grad else f(x;γ)
           x ← x + h·v + sqrt(2·alpha·h)·ξ,  ξ ~ N(0,I)
       """
   ```
2. **`src/aitchinson_flow/models/eqm.py::EqM.sample`** — add
   `method ∈ {"nag", "euler", "sde"}` and `alpha`/`use_grad` kwargs.
   Default behaviour unchanged.
3. **`src/aitchinson_flow/models/fmclr.py::FMonCLR.sample`** and
   **`logitkl_flow.py::LogitKLFlow.sample`** — add `method="sde"`
   route to `sde_flow_sample`.
4. **`scripts/eval_full.py`** — accept `--sample-method`,
   `--sample-alpha`, `--sample-use-grad` flags; pass through to
   `model.sample()`.  Persist these in the resulting `eval.json`
   under a `sample_kwargs` key.

No model retraining; no training-loop changes.  All eval-only.

### Sweep cells
`sweeps/phaseR_sde.yaml` — eval-only sweep over the three trained
checkpoints, written as a list of `{name, ckpt, sample_kwargs}`
entries.  `run_sweep.py` already supports this via the
"eval_only" branch.

| cell | model | use_grad | α (noise scale) | NFE |
|---|---|---|---:|---:|
| eqm_sde_a0p00_grad     | eqm   | True  | 0.00 | 128 | (= W1 Euler-on-grad control, expect KL_bi ≈ 1.39) |
| eqm_sde_a0p01_grad     | eqm   | True  | 0.01 | 128 |
| eqm_sde_a0p03_grad     | eqm   | True  | 0.03 | 128 |
| eqm_sde_a0p10_grad     | eqm   | True  | 0.10 | 128 |
| eqm_sde_a0p30_grad     | eqm   | True  | 0.30 | 128 |
| eqm_sde_a1p00_grad     | eqm   | True  | 1.00 | 128 |
| eqm_sde_a0p10_grad_n64 | eqm   | True  | 0.10 |  64 | (NFE robustness check) |
| fmclr_sde_a0p00        | fmclr | False | 0.00 | 128 | (= prior Euler control) |
| fmclr_sde_a0p10        | fmclr | False | 0.10 | 128 |
| fmclr_sde_a0p30        | fmclr | False | 0.30 | 128 |
| lkflow_sde_a0p00       | lkflow| False | 0.00 | 128 |
| lkflow_sde_a0p10       | lkflow| False | 0.10 | 128 |
| lkflow_sde_a0p30       | lkflow| False | 0.30 | 128 |

13 cells × ~2 min each (256 samples × 128 NFE, no training) ≈ 30 min wall-clock.

### Decision criteria
Compute the **best-α KL_bi** for each model.  Apply per-model:

| outcome | criterion | interpretation |
|---|---|---|
| **PASS**    | best KL_bi ≤ 0.40 | sampler stochasticity is the bound (R-H1 confirmed) |
| **PARTIAL** | best KL_bi ∈ (0.40, 0.97] | stochasticity helps but doesn't fully close the gap |
| **FAIL**    | best KL_bi > 0.97 | stochasticity not the bound; H1 falsified |

**Phase-level verdict (across all three models):**

- **Strong PASS** = ≥ 2 of 3 models PASS → headline writeup finding
  ("the continuous-on-simplex gap is a sampler-stochasticity gap").
  Proceed to Phase S as confirmation; skip Protocol-A's K=2 collapse.
- **Mixed** = 1 PASS or all PARTIAL → publishable as "stochasticity
  helps but is not the full story"; run Phase S to disambiguate.
- **Strong FAIL** = 0 PASS and ≥ 2 FAIL → R-H1 falsified; the bound
  is something deeper (most likely the dimensionality of the
  pushforward of σ-noise through a smooth velocity field).
  **Run Phase S anyway** — K=2 collapse is the cheapest second
  diagnostic regardless of R-outcome.

### Risk
- **EqM's energy field is ≈0 near the data manifold at γ=1** (per
  Phase F divergence-trace finding).  This is a problem only if you
  apply Langevin *at* γ=1 (which is Phase H, already failed).  The
  SDE sampler integrates γ from 0 → 1, so this risk does not apply
  here.  But sanity-check by logging `‖v‖` at γ ∈ {0.1, 0.5, 0.9}
  during the first cell — if any γ has ‖v‖ ≈ 0, the field is
  partially trivial and the Langevin step degenerates to pure
  diffusion in that region.
- **σ-noise injection on the simplex is mildly off-manifold.**  CLR
  features live in R^{K-1} (unconstrained), so noise injection is
  fine.  For LogitKLFlow (R^K) also fine.  For raw simplex
  parameterisations only, project back via softmax after each step.
  Our existing models all live in unconstrained spaces — no
  projection needed.

---

## §5  Phase S — K=2 binary collapse (training + eval)

**Hypothesis (S-H1):** at K=2, the per-position multimodality
disappears (each position is a binary choice), so DFM's discrete
sampler advantage over continuous flow ODEs should vanish.  If
EqM and DFM hit the same KL_bi at K=2, the K=27 gap is dominated
by per-position multimodality (which the ODE can't represent
deterministically).  If EqM still lags, the bound is something
else.

This phase complements Phase R: R tests the *mechanism*
(stochasticity), S tests the *regime* (multimodality vs K).

### Pre-conditions
- Phase R complete (we want R's verdict in hand before deciding
  whether S is interesting).  S still runs even if R is a strong
  PASS, as a sanity confirmation.

### Code changes
1. **`src/aitchinson_flow/data/text8_binary.py`** (new).  Map text8
   characters to {0, 1} via vowel/consonant (vowels = `aeiou` →
   class 0; everything else → 1; whitespace → 1).  Cache to
   `data/text8_binary_cache.pt`.  Same window length L=40.
2. **Config:** `text8_dataset.alphabet ∈ {"full", "binary"}`,
   default "full".  Plumbed through `data_sources.py` and the
   tokenizer.
3. **`scripts/eval_full.py`** — already KL-agnostic over K (uses
   actual histogram bins), so works for K=2 with no changes.

### Sweep cells
`sweeps/phaseS_K2.yaml`:

| cell | model | epochs | windows | data |
|---|---|---:|---:|---|
| eqm_K2_data50k_ep5    | eqm   | 5  | 50 000 | binary |
| dfm_K2_data50k_ep5    | dfm   | 5  | 50 000 | binary |
| fmclr_K2_data50k_ep5  | fmclr | 5  | 50 000 | binary |
| lkflow_K2_data50k_ep5 | lkflow| 5  | 50 000 | binary |

Eval each at NAG (EqM) / Euler (DFM, FMonCLR, LogitKLFlow) and at
the SDE sampler with α from the Phase-R best-cell value (re-uses
the new code path).

4 cells × ~14 min training + ~5 min eval (incl. SDE re-sampling)
≈ 80 min wall-clock.

### Decision criteria
Compare EqM-best vs DFM-best KL_bi at K=2:

| outcome | criterion | interpretation |
|---|---|---|
| **EQUAL**   | \|Δ KL_bi\| ≤ 0.05 | K=27 multimodality is the bound; continuous and discrete are equivalent at K=2 |
| **GAP**     | \|Δ KL_bi\| > 0.20 | bound is *not* multimodality alone; deeper structural issue |
| **AMBIG**   | otherwise        | inconclusive — log and move on |

Report the absolute numbers too — at K=2 the natural-language
bigram entropy is much lower (~1 bit/char) and KL_bi will be small
in absolute terms; the *relative* comparison is what matters.

### Combined Protocol-A verdict (Phase R + S)
Two-by-two outcome table for the writeup:

|         | R PASS                           | R FAIL                               |
|---------|----------------------------------|--------------------------------------|
| S EQUAL | the bound *is* sampler stochasticity, K=27 has a stochasticity ceiling no continuous ODE can clear → **strong headline finding** | sampler is *not* the bound but K=2 is fine; bound is K-dependent multimodality coverage → **moderate finding, frame mechanistically** |
| S GAP   | sampler is the bound at K=27, but K=2 still has a gap → check FMonCLR-K=2 to disambiguate → likely a *parameterisation* issue at K=2 | bound is structural and not driven by either lever → **honest negative**, frame as "open question" |

In all four quadrants the Protocol-A result is publishable.

### Hard cap: 2 hr total wall-clock for Phase S.

---

# Protocol B — W1 mechanistic follow-up

**Goal:** test whether training EqM with the conservative-gradient
target (instead of the raw velocity) closes the Euler-on-`f` vs
Euler-on-`∇⟨x,f⟩` gap.  This isolates "is the FM regression
target the lever?" cleanly.

## §6  Phase T — Conservative-gradient regression (training + eval)

**Hypothesis (T-H1):** the gap between Euler-on-`f` (KL_bi 1.682)
and Euler-on-`∇⟨x,f⟩` (1.391) on the same eqm_data50k_ep5_v2
checkpoint exists because FM regression supervises `f` directly,
leaving the data-pulling Jacobian information unsupervised.
Training a model whose *output* is `∇⟨x,f⟩` and whose *target* is
the FM target should make Euler-on-`f` (i.e. Euler on the model's
output) match the existing checkpoint's Euler-on-`∇⟨x,f⟩`
behaviour.

**Falsification path (T-H2):** if Euler-on-output of the new model
still trails NAG by the same ~22 %, the regression target was not
the lever, and the Jacobian extraction is doing something the FM
regression objective fundamentally cannot replicate.

### Pre-conditions
- `runs/eqm_data50k_ep5_v2/epoch_final.pt` exists (the comparison
  reference for the `f` vs `∇⟨x,f⟩` gap).
- Phase R complete (informs whether sampler stochasticity should
  be added on top in the eval).

### Code changes
1. **`src/aitchinson_flow/models/eqm_consgrad.py`** (new model).
   Same backbone as EqM, same input/output shapes.  Forward pass:

   ```python
   def forward(self, x, gamma, h_ctx=None):
       # standard transformer pass to get f-shaped tensor
       f = self.backbone(x, gamma, h_ctx)
       # compute ∇_x ⟨x, f⟩ via autograd
       energy = (x * f).sum(dim=(-1, -2))
       grad = torch.autograd.grad(energy.sum(), x, create_graph=True)[0]
       return grad   # this is the model's "v"
   ```

   Training loss: `L = ‖model(x_γ, γ) - (x_1 - x_γ)‖²` —
   identical to FM regression, but the model's output is now the
   conservative gradient itself.

2. **`src/aitchinson_flow/config.py`** — register `EqMConsGradConfig`
   (mostly identical to `EqMConfig`, omits the energy-hinge fields
   since this is for generation, not auditing).

3. **`src/aitchinson_flow/training/runner.py`** — already dispatches
   on model type; new model registers via the standard
   `ModelRegistry`.

4. **`scripts/eval_full.py`** — handle the new model type identically
   to EqM in the decode path (it predicts `∇⟨x,f⟩`-shaped tensors
   in CLR space, decoded the same way).

5. **`scripts/eval_w1_compare.py`** (new, ~80 lines).  Loads two
   checkpoints (the existing eqm and the new eqm_consgrad) and
   evaluates each at three sampler settings: (a) Euler on the
   model's output, (b) Euler on `∇⟨x, model_output⟩`, (c) NAG.
   For the new model, (a) is the natural sampler.  Reports the
   six-cell table.

### Sweep cells
`sweeps/phaseT_consgrad.yaml`:

| cell | model | platform | notes |
|---|---|---|---|
| eqm_consgrad_data50k_ep5         | eqm_consgrad | data_50k_ep5 | parity-compute training |
| eqm_consgrad_data50k_ep5_lr_5    | eqm_consgrad | data_50k_ep5 | lr=2.5e-4 (half default), in case 2nd-order autograd path destabilises |

Two cells × ~14 min training + ~3 min eval ≈ 35 min wall-clock.

### Decision criteria

| outcome | criterion | interpretation |
|---|---|---|
| **PASS**    | Euler-on-output(consgrad) KL_bi ≤ 1.45 (within 0.05 of NAG-on-eqm 1.382) | the regression target *was* the lever; W1's "gap is in the Jacobian" finding is fully explained |
| **PARTIAL** | KL_bi ∈ (1.45, 1.65] | partial — Jacobian extraction does some of the work, regression target does some |
| **FAIL**    | KL_bi > 1.65 | the regression target wasn't the lever — Jacobian extraction is doing something the regression can't replicate.  Stronger writeup claim: *FM regression is fundamentally lossy w.r.t. data-manifold sharpness* |

All three outcomes are publishable.  PASS gives you a clean
constructive finding ("here is a less wasteful FM training
recipe"); FAIL gives you a sharper mechanistic claim.

### Risk
- **Second-order autograd through the backbone is ~2.5× slower per
  step than first-order.**  Budget 35 min, hard-cap at 50 min.
  If training crashes with `create_graph=True` OOM at B=64, drop
  to B=32; document.
- **Numerical stability.**  Adam on a doubly-differentiated loss
  can be sensitive to lr.  The lr=2.5e-4 cell exists for this.
- **The hypothesis presumes the input-output shape match holds.**
  Verify in a 1-step smoke test that `model(x, γ)` returns a
  tensor the same shape as `(x_1 - x_γ)` before launching.

---

# Protocol C — Cascade audit

**Goal:** turn the Phase F / F-sanity findings about cascade
contamination into a methodological audit applicable to any
hallucination-detection method that probes autoregressive LM
hidden states for per-token / per-span localisation.

The deliverable is a small protocol + diagnostic table that any
practitioner can apply to their own probe.  The Phase U/V cells
demonstrate it on existing methods (locality-clean SE, plus
cascade-contaminated linear probe and EqM auditor) and on a
re-implemented SAPLMA-style probe so the writeup has at least one
*published-method* data point.

## §7  Phase U — Position-shuffle and uncorrupted-position controls (eval-only)

**Hypothesis (U-H1):** the diagnostic "AUROC at uncorrupted
positions in invalid sequences" cleanly separates locality-clean
methods (SE: 0.66) from cascade-contaminated methods (linear
probe on h_LLM: 0.97; trained EqM auditor: 0.97).

**Hypothesis (U-H2):** a second diagnostic — *position-shuffle
ablation* — should give a complementary signal.  Permute the order
of `(h_LLM, token, label)` triples within each sequence before
running the per-token classifier.  A locality-clean classifier
should retain its corrupted-position AUROC (it operates on local
features); a cascade-contaminated classifier's AUROC should drop
toward chance (the cascade signal is destroyed by the permutation).

### Pre-conditions
- `data/wiki_cache_gpt2.pt` exists (Phase F cache, 300 chunks).
- `runs/aud_gpt2_ctx/epoch_final.pt` exists (best F1 auditor).
- `runs/phaseF_uq.json` exists (the Phase F+ UQ comparison numbers
  to extend with these new diagnostics).

### Code changes
1. **`scripts/cascade_audit.py`** (new, ~200 lines).  Takes a list
   of method specs (each with a `score_fn(cache_chunk) →
   per_token_logits` callable) and produces, for each method,
   a 4-cell table:

   |                    | corrupted positions | uncorrupted positions |
   |--------------------|---------------------|------------------------|
   | sequence-aligned   | AUROC₁               | AUROC₂                  |
   | position-shuffled  | AUROC₃               | AUROC₄                  |

   Where:
   - AUROC₁: standard "Tok AUROC at corrupted positions" (the
     headline number reported in the literature).
   - AUROC₂: the locality control — AUROC at uncorrupted positions
     in invalid sequences, comparing to clean sequences.  The
     existing Phase-F-sanity diagnostic.
   - AUROC₃, AUROC₄: same, but with `(h_LLM, token, label)`
     triples permuted within each sequence (RNG seed fixed).

2. **Locality-clean variant: `SE_local`.**  Spilled Energy is
   already locality-clean by construction (uses position-local
   logits).  Verify by checking AUROC₁ ≈ AUROC₃ (shuffle-invariant).
   This is the gold-standard column for the writeup.

3. **`scripts/plot_phaseU.py`** — produces a single grouped bar
   chart per method, four bars per group (AUROC₁..₄), with a
   horizontal line at 0.5.  Locality-clean methods have flat bars
   across columns; cascade-contaminated methods show a
   characteristic AUROC₂ ≈ AUROC₁ (cascade flag fires across
   positions) and AUROC₃, AUROC₄ both near 0.5 (shuffle destroys
   the cascade).

### Sweep cells
No training. `cascade_audit.py` runs all method specs in one process.

| method | type | expected AUROC₁ | expected AUROC₂ | expected AUROC₃ |
|---|---|---:|---:|---:|
| Spilled Energy             | locality-clean (zero-train) | ~0.97 | ~0.66 | ~0.97 |
| Linear probe on h_LLM      | cascade-contaminated        | ~0.99 | ~0.97 | ~0.50 |
| BLR-Laplace on h_LLM       | cascade-contaminated        | ~0.97 | ~0.97 | ~0.50 |
| SVGP on h_LLM              | cascade-contaminated        | ~0.98 | ~0.97 | ~0.50 |
| EqM auditor (aud_gpt2_ctx) | cascade-contaminated        | ~0.99 | ~0.97 | ~0.50 |
| top-K entropy on logits    | locality-clean              | ~0.95 | ~0.65 | ~0.95 |

The numbers above are predictions; actual results go in
`runs/phaseU_cascade_audit.{md,json}`.  Wall-clock ≈ 15 min.

### Decision criteria

This phase **always lands the methodology contribution**, regardless
of numbers.  The four-AUROC diagnostic *is* the contribution; the
results just populate the table for the writeup.

The **strength** of the writeup claim depends on which of the three
following patterns shows up:

| pattern | claim |
|---|---|
| **Clean separation**: locality-clean methods have flat (AUROC₁ ≈ AUROC₃) bars; cascade-contaminated methods have AUROC₃, AUROC₄ both near 0.5 | strong claim — propose the four-AUROC table as a required diagnostic for any per-token AUROC paper |
| **Partial**: shuffle reduces but doesn't eliminate cascade-method AUROC | moderate claim — diagnostic is informative but not definitive |
| **No separation**: locality-clean methods *also* drop under shuffle | weak claim — fall back to the AUROC₂-only diagnostic, which is already validated by Phase F-sanity |

### Hard cap: 30 min wall-clock.

---

## §8  Phase V — SAPLMA-style probe replication (training + eval)

**Hypothesis (V-H1):** the SAPLMA recipe (3-layer MLP probe on
LM hidden states for per-token hallucination detection) is
representative of a class of published methods that report per-token
AUROC.  When subjected to the four-AUROC diagnostic from Phase U,
SAPLMA-style probes should show the cascade-contamination signature.

**Why this phase exists:** Phase U's headline target is methodology,
but the writeup is much stronger if at least one *published-recipe*
method shows the diagnostic firing.  Re-implementing SAPLMA is
~80 lines of code (a 3-layer MLP on hidden states with mean-pool
inference), and the WikiText-2 span-corruption setup is already
cached.

### Pre-conditions
- Phase U complete (we want its diagnostic numbers in hand for
  comparison to the SAPLMA cell).

### Code changes
1. **`src/aitchinson_flow/baselines/saplma.py`** (new, ~80 lines).
   Standard recipe per Azaria & Mitchell 2023 §3:
   - 3-layer MLP, hidden dims 256→128→64, ReLU, sigmoid output.
   - Input: GPT-2 last-hidden-state (768-dim) at each position.
   - Trained on per-token labels (corrupted vs clean) with BCE.
   - Inference: per-token sigmoid + mean-pool over the answer span
     (this is the published per-claim aggregation).
2. **`scripts/train_saplma.py`** (~50 lines).  Standard training
   loop.  Reuses `WikiAuditorDataset`.
3. Add SAPLMA as a method spec in `cascade_audit.py`.

### Sweep cell
| cell | training data | epochs | notes |
|---|---|---:|---|
| saplma_wiki | wiki_cache_gpt2 (240 train chunks) | 25 | matches MLP-probe budget in published works |

One cell × ~15 min training + ~3 min eval = 18 min wall-clock.

### Decision criteria

| outcome | criterion | interpretation |
|---|---|---|
| **DIAGNOSTIC FIRES** | SAPLMA AUROC₁ > 0.95 *and* AUROC₂ > 0.85 *and* AUROC₃ < 0.70 | the cascade-audit diagnostic identifies cascade contamination in a recipe directly drawn from a high-citation-count published paper.  **Headline methodological finding.** |
| **PARTIAL**          | AUROC₁ > 0.95 but AUROC₂ < 0.80 | SAPLMA's training somehow reduces cascade leak; weaker claim |
| **NEGATIVE**         | AUROC₁ < 0.85 | SAPLMA doesn't reproduce strongly enough on this data — try TruthfulQA or HaluEval-QA in a follow-up; defer the claim |

### Risk
- **HaluEval-QA cache (Phase K, 10 k pairs) is also a viable
  testbed** for SAPLMA + cascade audit.  But Phase K already
  established that SE is at chance there (real hallucinations
  don't spike per-token NLL), so the cascade comparison would lose
  its locality-clean baseline.  The wiki span-corruption setup is
  the right one.
- **80 lines of replication might miss SAPLMA's exact training
  details.**  If AUROC₁ < 0.85, check Azaria & Mitchell 2023
  Appendix B for hidden-dim, lr, and dropout choices, retry once.

### Hard cap: 1 hr wall-clock.

---

## §9  Decision tree — which phases to run in which order

```
Pre-flight checks (§2)
    │
    ▼
Phase R (SDE on continuous methods, ~30 min, eval-only)
    │
    ├── Strong PASS  → run Phase S as confirmation, skip Phase T's lr_5 cell
    ├── Mixed       → run Phase S
    ├── Strong FAIL → run Phase S anyway (cheap second diagnostic)
    │
    ▼
Phase S (K=2 collapse, ~80 min)
    │
    ▼  [Protocol A complete; writeup contribution locked]
    │
Phase T (consgrad regression, ~35 min)
    │
    ▼  [Protocol B complete]
    │
Phase U (cascade audit, ~30 min, eval-only)
    │
    ▼
Phase V (SAPLMA replication, ~18 min)
    │
    ▼  [Protocol C complete]
```

Total wall-clock for the full sequence: **≈ 3 hr**, mostly
eval-only or short training runs.  Suitable for a single session.

If time is tight, the priority order — drop from the bottom up — is:

1. **Phase R**  (cheapest, biggest writeup leverage) — *do not skip*.
2. **Phase U**  (turns existing data into a writeup contribution) —
   *do not skip*.
3. **Phase S**  (confirms or sharpens R's finding).
4. **Phase T**  (sharpens W1 finding; can defer if Phase R succeeds
   and absorbs the W1 framing into the broader stochasticity claim).
5. **Phase V**  (publication-grade reinforcement of Phase U).

---

## §10  Compute budget summary

| phase | training | eval | total wall-clock | hard cap |
|---|---:|---:|---:|---:|
| R |     0 |  30 min |  30 min |  45 min |
| S | 70 min |  10 min |  80 min | 120 min |
| T | 30 min |   5 min |  35 min |  50 min |
| U |     0 |  15 min |  15 min |  30 min |
| V | 15 min |   3 min |  18 min |  60 min |
| **total** | **115 min** | **63 min** | **≈ 3 hr** | **≈ 5 hr (worst case)** |

GPU memory: every cell fits in 8 GB at B=32 (the constraint that
already applied to Phase 14/15).  Phase T at B=64 with
`create_graph=True` may OOM; fall back to B=32 if so (already in
risk register).

---

## §11  Writeup deliverable mapping

Each phase produces a specific writeup artefact.  Update
`REPORT.md` after each phase completes.

### Protocol A → Section 4 ("Localising the bound")
- **Phase R figure:** KL_bi vs α (noise scale) for each of EqM,
  FMonCLR, LogitKLFlow.  Three lines, x-axis log α, y-axis KL_bi
  with a horizontal dashed line at DFM's 0.148.  Source:
  `runs/phaseR_sde.png`.
- **Phase S table:** the 2×4 K=2 KL_bi grid (4 models × NAG/SDE).
  Source: `runs/phaseS_K2.json`.
- **Combined text:** the 2×2 outcome quadrant from §5.

### Protocol B → Section 3 ("What FM regression learns vs uses")
- **Phase T table:** 6-cell comparison (2 models × 3 samplers,
  Euler-on-f / Euler-on-∇⟨x,f⟩ / NAG).  The headline number is
  the diagonal: does Euler-on-output of the consgrad-trained
  model match NAG-on-eqm?
- **Combined text:** mechanistic claim + falsification logic.

### Protocol C → Section 5 ("Per-token AUROC: a methodological caveat")
- **Phase U figure:** the four-AUROC bar chart per method.
  Source: `runs/phaseU_cascade_audit.png`.
- **Phase V row:** the SAPLMA-style probe added to the same chart.
- **Combined text:** the diagnostic protocol (4 numbers per method,
  shuffle ablation), plus a paragraph identifying the class of
  published methods to which it applies.

### Capstone landing → Section 6 ("Constructive proposal")
- Already done from Phase F+ (SVGP-on-h_LLM with 3400× ECE
  improvement and a free epistemic-uncertainty channel).  No new
  work in this protocol; the section just references the existing
  `runs/phaseF_uq.png`.

---

## §12  Risk register

| risk | mitigation | trigger to abort phase |
|---|---|---|
| Phase R: EqM SDE diverges at high α | clip x to a CLR bounding box; log per-step ‖x‖ | ‖x‖ > 100 at any step |
| Phase R: f ≈ 0 in some γ region trivialises Langevin | log ‖v‖ at γ ∈ {0.1, 0.5, 0.9} for the first cell; document if any region is ≈ 0 | Phase R proceeds either way; the result is still informative |
| Phase S: K=2 entropy is near 1 bit, KL_bi is small in absolute terms | report Δ KL_bi and ratio to the K=27 result side-by-side | n/a |
| Phase T: 2nd-order autograd OOM | drop B 64 → 32 (already pre-empted) | OOM at B=32 → defer phase to a higher-mem environment |
| Phase T: doubly-differentiated loss unstable | lr=2.5e-4 cell pre-empts; if both diverge, bump grad-clip to 0.5 | NaN loss in epoch 1 of both cells |
| Phase U: shuffle ablation destroys SE too | report and weaken the claim to "AUROC₂-only diagnostic" (still valid) | n/a |
| Phase V: SAPLMA replication AUROC₁ < 0.85 | check Azaria & Mitchell §B for hyperparams, retry once | AUROC₁ < 0.85 after retry → defer to follow-up session |
| Any phase: cell overruns hard cap by 1.3× | kill, log partial result as NEGATIVE | n/a (this is the trigger) |

---

## §13  Termination criteria for the protocol as a whole

The protocol is **complete** when any of:

1. All five phases have run to completion (PASS, PARTIAL, FAIL,
   DIAGNOSTIC FIRES, NEGATIVE — all valid terminal states).
2. Phase R is a Strong PASS *and* Phase U+V have run.  (This is
   the minimum publishable arc; T and S are confirmations.)
3. Total wall-clock budget exhausted at the §10 hard caps
   (≈ 5 hr).  Whatever has run, run.

The protocol is **abandoned** if:

- Pre-flight checks fail and cannot be resolved in the same
  session (missing checkpoints, environment broken).  Document and
  reschedule.
- Phase R AND Phase U both fail to produce a writeup-worthy
  result.  This is unlikely (Phase U lands the methodology
  contribution regardless of numbers), but if it happens, the
  fallback writeup is the existing Phase A–Q material with no new
  contributions.

---

## §14  Decision-log discipline

For each phase, append one entry per cell to
`runs/DECISION_LOG.md`:

```
## [<UTC timestamp>] <run-name>
- Hypothesis: <one line — restating the phase's H1>
- Result: <key numbers — KL_bi, AUROC, etc.>
- Decision: <PASS / PARTIAL / FAIL / DIAGNOSTIC FIRES / NEGATIVE>
- Next: <next cell name, or "Phase X complete">
```

Phase-summary entries (after the last cell in a phase) include the
`Combined verdict` block from the §4–§8 decision tables.

---

## §15  Files added/modified by this protocol (planning aid)

```
src/aitchinson_flow/
├── sampling/
│   └── sde.py                          # NEW — Phase R
├── models/
│   ├── eqm.py                          # MODIFIED — sample() method dispatch (Phase R)
│   ├── fmclr.py                        # MODIFIED — sample() method dispatch (Phase R)
│   ├── logitkl_flow.py                 # MODIFIED — sample() method dispatch (Phase R)
│   └── eqm_consgrad.py                 # NEW — Phase T
├── data/
│   └── text8_binary.py                 # NEW — Phase S
├── baselines/
│   └── saplma.py                       # NEW — Phase V
├── config.py                           # MODIFIED — EqMConsGradConfig (Phase T), text8 alphabet (Phase S)
└── training/
    └── data_sources.py                 # MODIFIED — binary-alphabet route (Phase S)

scripts/
├── eval_full.py                        # MODIFIED — sample-method/alpha/use-grad flags (Phase R)
├── eval_w1_compare.py                  # NEW — Phase T
├── cascade_audit.py                    # NEW — Phase U
├── plot_phaseU.py                      # NEW — Phase U
└── train_saplma.py                     # NEW — Phase V

sweeps/
├── phaseR_sde.yaml                     # NEW
├── phaseS_K2.yaml                      # NEW
├── phaseT_consgrad.yaml                # NEW
└── phaseV_saplma.yaml                  # NEW

data/
└── text8_binary_cache.pt               # NEW — Phase S

runs/
├── phaseR_sde.{json,png}               # NEW
├── phaseS_K2.{json,png}                # NEW
├── phaseT_consgrad.{json,png}          # NEW
├── phaseU_cascade_audit.{json,md,png}  # NEW
├── phaseV_saplma.{json,md}             # NEW
├── DECISION_LOG.md                     # APPENDED
└── REPORT.md                           # APPENDED with §3, §4, §5 updates
```

---

## §16  Quick-reference cheatsheet

```bash
# Phase R (eval-only, ~30 min)
python scripts/run_sweep.py sweeps/phaseR_sde.yaml --eval-only

# Phase S (training + eval, ~80 min)
python scripts/cache_text8_binary.py        # one-time, ~3 min
python scripts/run_sweep.py sweeps/phaseS_K2.yaml

# Phase T (training + eval, ~35 min)
python scripts/run_sweep.py sweeps/phaseT_consgrad.yaml
python scripts/eval_w1_compare.py \
    --eqm-ckpt runs/eqm_data50k_ep5_v2/epoch_final.pt \
    --consgrad-ckpt runs/eqm_consgrad_data50k_ep5/epoch_final.pt

# Phase U (eval-only, ~15 min)
python scripts/cascade_audit.py --output runs/phaseU_cascade_audit.json
python scripts/plot_phaseU.py runs/phaseU_cascade_audit.json

# Phase V (training + eval, ~18 min)
python scripts/train_saplma.py --output runs/saplma_wiki/
python scripts/cascade_audit.py --include saplma --output runs/phaseU_with_saplma.json
python scripts/plot_phaseU.py runs/phaseU_with_saplma.json

# Final summary
python scripts/build_report_section.py --phases R,S,T,U,V
```

---

End of protocol.
