# EqM Training Improvement Plan — text8 (capstone)


> **HISTORICAL — superseded 2026-05-07, verdict inverted by the final paper.** Written 2026-05-06 as an improvement plan for EqM generation at the development scale (L=40, d_model=1024, 8 layers, B=64). Its target of bigram KL ≤ 0.50 was never reached: the full scaling sweep tops out at KL_bi 1.38, roughly an order of magnitude above the Discrete Flow Matching control at the same scale (0.148) — see the paper's appendix §Budget, capacity, and conditioning.
>
> **Its Phase 8 framing is the opposite of the paper's finding.** The paper does not claim continuous flow on the simplex matches the discrete baseline; it reports that no EqM variant generates text, and selects a *transport* model (Dirichlet Flow Matching) as the generator. For the settled verdicts read `chapters/results.tex` §The Negative Result.
>
> Phase-A outcomes for this line are recorded in `TRAINING_PROTOCOL.md` §6 Phase A and `runs/DECISION_LOG.md`. Phase 5's n-gram-aware loss WAS run and the paper does not report it: `runs/ng_bg05_data50k/` and `runs/ng_bg10_data50k/` hold the numbers, and this document plus `RESULTS.md`'s Phase-5 derivation are their only description. Retained because three code/config comments cite this file.
> Dead pointers: `scripts/diagnose_overshoot.py` is now `scripts/legacy/`; `runs/best_so_far.pt` does not exist; the surviving baseline checkpoint is `checkpoints/baseline_5ep/epoch_final.pt`.

> **Audience**: a fresh Claude Code session running on a single 20 GB A100
> MIG slice. You will *supervise* this plan: pick the next experiment, kick
> it off, summarise its result, and decide whether to branch or continue.
> You are expected to keep running across many work blocks; persist results
> to disk and the live decision log so the work survives context resets.
>
> **Goal**: improve the unconditional generation quality of the EqM
> (equilibrium flow matching) model on text8 from "per-position char
> distributions match GT, but text is character soup" to "bigram + trigram
> structure looks like English". Without changing the conservative-grad
> training framework or the simplex / CLR geometry.
>
> **Ground state to start from**: `checkpoints/baseline_5ep/epoch_final.pt`,
> the 5-epoch model whose sampler was diagnosed and patched in
> `SAMPLER_FINDINGS.md`. The sampler patch (per-position grad clip + best
> iterate) is already in `src/aitchinson_flow/models/eqm.py`.

## 0. What you must read first

1. `CLAUDE.md` — project conventions and load-bearing decisions.
2. `SESSION_SUMMARY.md` — diagnosis history, known dead code, open questions.
3. `SAMPLER_FINDINGS.md` — the sampler fix that produced the current
   baseline numbers.

Three load-bearing facts from those docs you must not forget:

- The training step is `_eqm_loss` in `src/aitchinson_flow/models/eqm.py`.
  It regresses `∇_x⟨x, f(x)⟩` (a *conservative gradient*, computed via
  `torch.autograd.grad(..., create_graph=True)`) to the FM target. **Do not**
  switch to flash attention or to a plain `f(x)` regression unless you have
  decided to abandon the conservative-grad framework — both break the
  training step.
- The aux CE branch at `s.lambda_ce > 0` and `γ ≥ s.ce_min_gamma` is the
  reason the model doesn't collapse to "all space". Don't disable it
  without an explicit ablation justifying the change.
- `cfg.eqm.source_sigma` is the *single* σ used at both train and sample.
  A mismatch silently OOD-s inference.

## 1. Baseline — what we're trying to beat

The 5-epoch baseline (`checkpoints/baseline_5ep/epoch_final.pt`) with the
patched sampler reaches:

| Metric | Value |
|---|---:|
| Unigram KL (gen vs train) | **0.053** |
| Bigram KL (gen vs train) | **1.65** |
| H_gen / H_gt | **0.952** |
| ‖∇E‖ at gen / ‖∇E‖ at GT | 0.192 / 0.178 |
| Sample example | `'h hve iuneosfett ooriruwttiseen  aigmie '` |

The unigram and energy-minimum metrics are essentially solved — generated
samples sit on the data manifold's energy minima as tightly as GT does.
**The remaining failure is bigram-and-up coherence**: per-position
distributions match, but the joint distribution across positions is
nonsense. Bigram KL is still ~1.65 nats — at 25 % of unigram KL we'd
already be qualitatively reading words.

**Headline success criterion** for this plan: bigram KL ≤ **0.50 nats**
with the unigram KL stable under 0.10. A stretch goal of trigram KL ≤ 1.0
nats (a metric we'll add) tells us we crossed into "looks like English".

## 2. Working hypotheses

In rough decreasing prior probability:

1. **Undertrained.** 5 epochs at 10k windows is ~50k token-window updates;
   text8 has 100M characters. The model is in the "memorising marginals,
   not joints" regime. Longer training and / or more data should move
   bigram KL faster than any architectural change.
2. **No time conditioning.** The transformer takes only `x_γ` as input.
   It must *infer* γ from the structure of `x_γ`, which gets harder near
   `γ ≈ 1` where `x_γ ≈ x_1`. DFM (which has explicit `t` conditioning) is
   the canonical comparator and should beat us at small compute.
3. **No multi-position term.** The aux CE is per-position; nothing in the
   loss penalises invalid digraphs. An n-gram likelihood term over the
   implied-x1 reconstruction is the cheapest addition.
4. **Backbone too narrow OR too deep for the data.** d=1024 / 8 layers /
   100M params is over-parameterised for 10k windows. Either it overfits
   per-position attractors (no joint structure to learn) or the data
   ceiling is the actual bottleneck.
5. **Energy field still has weak repulsive directions inside basins**
   (`SAMPLER_FINDINGS.md` §7). Plain GD overshoots even after the basin is
   reached. This may resolve itself with longer training; if not, a
   smoothness/curvature regulariser on `f` is worth trying.

## 3. Compute budget & GPU sizing

Target: single 20 GB A100 MIG. The current 5-epoch run trained in ~8 min
on a consumer GPU at d=1024, 8L, B=64. On A100 expect ~3-4× faster, so
budget assumptions:

- 5 epochs at default (d=1024, 8L, B=64, 10k windows) ≈ **2-3 min**.
- 25 epochs at default ≈ **15 min**.
- 50 epochs at default ≈ **30 min**.
- d=1536, 12L (~340 M params) ≈ **2-3× slower** but fits in 20 GB.
- The MATH SDPA backend means attention is O(L²) memory but L=40 is fine.

Total compute budget for the plan: **~12 hours** of GPU time, comfortably
fits in a session-day. Each phase below has an estimate; abort early if
you start trending over.

## 4. Tooling you must build first (Phase −1)

Before running any training, set up these helpers in this order. Do not
skip — every later phase depends on logging.

### 4.1 `scripts/eval_full.py` — the canonical scorecard

A single eval script that loads a checkpoint and writes a JSON line with:

```json
{
  "ckpt": "...", "epoch": ..., "global_step": ...,
  "n_samples": 256, "sample_steps": 200,
  "unigram_kl": ..., "bigram_kl": ..., "trigram_kl": ...,
  "H_gen": ..., "H_gt": ..., "H_ratio": ...,
  "grad_at_gen": ..., "grad_at_gt": ...,
  "samples": ["...", "...", ...]    # 8 example argmax decodes
}
```

`trigram_kl` is new. Implement counts via the same per-row loop in
`scripts/quick_eval.py:bigram_kl`, with one extra index. Accept a
`--ckpt`, `--n`, `--steps`, `--out` flag. Default `--n=256` (was 64) for
tighter KL estimates — the diagnostic showed B=32 is noisy on bigram KL.

### 4.2 `scripts/run_sweep.py` — orchestration

Reads a YAML or JSON list of `{name, overrides}` entries, for each one:

1. Builds `cfg` from defaults, applies overrides (use `dataclasses.replace`).
2. Runs `fit()` with `history_out=[]`.
3. Evaluates the final checkpoint with `eval_full.py`.
4. Writes results to `runs/<run-name>/{config.json,history.jsonl,eval.json,
   epoch_final.pt}` plus appends one row to `runs/sweep_results.jsonl`.

Idempotent: if `runs/<run-name>/eval.json` exists, skip. This is what lets
you resume after a context reset.

### 4.3 `runs/DECISION_LOG.md` — your supervisor diary

Append-only. After every experiment finishes write a 5-line entry:

```
## [<UTC timestamp>] <run-name>
- Hypothesis: <one line — what would this run prove or kill?>
- Result: KL_uni=<...> KL_bi=<...> KL_tri=<...> H_ratio=<...>
- Decision: <continue|branch to phase X|abandon and explain>
- Next: <run-name of next experiment>
```

The next session reads this log to know where to pick up. **Maintain it
even when running automated sweeps** — one entry per sweep cell.

### 4.4 In-training trigram-KL probe

Extend `_unigram_kl_probe` in `src/aitchinson_flow/training/runner.py` to
also report bigram and trigram KL. Currently it only reports unigram —
which we've solved. Bigram KL is the metric we're optimising; you cannot
afford to miss its trajectory by training-budget end.

Cap the probe at `n=64, steps=200` so it stays cheap (already the default).

## 5. Phase plan

Each phase is independently evaluable. Run phases in order, but feel free
to skip a phase if a later one's hypothesis is more compelling given
prior results. Update `DECISION_LOG.md` whenever you skip.

Where a phase has a sweep, list the configs as a YAML literal you can
pass straight to `scripts/run_sweep.py` — no need to invent the format on
the fly.

---

### Phase 0 — Reproduce the patched-sampler baseline (sanity)

**Goal**: make sure you can hit the headline numbers from §1 before you
trust your own sweep results.

1. Run `python scripts/eval_full.py --ckpt checkpoints/baseline_5ep/epoch_final.pt --n 256 --steps 200`.
2. Compare to the table in §1. Trigram KL is new; record whatever number
   you get as the trigram baseline.
3. **Stop and ask the user** if KL_uni > 0.080 or KL_bi > 1.90 — that
   means either the sampler patch regressed or the GPU is non-deterministic.

Compute: ~30 s.

---

### Phase 1 — Epoch scaling (validates the "undertrained" hypothesis)

**Hypothesis**: 5 → 25 → 50 epochs at the *same* config moves bigram KL
substantially. If yes, longer training is the cheapest path; if no, the
problem is architectural and we should jump to phases 3–5.

**Sweep**:

```yaml
- name: ep10_default
  overrides: { training.epochs: 10 }
- name: ep25_default
  overrides: { training.epochs: 25 }
- name: ep50_default
  overrides: { training.epochs: 50 }
- name: ep25_lr2x
  overrides: { training.epochs: 25, training.lr: 6.0e-4 }
- name: ep25_lr_5x
  overrides: { training.epochs: 25, training.lr: 1.5e-4 }
```

Compute: ~75 min total (3 + 15 + 30 + 15 + 15).

**Decision**:
- KL_bi at ep25 < 1.0 → continue scaling (try ep100). The cheapest path.
- KL_bi at ep25 in [1.0, 1.5] → diminishing returns; bring forward Phase 4
  (time conditioning) and Phase 5 (n-gram loss) before more scaling.
- KL_bi at ep25 ≥ 1.5 → epoch scaling is *not* the bottleneck. Skip
  Phase 2's "smaller models" cell, jump to Phases 3–5.

Save the best epoch checkpoint as `runs/best_so_far.pt` symlink.

---

### Phase 2 — Backbone scaling (validates "the model is over/under-sized")

**Hypothesis**: at 5–25 epochs and 10 k windows, d=1024 / 8L is the wrong
shape. Either smaller is enough (overfit per-position) or wider is
needed for joint structure.

**Sweep**, all at the best `epochs` from Phase 1 (default to 25 if Phase 1
disagrees with itself):

```yaml
- name: bb_d256_l4    # tiny — does it suffice?
  overrides: { transformer.d_model: 256, transformer.nhead: 4, transformer.num_layers: 4 }
- name: bb_d512_l6    # mid
  overrides: { transformer.d_model: 512, transformer.nhead: 8, transformer.num_layers: 6 }
- name: bb_d768_l8    # ⅔ of default
  overrides: { transformer.d_model: 768, transformer.nhead: 8, transformer.num_layers: 8 }
- name: bb_d1024_l8   # default — re-run for variance estimate
  overrides: {}
- name: bb_d1024_l12  # deeper
  overrides: { transformer.num_layers: 12 }
- name: bb_d1536_l8   # wider
  overrides: { transformer.d_model: 1536, transformer.nhead: 16 }
- name: bb_d1536_l12  # both — fits in 20 GB; ~340 M params
  overrides: { transformer.d_model: 1536, transformer.nhead: 16, transformer.num_layers: 12 }
```

Compute: ~3-4 hr total at 25 epochs each.

**Decision**:
- Pareto-plot KL_bi vs param count. The "knee" config wins.
- If d=256/l=4 is competitive on KL_bi: the data is the bottleneck, not
  the model. Skip remaining backbone scaling, jump to Phase 3 (data).
- If d=1536/l=12 is the only clear winner: capacity is the bottleneck,
  but probably together with data. Run Phase 3 next, then return to
  scaling with the bigger data ceiling.

Be careful with `nhead`: must divide `d_model`. Don't auto-set a config
that would crash at construction.

---

### Phase 3 — Data ceiling (validates "10k windows is too few")

**Hypothesis**: `cfg.text8_dataset.max_train_windows = 10_000` was a
demo-time choice. text8 has ~100M chars / 40-char window = 2.5 M
windows. Removing the cap should give the model more joints to fit.

**Sweep** (best backbone from Phase 2, best epoch count from Phase 1):

```yaml
- name: data_50k
  overrides: { text8_dataset.max_train_windows: 50000 }
- name: data_200k
  overrides: { text8_dataset.max_train_windows: 200000 }
- name: data_full
  overrides: { text8_dataset.max_train_windows: null }    # uncapped
- name: data_full_L64    # longer windows, more bigram coverage per sample
  overrides:
    text8_dataset.max_train_windows: null
    text8_dataset.L: 64
    training.L: 64
```

Note: changing `L` requires re-checking the position embedding
construction — the model is built with `nn.Embedding(L, d)`, so the
position table will be size 64 instead of 40. This is fine; just don't
load a 40-position checkpoint into a 64-position model.

Compute: ~3-4 hr total.

**Decision**: take the best of `{best Phase-2 backbone × best data
ceiling}` forward as the "platform" config for Phases 4-7.

---

### Phase 4 — Time conditioning (architectural hypothesis)

**Hypothesis**: the model has to infer γ from `x_γ`'s structure; giving
it γ explicitly should help, especially near γ ≈ 1 where the input is
almost on-manifold and the velocity field is the most informative.

**Implementation**:

Modify `TransformerBackbone.forward` (in
`src/aitchinson_flow/transformer_backbone.py`) to accept an optional
`gamma` arg, embed it via `_sinusoidal_embedding(gamma, d_model)` (already
defined in that file for DFM), and *add* it to the per-token hidden
state (or concat then project). Only active when `cfg.eqm.time_conditioning
= True`.

Pass γ from `_eqm_loss` and `_compute_grad`/`sample`. **Crucially**: at
sample time you don't know γ — the iterate is unconditioned. Two options:
(a) condition on a fixed γ=1 at sample time (the data-manifold endpoint);
(b) treat γ as a parameter the sampler learns (an extra dim in the
energy gradient). Default to (a); ablate (b) only if (a) underperforms.

**Sweep**:

```yaml
- name: tc_off    # control: best platform config
  overrides: {}
- name: tc_add    # γ added to hidden state
  overrides: { eqm.time_conditioning: "add" }
- name: tc_concat # γ concatenated then projected
  overrides: { eqm.time_conditioning: "concat" }
```

Compute: ~1 hr.

**Risk**: time conditioning at sample time is the awkward part. If both
variants underperform the control, `eqm.time_conditioning="off"` stays
and we accept that the conservative-grad framework can't use γ as easily
as DFM can. Document the negative result rather than retrying.

---

### Phase 5 — n-gram-aware loss (additional loss parameter)

**Hypothesis**: adding a bigram (or bigram + trigram) likelihood term to
`_eqm_loss` directly penalises invalid digraphs. The aux CE branch is
already per-position; bolt the n-gram term onto it.

**Implementation**: in `_eqm_loss`, after computing
`pred_x1 = x_γ - λ·grad_g` and `log_probs = log_softmax(pred_x1)`, also
compute the bigram log-prob:

```python
log_p_bigram = log_probs[:, :-1, :, None] + log_probs[:, 1:, None, :]  # (B, L-1, K, K)
# pull out the bigram observed in token_ids
bg_target = token_ids[:, :-1] * K + token_ids[:, 1:]                   # (B, L-1)
bg_nll = F.nll_loss(log_p_bigram.reshape(-1, K*K), bg_target.reshape(-1))
total_loss += s.lambda_bigram * bg_nll
```

Mask to γ ≥ ce_min_gamma like the unigram CE. Trigram analogous;
prohibitive if `K^3 = 19,683` is too memory-heavy at L=40, B=64 — measure
before adding.

Add to `EqM` config: `lambda_bigram: float = 0.5`,
`lambda_trigram: float = 0.0` (default off).

**Sweep**:

```yaml
- name: ng_bg05
  overrides: { eqm.lambda_bigram: 0.5 }
- name: ng_bg10
  overrides: { eqm.lambda_bigram: 1.0 }
- name: ng_bg05_tg02
  overrides: { eqm.lambda_bigram: 0.5, eqm.lambda_trigram: 0.2 }
- name: ng_bg00_tg05  # trigram only (bigram falls out as marginal)
  overrides: { eqm.lambda_bigram: 0.0, eqm.lambda_trigram: 0.5 }
```

Compute: ~1.5 hr.

**Decision**: take the best `lambda_bigram, lambda_trigram` forward.
Watch for unigram drift — if KL_uni rises above 0.10, the n-gram term is
distorting the per-position distribution; back off the weight.

---

### Phase 6 — Loss / γ-sampling ablations (low-priority but cheap)

**Hypothesis**: `gamma_power=0.5`, `lambda_ce=0.5`, `ce_min_gamma=0.5`,
`decay_strategy="linear"` were all set heuristically. Some may be
over-tuned to 5-epoch / 10k-windows; revisit at the new platform.

**Sweep**:

```yaml
- name: abl_lambda_ce_2x
  overrides: { eqm.lambda_ce: 1.0 }
- name: abl_lambda_ce_05x
  overrides: { eqm.lambda_ce: 0.25 }
- name: abl_gp_uniform
  overrides: { eqm.gamma_power: 1.0 }
- name: abl_gp_low
  overrides: { eqm.gamma_power: 0.25 }
- name: abl_decay_truncated
  overrides: { eqm.decay_strategy: "truncated", eqm.decay_a: 0.2 }
- name: abl_loss_hilbert_soft
  overrides: { loss.mode: "hilbert_soft" }   # SESSION_SUMMARY.md §5.4
```

Compute: ~1.5 hr.

**Decision**: keep the default unless an ablation moves KL_bi by ≥10 %.
Most of these will be no-ops; the useful ones graduate into the platform.

---

### Phase 7 — Training-time stability (optional, low prior)

**Hypothesis**: `SAMPLER_FINDINGS.md` §7 noted plain GD overshoots even
inside basins on the 5-epoch model — implies weak repulsive directions in
`⟨x, f(x)⟩`. A curvature penalty on `f` could fix this. Two cheap options:

- **Spectral norm on velocity head** (`VelocityHead.proj`): wrap with
  `torch.nn.utils.spectral_norm`. Caps the head's Lipschitz constant.
- **Energy-gradient penalty** during training:
  `lambda_smooth · ‖grad_g‖_F^2` averaged over the batch. Penalises peaky
  energies in regions the FM target wouldn't visit anyway.

Only run this phase if Phase 1's ‖∇E‖-at-gen *doesn't* converge to ‖∇E‖-at-GT
within ~5 % at the longer training run. If it does, the field is fine and
this phase is unnecessary.

---

### Phase 8 — DFM head-to-head (capstone narrative)

Run the DFM model (already in the repo, `cfg.training.model_name = "DFM"`)
at the *best EqM platform's* compute budget (same backbone, same epochs,
same data). Compare:

| Metric | EqM | DFM |
|---|---|---|
| KL_uni | | |
| KL_bi | | |
| KL_tri | | |
| H_ratio | | |

This is the headline of the capstone — "continuous flow on the simplex
matches / beats the discrete baseline at parity compute". If EqM loses by
>20 % on bigram KL, that's the honest result and worth a separate
discussion before further iteration.

Compute: ~30-60 min.

---

### Phase 9 — Final long run

Take the winner and train for as long as the remaining compute budget
allows (typically 100-200 epochs at the chosen platform). Save
intermediate checkpoints every 25 epochs. Generate the final
`SAMPLER_FINDINGS.md`-style table for the writeup.

## 6. Things to NOT do

1. **Don't switch to flash attention** in `TransformerBackbone`. The
   conservative-gradient training step needs second-order autograd through
   attention; flash silently breaks it. (`SESSION_SUMMARY.md` mentions
   this; verified by `transformer_backbone.py:51` using `SDPBackend.MATH`.)
2. **Don't change the σ-matching invariant** between training (`x0 ~
   σ·N(0,I)`) and sampling (init at `σ·randn`). Both read from
   `cfg.eqm.source_sigma`. If you raise σ, both move.
3. **Don't disable the aux CE branch** (`s.lambda_ce > 0`). Without it the
   model collapses to the unigram-mode minimum (the "all space" failure
   mode in `SESSION_SUMMARY.md`).
4. **Don't rely on `model.bpd()`**. It's tautological — start from GT,
   recover GT. Use the n-gram KLs in `eval_full.py` instead.
5. **Don't replace `EqM.sample`'s grad-clip / best-iterate logic**. It's
   the patch from `SAMPLER_FINDINGS.md`; verified to drop sampler-side KL
   by ~30 %. If you want a cleaner sampler, ablate explicitly with one
   sweep cell, not by removing the patch wholesale.
6. **Don't commit checkpoints**. They're hundreds of MB. The repo's
   `.gitignore` should already exclude `runs/` and `checkpoints/` — check
   before committing.

## 7. Operating cadence (you, the supervisor)

After each kicked-off run finishes:

1. Read `runs/<name>/eval.json`.
2. Append a 5-line entry to `runs/DECISION_LOG.md` (template in §4.3).
3. Decide: continue the current phase, branch to a different phase, or
   abandon the line. Justify in the log.
4. Update the live "best" symlink: `ln -sf <name>/epoch_final.pt
   runs/best_so_far.pt`.
5. Kick off the next run.

If you reach end-of-session with phases unfinished, leave the next
session a one-paragraph hand-off note at the top of `DECISION_LOG.md`
under a `## NEXT SESSION` heading: what to run first, what to expect.

## 8. Data the writeup will need

Hold these aside as you sweep so the capstone writeup isn't a rerun:

- The **Phase 1 bar chart**: KL_bi vs epochs, EqM, single line.
- The **Phase 2 Pareto**: KL_bi vs param count.
- The **Phase 4–5 ablation table**: a clean before/after for each
  intervention.
- The **Phase 8 EqM-vs-DFM table** at parity compute.
- A **sample diversity grid** at three checkpoints (5 ep, mid, final),
  8 samples each, argmax + top-3.
- The trajectory plots from `scripts/diagnose_overshoot.py` *re-run on
  the final model* — does the overshoot we patched in Phase 0 still
  exist after longer training? Probably less so; the answer to that
  question goes into the discussion.

## 9. Quick checklist for kicking off a run

```
cfg overrides written in YAML        ✅
runs/<name>/ does not already exist  ✅
GPU memory headroom > 2 GB at start  ✅
DECISION_LOG entry written before kickoff (hypothesis, decision criterion) ✅
training history saved as JSONL (history_out=[])                         ✅
eval_full.py invoked at end                                              ✅
DECISION_LOG entry updated with result                                   ✅
```

If any of those isn't true, fix it before the next run.
