# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Capstone research code for **continuous flow-matching on the simplex** applied to character-level text8 (K=27 vocab; default L=40 windows, **L=256** for the publication-length benchmark). Several model families live side-by-side:

- **EqM** — Equilibrium Flow Matching (Wang & Du 2025, arXiv:2510.02300). Trains a velocity field `f(x)` on CLR (centered log-ratio) features; uses the **conservative gradient** `∇_x ⟨x, f(x)⟩` of an implicit energy as the FM target. Sampling is NAG-GD (Nesterov-accelerated gradient descent) on that energy. Variants: `EqM_OneHot`, `EqMLatent` (learned-embedding space).
- **DFM / DirichletFM** — discrete / Dirichlet Flow Matching baselines (`models/dfm.py`, `models/dirichlet_fm.py`; Stark et al. 2024, arXiv:2402.05841). CE on the denoiser, Euler sampling. `DFM` exposes the **only peer-comparable BPC** via `DFM.elbo_bpc` — a genuine D3PM uniform-process **variational NLL upper bound** (honest peer: **D3PM-uniform ≈1.61**; SEDD 1.32 / D3PM-absorb 1.45 / MDLM are absorbing/score methods, reference only). `DFM.bpd()` now returns this bound; the old uniform-t training CE is kept as `DFM.denoiser_ce_bpc` (a diagnostic, **not** a likelihood). A `<0.5`-BPC sanity guard refuses identity/near-clean artifacts.
- **SFLM / SFLMEBM** — hyperspherical flow (arXiv:2605.11125); `SFLMEBM` reads it as an EqM-style EBM (log-sum-exp energy, Riemannian-GD sampler). `SFLMEBM_FM` is the same class with the Riemannian-FM target (`cfg.sflm_ebm.lambda_fm>0`).
- **OOD heads (post-hoc, on frozen `DirichletFM` features)** — the current recommended detector is **`BayesLinHead`** (`scripts/ood_bayes_linear.py`; math in `docs/bayes_linear_ood.md`, results in **`bench_ood_final/blr*/`**): a linear **energy** head (per-token hinge — valid→0, corrupt≥margin) plus a closed-form **Laplace / Bayesian-linear variance** `Var(z)=zᵀ(Φ+λI)⁻¹z` (training-free Mahalanobis-to-ID density), read **per-token** (localization heatmaps) and **per-sequence** (position-mean). It **superseded** the earlier **`DirichletFMSvgp`** Sparse Variational GP head. Both read **deterministic Dirichlet-mean** features (a stochastic Dirichlet sample collapses clean-vs-corrupt features — the mean is load-bearing). Two passes: energy at `t_eval=4.5`, variance at a second pass `t_var=7.5` — read at 4.5 the variance signal *inverts* (see `docs/bayes_linear_ood.md`'s banner and the paper's methods §BayesLinHead).

The repo investigates three objectives — **(1) unconditional generation, (2) conditional recovery, (3) OOD detection**. **The paper is finished; its verdicts are authoritative** — see `README.md` (the index), `REPRODUCE.md`, and `docs/paper-map.md`. `EVAL_ASSESSMENT.md` is the historical eval reference (its per-objective verdicts predate the full-corpus run and three are now wrong; see its banner). The authoritative arm list is `scripts/bench_sflm_ebm.py:MODELS` plus the factory registry. The **capstone paper** is written in `../capstone-paper/`; the paper's chapters are the source of truth for every claim (its early `PAPER_BLUEPRINT.md` is a non-authoritative planning snapshot).

**Cloud-run hardening (L256).** For the L=256 cloud workstream the repo was hardened: `cfg.transformer.grad_checkpointing` + a memory-fallback ladder in `train_for_sflm_bench.py` (grad-ckpt → halve batch → L=128 → mark-failed-and-continue) handle the 20 GB MIG OOM/NVML asserts; `--seeds 42,43,44`, idempotent skip-if-done, `--full-split` (lazy CLR features), **early stopping (`--early-stop-patience`, restores the best val ckpt as `epoch_final.pt`) + the 36 h cap (`--max-hours`)**, and the E1b length override (`--length`) are implemented; the X1/X2/X3 harness (`scripts/eval_all.py`, `scripts/manifest.py`, `scripts/run_experiment.py`, `RESULTS_README.md`) and **all** experiment scripts now exist (E2a `ablate_training_signal.py`, E2b `ablate_sampler.py`, E2d `probe_field_geometry.py`, E2e `ablate_hyperparams.py`, E4d `eval_ood_baselines.py`, E5a `train_logit_kl_flow.py`, E6a `cache_llm_features.py`, E6b `train_eqm_auditor.py`, E7 `aggregate_results.py`). Only E5b (SFM √p) lacks a dedicated model (runs via the `SFLM` arm). See `CLUSTER_RUNBOOK_L256.md` / `capstone_experiment_runbook.md`.

`SESSION_SUMMARY.md` documents the diagnosis/fixes that brought the EqM model from mode-collapse to ~94% of corpus unigram entropy. Read it before changing the EqM training step or sampler — many of the choices (conservative-grad sampling, matched train/sample σ, aux CE on implied-x1, γ-importance sampling) are fixes for specific failure modes documented there — but note the paper's later finding that the aux CE does **not** act as a per-token anchor (masked to γ≥0.5 it is solved before the first epoch ends; `appendix` §Training-time collapse). SESSION_SUMMARY's contrary verdict carries a banner. The **theory of what fails and why** is in `NOTE_WHY_EBM_INIT_STUCK.md` (training-time collapse) and `NOTE_WHY_UNCONDITIONAL_FAILS.md` (sampling-time failure).

## Commands

Project uses `uv` (see `uv.lock`) and Python 3.13.

- Train EqM (default model): `python main.py` — flags: `--out-dir`, `--epochs`, `--lr`. The CLI is intentionally minimal; everything else lives in `Config` (`src/aitchinson_flow/config.py`).
- Lint: `uv run ruff check` / `uv run ruff format`.
- Evaluate a trained checkpoint: `python scripts/evaluate_eqm.py` (writes a multi-panel figure to `scripts/evaluate_eqm.png`).
- **Canonical evals** (see `EVAL_ASSESSMENT.md` for which metric serves which objective): generation `scripts/eval_generation.py`, recovery `scripts/recovery_check.py`, OOD bench `scripts/bench_sflm_ebm.py`. One-off / superseded / broken runners now live in `scripts/legacy/` (e.g. `compare_gt_vs_gen.py`, `eval_plots.py`, `eval_nfe_comparison.py`, `eval_healing_demo.py`, `diagnose_*.py`).
- `run_files/run_text8.sh` references a `run.py` entry point that does **not** exist (now carries a deprecation header). Treat it as a fossil; use `main.py` / `scripts/train_for_sflm_bench.py` instead.

There is no test suite.

## Evaluation & objectives

**`EVAL_ASSESSMENT.md` is the eval reference** — per objective it states which metric to report, the headline numbers, and *what works / fails and why* (with theory + peer citations). Canonical evals:

- **Obj 1 (generation):** `scripts/eval_generation.py` → n-gram **KL_bi/KL_tri**, **H_ratio**, samples; **BPC** only from `DFM` (MC-ELBO). Identity-path arms (`EqM`/`EqMLatent`/`SFLM`/`EqM_OneHot`) have **no comparable BPC** — their `bpd()` is recovery-from-a-tiny-perturbation, so PPL≈1.0/BPC≈0; the bench prints `—` (`generation_metric_valid=False`). Don't resurrect those as a likelihood.
- **Obj 2 (recovery):** `scripts/recovery_check.py` → **Δ@α = token_acc − token_acc_perturbed** (headline **Δ@.50**). Don't lead with KL here (deterministic refs look better on KL while doing zero recovery work).
- **Obj 3 (OOD):** word-level corruption AUROC, per the paper. **The paper's headline is substitution at the word unit** (denoiser NLL 0.981 vs GPT-2 ≤ 0.75, `tab:ood-word`), with shuffle close behind (0.967); the old "headline the shuffle axis, substitution is trivially ~1.0" guidance predates the word-unit protocol and is retired. **The strongest localizer is the training-free denoiser NLL**; `BayesLinHead` (`scripts/ood_bayes_linear.py`, results `bench_ood_final/blr*/`) is the strongest *fitted* head and the two-pass variance is a genuine density channel (matches the energy at rate 0.15 sequence-level). The earlier hinge-**SVGP** is retired (variance collapses by concentration of measure; `DFM_SVGP_FINDINGS.md`). Native EqM energy contributed nothing (`results` §What survives).

### Spilled energy — the external-LM baseline (corrected 2026-07-13)

The repo called its GPT-2 baseline "spilled energy" but computed **per-token NLL**. They are different quantities and the distinction is load-bearing:

- **NLL (same step):** `logsumexp(logits[i-1]) − logits[i-1][x_i]` = `−log p(x_i | x_<i)`.
- **Spilled energy (CROSS step)** — Minut, Dewidar & Masi, *Spilled Energy in LLMs*, ICLR 2026 (arXiv:2602.18671), Def 4.1 / Eq. 8: `ΔE_i = logsumexp(logits[i]) − logits[i-1][x_i]` — the logit energy is read at step `i-1`, the marginal energy at step `i`; the chain rule says they cancel, and the residual is the signal. They differ by `logsumexp(logits[i]) − logsumexp(logits[i-1])`.

**Sign:** follow the authors' code (`OmnAI-Lab/spilled-energy`, `energy.py`: `delta = -E_margin + E`), **not** the paper's Eq. 8 prose, which has a sign typo. A flip inverts AUROC. Validated by `scripts/validate_spilled_energy.py` (exact match vs the reference implementation).

**Zero-property:** ΔE ≈ 0 only on text the LM *models correctly* — measured **−0.30 on in-domain English** vs **+3.4 on clean text8** (GPT-2 finds lowercase/unpunctuated text8 genuinely OOD). So a large clean-text8 ΔE is **not** a bug; check the property on in-domain English.

**ΔE does not localize.** By construction it straddles two decoding steps, so per-token AUROC on `replace` is ~0.48–0.59 (≈chance) while **sequence** AUROC is ~1.0. It is a sequence-level signal. The bench therefore runs **two** GPT-2 arms: `gpt2_se` (real ΔE) and **`gpt2_nll`** (same-step NLL — the honest per-token comparator, and what the old mislabelled number actually was). Claim "our NLL beats **GPT2_NLL** per-token"; beating ΔE at localization is a straw man.

**Evaluation protocol (char model vs BPE LM).** Attributing a BPE score *down* onto characters is ill-posed (the old code spread it uniformly over the token's chars → smeared localization). Pooling *up* is exact. So each model is scored in its **native unit**, and the comparison happens at a **common** one:

> **flow-matching detectors → CHARACTER · GPT-2 baselines → BPE TOKEN · both → WORD (the head-to-head)**

`DETECTOR_KEYS[*]["unit"]` encodes this; `_bench_common.word_metrics` (char scorers) / `word_metrics_from_segments` (BPE scorers) do the pooling, with **both** `max`- and `mean`-pool reported (max suits a single bad char, mean suits weak signal spread over a word — the false-info regime). Heatmaps render each detector in its own units: char cells for FM, **BPE-token cells for GPT-2** (`heal_style_examples_bpe`). Sequence level is comparable *only* under per-**character** normalisation (bits-per-char; = mean over chars = `sum(BPE)/n_chars`) — a mean over BPE tokens is **not** comparable.

**Why word level is required, not just fairer:** character corruption **shatters GPT-2's tokenization**. At replace@0.15 the text re-tokenizes from ~13.5k → **22.8k BPE tokens (+70%)** and **40% of BPE tokens touch a corrupted char** (not 15%); at 0.5 it is 77%. The BPE unit is *itself a function of the corruption*, so BPE-level localization is degenerate for char noise (SE 0.48, NLL 0.56 ≈ chance). `falseinfo` (word swaps) leaves tokenization intact (+2.4%, prevalence tracks the rate) — BPE numbers are meaningful there. **Words are the only unit stable across both tokenizations.**

**Corrected GPT-2 numbers** (seq | BPE | word-max): replace@0.15 SE `1.000|0.480|0.727`, NLL `1.000|0.557|0.738`; falseinfo@0.15 NLL `0.893|0.768|0.853`. Note GPT-2's 1.000 sequence AUROC on replace/shuffle is *cheap* (it detects "not English" via the tokenization blow-up), and **GPT-2_NLL is genuinely strong on false-info at the sequence level** (the better triage model there, `results` §ood-falseinfo) — though at the word unit the supervised falseinfo head leads on the final model (0.812 vs NLL 0.807, and the refit head 0.912 vs GPT-2 0.837 on transfer).

Renamed (behaviour unchanged, name was wrong): `bench_sflm_ebm._spilled_energy` → `_per_position_nll`; `wiki._spilled_energy_per_pos` → `_per_position_nll`; `sflm_ebm.spilled_energy` → `per_position_nll` (a non-causal denoiser has no adjacent decoding step, so cross-step ΔE is *undefined* there); `train_eqm_auditor._spilled_energy` → `_per_position_nll` (its JSON key `spilled_energy_baseline` → `gpt2_nll_baseline`).

Peer-comparable text8 BPC was a planned deliverable that **the paper dropped**: it reports no BPC for any of our models (`bpd()` is marked diagnostic-only in `tab:config`; the only BPC it cites is Statistical FM's published 1.39). The L=256 `DFM` arm's own stored bound is 4.10 (`runs/sflm_bench_a100_20g_L256/DFM/eval_all.json`), far off the 1.32–1.47 frontier — a budget statement, not a frontier claim. `bench_sflm_ebm.py` is now robust to per-arm OOM at L=256 (each readout wrapped; `SFLMEBM.position_uncertainty` chunks over the batch).

**L256 status (final, 2026-08):** the full-text8 run **finished** — the paper's model is `runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM_converge/epoch_final.pt` (md5 `9946dce9…`, a genuine annealed final; the interrupted legs' history is narrated in `results` §Experimental Setup and REPRODUCE.md §T3). Two unreported scaling points live in `runs/_archive/sflm_bench_a100_20g_L256_d1280L14{,_b16}` (30k/30ep at d1280: KL_bi 0.26 — *worse* than the reported 0.208 at d1024; 50k/50ep: KL_bi 0.14) — the `_archive` banner names the arm-name decoy in the first one. These `DirichletFM` rows report `bpc=—` (identity-path readout) — a peer-comparable BPC still requires the **`DFM` (Discrete-FM) `elbo_bpc`** route at L=256, **not** `DirichletFM.bpd()`. Don't confuse `DirichletFM` (Dirichlet FM, the L256 generator/healer) with `DFM` (Discrete FM, the only peer-BPC arm) — they share the "DFM" abbreviation and a backbone.

## Repository layout

`main.py` (train EqM) · `src/aitchinson_flow/` (package: `config`, `models/`, `data/`, `training/`, `geometry`, `sampling`) · `scripts/` (canonical entry points) · **`scripts/legacy/`** (one-off / superseded / broken runners — `diagnose_*`, `eval_plots.py`, `compare_gt_vs_gen.py`, …) · `sweeps/` (12 active configs) + **`sweeps/archive/`** (phase/POC) · `runs/` (outputs; **checkpoints are untracked**; `runs/DECISION_LOG.md` is the source-of-truth diary) · **`docs/archive/`** (superseded planning docs). `runs/` was physically pruned in the 2026-08-20 cleanup (weights of unpublished dev runs deleted; every result JSON kept and tracked; see `../capstone-paper/review/2026-08-20-experiments-cleanup.md` for the signed plan). The T2-enabling checkpoint set (~10 GB) is enumerated in `REPRODUCE.md`.

## Architecture

### Config-as-dataclass

`Config` (`src/aitchinson_flow/config.py`) is the single source of truth — `TrainingConfigs`, `Text8DataConfig`, `TransformerConfig`, `EqM`, `DFMConfig`, `LoaderSettings`, `LossConfig`, `TransformationConfig` are nested dataclasses. The CLI only overrides three fields; experiments are run by editing the dataclass defaults or by mutating `cfg` in code (`dataclasses.replace`). `cfg.training.model_name` selects the model.

### Model registry

Models register builders by name via `aitchinson_flow.models.factory.register("Name")`. `build_model(cfg)` looks up `cfg.training.model_name` (e.g. `EqM`, `EqM_OneHot`, `EqMLatent`, `DFM`, `DirichletFM`, `DirichletFMSvgp`, `SFLM`, `SFLMEBM`) in `REGISTRY`; the full set benchmarked side-by-side is `scripts/bench_sflm_ebm.py:MODELS`. **Importing `aitchinson_flow.models` is what populates the registry** — `main.py` imports it for its side effect (see comment on the import).

Models implement the `GenerativeTrainingModel` protocol in `models/base.py`: a `training_step(batch, step) -> LossDict` that must include the key `TRAINING_LOSS_KEY = "loss"`, and an `eval_step(batch) -> LossDict`. Extra dict entries become postfix metrics in the tqdm bar and history.

### Training loop

`training/runner.py:fit` is the top-level loop: build optimizer/scheduler, optional resume, per-epoch `train_epoch` / `evaluate`, optional `_unigram_kl_probe` (samples sequences and compares unigram distribution to the training corpus — the canonical mode-collapse detector), checkpoint every N epochs, and a final `epoch_final.pt` saved **without** optimizer state for inference-only use. The probe is gated by `cfg.training.sample_eval_every`.

### Data pipeline

Text8 loads via HuggingFace (`afmck/text8`) → ASCII chars → `CHAR2ID` (lowercase a-z + space, K=27) → length-L windows. `CharWindowDataset` precomputes label-smoothed CLR-style features (`token_ids_to_features`). Windows can be cached to disk under `cache_dir/windows/<md5>.pt`.

`CorruptingCollate` produces both `x` (smoothed features, possibly position-mixed) and `token_ids` (clean integer labels) per batch. Both keys are used: EqM consumes `x` for the FM regression and `token_ids` for the aux CE on the implied-x1 reconstruction; DFM uses only `token_ids`.

### EqM training step (the critical path)

`models/eqm.py:_eqm_loss` — read this whole function before editing it:

1. `x0 = source_sigma · randn` centered on V_d (zero-mean across K). **Same σ is used at sample time** (`cfg.eqm.source_sigma`); a mismatch silently OOD's inference.
2. `gamma = U(0,1)**gamma_power` with `gamma_power=0.5` upweights γ≈1 (the signal regime).
3. Interpolate `x_γ = (1−γ)·x0 + γ·x1`, target `u_tgt = c(γ)·(x0 − x1)`.
4. Forward `v = f(x_γ)`. Compute `grad_g = ∇_{x_γ} ⟨x_γ, v⟩` via `torch.autograd.grad(..., create_graph=True)`. The model is the **conservative gradient of `⟨x, f(x)⟩`**, not `f` itself.
5. Flow loss: `loss_fn(grad_g, u_tgt)` — `loss_fn` comes from the loss registry in `losses.py` (`mse` / `hilbert` / `hilbert_soft`).
6. Aux CE on the implied-x1 reconstruction `pred_x1 = x_γ − λ·grad_g` against `token_ids`, masked to γ ≥ `ce_min_gamma`. It was *intended* as a per-token anchor against unigram collapse; the paper shows it does not act as one — the mask puts the term where the label is already visible, so it is solved before the first epoch ends (`appendix` §Training-time collapse). It is kept because every published EqM arm trained with it.

Sampling (`eqm.sample`) runs NAG-GD on the **same conservative gradient** (`_compute_grad`). Train and inference must use the same field; a previous bug where sampling used raw `f(x)` is documented in `SESSION_SUMMARY.md`.

### Backbone constraint

`TransformerBackbone` runs attention under `sdpa_kernel(SDPBackend.MATH)` on purpose — the FlashAttention backend doesn't support the second-order autograd needed by `create_graph=True` on the conservative gradient. Don't switch to flash unless you've also moved off the `⟨x, f(x)⟩` formulation.

### Geometry primitives

`geometry.py` holds Aitchison-simplex utilities: `hilbert_distance` (variation norm, used as a metric), `nielsen_soft_hilbert_distance` (LSE-smoothed, differentiable), `ilr` / `ilr_inv` (orthogonal Helmert basis between CLR `R^K` and ILR `R^{K-1}`). ILR is used **only** for visualization in eval scripts; the model trains and samples in CLR. `volume_penalty` is implemented but not wired in — `lambda_vol`/`lambda_mse`/`alpha` in the EqM config are dead knobs.

### Known dead code / leftovers

Documented in `SESSION_SUMMARY.md` §5.6 — `volume_penalty`, several EqM config fields, and `Text8DataConfig.enabled`; separately, `run.py` referenced by `run_files/run_text8.sh` does not exist (the script carries a deprecation header). Don't be misled by them; they aren't reachable from `main.py`. Also **retired**: the GPT-2/WikiText "auditor" line (Phase F/H — the trained EqM auditor added nothing beyond a linear probe / spilled energy; see `EVAL_ASSESSMENT.md` §Obj3) and the identity-path BPC/PPL (a recovery artifact, never a likelihood).
