# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Capstone research code for **continuous flow-matching on the simplex** applied to character-level text8 (K=27 vocab; default L=40 windows, **L=256** for the publication-length benchmark). Several model families live side-by-side:

- **EqM** — Equilibrium Flow Matching (Wang & Du 2025, arXiv:2510.02300). Trains a velocity field `f(x)` on CLR (centered log-ratio) features; uses the **conservative gradient** `∇_x ⟨x, f(x)⟩` of an implicit energy as the FM target. Sampling is NAG-GD (Nesterov-accelerated gradient descent) on that energy. Variants: `EqM_OneHot`, `EqMLatent` (learned-embedding space).
- **DFM / DirichletFM** — discrete / Dirichlet Flow Matching baselines (`models/dfm.py`, `models/dirichlet_fm.py`; Stark et al. 2024, arXiv:2402.05841). CE on the denoiser, Euler sampling. `DFM` exposes the **only peer-comparable BPC** via `DFM.elbo_bpc` — a genuine D3PM uniform-process **variational NLL upper bound** (honest peer: **D3PM-uniform ≈1.61**; SEDD 1.32 / D3PM-absorb 1.45 / MDLM are absorbing/score methods, reference only). `DFM.bpd()` now returns this bound; the old uniform-t training CE is kept as `DFM.denoiser_ce_bpc` (a diagnostic, **not** a likelihood). A `<0.5`-BPC sanity guard refuses identity/near-clean artifacts.
- **SFLM / SFLMEBM** — hyperspherical flow (arXiv:2605.11125); `SFLMEBM` reads it as an EqM-style EBM (log-sum-exp energy, Riemannian-GD sampler). `SFLMEBM_FM` is the same class with the Riemannian-FM target (`cfg.sflm_ebm.lambda_fm>0`).
- **SVGP OOD head** (`DirichletFMSvgp`) — post-hoc Sparse Variational GP, hinge-trained on frozen features, the recommended OOD detector.

The repo investigates three objectives — **(1) unconditional generation, (2) conditional recovery, (3) OOD detection**. Verdicts and the *which-eval-and-why* analysis live in **`EVAL_ASSESSMENT.md`** (start there; `README.md` is the index). The authoritative arm list is `scripts/bench_sflm_ebm.py:MODELS` plus the factory registry.

**Cloud-run hardening (L256).** For the L=256 cloud workstream the repo was hardened: `cfg.transformer.grad_checkpointing` + a memory-fallback ladder in `train_for_sflm_bench.py` (grad-ckpt → halve batch → L=128 → mark-failed-and-continue) handle the 20 GB MIG OOM/NVML asserts; `--seeds 42,43,44`, idempotent skip-if-done, and `--full-split` (lazy CLR features) are implemented; the X1/X2/X3 harness (`scripts/eval_all.py`, `scripts/manifest.py`, `scripts/run_experiment.py`, `RESULTS_README.md`) and the E2a/E2b/E2d/E4d/E5a/E6a/E7 scripts now exist. See `CLUSTER_RUNBOOK_L256.md` / `capstone_experiment_runbook.md`.

`SESSION_SUMMARY.md` documents the diagnosis/fixes that brought the EqM model from mode-collapse to ~94% of corpus unigram entropy. Read it before changing the EqM training step or sampler — many of the choices (conservative-grad sampling, matched train/sample σ, aux CE on implied-x1, γ-importance sampling) are load-bearing fixes for specific failure modes documented there. The **theory of what fails and why** is in `NOTE_WHY_EBM_INIT_STUCK.md` (training-time collapse) and `NOTE_WHY_UNCONDITIONAL_FAILS.md` (sampling-time failure).

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
- **Obj 3 (OOD):** `scripts/bench_sflm_ebm.py` → corruption-ladder AUROC, anchored to the external **`gpt2_baseline`** row (GPT-2 spilled energy, `--ref-lm gpt2`). **Headline the shuffle (order) axis**, not substitution (trivially ~1.0). Native EBM energy is at chance on shuffle; the hinge-**SVGP** detector (`svgp_corruption_sweep.json`) recovers it.

Peer-comparable text8 BPC (vs SEDD 1.32 / D3PM 1.45 / MDLM ≤1.38 / SFM 1.39; frontier 1.32–1.47) needs **L=256** on the standard last-5M test split. That run is GPU-bound (20 GB) and is handed off in **`CLUSTER_RUNBOOK_L256.md`** (recipe for a fresh session). `bench_sflm_ebm.py` is now robust to per-arm OOM at L=256 (each readout wrapped; `SFLMEBM.position_uncertainty` chunks over the batch).

## Repository layout

`main.py` (train EqM) · `src/aitchinson_flow/` (package: `config`, `models/`, `data/`, `training/`, `geometry`, `sampling`) · `scripts/` (canonical entry points) · **`scripts/legacy/`** (one-off / superseded / broken runners — `diagnose_*`, `eval_plots.py`, `compare_gt_vs_gen.py`, …) · `sweeps/` (8 active configs) + **`sweeps/archive/`** (phase/POC) · `runs/` (outputs; **checkpoints are untracked**; `runs/DECISION_LOG.md` is the source-of-truth diary) · **`docs/archive/`** (superseded planning docs). `runs/` was triaged in `EVAL_ASSESSMENT.md` but **not physically moved** (untracked ~19 GB checkpoints + hardcoded path refs).

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
6. Aux CE on the implied-x1 reconstruction `pred_x1 = x_γ − λ·grad_g` against `token_ids`, masked to γ ≥ `ce_min_gamma`. This anchors per-token attractors and is what prevents mode collapse to the unigram peak.

Sampling (`eqm.sample`) runs NAG-GD on the **same conservative gradient** (`_compute_grad`). Train and inference must use the same field; a previous bug where sampling used raw `f(x)` is documented in `SESSION_SUMMARY.md`.

### Backbone constraint

`TransformerBackbone` runs attention under `sdpa_kernel(SDPBackend.MATH)` on purpose — the FlashAttention backend doesn't support the second-order autograd needed by `create_graph=True` on the conservative gradient. Don't switch to flash unless you've also moved off the `⟨x, f(x)⟩` formulation.

### Geometry primitives

`geometry.py` holds Aitchison-simplex utilities: `hilbert_distance` (variation norm, used as a metric), `nielsen_soft_hilbert_distance` (LSE-smoothed, differentiable), `ilr` / `ilr_inv` (orthogonal Helmert basis between CLR `R^K` and ILR `R^{K-1}`). ILR is used **only** for visualization in eval scripts; the model trains and samples in CLR. `volume_penalty` is implemented but not wired in — `lambda_vol`/`lambda_mse`/`alpha` in the EqM config are dead knobs.

### Known dead code / leftovers

Documented in `SESSION_SUMMARY.md` §5.6 — `volume_penalty`, several EqM config fields, `Text8DataConfig.enabled`, and `run.py` referenced by `run_files/run_text8.sh` (now carries a deprecation header). Don't be misled by them; they aren't reachable from `main.py`. Also **retired**: the GPT-2/WikiText "auditor" line (Phase F/H — the trained EqM auditor added nothing beyond a linear probe / spilled energy; see `EVAL_ASSESSMENT.md` §Obj3) and the identity-path BPC/PPL (a recovery artifact, never a likelihood).
