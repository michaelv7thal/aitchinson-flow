# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Capstone research code for **continuous flow-matching on the simplex** applied to character-level text8 (K=27 vocab, L=40 windows). Two model families live side-by-side:

- **EqM** — Equilibrium Flow Matching. Trains a velocity field `f(x)` on CLR (centered log-ratio) features; uses the **conservative gradient** `∇_x ⟨x, f(x)⟩` of an implicit energy as the FM target. Sampling is NAG-GD (Nesterov-accelerated gradient descent) on that energy.
- **DFM** — Discrete Flow Matching (Gat et al. 2024 baseline). Uniform-source probability path `p_t = κ_t·δ(x1) + (1−κ_t)·U(K)`, CE on the denoiser, Euler sampling on the probability velocity.

`SESSION_SUMMARY.md` documents the diagnosis/fixes that brought the EqM model from mode-collapse to ~94% of corpus unigram entropy. Read it before changing the EqM training step or sampler — many of the choices (conservative-grad sampling, matched train/sample σ, aux CE on implied-x1, γ-importance sampling) are load-bearing fixes for specific failure modes documented there.

## Commands

Project uses `uv` (see `uv.lock`) and Python 3.13.

- Train EqM (default model): `python main.py` — flags: `--out-dir`, `--epochs`, `--lr`. The CLI is intentionally minimal; everything else lives in `Config` (`src/aitchinson_flow/config.py`).
- Lint: `uv run ruff check` / `uv run ruff format`.
- Evaluate a trained checkpoint: `python scripts/evaluate_eqm.py` (writes a multi-panel figure to `scripts/evaluate_eqm.png`). Other scripts in `scripts/` are one-off experiment runners — `compare_gt_vs_gen.py`, `eval_plots.py`, `eval_nfe_comparison.py`, `eval_healing_demo.py`, `plot_simplex.py`.
- `run_files/run_text8.sh` references a `run.py` entry point that does **not** exist in the current tree (it expects flags like `--model eqm --preset cluster`). Treat it as a fossil; use `main.py` instead.

There is no test suite.

## Architecture

### Config-as-dataclass

`Config` (`src/aitchinson_flow/config.py`) is the single source of truth — `TrainingConfigs`, `Text8DataConfig`, `TransformerConfig`, `EqM`, `DFMConfig`, `LoaderSettings`, `LossConfig`, `TransformationConfig` are nested dataclasses. The CLI only overrides three fields; experiments are run by editing the dataclass defaults or by mutating `cfg` in code (`dataclasses.replace`). `cfg.training.model_name` selects the model.

### Model registry

Models register builders by name via `aitchinson_flow.models.factory.register("Name")`. `build_model(cfg)` looks up `cfg.training.model_name` ("EqM", "DFM") in `REGISTRY`. **Importing `aitchinson_flow.models` is what populates the registry** — `main.py` imports it for its side effect (see comment on the import).

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

Documented in `SESSION_SUMMARY.md` §5.6 — `volume_penalty`, several EqM config fields, `Text8DataConfig.enabled`, and `run.py` referenced by `run_files/run_text8.sh`. Don't be misled by them; they aren't reachable from `main.py`.
