# Aitchison Flow: Two-Stage Bayesian Auditor

Uncertainty quantification for discrete-token sequences using:

- simplex-aware geometry (Aitchison/log-ratio coordinates),
- Equilibrium Matching (EqM) for manifold-aware velocity learning,
- sparse Gaussian Processes (GPs) for calibrated uncertainty signals.

The repo supports both a single-stage auditor and a **two-stage Bayesian Auditor**:

1. **Stage 1**: EqM + Hilbert-family training on valid data only (learn geometry of the valid manifold).
2. **Stage 2**: freeze Stage 1 backbone, train latent projection + GP contrastively on valid vs invalid.
3. **Compose** both into one inference model (`BayesianAuditor`).

---

## Theory (What The Model Is Doing)

### 1) Discrete tokens -> simplex geometry

Tokens are converted to continuous features via:

`token ids -> one-hot -> (label smoothing or eps path) -> log -> ILR/CLR`

- **Label smoothing** (`alpha > 0`) moves one-hot points into the simplex interior with  
  `(1 - alpha) * one_hot + alpha / K`.
- **Log-space** linearizes multiplicative/probability-ratio structure.
- **ILR** (default) maps to an unconstrained Euclidean chart (`K-1` dims).
- **CLR** is available for ablation (`K` dims, sum-to-zero constrained).

### 2) Stage 1 geometric objective

Stage 1 learns a velocity field on interpolants between uniform noise and valid sequences.

- EqM target direction in this codebase:
  `u_tgt = c(gamma) * (log_x0 - log_x1)` (data -> noise convention)
- Inference integration follows:
  `x <- x - v_theta(x) * dt` (noise -> data)

Stage 1 also exposes an explicit geometric score based on soft Hilbert distance:

- `g(x) = - d_H(f(x), x)`  (sequence energy, averaged over tokens)
- `ood_score(x) = -g(x) = d_H(f(x), x)` (higher means more OOD)

where `f(x)` is the Stage 1 forward output in the same coordinate space.

### 3) Stage 2 contrastive GP objective

On top of frozen Stage 1 backbone features:

- `backbone`: frozen
- `latent_head`: trainable
- `gp`: trainable

The GP is trained with contrastive valid/invalid supervision and KL regularization to separate in-distribution vs anomalous behavior, while keeping valid energy anchored.
Stage 2 does **not** include Hilbert-distance penalties directly; Hilbert geometry is learned in Stage 1 and transferred through the frozen backbone. By default in Stage 2, GP aleatoric noise (`log_noise_var`) is fixed and the objective focuses on epistemic/contrastive learning.

### 4) Two complementary UQ signals

- **Stage 1 geometric score** (`d_H`-based): sequence-level OOD sensitivity without invalid-label training.
- **Stage 2 GP variance/energy outputs**: finer uncertainty structure after contrastive calibration.

---

## Code Architecture

Core package: `src/aitchinson_flow/`

- `config.py`  
  Typed dataclass config for model/training/data/benchmark.
- `geometry.py`  
  ILR/CLR and Hilbert-geometry primitives.
- `loss.py`  
  Velocity loss builders (Hilbert-family + MSE-family options).
- `transformer_backbone.py`  
  Shared transformer representation stack (`TransformerBackbone`, `VelocityHead`, `LatentHead`).
- `models/`
  - `bayesian_auditor_stage1.py`: Stage 1 EqM + geometric scoring
  - `bayesian_auditor_stage2.py`: Stage 2 frozen-backbone contrastive GP
  - `bayesian_auditor.py`: composed inference model + stage composition helpers
  - `flow_matching.py`, `equilibrium.py`, `bayesian_generator.py`, etc.
  - `factory.py`: model registry (`cfg.training.model_name`)
- `gp/`  
  Sparse GP implementation and algebra/kernels.
- `training/`
  - `runner.py`: `fit(...)`
  - `loops.py`: train/eval loops
  - `optim.py`: optimizer/scheduler builders
  - `checkpoint.py`: checkpoint IO
- `data/`
  - `transforms/discrete.py`: token -> ILR/CLR feature transform
  - `feature_dim.py`: transform-aware feature dimensionality helper
  - `text8_datamodule.py`: text8 dataset + corruption plumbing

Benchmark package: `benchmarks/`

- `runner.py`: scale + ablation sweeps
- `tasks/text_audit.py`: metrics/AUROC reporting (auditor/residual/energy/spilled)
- `corruption.py`: invalid sample construction
- `plots.py`: benchmark plotting utilities
- `run_bench.py`: simple programmable benchmark example

Scripts:

- `scripts/two_stage_train.py`: first-class Stage1 -> Stage2 -> compose workflow
- `scripts/scale_sweep.py`: utility sweep script

---

## Getting Started

### Prerequisites

- Python `>=3.11`
- Linux/macOS recommended
- GPU optional (CPU works for smoke tests)

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,benchmarks]"
```

If you use `uv`, `pyproject.toml` already includes a CUDA index setup for Linux torch wheels.

---

## Quickstart

### 1) Run tests

```bash
source .venv/bin/activate
pytest -q
```

### 2) Run benchmark sweep entrypoint

The package script is exposed as `bench-scaling`:

```bash
source .venv/bin/activate
bench-scaling
```

Results are written to `results/benchmark/benchmark_latest.json` by default.

### 3) Programmatic benchmark example

See `benchmarks/run_bench.py` for a minimal config-and-run script:

```bash
source .venv/bin/activate
python benchmarks/run_bench.py
```

---

## Two-Stage Workflow (Train -> Compose)

Use the dedicated orchestration script:

```bash
source .venv/bin/activate
python scripts/two_stage_train.py \
  --out-dir checkpoints/two_stage/baseline \
  --stage1-epochs 10 \
  --stage2-epochs 5
```

Artifacts:

- `stage1.pt` (Stage 1 model checkpoint)
- `stage2.pt` (Stage 2 model checkpoint)
- `fused.pt` (composed `BayesianAuditor`)
- `orchestration.json` (manifest/config snapshot)

Useful options:

- `--random-stage2-backbone`: Stage 2 random-backbone ablation.
- `--smoke`: tiny CPU smoke run.

---

## Running Ablations

The benchmark scale grid accepts architecture and ablation overrides per run.
Supported ablation keys include:

- `velocity_loss` (e.g. `soft_hilbert`, `hard_hilbert`, `clr_mse`, `ilr_mse`)
- `transform_mode` (`ilr` or `clr`)
- `label_smoothing` (float in `[0,1)`)
- `model_name` (e.g. `bayesian_auditor_stage1`, `bayesian_auditor`)

Example (programmatic):

```python
from aitchinson_flow.config import Config
from benchmarks.runner import run_benchmark

cfg = Config()
cfg.benchmark.scale_grid = [
    {"d_model": 128, "num_layers": 4, "nhead": 8, "velocity_loss": "soft_hilbert", "transform_mode": "ilr"},
    {"d_model": 128, "num_layers": 4, "nhead": 8, "velocity_loss": "clr_mse", "transform_mode": "ilr"},
    {"d_model": 128, "num_layers": 4, "nhead": 8, "velocity_loss": "soft_hilbert", "transform_mode": "clr"},
]
results = run_benchmark(cfg)
```

`text_audit` output includes AUROCs and `ablation_tags` for easy slicing/aggregation.

---

## Key Config Knobs

From `Config()`:

- `cfg.training.model_name`: model registry key
- `cfg.training.velocity_loss`: Stage1 velocity objective family
- `cfg.equilibrium.*`: EqM schedule + generation settings
- `cfg.hf_dataset.label_smoothing`: explicit simplex-interior smoothing
- `cfg.hf_dataset.transform_mode`: `ilr` vs `clr`
- `cfg.benchmark.*`: scale sweeps, corruption params, plotting, reporting

---

## Composition API (Manual)

If you already have checkpoints/states:

- `aitchinson_flow.models.compose_auditor_from_stages(...)`
- `aitchinson_flow.models.load_auditor_from_stage_checkpoints(...)`

These build an inference-ready `BayesianAuditor` by fusing Stage 1 backbone + Stage 2 latent/GP weights.

---

## Development Notes

- Model registry keys are defined through `@register(...)` in `models/factory.py`.
- Training loop entrypoint is `aitchinson_flow.training.runner.fit`.
- Checkpoints are saved every `cfg.training.checkpoint_every` epochs to `cfg.training.checkpoint_dir`.
- The repository currently uses strict typing/linting tooling (`mypy`, `ruff`) in optional dev dependencies.

---

## License

MIT (see `pyproject.toml`).
