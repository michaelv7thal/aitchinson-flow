# Aitchison Flow: Two-Stage Bayesian Auditor

Uncertainty quantification for discrete-token sequences using:

- simplex-aware geometry (Aitchison/log-ratio coordinates),
- Equilibrium Matching (EqM) for manifold-aware velocity learning,
- sparse Gaussian Processes (GPs) for calibrated uncertainty signals.

The repo supports both a single-stage auditor and a **two-stage Bayesian Auditor**:

1. **Stage 1**: EqM + Hilbert-family training on valid data only (learn geometry of the valid manifold).
2. **Stage 2**: freeze Stage 1 backbone, train latent projection + GP contrastively on valid vs invalid.
3. **Compose** both into one inference model (`BayesianAuditor`).

### Two coexisting task paths

The same Stage 1 / Stage 2 infrastructure drives two distinct auditors that produce separate checkpoints:

- **Path A — char-level text8 OOD (K=27, L=30).** The established flow; text corruption provides invalid negatives.
- **Path B — byte-level Q+A hallucination auditor (K=256, L=128).** Trained on `[Question][Answer]` pairs from a Hugging Face dataset (default `trivia_qa/rc.nocontext`). In-batch cross-question-swap supplies training negatives; an HF causal LM (e.g. `gpt2`) supplies eval-time hallucinated answers. The GP loss can be restricted to answer-span tokens via `cfg.gp.score_answer_tokens_only`.

Path selection is config-driven (`cfg.dataset.K`, `cfg.training_data.source`); both paths share the same `fit()` loop, the same `BayesianAuditorStage1`/`Stage2` classes, and the same geometry/GP modules.

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
  - `text8_datamodule.py`: text8 dataset + corruption plumbing (Path A)
  - `trivia_datamodule.py`: char-level trivia Q+A data (Path A, K=27)
  - `byte_vocab.py`: UTF-8 byte codec + role markers (Path B)
  - `bytes_datamodule.py`: raw-text corpus → byte windows for Path B Phase 1
  - `qa_datamodule.py`: `[Q][A]` byte-encoded pairs for Path B Phase 2
  - `qa_negatives.py`: cross-question-swap training negatives

Benchmark package: `benchmarks/`

- `runner.py`: scale + ablation sweeps
- `tasks/text_audit.py`: metrics/AUROC reporting (Path A, auditor/residual/energy/spilled)
- `tasks/trivia_audit.py`: char-level trivia OOD AUROC (Path A)
- `tasks/hallucination_audit.py`: byte-level Q+A hallucination AUROC (Path B)
- `corruption.py`: invalid sample construction
- `plots.py`: benchmark plotting utilities
- `run_bench.py`: simple programmable benchmark example

Scripts:

- `scripts/two_stage_train.py`: Path A — first-class Stage1 -> Stage2 -> compose workflow (text8)
- `scripts/phase1_train_bytes.py`: Path B — retrain the Stage 1 backbone at K=256 on bytes
- `scripts/phase2_train_qa.py`: Path B — Stage 2 Q+A head on top of the byte-level backbone
- `scripts/phase2_eval_hallucination.py`: Path B — AUROC eval against LLM-generated answers
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

## Path A: Two-Stage text8 Workflow (Train -> Compose)

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

## Path B: Byte-level Q+A Hallucination Auditor

Path B produces a **separate** set of checkpoints; it does not replace Path A. The flow is: retrain Phase 1 at `K=256` on a byte-level corpus, train Phase 2 on `[Q][A]` byte pairs, then evaluate against LLM-generated answers.

```bash
# Phase 1: retrain the backbone at K=256 on raw bytes.
python scripts/phase1_train_bytes.py \
  --out-dir checkpoints/phase1_bytes/baseline \
  --epochs 10 --L 512

# Phase 2: train Stage 2 GP head on byte-encoded [Q][A] pairs.
python scripts/phase2_train_qa.py \
  --out-dir checkpoints/phase2_qa/baseline \
  --stage1-backbone-ckpt checkpoints/phase1_bytes/baseline/stage1.pt \
  --epochs 10

# Eval: have gpt2 answer val questions, score [Q][A_llm] pairs, report AUROC.
python scripts/phase2_eval_hallucination.py \
  --stage2-ckpt checkpoints/phase2_qa/baseline/stage2.pt \
  --answer-model-id gpt2 --max-val-samples 200
```

Key knobs (see `QADatasetConfig`, `AnswerGeneratorConfig`, `GPConfig.score_answer_tokens_only`):

- `cfg.qa_dataset.hf_path` / `.name` — HF dataset (default `trivia_qa` / `rc.nocontext`).
- `cfg.qa_dataset.max_question_bytes` / `.max_answer_bytes` — per-side UTF-8 byte caps.
- `cfg.answer_generator.model_id` / `.prompt_template` — the eval-time hallucinator.
- `cfg.gp.score_answer_tokens_only = True` — restrict the GP loss (and eval score) to answer-span positions only.
- `cfg.training_data.qa_skip_llm_eval = True` — plumbing-only smoke path that skips LLM instantiation and uses cross-question-swap as a placeholder.

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
