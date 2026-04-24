# Aitchison Flow: Three-Component Uncertainty Quantification

Calibrated, per-token uncertainty quantification for discrete-token sequences using:

- **Simplex-aware geometry** — Aitchison/ILR coordinates for token distributions
- **Equilibrium Matching (EqM)** — manifold-aware velocity learning on valid sequences only
- **Sparse Gaussian Processes (SVGP)** — calibrated epistemic energy + variance after contrastive training
- **Spilled energy** — training-free anomaly score from frozen LLM logits

Each token position receives up to five signals: structural energy, structural variance, contextual energy, contextual variance, and spilled energy. All three components share architecture code, loss functions, and GP implementation; they differ only in input space.

---

## Three UQ Components

### Component 1 — Structural Geometry UQ

Does this sequence look structurally valid independent of meaning? (correct spelling, valid syntax, biologically plausible nucleotides)

```
token IDs → one-hot → ILR → Stage 1 EqM backbone → Stage 2 SVGP
                                                           ↓
                                          energy (L,)  variance (L,)
```

### Component 2 — Contextual Semantic UQ

Does the LLM treat this context as plausible? Valid contexts produce exponential-decay top-K distributions; surprising or invalid contexts produce flat distributions.

```
raw text → frozen LLM → top-K softmax → re-normalize → ILR → Stage 1 EqM → Stage 2 SVGP
                                                                                    ↓
                                                               energy (L,)  variance (L,)
```

### Component 3 — Spilled Energy (training-free)

Internal consistency of the autoregressive distribution. Requires no training.

```
ΔE(x_i) = −logsumexp(logits[i]) + logits[i, token_id[i+1]]
anomaly(x) = −mean_i(ΔE(x_i))
```

### Signal interpretation

| Structural energy | Structural variance | Interpretation |
|---|---|---|
| Low | Low | Structurally certain token |
| Low | High | Plausible but model is unsure |
| High | Low | Confidently out-of-distribution |
| High | High | Anomalous and uncertain |

---

## Architecture

```
                  Input Token Sequence
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
 Component 1       Component 2       Component 3
 Structural UQ     Contextual UQ     Spilled Energy
                                     (training-free)
 one-hot → ILR     top-K probs →     ΔE = -logsumexp
 → EqM Flow        re-norm → ILR        + logit[t+1]
 → Sparse GP       → EqM Flow
                   → Sparse GP
        │                 │                 │
        └─────────────────┴─────────────────┘
                          │
               Unified UQ Report (5 signals / token)
```

### Two-stage architecture

Stage 1 learns the geometry of valid sequences without ever seeing invalid samples — preventing the backbone from collapsing to a trivial discriminator. Stage 2 then trains the GP contrastively on frozen Stage 1 features.

| Stage | Trainable | Objective |
|---|---|---|
| Stage 1 | Transformer backbone + velocity head | Hilbert-family velocity loss on valid data |
| Stage 2 | Latent head + SVGP | Contrastive energy hinge + KL regularization |
| Inference | Composed `BayesianAuditor` | Energy + variance per token |

---

## Code Layout

```
src/aitchinson_flow/
├── config.py                    # All typed dataclasses (Config + 20 sub-configs)
├── geometry.py                  # ILR, CLR, Hilbert distances
├── loss.py                      # Velocity loss builders
├── transformer_backbone.py      # Shared backbone + heads
│
├── gp/                          # Sparse GP
│   ├── gp.py                    # SparseGP: Matérn 5/2, SVGP, predictive
│   ├── _svgp_algebra.py         # Titsias predictive + KL
│   └── _svgp_kernels.py         # Matérn kernel, adaptive Cholesky
│
├── models/
│   ├── bayesian_auditor_stage1.py   # EqM backbone training
│   ├── bayesian_auditor_stage2.py   # Frozen backbone + contrastive GP
│   ├── bayesian_auditor.py          # Composed Stage 1+2 inference
│   ├── flow_matching.py             # Time-conditioned CFM
│   ├── equilibrium.py               # Time-independent EqM
│   ├── llm_projection.py            # TokenEmbeddingToSimplex
│   └── factory.py                   # Model registry
│
├── data/
│   ├── text8_datamodule.py          # Char-level text8 (K=27, Path A)
│   ├── llm_topk_probs_datamodule.py # Component 2 top-K probability pipeline
│   ├── llm_embedding_datamodule.py  # Path B frozen-embedding pipeline
│   ├── qa_datamodule.py             # Byte-level Q+A trivia (Path C)
│   ├── dna_datamodule.py            # Nucleotide sequences (K=4/5)
│   ├── medical_datamodule.py        # Clinical text (K=47)
│   ├── corruption.py                # Swap / drop / insert / replace corruptions
│   ├── teachers/causal_lm.py        # Frozen HF CausalLM + top_k_probs()
│   └── transforms/discrete.py       # Token IDs → ILR/CLR features
│
├── training/
│   ├── runner.py                # fit() — epoch loop + checkpointing
│   ├── data_sources.py          # Source string → DataModule dispatcher
│   ├── loops.py                 # train_epoch(), evaluate()
│   ├── optim.py                 # Optimizer + scheduler builders
│   └── checkpoint.py            # Save / load helpers
│
├── metrics/
│   ├── auroc.py                 # NaN-safe AUROC
│   ├── spilled_energy.py        # Marginal + spilled energy (training-free)
│   └── calibration.py           # ECE + reliability diagrams for GP variance
│
├── analysis/
│   ├── token_heatmaps.py        # Multi-signal stacked token heatmaps
│   ├── component_correlation.py # Pearson/Spearman between C1/C2/C3 energies
│   └── failure_modes.py         # Detect cross-component disagreements
│
├── healing/
│   ├── targeted_resample.py     # Mask high-energy tokens → resample → iterate
│   ├── simplex_project.py       # EqM backward → project to nearest valid token
│   └── beam_rerank.py           # Beam reranking with energy penalty
│
└── plots/plots.py               # Benchmark heatmaps, stage diagnostic plots

benchmarks/
├── runner.py                    # ComponentTaskSpec, run_unified_benchmark()
├── results_schema.py            # BenchmarkResultRow, FullBenchmarkResults
└── tasks/
    ├── text_audit.py            # Component 1 text8 AUROC
    ├── trivia_audit.py          # Component 2 Q+A AUROC
    ├── hallucination_audit.py   # Component 2 hallucination detection
    ├── dna_audit.py             # Component 1 DNA sequences
    ├── medical_audit.py         # Component 1+2 clinical text
    └── healing_audit.py         # Phase 4 healing evaluation

scripts/
├── two_stage_train.py           # Canonical training entry point
├── run_full_benchmark.py        # Phase 3 unified benchmark matrix
├── run_healing_eval.py          # Phase 4 healing evaluation
├── phase1_train_bytes.py        # Path C backbone training
├── phase2_train_qa.py           # Path C Q+A GP head
└── phase2_eval_hallucination.py # Path C hallucination eval
```

---

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,benchmarks]"
```

`uv` is also supported; `pyproject.toml` includes a CUDA index for Linux torch wheels.

---

## Quickstart

### Path A — Char-level text8 (Component 1)

```bash
python scripts/two_stage_train.py \
  --out-dir checkpoints/text8 \
  --stage1-epochs 10 \
  --stage2-epochs 5
```

### Path B — Pretrained-LLM top-K (Component 2)

```bash
python scripts/two_stage_train.py \
  --out-dir checkpoints/path_b \
  --stage1-epochs 5 \
  --stage2-epochs 3 \
  --training-data-source llm_topk_probs
```

The frozen LLM is configured via `cfg.teacher` (default: `gpt2`). Any HF causal LM works.

### Path C — Byte-level Q+A hallucination auditor

```bash
# Stage 1: train backbone at K=256 on raw bytes
python scripts/phase1_train_bytes.py \
  --out-dir checkpoints/phase1_bytes \
  --epochs 10 --L 512

# Stage 2: train GP head on [Q][A] pairs
python scripts/phase2_train_qa.py \
  --out-dir checkpoints/phase2_qa \
  --stage1-backbone-ckpt checkpoints/phase1_bytes/stage1.pt \
  --epochs 10

# Eval: score LLM-generated answers, report AUROC
python scripts/phase2_eval_hallucination.py \
  --stage2-ckpt checkpoints/phase2_qa/stage2.pt \
  --answer-model-id gpt2 --max-val-samples 200
```

### Full benchmark matrix (all components × all tasks)

```bash
python scripts/run_full_benchmark.py --out-dir results/
```

Produces `results/full_benchmark.json` with one row per `(component, task, scale)`.

### Healing evaluation

```bash
python scripts/run_healing_eval.py \
  --stage2-ckpt checkpoints/text8/stage2.pt \
  --out-dir results/healing/
```

---

## Benchmark Tasks

| Task | Component | Vocabulary | Valid | Invalid |
|---|---|---|---|---|
| `text8_audit` | 1 | K=27 chars | text8 windows | swap / drop / insert corruptions |
| `trivia_audit` | 2 | byte-level | correct Q+A pairs | cross-question swap |
| `hallucination_audit` | 2 | byte-level | correct answers | LLM-generated answers |
| `dna_audit` | 1 | K=4/5 nucleotides | real genomic sequences | point mutations, frameshifts |
| `medical_audit` | 1+2 | K=47 chars | clinical notes | drug misspellings, unit corruption |
| `healing_audit` | 1+2+3 | — | — | energy/variance before vs after repair |

### Evaluation metrics

| Metric | Description |
|---|---|
| `auroc_energy` | Separability (valid / invalid) using GP energy |
| `auroc_variance` | Separability using GP variance alone |
| `auroc_combined` | Energy + variance joint score |
| `auroc_spilled` | Spilled energy as training-free baseline |
| `energy_gap` | `mean_energy(invalid) − mean_energy(valid)` |
| `variance_ratio` | `mean_var(invalid) / mean_var(valid)` |
| `per_token_auroc` | Token-level AUROC at corruption sites |

---

## OOD Healing

Three strategies are implemented in `src/aitchinson_flow/healing/`:

| Strategy | File | Description |
|---|---|---|
| Beam reranking | `beam_rerank.py` | `score = log_p − λ·mean_energy`; penalizes high-energy candidates |
| Targeted resampling | `targeted_resample.py` | Mask tokens where `energy[t] > threshold`, resample via LLM/EqM, iterate |
| Simplex projection | `simplex_project.py` | Run EqM ODE backward on flagged tokens, snap to nearest valid token via ILR inverse + argmax |

Configure via `HealingConfig(strategy, threshold, max_iter, lambda_energy)`.

---

## Key Config Knobs

All configuration lives in `Config()` from `src/aitchinson_flow/config.py`.

```python
from aitchinson_flow.config import Config
cfg = Config()

cfg.dataset.K           # vocabulary size (27=text8, 256=bytes, 4=DNA, 50=top-K)
cfg.dataset.L           # sequence length
cfg.training.velocity_loss     # "soft_hilbert" | "hard_hilbert" | "clr_mse" | "ilr_mse"
cfg.training_data.source       # "raw_text" | "llm_topk_probs" | "qa_pairs" | "dna" | "medical"
cfg.teacher.model_id           # frozen LLM for Component 2 (e.g. "gpt2", "Qwen/Qwen2-1.5B")
cfg.gp.num_inducing            # SVGP inducing point count
cfg.equilibrium.ode_steps      # EqM integration steps at inference
cfg.healing.strategy           # "beam_rerank" | "targeted_resample" | "simplex_project"
cfg.healing.threshold          # energy threshold for targeted resampling
cfg.benchmark.scale_grid       # list of (d_model, nhead, num_layers) dicts
```

---

## Analysis Tools

```python
from aitchinson_flow.analysis.token_heatmaps import plot_token_heatmap
from aitchinson_flow.analysis.component_correlation import compute_correlations
from aitchinson_flow.analysis.failure_modes import find_disagreements
from aitchinson_flow.metrics.calibration import expected_calibration_error
```

- `token_heatmaps.py` — stacked heatmap of all five UQ signals aligned to token positions
- `component_correlation.py` — Pearson/Spearman correlation matrix between Component 1, 2, and 3 energy signals
- `failure_modes.py` — identifies sequences where components disagree (e.g., structurally valid but contextually anomalous)
- `calibration.py` — ECE and reliability diagrams for GP variance estimates

---

## Composition API

```python
from aitchinson_flow.models import compose_auditor_from_stages, load_auditor_from_stage_checkpoints

# From existing Stage 1 + Stage 2 objects
auditor = compose_auditor_from_stages(stage1, stage2)

# From saved checkpoints
auditor = load_auditor_from_stage_checkpoints(
    stage1_path="checkpoints/text8/stage1.pt",
    stage2_path="checkpoints/text8/stage2.pt",
    cfg=cfg,
)

# Inference
energy, variance = auditor.per_token_uq(log_x)   # both shape (B, L)
ood = auditor.ood_score(log_x)                    # shape (B,)
```

---

## Ablations

```python
from aitchinson_flow.config import Config
from benchmarks.runner import run_unified_benchmark

cfg = Config()
cfg.benchmark.scale_grid = [
    {"d_model": 64,  "num_layers": 2, "nhead": 4, "velocity_loss": "soft_hilbert"},
    {"d_model": 128, "num_layers": 4, "nhead": 8, "velocity_loss": "soft_hilbert"},
    {"d_model": 128, "num_layers": 4, "nhead": 8, "velocity_loss": "clr_mse"},
]
results = run_unified_benchmark(cfg)
```

Supported ablation keys: `velocity_loss`, `transform_mode` (`ilr` / `clr`), `label_smoothing`, `model_name`.

---

## Tests

```bash
pytest -q
```

Coverage includes geometry, transforms, all datamodules, GP, spilled energy, calibration, benchmark schema, two-stage orchestration, ablation plumbing, and unified benchmark output.

---

## CLI Entry Point

```bash
bench-scaling   # runs benchmarks/runner:main — scale-grid AUROC sweep
```

---

## License

MIT (see `pyproject.toml`).
