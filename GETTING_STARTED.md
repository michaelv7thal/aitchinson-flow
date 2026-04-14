# Getting Started

End-to-end audit pipeline on **text8**: train a Bayesian auditor on valid sequences, evaluate on valid + corrupted, and compare against the training-free spilled-energy baseline via AUROC and plots.

## 1. Install

```bash
uv sync                       # resolve from pyproject.toml
uv sync --extra benchmarks    # for the benchmark sweep (transformers / datasets / sklearn)
```

## 2. Run the tests

```bash
uv run pytest -q
```

## 3. Run the benchmark on text8

Create `run_bench.py`:

```python
from aitchinson_flow.config import Config
from benchmarks.runner import run_benchmark

cfg = Config()

# --- data ---
cfg.benchmark.data_source = "text8"    # char-level text8 with real splits
cfg.text8_dataset.split_ratios = (0.9, 0.05, 0.05)
cfg.text8_dataset.train_corrupt_rate = 0.15   # contrastive training
cfg.text8_dataset.eval_corrupt_rate = 0.30    # sharper gap at eval
cfg.text8_dataset.max_train_windows = 20_000
cfg.text8_dataset.max_eval_windows = 2_000

# --- model ---
cfg.training.model_name = "bayesian_auditor"   # or "per_token_bayesian_auditor"
cfg.dataset.K = 27     # text8 alphabet (26 letters + space)
cfg.dataset.L = 64     # window length

# --- sweep + training ---
cfg.benchmark.scale_grid = [
    {"d_model": 128, "num_layers": 4, "nhead": 8},
]
cfg.benchmark.train_before_eval = True
cfg.benchmark.train_epochs = 10
cfg.benchmark.save_plots = True

results = run_benchmark(cfg)
print(results)
```

```bash
uv run python run_bench.py
```

## 4. Where to find outputs

```
results/benchmark/
├── benchmark_latest.json              # per-scale metrics (loss, AUROC, separation…)
└── <scale_tag>/                       # e.g. d_model128_nhead8_num_layers4/
    ├── roc.png                        # ROC: auditor vs spilled energy
    ├── hist_auditor.png               # valid vs invalid GP variance
    ├── hist_spilled.png               # valid vs invalid spilled-energy score
    ├── sequence_spilled.png           # per-position spilled ΔE mean ± std
    └── loss_curve.png                 # per-epoch training losses
```

Key metrics in `benchmark_latest.json`:

| Metric | Meaning |
|---|---|
| `auroc_auditor` | AUROC of GP variance separating valid vs corrupted |
| `auroc_spilled` | AUROC of −mean(ΔE) separating valid vs corrupted |
| `spilled_separation` | mean-anomaly gap (invalid − valid) |
| `loss`, `flow_loss`, `mean_loss`, `var_loss` | eval-time model losses |

## 5. Swap out the dataset

- **LM-generated synthetic (original path):** `cfg.benchmark.data_source = "lm_teacher"`; optionally `cfg.benchmark.use_text8_prompts = True` to seed GPT-2 with text8 prefixes.
- **Custom HF dataset:** set `cfg.hf_dataset.enabled = True` and point `cfg.hf_dataset.path` at a dataset with an `input_ids` column — uses [`HFDataModule`](src/aitchinson_flow/data/transforms/hf_datamodule.py).

## 6. Swap the model

Registered under [`models/factory.py`](src/aitchinson_flow/models/factory.py):

- `flow_matching` — velocity-only baseline (no contrast / OOD score).
- `bayesian_generator` — GP energy, no contrastive loss.
- `bayesian_auditor` — contrastive valid/invalid hinge on GP mean & variance.
- `per_token_bayesian_auditor` — per-position GP (optionally conditioned on teacher hidden states).

Set `cfg.training.model_name` accordingly. Both auditor variants expose `score_per_sample(log_x) → (B,)` used for AUROC.

## 7. Architecture reference

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full data / model / runner diagram.
