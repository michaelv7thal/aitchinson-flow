from aitchinson_flow.config import Config
from benchmarks.runner import run_benchmark

cfg = Config()

# --- data ---
cfg.benchmark.data_source = "text8"  # char-level text8 with real splits
cfg.text8_dataset.train_corrupt_rate = 0.15  # contrastive training
cfg.text8_dataset.eval_corrupt_rate = 0.30  # sharper gap at eval
cfg.text8_dataset.max_train_windows = 20_000
cfg.text8_dataset.max_eval_windows = 2_000

# --- model ---
cfg.training.model_name = "bayesian_auditor"  # or "per_token_bayesian_auditor"
cfg.dataset.K = 27  # text8 alphabet (26 letters + space)
cfg.dataset.L = 64  # window length

# --- sweep + training ---
cfg.benchmark.scale_grid = [
    {"d_model": 128, "num_layers": 4, "nhead": 8},
    {"d_model": 256, "num_layers": 6, "nhead": 8},
    {"d_model": 512, "num_layers": 8, "nhead": 8},
]
cfg.benchmark.train_before_eval = True
cfg.benchmark.train_epochs = 20
cfg.benchmark.save_plots = True

results = run_benchmark(cfg)
print(results)
