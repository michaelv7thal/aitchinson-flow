"""Quick smoke run for Dirichlet Flow Matching (Stark et al. 2024).

Trains a small DirichletFM on a slice of text8 and prints a per-epoch report:
training/eval CE loss, unigram/bigram/trigram KL against the train corpus,
and a few decoded samples. Used to verify end-to-end correctness.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

import aitchinson_flow.models  # noqa: E402,F401  — populate registry
from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.training import (  # noqa: E402
    build_training_datamodule,
    seed_all,
    fit,
)


def main() -> None:
    cfg = Config()

    # Model selection
    cfg.training = replace(
        cfg.training,
        model_name="DirichletFM",
        epochs=3,
        lr=3e-4,
        B=64,
        L=64,
        eval_every=1,
        sample_eval_every=1,
        sample_eval_n=64,
        sample_eval_steps=100,  # NFE for the sampler
        checkpoint_every=999,  # don't save checkpoints
        checkpoint_dir=str(REPO / "runs" / "dirichlet_fm_smoke"),
        log_every=20,
        seed=0,
    )

    # Smaller transformer to keep this brisk
    cfg.transformer = replace(
        cfg.transformer,
        d_model=256,
        nhead=4,
        num_layers=4,
        d_latent=256,
        dropout=0.0,
    )

    # Smaller dataset slice
    cfg.text8_dataset = replace(
        cfg.text8_dataset,
        L=cfg.training.L,
        batch_size=cfg.training.B,
        max_train_windows=20_000,
        max_eval_windows=2_000,
    )

    cfg.dirichlet_fm = replace(
        cfg.dirichlet_fm,
        t_max=8.0,        # paper default α_max
        sample_nfe=100,   # paper default
    )

    # Disable W&B for smoke
    cfg.wandb = replace(cfg.wandb, enabled=False)

    Path(cfg.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    seed_all(cfg.training.seed)

    datamodule, _ = build_training_datamodule(cfg)
    history: list[dict[str, float]] = []
    fit(cfg=cfg, datamodule=datamodule, history_out=history)

    print("\n==== history ====")
    for i, h in enumerate(history):
        print(f"epoch {i + 1}:")
        for k in sorted(h):
            print(f"  {k}: {h[k]:.4f}")


if __name__ == "__main__":
    main()
