"""EqM training entry point — text8 character-level flow matching."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path


def bootstrap_repo_paths(anchor: Path) -> Path:
    """Add ``src`` and repo root to ``sys.path`` for direct script execution."""
    repo_root = anchor.resolve().parent
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))
    return repo_root


_REPO_ROOT = bootstrap_repo_paths(Path(__file__))

import aitchinson_flow.models  # noqa: E402,F401  — populate model REGISTRY
from aitchinson_flow.config import Config
from aitchinson_flow.training import (
    build_training_datamodule,
    seed_all,
    fit,
    evaluate,
    build_wandb_logger,
)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=str, default="checkpoints")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument(
        "--dirichlet",
        action="store_true",
        help="train with Dirichlet-sampled CLR data (Phase 1+).",
    )
    p.add_argument("--alpha-peak", type=float, default=None,
                   help="Dirichlet target-class concentration (override config default).")
    p.add_argument("--alpha-base", type=float, default=None,
                   help="Dirichlet off-target concentration (override config default).")
    p.add_argument(
        "--latent",
        action="store_true",
        help="train EqMLatent (learned-embedding EqM) instead of simplex EqM.",
    )
    p.add_argument("--d-embed", type=int, default=None,
                   help="EqMLatent embedding dim (overrides cfg.embedding.d_embed).")
    p.add_argument("--loss-mode", type=str, default=None,
                   choices=("mse", "hilbert", "hilbert_soft", "hilbert_soft_softmax"),
                   help="Override cfg.loss.mode (Phase 5 ablation).")
    p.add_argument("--gradient-lambda", type=float, default=None,
                   help="Override cfg.eqm.gradient_lambda (Phase 4 retune).")
    p.add_argument("--gamma-power", type=float, default=None,
                   help="Override cfg.eqm.gamma_power (Phase 4 retune).")
    p.add_argument("--decay-strategy", type=str, default=None,
                   choices=("linear", "truncated", "piecewise"),
                   help="Override cfg.eqm.decay_strategy (Phase 4 retune).")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--wandb",
        action="store_true",
        help="enable W&B logging (also enabled if WANDB_PROJECT env is set)",
    )
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-name", type=str, default=None)
    p.add_argument(
        "--wandb-tags",
        type=str,
        default=None,
        help="comma-separated tags",
    )
    return p


def main(argv: list[str] | None = None):
    args = _build_argparser().parse_args(argv)

    cfg = Config()
    overrides = {"checkpoint_dir": args.out_dir}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.lr is not None:
        overrides["lr"] = args.lr
    if args.seed is not None:
        overrides["seed"] = args.seed
    cfg.training = replace(cfg.training, **overrides)

    transform_overrides: dict[str, object] = {}
    if args.dirichlet:
        transform_overrides["dirichlet_sampling"] = True
    if args.alpha_peak is not None:
        transform_overrides["dirichlet_alpha_peak"] = args.alpha_peak
    if args.alpha_base is not None:
        transform_overrides["dirichlet_alpha_base"] = args.alpha_base
    if transform_overrides:
        cfg.transformation = replace(cfg.transformation, **transform_overrides)

    if args.loss_mode is not None:
        cfg.loss = replace(cfg.loss, mode=args.loss_mode)

    if args.latent:
        cfg.training = replace(cfg.training, model_name="EqMLatent")
        embed_overrides: dict[str, object] = {"enabled": True}
        if args.d_embed is not None:
            embed_overrides["d_embed"] = args.d_embed
        cfg.embedding = replace(cfg.embedding, **embed_overrides)

    eqm_overrides: dict[str, object] = {}
    if args.gradient_lambda is not None:
        eqm_overrides["gradient_lambda"] = args.gradient_lambda
    if args.gamma_power is not None:
        eqm_overrides["gamma_power"] = args.gamma_power
    if args.decay_strategy is not None:
        eqm_overrides["decay_strategy"] = args.decay_strategy
    if eqm_overrides:
        cfg.eqm = replace(cfg.eqm, **eqm_overrides)

    wandb_enabled = args.wandb or bool(os.environ.get("WANDB_PROJECT"))
    if wandb_enabled:
        run_name = (
            args.wandb_name
            or os.environ.get("WANDB_NAME")
            or f"{cfg.training.model_name}_{datetime.now():%Y%m%d-%H%M%S}"
        )
        tags = (
            tuple(t.strip() for t in args.wandb_tags.split(",") if t.strip())
            if args.wandb_tags
            else ()
        )
        wandb_overrides = {"enabled": True, "run_name": run_name, "tags": tags}
        if args.wandb_project is not None:
            wandb_overrides["project"] = args.wandb_project
        cfg.wandb = replace(cfg.wandb, **wandb_overrides)

    Path(cfg.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    seed_all(cfg.training.seed)

    datamodule, _ = build_training_datamodule(cfg)
    wandb_logger = build_wandb_logger(cfg, run_dir=Path(cfg.training.checkpoint_dir))
    model = fit(cfg=cfg, datamodule=datamodule, wandb_logger=wandb_logger)

    val_loader = datamodule.val_dataloader()
    if val_loader is not None:
        evaluate(model=model, loader=val_loader, device=cfg.training.device)


if __name__ == "__main__":
    main()
