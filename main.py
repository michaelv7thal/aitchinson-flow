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
    cfg.training = replace(cfg.training, **overrides)

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
