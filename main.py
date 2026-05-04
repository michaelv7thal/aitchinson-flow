"""EqM training entry point — text8 character-level flow matching."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
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
)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=str, default="checkpoints")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
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

    Path(cfg.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    seed_all(cfg.training.seed)

    datamodule, _ = build_training_datamodule(cfg)
    model = fit(cfg=cfg, datamodule=datamodule)

    val_loader = datamodule.val_dataloader()
    if val_loader is not None:
        evaluate(model=model, loader=val_loader, device=cfg.training.device)


if __name__ == "__main__":
    main()
