import argparse
import sys

from copy import deepcopy
from pathlib import Path
from dataclasses import replace
from typing import Any


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
    DataModule,
    fit,
    evaluate,
)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=str, default="checkpoints/manual/stage1")
    p.add_argument("--stage1-epochs", type=int, default=25)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Use a tiny CPU config for end-to-end smoke testing.",
    )
    p.add_argument("--lm-key", type=str, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--seq-length", type=int, default=None)
    p.add_argument("--char-window-length", type=int, default=None)
    p.add_argument("--corrupt-rate", type=float, default=None)
    p.add_argument("--no-renormalize", action="store_true")

    return p


def _build_training_datamodule(cfg: Config) -> tuple[DataModule, dict[str, Any]]:
    """Build the configured training data source (raw text or LLM-generated)."""
    return build_training_datamodule(cfg)


def _set_config(base: Config, *, epochs: int, ckpt_dir: Path) -> Config:
    cfg = deepcopy(base)
    cfg.training = replace(
        cfg.training,
        model_name="bayesian_auditor_stage1",
        epochs=epochs,
        checkpoint_dir=str(ckpt_dir),
    )
    return cfg


def main(argv: list[str] | None = None):
    args = _build_argparser().parse_args(argv)

    cfg = Config()

    datamodule, _ = _build_training_datamodule(cfg)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_cfg = _set_config(
        cfg, epochs=args.stage1_epochs, ckpt_dir=out_dir / "stage1_ckpts"
    )
    seed_all(set_cfg.training.seed)

    model = fit(cfg=cfg, datamodule=datamodule)

    val_dataloader = datamodule.val_dataloader()

    if val_dataloader is not None:
        evals = evaluate(model=model, loader=val_dataloader, device=cfg.training.device)


if __name__ == "__main__":
    main()
