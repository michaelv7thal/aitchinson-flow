from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from aitchinson_flow.config import Config


def config_checkpoint_dict(cfg: Config) -> dict[str, Any]:
    d = asdict(cfg)
    d["training"]["device"] = str(cfg.training.device)
    return d


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    cfg: Config,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    global_step: int,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "cfg": config_checkpoint_dict(cfg),
        "epoch": epoch,
        "global_step": global_step,
        "optimizer_state_dict": optimizer.state_dict()
        if optimizer is not None
        else None,
    }

    if extra_metadata:
        payload["extra_metadata"] = dict(extra_metadata)

    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device | None = None,
) -> tuple[dict[str, Any], int, int]:
    """Returns (cfg_dict, epoch, global_step). Restore cfg yourself if needed."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["cfg"], int(ckpt["epoch"]), int(ckpt["global_step"])
