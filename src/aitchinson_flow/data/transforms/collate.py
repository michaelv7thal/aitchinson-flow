"""Stack per-sample dict batches for ``DataLoader``."""

from __future__ import annotations

from typing import Any

import torch


def collate_tensor_dict(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate empty batch")

    out: dict[str, Any] = {}

    for k in batch[0]:
        v0 = batch[0][k]

        if isinstance(v0, torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        else:
            out[k] = [b[k] for b in batch]

    return out
