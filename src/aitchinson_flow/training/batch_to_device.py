from __future__ import annotations

from typing import Any

import torch


def to_device(batch: Any, device: torch.device) -> Any:
    """Move a tensor / dict-of-tensors batch onto ``device``.

    Non-tensor values pass through unchanged so callers can attach
    arbitrary metadata (e.g. tags, masks, booleans) to batches
    without a special case.
    """
    if isinstance(batch, dict):
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

    if torch.is_tensor(batch):
        return batch.to(device)
    return batch
