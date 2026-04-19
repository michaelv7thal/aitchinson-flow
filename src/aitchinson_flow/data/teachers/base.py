from typing import Any, Protocol
import torch


class TeacherBackend(Protocol):
    def __call__(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> dict[str, torch.Tensor]:
        """Return dict of CPU or same-device tensors; keys depend on task."""
        ...
