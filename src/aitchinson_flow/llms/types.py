from __future__ import annotations

from typing import Protocol

import torch


class CausalLMForInference(Protocol):
    """Frozen causal LM: token ids input → logits output."""

    @property
    def device(self) -> torch.device: ...

    def forward_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits (B, L, K) for given input ids. Must be called inside inference_mode."""
        ...

    def encode_text(
        self, text: str, *, max_length: int
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Single string → (1, L) ids, mask; for cache loops."""
        ...

    def generate_ids(
        self,
        *,
        batch_size: int,
        max_new_tokens: int,
        prompt_ids: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """Generate ids (B, L) for given prompt ids and attention mask. Must be called inside inference_mode."""
        ...
