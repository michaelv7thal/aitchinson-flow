"""Path B learned projection: frozen LLM embeddings → simplex coordinates.

The Path B datamodule emits per-token input embeddings from a frozen LLM
(``(B, L, d_embed)``). Those embeddings carry content, but they do **not**
live on the simplex — the Bayesian auditor's geometry requires ILR/CLR rows.

This module is a minimal trainable bridge:

    embeddings → Linear(d_embed → K) → log_softmax → ILR/CLR → (B, L, K-1)

Because the head is deterministic (no sampling, no dropout), the same
``token_id`` maps to the same simplex point regardless of context. That is
the key property Path B's earlier sorted-top-K feature lacked.

Training: the projection weights train jointly with the Stage 1 backbone.
Stage 2 freezes the projection alongside the backbone (contrastive GP never
updates it). ``compose_auditor_from_stages`` copies ``llm_projection.*`` from
the Stage 1 checkpoint the same way it copies ``backbone.*``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from aitchinson_flow.data.transforms.discrete import project_log_to_simplex_features


class TokenEmbeddingToSimplex(nn.Module):
    """Map frozen LLM token embeddings to ILR/CLR simplex coordinates.

    Args:
        llm_embed_dim: Input embedding dim (``d_embed``) reported by the LLM.
        K: Simplex output dimension — same as ``cfg.dataset.K`` elsewhere in
            the codebase. Output shape is ``(..., K-1)`` for ILR or ``(..., K)``
            for CLR.
        transform_mode: ``"ilr"`` (default) or ``"clr"``. Matches
            ``cfg.hf_dataset.transform_mode`` so the backbone's input contract
            is preserved across Path A and Path B.
    """

    def __init__(
        self,
        llm_embed_dim: int,
        K: int,
        *,
        transform_mode: str = "ilr",
    ) -> None:
        super().__init__()
        if llm_embed_dim < 1:
            raise ValueError(f"llm_embed_dim must be >= 1, got {llm_embed_dim}")
        if K < 2:
            raise ValueError(f"K must be >= 2 (simplex needs >=2 components), got {K}")
        self.proj = nn.Linear(llm_embed_dim, K)
        self._transform_mode = transform_mode.lower()
        if self._transform_mode not in ("ilr", "clr"):
            raise ValueError(
                f"transform_mode must be 'ilr' or 'clr', got {transform_mode!r}"
            )

    @property
    def transform_mode(self) -> str:
        return self._transform_mode

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """``(B, L, d_embed)`` embeddings → ``(B, L, K-1)`` ILR (or ``(B, L, K)`` CLR).

        The projection is deterministic by construction — no dropout, no
        sampling — so identical ``token_id`` sequences produce identical
        features.
        """
        if embeddings.ndim != 3:
            raise ValueError(
                f"embeddings must be 3D (B, L, d_embed), got shape {tuple(embeddings.shape)}"
            )
        logits = self.proj(embeddings)
        log_p = F.log_softmax(logits, dim=-1)
        return project_log_to_simplex_features(log_p, mode=self._transform_mode)
