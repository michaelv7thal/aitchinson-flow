"""Unit tests for Path B's learned projection head."""

from __future__ import annotations

import pytest
import torch

from aitchinson_flow.models.llm_projection import TokenEmbeddingToSimplex


class TestShapeAndMode:
    def test_ilr_shape(self) -> None:
        proj = TokenEmbeddingToSimplex(llm_embed_dim=8, K=6, transform_mode="ilr")
        out = proj(torch.randn(2, 5, 8))
        assert out.shape == (2, 5, 5)
        assert torch.isfinite(out).all()

    def test_clr_shape(self) -> None:
        proj = TokenEmbeddingToSimplex(llm_embed_dim=8, K=6, transform_mode="clr")
        out = proj(torch.randn(2, 5, 8))
        assert out.shape == (2, 5, 6)
        # CLR rows sum to zero by construction.
        assert torch.allclose(out.sum(dim=-1), torch.zeros(2, 5), atol=1e-5)

    def test_rejects_2d_input(self) -> None:
        proj = TokenEmbeddingToSimplex(llm_embed_dim=4, K=3)
        with pytest.raises(ValueError, match="embeddings must be 3D"):
            proj(torch.randn(4, 4))

    def test_rejects_invalid_transform_mode(self) -> None:
        with pytest.raises(ValueError, match="transform_mode"):
            TokenEmbeddingToSimplex(llm_embed_dim=4, K=3, transform_mode="pca")


class TestDeterminism:
    def test_same_embeddings_same_output(self) -> None:
        """Deterministic by construction — same input → same output."""
        torch.manual_seed(0)
        proj = TokenEmbeddingToSimplex(llm_embed_dim=8, K=5)
        x = torch.randn(2, 3, 8)
        out_a = proj(x)
        out_b = proj(x.clone())
        assert torch.allclose(out_a, out_b)

    def test_same_token_same_feature_across_positions(self) -> None:
        """Identical embeddings at different positions → identical features.

        This is the property top-K sampling lacked: a given token should map
        to the same simplex point everywhere it appears.
        """
        proj = TokenEmbeddingToSimplex(llm_embed_dim=8, K=5)
        emb = torch.randn(1, 1, 8)
        # Tile the same embedding across L positions.
        tiled = emb.expand(1, 4, 8).contiguous()
        out = proj(tiled)
        for t in range(1, out.shape[1]):
            assert torch.allclose(out[0, 0], out[0, t])


class TestGradientFlow:
    def test_gradient_reaches_projection_weight(self) -> None:
        proj = TokenEmbeddingToSimplex(llm_embed_dim=6, K=4)
        embeddings = torch.randn(2, 3, 6, requires_grad=False)
        out = proj(embeddings)
        out.sum().backward()
        assert proj.proj.weight.grad is not None
        assert torch.isfinite(proj.proj.weight.grad).all()
        # Some non-zero signal made it back.
        assert proj.proj.weight.grad.abs().sum() > 0.0
