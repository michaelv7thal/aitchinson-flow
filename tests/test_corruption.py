"""Tests for benchmark corruption utilities."""

from __future__ import annotations

import torch
import pytest

from benchmarks.corruption import build_invalid_batch, corrupt_token_ids, partially_shuffle_token_ids


class TestCorruptTokenIds:
    def test_shape_preserved(self) -> None:
        ids = torch.randint(0, 100, (8, 20))
        out = corrupt_token_ids(ids, vocab_size=100, corrupt_rate=0.5)
        assert out.shape == ids.shape
        assert out.dtype == ids.dtype

    def test_some_tokens_changed(self) -> None:
        torch.manual_seed(0)
        ids = torch.zeros(4, 50, dtype=torch.long)
        out = corrupt_token_ids(ids, vocab_size=100, corrupt_rate=0.5, seed=0)
        assert (out != ids).any(), "at least some tokens should change"

    def test_zero_rate_no_change(self) -> None:
        ids = torch.randint(0, 100, (4, 20))
        out = corrupt_token_ids(ids, vocab_size=100, corrupt_rate=0.0, seed=0)
        torch.testing.assert_close(out, ids)

    def test_full_rate_all_changed(self) -> None:
        ids = torch.zeros(4, 50, dtype=torch.long)
        out = corrupt_token_ids(ids, vocab_size=100, corrupt_rate=1.0, seed=0)
        # With vocab_size=100 and all ids=0, almost all should change
        changed = (out != ids).float().mean()
        assert changed > 0.95, f"expected nearly all changed, got {changed:.2f}"

    def test_values_in_range(self) -> None:
        V = 50
        ids = torch.randint(0, V, (8, 20))
        out = corrupt_token_ids(ids, vocab_size=V, corrupt_rate=0.5, seed=42)
        assert (out >= 0).all() and (out < V).all()

    def test_deterministic_with_seed(self) -> None:
        ids = torch.randint(0, 100, (4, 20))
        out1 = corrupt_token_ids(ids, vocab_size=100, corrupt_rate=0.3, seed=123)
        out2 = corrupt_token_ids(ids, vocab_size=100, corrupt_rate=0.3, seed=123)
        torch.testing.assert_close(out1, out2)


class TestBuildInvalidBatch:
    def test_adds_expected_keys(self) -> None:
        B, L, V, K = 4, 10, 50, 27
        batch: dict[str, torch.Tensor] = {
            "log_x": torch.randn(B, L, K),
            "token_ids": torch.randint(0, V, (B, L)),
            "logits": torch.randn(B, L, V),
        }
        out = build_invalid_batch(batch, K=K, corrupt_rate=0.3, seed=0)
        assert "log_x_invalid" in out
        assert "token_ids_invalid" in out
        assert "logits_invalid" in out
        assert out["log_x_invalid"].shape == (B, L, K - 1)
        assert out["token_ids_invalid"].shape == (B, L)
        assert out["logits_invalid"].shape == (B, L, V)

    def test_invalid_differs_from_valid(self) -> None:
        B, L, V, K = 2, 20, 100, 27
        batch: dict[str, torch.Tensor] = {
            "log_x": torch.randn(B, L, K),
            "token_ids": torch.randint(0, V, (B, L)),
            "logits": torch.randn(B, L, V),
        }
        build_invalid_batch(batch, K=K, corrupt_rate=0.5, seed=0)
        assert not torch.equal(batch["token_ids"], batch["token_ids_invalid"])


class TestPartialShuffle:
    def test_shape_preserved(self) -> None:
        ids = torch.randint(0, 10, (3, 12))
        out = partially_shuffle_token_ids(ids, shuffle_rate=0.5, seed=123)
        assert out.shape == ids.shape

    def test_zero_rate_no_change(self) -> None:
        ids = torch.randint(0, 10, (3, 12))
        out = partially_shuffle_token_ids(ids, shuffle_rate=0.0, seed=123)
        torch.testing.assert_close(out, ids)
