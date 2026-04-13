"""Tests for spilled-energy metric utilities."""

from __future__ import annotations

import torch
import pytest

from aitchinson_flow.metrics.spilled_energy import (
    compute_spilled_energy,
    compute_spilled_energy_batch,
    hard_negative_ids,
    marginal_energy,
    sequence_anomaly_score,
)


class TestMarginalEnergy:
    def test_shape(self) -> None:
        logits = torch.randn(10, 50)
        out = marginal_energy(logits)
        assert out.shape == (10,)

    def test_uniform_logits(self) -> None:
        V = 100
        logits = torch.zeros(5, V)
        out = marginal_energy(logits)
        expected = -torch.tensor(V, dtype=torch.float32).log()
        torch.testing.assert_close(out, expected.expand(5), atol=1e-5, rtol=1e-5)

    def test_always_nonpositive(self) -> None:
        logits = torch.randn(20, 30)
        out = marginal_energy(logits)
        assert (out <= 1e-6).all(), "marginal energy should be <= 0"


class TestComputeSpilledEnergy:
    def test_output_shape(self) -> None:
        L, V = 10, 50
        logits = torch.randn(L, V)
        token_ids = torch.randint(0, V, (L,)).tolist()
        out = compute_spilled_energy(logits, token_ids)
        assert out.shape == (L - 1,)

    def test_perfect_prediction(self) -> None:
        """When the LM assigns all mass to the true next token, spilled ~ 0."""
        L, V = 5, 10
        token_ids = [3, 7, 1, 9, 2]
        logits = torch.full((L, V), -100.0)
        for i in range(L - 1):
            logits[i, token_ids[i + 1]] = 100.0
        out = compute_spilled_energy(logits, token_ids)
        assert out.shape == (L - 1,)
        assert (out > -1.0).all(), "spilled should be near 0 for perfect prediction"

    def test_wrong_prediction_is_more_negative(self) -> None:
        """When the LM is confident in the wrong token, spilled is highly negative."""
        L, V = 3, 10
        token_ids_good = [0, 5, 3]
        token_ids_bad = [0, 8, 3]

        logits = torch.full((L, V), -10.0)
        logits[0, 5] = 10.0  # position 0 predicts token 5 strongly
        logits[1, 3] = 10.0

        sp_good = compute_spilled_energy(logits, token_ids_good)
        sp_bad = compute_spilled_energy(logits, token_ids_bad)
        assert sp_bad[0] < sp_good[0], "wrong token should give more negative spilled energy"

    def test_length_one_sequence(self) -> None:
        logits = torch.randn(1, 10)
        out = compute_spilled_energy(logits, [5])
        assert out.shape == (0,)


class TestSequenceAnomalyScore:
    def test_sign_convention(self) -> None:
        spilled_negative = torch.tensor([-5.0, -3.0, -4.0])
        spilled_positive = torch.tensor([1.0, 2.0, 3.0])
        assert sequence_anomaly_score(spilled_negative) > sequence_anomaly_score(spilled_positive)


class TestComputeSpilledEnergyBatch:
    def test_matches_unbatched(self) -> None:
        B, L, V = 4, 8, 20
        logits = torch.randn(B, L, V)
        ids = torch.randint(0, V, (B, L))
        batch_out = compute_spilled_energy_batch(logits, ids)
        assert batch_out.shape == (B, L - 1)

        for b in range(B):
            single = compute_spilled_energy(logits[b], ids[b].tolist())
            torch.testing.assert_close(batch_out[b], single, atol=1e-5, rtol=1e-5)

    def test_gradient_flows(self) -> None:
        logits = torch.randn(2, 5, 10, requires_grad=True)
        ids = torch.randint(0, 10, (2, 5))
        out = compute_spilled_energy_batch(logits, ids)
        out.sum().backward()
        assert logits.grad is not None


class TestHardNegativeIds:
    def test_position_zero_unchanged(self) -> None:
        logits = torch.randn(5, 10)
        token_ids = [3, 7, 1, 9, 2]
        hard = hard_negative_ids(logits, token_ids)
        assert hard[0] == token_ids[0]

    def test_all_positions_changed(self) -> None:
        """When true token is NOT in top-k, all positions 1..L-1 should change."""
        V = 100
        L = 5
        token_ids = [99] * L  # token 99 — unlikely to be in top-5
        logits = torch.zeros(L, V)
        logits[:, :5] = 10.0  # top-5 are tokens 0..4
        hard = hard_negative_ids(logits, token_ids, top_k=5)
        for i in range(1, L):
            assert hard[i] != token_ids[i], f"position {i} should be changed"
            assert hard[i] in range(5), f"hard neg should be in top-k"

    def test_output_length(self) -> None:
        logits = torch.randn(8, 20)
        token_ids = torch.randint(0, 20, (8,)).tolist()
        hard = hard_negative_ids(logits, token_ids)
        assert len(hard) == len(token_ids)
