"""Unit tests for compositional geometry primitives.

Covers the orthonormal Helmert basis used by the ILR transform and basic
round-trip invariants. These tests directly exercise the fix for review
item C3 (incorrect Helmert scaling).
"""

from __future__ import annotations

import math

import pytest
import torch

from aitchinson_flow.geometry import _helmert_matrix, ilr, ilr_inv


class TestHelmertMatrix:
    @pytest.mark.parametrize("K", [2, 3, 5, 27, 64])
    def test_columns_are_orthonormal(self, K: int) -> None:
        psi = _helmert_matrix(K, device=torch.device("cpu"), dtype=torch.float64)
        gram = psi.T @ psi
        identity = torch.eye(K - 1, dtype=torch.float64)
        assert torch.allclose(gram, identity, atol=1e-10), (
            f"Helmert basis must be orthonormal for K={K}; max deviation "
            f"{(gram - identity).abs().max().item():.3e}"
        )

    @pytest.mark.parametrize("K", [3, 5, 27])
    def test_columns_sum_to_zero(self, K: int) -> None:
        """Every ILR basis vector lies in the centered hyperplane."""
        psi = _helmert_matrix(K, device=torch.device("cpu"), dtype=torch.float64)
        col_sums = psi.sum(dim=0)
        assert torch.allclose(
            col_sums, torch.zeros_like(col_sums), atol=1e-12
        ), f"column sums must be zero for K={K}, got {col_sums}"

    def test_small_example_matches_closed_form(self) -> None:
        """K=3 should match the textbook 3-part Helmert contrast."""
        psi = _helmert_matrix(3, device=torch.device("cpu"), dtype=torch.float64)
        expected_col0 = torch.tensor([1.0 / math.sqrt(2), -1.0 / math.sqrt(2), 0.0])
        expected_col1 = torch.tensor(
            [1.0 / math.sqrt(6), 1.0 / math.sqrt(6), -2.0 / math.sqrt(6)]
        )
        assert torch.allclose(psi[:, 0].double(), expected_col0.double(), atol=1e-12)
        assert torch.allclose(psi[:, 1].double(), expected_col1.double(), atol=1e-12)


class TestILRRoundTrip:
    @pytest.mark.parametrize("K", [3, 5, 27])
    def test_ilr_inv_roundtrip(self, K: int) -> None:
        torch.manual_seed(0)
        y = torch.randn(8, K - 1, dtype=torch.float64)
        log_x = ilr_inv(y, K=K)
        y_recovered = ilr(log_x)
        assert torch.allclose(y, y_recovered, atol=1e-10), (
            f"ilr(ilr_inv(y)) should recover y for K={K}; max deviation "
            f"{(y - y_recovered).abs().max().item():.3e}"
        )

    @pytest.mark.parametrize("K", [3, 5, 27])
    def test_ilr_is_isometry(self, K: int) -> None:
        """Euclidean distance in ILR space equals Aitchison distance in CLR."""
        torch.manual_seed(1)
        log_x = torch.randn(4, K, dtype=torch.float64)
        log_y = torch.randn(4, K, dtype=torch.float64)
        clr_x = log_x - log_x.mean(dim=-1, keepdim=True)
        clr_y = log_y - log_y.mean(dim=-1, keepdim=True)
        aitch_dist = (clr_x - clr_y).norm(dim=-1)
        ilr_dist = (ilr(log_x) - ilr(log_y)).norm(dim=-1)
        assert torch.allclose(aitch_dist, ilr_dist, atol=1e-10), (
            f"ILR must be an isometry for K={K}; max deviation "
            f"{(aitch_dist - ilr_dist).abs().max().item():.3e}"
        )
