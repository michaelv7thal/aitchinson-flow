"""Beam search reranking with energy penalty (Phase 4, Strategy 1).

Reranking score (per candidate):
    score(candidate) = log_p(candidate | context) − λ · mean_energy(candidate)

Candidates with high structural or contextual energy are penalized; the
candidate with the highest combined score is selected.

Usage — two entry points:

1. ``BeamRerankScorer.score_candidates(log_x_list, log_p_list)``
   Given pre-generated beam candidates as ILR tensors + their LM log-probs,
   returns the per-candidate combined scores and the index of the best.

2. ``BeamRerankScorer.rerank_batch(log_x_batch, log_p_batch)``
   Operates on a 3-D batch ``(B, n_beams, L, D)`` and ``(B, n_beams)`` log-probs,
   returns the best candidate per sequence.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class BeamRerankResult:
    """Output of :meth:`BeamRerankScorer.score_candidates`."""

    scores: torch.Tensor          # (n_beams,) or (B, n_beams) combined scores
    best_idx: torch.Tensor        # () or (B,) index of best candidate per sequence
    best_log_x: torch.Tensor      # (B, L, D) or (L, D) best-scoring ILR sequence
    energy: torch.Tensor          # (n_beams,) or (B, n_beams) auditor energy per candidate
    log_p: torch.Tensor           # (n_beams,) or (B, n_beams) LM log-probs (input echo)


class BeamRerankScorer:
    """Rerank beam candidates using ``log_p − λ · energy``.

    Args:
        auditor: Model with ``ood_score(log_x) → (B,)`` sequence-level energy.
        lambda_energy: Penalty weight λ; higher values penalize anomalous
            candidates more aggressively.
    """

    def __init__(
        self,
        auditor: nn.Module,
        lambda_energy: float = 1.0,
    ) -> None:
        if lambda_energy < 0.0:
            raise ValueError(
                f"lambda_energy must be >= 0 (negative values reward anomalous sequences), "
                f"got {lambda_energy}"
            )
        self.auditor = auditor
        self.lambda_energy = lambda_energy

    @torch.no_grad()
    def score_candidates(
        self,
        log_x_list: list[torch.Tensor],
        log_p_list: list[float] | list[torch.Tensor],
    ) -> BeamRerankResult:
        """Score a list of beam candidates for a single sequence.

        Args:
            log_x_list: List of ``n_beams`` ILR tensors, each ``(L, D)`` or ``(1, L, D)``.
            log_p_list: List of ``n_beams`` LM log-probabilities (scalars or 0-d tensors).

        Returns:
            :class:`BeamRerankResult` with scores, best index, and best candidate.
        """
        if len(log_x_list) != len(log_p_list):
            raise ValueError(
                f"log_x_list ({len(log_x_list)}) and log_p_list ({len(log_p_list)}) "
                f"must have the same length."
            )
        if not log_x_list:
            raise ValueError("log_x_list must be non-empty.")

        ood_score_fn = getattr(self.auditor, "ood_score", None)
        if ood_score_fn is None:
            raise AttributeError("auditor must implement ood_score(log_x)")

        # Stack candidates into a single batch for a single forward pass
        device = log_x_list[0].device
        stacked = torch.stack(
            [lx.squeeze(0) if lx.dim() == 3 else lx for lx in log_x_list],
            dim=0,
        )  # (n_beams, L, D)

        energy = ood_score_fn(stacked).detach().cpu()    # (n_beams,)
        log_p = torch.tensor(
            [float(lp) for lp in log_p_list], dtype=torch.float32
        )  # (n_beams,)

        scores = log_p - self.lambda_energy * energy     # (n_beams,)
        best_idx = scores.argmax()                       # scalar

        return BeamRerankResult(
            scores=scores,
            best_idx=best_idx,
            best_log_x=stacked[best_idx.item()],
            energy=energy,
            log_p=log_p,
        )

    @torch.no_grad()
    def rerank_batch(
        self,
        log_x_batch: torch.Tensor,
        log_p_batch: torch.Tensor,
    ) -> BeamRerankResult:
        """Rerank beam candidates for a full batch of sequences.

        Args:
            log_x_batch: ``(B, n_beams, L, D)`` ILR candidates.
            log_p_batch: ``(B, n_beams)`` LM log-probabilities.

        Returns:
            :class:`BeamRerankResult` where ``best_log_x`` is ``(B, L, D)``
            and ``best_idx`` is ``(B,)``.
        """
        if log_x_batch.dim() != 4:
            raise ValueError(
                f"log_x_batch must be 4-D (B, n_beams, L, D), got {tuple(log_x_batch.shape)}"
            )
        if log_p_batch.dim() != 2:
            raise ValueError(
                f"log_p_batch must be 2-D (B, n_beams), got {tuple(log_p_batch.shape)}"
            )
        B, n_beams, L, D = log_x_batch.shape

        ood_score_fn = getattr(self.auditor, "ood_score", None)
        if ood_score_fn is None:
            raise AttributeError("auditor must implement ood_score(log_x)")

        # Flatten to (B*n_beams, L, D), score, reshape to (B, n_beams)
        flat = log_x_batch.view(B * n_beams, L, D)
        energy_flat = ood_score_fn(flat).detach().cpu()          # (B*n_beams,)
        energy = energy_flat.view(B, n_beams)                    # (B, n_beams)

        log_p = log_p_batch.cpu().to(dtype=torch.float32)        # (B, n_beams)
        scores = log_p - self.lambda_energy * energy              # (B, n_beams)
        best_idx = scores.argmax(dim=-1)                          # (B,)

        # Gather best candidate per sequence
        gather_idx = best_idx.view(B, 1, 1, 1).expand(B, 1, L, D)
        best_log_x = log_x_batch.gather(dim=1, index=gather_idx).squeeze(1)  # (B, L, D)

        # Gather best score / energy / log_p for diagnostics
        best_scores = scores.gather(dim=1, index=best_idx.unsqueeze(1)).squeeze(1)
        best_energy = energy.gather(dim=1, index=best_idx.unsqueeze(1)).squeeze(1)
        best_log_p = log_p.gather(dim=1, index=best_idx.unsqueeze(1)).squeeze(1)

        return BeamRerankResult(
            scores=best_scores,
            best_idx=best_idx,
            best_log_x=best_log_x,
            energy=best_energy,
            log_p=best_log_p,
        )
