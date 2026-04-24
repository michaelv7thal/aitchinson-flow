"""Targeted re-sampling healer (Phase 4, Strategy 2).

Workflow per iteration:
  1. ``auditor.per_token_uq(log_x)`` → per-token energy ``(B, L)``.
  2. Flag positions where energy > threshold.
  3. Run ``healer.integrate()`` on the full batch; accept healed ILR values
     *only* at flagged positions (leaving non-flagged positions unchanged).
  4. Re-evaluate; repeat up to ``max_iter`` times.

When ``token_ids`` and a live LLM are supplied, step 3 uses the LLM's greedy
prediction at each flagged position conditioned on the original left context,
then converts back to ILR coordinates.  This is the full-fidelity path; the
EqM-only path serves as a training-free fallback that works without an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from aitchinson_flow.geometry import ilr_inv


@dataclass
class TargetedResampleResult:
    """Output of :meth:`TargetedResampler.heal`."""

    healed_log_x: torch.Tensor   # (B, L, D) final ILR sequences
    token_mask: torch.Tensor      # (B, L) bool — positions that were updated
    seq_mask: torch.Tensor        # (B,)   bool — sequences with ≥1 updated position
    n_iter: int                   # iterations taken (≤ max_iter)
    pre_energy: torch.Tensor      # (B,) sequence-level OOD score before healing
    post_energy: torch.Tensor     # (B,) sequence-level OOD score after healing
    success_rate: float           # fraction of flagged seqs where energy dropped


class TargetedResampler:
    """Per-position healer: project high-energy token positions toward the valid manifold.

    Args:
        auditor: Any model with ``per_token_uq(log_x) → (energy, variance, noise)``
            and ``ood_score(log_x) → (B,)``.
        healer: Any model with ``integrate(log_x, steps?, dt?) → log_x``.
        threshold: Per-token energy threshold; positions above this are flagged.
        max_iter: Maximum healing iterations.
        steps: EqM integration steps per pass (``None`` → healer default).
        dt: EqM step size (``None`` → healer default).
    """

    def __init__(
        self,
        auditor: nn.Module,
        healer: nn.Module,
        threshold: float = 1.0,
        max_iter: int = 5,
        steps: int | None = None,
        dt: float | None = None,
    ) -> None:
        self.auditor = auditor
        self.healer = healer
        self.threshold = threshold
        self.max_iter = max_iter
        self.steps = steps
        self.dt = dt

    @torch.no_grad()
    def heal(
        self,
        log_x: torch.Tensor,
        *,
        token_ids: torch.Tensor | None = None,
        K: int | None = None,
        lm: Any | None = None,
        eps: float = 1e-8,
        label_smoothing: float = 0.0,
        transform_mode: str = "ilr",
    ) -> TargetedResampleResult:
        """Heal a batch by targeted per-position resampling.

        Args:
            log_x: ``(B, L, D)`` ILR-encoded sequences (D = K−1).
            token_ids: ``(B, L)`` integer token IDs for LLM-based resampling.
            K: Vocabulary size; required when ``token_ids`` is provided.
            lm: Object with ``forward_logits(input_ids) → (B, L, V)``; enables
                LLM-based resampling at flagged positions.
            eps: Label-smoothing epsilon for re-encoding (EqM path only).
            label_smoothing: Label-smoothing coefficient for re-encoding.
            transform_mode: ``"ilr"`` or ``"clr"``.

        Returns:
            :class:`TargetedResampleResult` with healed sequences and diagnostics.
        """
        per_token_uq_fn = getattr(self.auditor, "per_token_uq", None)
        ood_score_fn = getattr(self.auditor, "ood_score", None)
        integrate_fn = getattr(self.healer, "integrate", None)
        if per_token_uq_fn is None:
            raise AttributeError("auditor must implement per_token_uq(log_x)")
        if ood_score_fn is None:
            raise AttributeError("auditor must implement ood_score(log_x)")
        if integrate_fn is None:
            raise AttributeError("healer must implement integrate(log_x, ...)")

        pre_energy: torch.Tensor = ood_score_fn(log_x).detach()
        log_x_healed = log_x.clone()
        cumulative_mask = torch.zeros(log_x.shape[:2], dtype=torch.bool, device=log_x.device)

        integrate_kw: dict[str, Any] = {}
        if self.steps is not None:
            integrate_kw["steps"] = self.steps
        if self.dt is not None:
            integrate_kw["dt"] = self.dt

        n_iter = 0
        for _ in range(self.max_iter):
            n_iter += 1
            energy, _var, _noise = per_token_uq_fn(log_x_healed)  # (B, L)
            token_mask = (energy > self.threshold)                  # (B, L)
            if not token_mask.any():
                break

            cumulative_mask |= token_mask

            if lm is not None and token_ids is not None and K is not None:
                log_x_healed = _resample_via_lm(
                    log_x_healed,
                    token_ids=token_ids,
                    token_mask=token_mask,
                    lm=lm,
                    K=K,
                    eps=eps,
                    label_smoothing=label_smoothing,
                    transform_mode=transform_mode,
                )
            else:
                # EqM projection: integrate full batch, accept only at flagged positions
                log_x_projected = integrate_fn(log_x_healed, **integrate_kw)
                log_x_healed = torch.where(
                    token_mask.unsqueeze(-1).expand_as(log_x_healed),
                    log_x_projected,
                    log_x_healed,
                )

        post_energy: torch.Tensor = ood_score_fn(log_x_healed).detach()
        seq_mask = cumulative_mask.any(dim=-1)

        n_flagged = int(seq_mask.sum().item())
        if n_flagged > 0:
            improved = post_energy[seq_mask] < pre_energy[seq_mask]
            success_rate = float(improved.float().mean().item())
        else:
            success_rate = 0.0

        return TargetedResampleResult(
            healed_log_x=log_x_healed,
            token_mask=cumulative_mask,
            seq_mask=seq_mask,
            n_iter=n_iter,
            pre_energy=pre_energy,
            post_energy=post_energy,
            success_rate=success_rate,
        )


def _resample_via_lm(
    log_x: torch.Tensor,
    *,
    token_ids: torch.Tensor,
    token_mask: torch.Tensor,
    lm: Any,
    K: int,
    eps: float,
    label_smoothing: float,
    transform_mode: str,
) -> torch.Tensor:
    """Replace flagged positions using the LLM's greedy next-token prediction.

    For each flagged position ``(b, t)`` we use the LLM's logits at that
    position (conditioned on the original left context ``token_ids[b, :t]``)
    to pick the most likely token, then re-encode it in ILR/CLR space.

    Args:
        log_x: ``(B, L, D)`` ILR-encoded sequences.
        token_ids: ``(B, L)`` integer token IDs (original sequence).
        token_mask: ``(B, L)`` bool mask of positions to replace.
        lm: Object with ``forward_logits(input_ids) → (B, L, V)``.
        K: Vocabulary size for re-encoding.
        eps / label_smoothing / transform_mode: passed to ``token_ids_to_features``.

    Returns:
        Updated ``(B, L, D)`` tensor with flagged positions replaced.
    """
    from aitchinson_flow.data.transforms.discrete import token_ids_to_features  # noqa: PLC0415

    # Greedy predictions at every position — (B, L)
    logits = lm.forward_logits(input_ids=token_ids)       # (B, L, V)
    predicted_ids = logits.argmax(dim=-1).cpu()            # (B, L)

    B, L, D = log_x.shape
    log_x_updated = log_x.clone()

    for b in range(B):
        flagged_positions = token_mask[b].nonzero(as_tuple=True)[0]
        if flagged_positions.numel() == 0:
            continue
        # Build updated token-id row: original everywhere except flagged positions
        updated_ids = token_ids[b].cpu().clone()
        updated_ids[flagged_positions] = predicted_ids[b][flagged_positions]
        # Re-encode updated positions only
        for pos in flagged_positions.tolist():
            feat = token_ids_to_features(
                updated_ids[pos : pos + 1],
                K=K,
                eps=eps,
                label_smoothing=label_smoothing,
                transform_mode=transform_mode,
            )  # (1, D)
            log_x_updated[b, pos] = feat[0].to(log_x.device)

    return log_x_updated
