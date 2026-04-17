"""Self-healing pipeline: flag OOD sequences by GP variance, then project via EqM flow."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class HealingResult:
    """Output of a single HealingPipeline.heal_batch() call."""
    pre_var: torch.Tensor    # (B,) GP variance before healing
    post_var: torch.Tensor   # (B,) GP variance after healing (unchanged for non-flagged)
    healed_log_x: torch.Tensor  # (B, L, K-1) sequences (healed where flagged)
    healed_mask: torch.Tensor   # (B,) bool — which sequences were flagged and healed
    success_rate: float         # fraction of flagged sequences whose post_var < threshold


class HealingPipeline:
    """Pairs a scorer (BayesianAuditor) with a healer (EquilibriumAuditor).

    Workflow:
      1. Score batch with auditor → GP variance per sequence.
      2. Flag sequences above ``var_threshold``.
      3. Run EqM flow (``healer.integrate()``) on flagged sequences.
      4. Re-score healed sequences.

    Both ``auditor`` and ``healer`` should already be on the correct device and in eval mode.
    """

    def __init__(
        self,
        auditor: nn.Module,
        healer: nn.Module,
        var_threshold: float = 0.5,
    ) -> None:
        self.auditor = auditor
        self.healer = healer
        self.var_threshold = var_threshold

    @torch.no_grad()
    def heal_batch(
        self,
        log_x: torch.Tensor,
        steps: int | None = None,
        dt: float | None = None,
    ) -> HealingResult:
        """Heal a batch of sequences.

        Args:
            log_x: (B, L, K-1) ILR-encoded sequences.
            steps: Euler steps for the healer (defaults to healer's config).
            dt: Step size (defaults to healer's config).

        Returns:
            HealingResult with pre/post variance, healed sequences, and success rate.
        """
        ood_score_fn = getattr(self.auditor, "ood_score", None)
        if ood_score_fn is None:
            raise AttributeError("auditor must implement ood_score(log_x)")
        integrate_fn = getattr(self.healer, "integrate", None)
        if integrate_fn is None:
            raise AttributeError("healer must implement integrate(log_x, steps, dt)")

        pre_var: torch.Tensor = ood_score_fn(log_x)  # (B,)
        needs_healing = pre_var > self.var_threshold   # (B,) bool

        healed_log_x = log_x.clone()
        post_var = pre_var.clone()

        if needs_healing.any():
            ood_x = log_x[needs_healing]
            integrate_kwargs: dict = {}
            if steps is not None:
                integrate_kwargs["steps"] = steps
            if dt is not None:
                integrate_kwargs["dt"] = dt
            healed_x = integrate_fn(ood_x, **integrate_kwargs)
            healed_log_x[needs_healing] = healed_x
            post_var[needs_healing] = ood_score_fn(healed_x)

        healed_and_fixed = needs_healing & (post_var < self.var_threshold)
        n_flagged = int(needs_healing.sum().item())
        success_rate = (
            float(healed_and_fixed.sum().item()) / n_flagged if n_flagged > 0 else 0.0
        )

        return HealingResult(
            pre_var=pre_var,
            post_var=post_var,
            healed_log_x=healed_log_x,
            healed_mask=needs_healing,
            success_rate=success_rate,
        )
