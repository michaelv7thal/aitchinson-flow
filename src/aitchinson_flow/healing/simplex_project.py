"""Simplex projection healer (Phase 4, Strategy 3).

Workflow:
  1. Score sequences; flag those above ``energy_threshold`` (sequence-level OOD score).
  2. Run ``healer.integrate()`` to project flagged sequences toward the valid manifold.
  3. Convert healed ILR coordinates back to the probability simplex via ``ilr_inv``.
  4. Recover the nearest valid token at each position via ``argmax`` over the K-dim simplex.
  5. Re-encode the recovered token IDs as ILR features.
  6. Re-evaluate and return diagnostics.

This strategy is training-free at evaluation time (only the EqM healer is used;
no separate GP or LLM is required).  The nearest-token snap guarantees that the
output lies exactly on the character/token vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from aitchinson_flow.geometry import ilr_inv


@dataclass
class SimplexProjectResult:
    """Output of :meth:`SimplexProjectHealer.heal`."""

    healed_log_x: torch.Tensor       # (B, L, D) ILR coords after projection + snap
    nearest_token_ids: torch.Tensor  # (B, L) argmax token IDs in the K-dim simplex
    seq_mask: torch.Tensor           # (B,)   bool — sequences that were projected
    pre_energy: torch.Tensor         # (B,) OOD score before healing
    post_energy: torch.Tensor        # (B,) OOD score after healing
    energy_reduction: float          # mean energy drop for projected sequences
    success_rate: float              # fraction where post_energy < pre_energy


class SimplexProjectHealer:
    """Project high-energy sequences to the valid manifold and snap to nearest tokens.

    Args:
        auditor: Model with ``ood_score(log_x) → (B,)``.
        healer: Model with ``integrate(log_x, steps?, dt?) → log_x`` and config
            exposing ``cfg.dataset.K`` for vocabulary size.
        energy_threshold: Sequence-level OOD score above which healing is applied.
        steps: EqM integration steps (``None`` → healer default).
        dt: EqM step size (``None`` → healer default).
        re_encode: If ``True`` (default), re-encode the snapped token IDs back to ILR
            before returning ``healed_log_x`` and re-evaluating energy.  If ``False``,
            ``healed_log_x`` is the raw EqM-projected ILR (no vocabulary snap).
        eps: Label-smoothing epsilon for re-encoding (used when ``re_encode=True``).
        label_smoothing: Mix coefficient for re-encoding.
        transform_mode: ``"ilr"`` or ``"clr"`` for re-encoding.
    """

    def __init__(
        self,
        auditor: nn.Module,
        healer: nn.Module,
        energy_threshold: float = 1.0,
        steps: int | None = None,
        dt: float | None = None,
        re_encode: bool = True,
        eps: float = 1e-8,
        label_smoothing: float = 0.0,
        transform_mode: str = "ilr",
    ) -> None:
        self.auditor = auditor
        self.healer = healer
        self.energy_threshold = energy_threshold
        self.steps = steps
        self.dt = dt
        self.re_encode = re_encode
        self.eps = eps
        self.label_smoothing = label_smoothing
        self.transform_mode = transform_mode

    @torch.no_grad()
    def heal(self, log_x: torch.Tensor, K: int) -> SimplexProjectResult:
        """Project high-energy sequences to the valid manifold.

        Args:
            log_x: ``(B, L, D)`` ILR-encoded sequences (D = K−1).
            K: Vocabulary size; used for ``ilr_inv`` and re-encoding.

        Returns:
            :class:`SimplexProjectResult` with healed sequences and diagnostics.
        """
        ood_score_fn = getattr(self.auditor, "ood_score", None)
        integrate_fn = getattr(self.healer, "integrate", None)
        if ood_score_fn is None:
            raise AttributeError("auditor must implement ood_score(log_x)")
        if integrate_fn is None:
            raise AttributeError("healer must implement integrate(log_x, ...)")

        pre_energy: torch.Tensor = ood_score_fn(log_x).detach()   # (B,)
        needs_healing = pre_energy > self.energy_threshold          # (B,) bool

        healed_log_x = log_x.clone()
        nearest_token_ids = torch.zeros(
            log_x.shape[:2], dtype=torch.long, device=log_x.device
        )

        if needs_healing.any():
            integrate_kw: dict = {}
            if self.steps is not None:
                integrate_kw["steps"] = self.steps
            if self.dt is not None:
                integrate_kw["dt"] = self.dt

            # 1. Project flagged sequences toward valid manifold
            ood_x = log_x[needs_healing]                              # (M, L, D)
            projected = integrate_fn(ood_x, **integrate_kw)           # (M, L, D)

            # 2. ILR inverse → log-probabilities over K vocab items
            log_probs = ilr_inv(projected, K)                         # (M, L, K)

            # 3. Nearest token by argmax
            tok_ids = log_probs.argmax(dim=-1)                        # (M, L)
            nearest_token_ids[needs_healing] = tok_ids

            if self.re_encode:
                # 4. Re-encode snapped token IDs → ILR features for consistent geometry
                re_encoded = _encode_token_ids_batch(
                    tok_ids,
                    K=K,
                    eps=self.eps,
                    label_smoothing=self.label_smoothing,
                    transform_mode=self.transform_mode,
                    device=log_x.device,
                )
                healed_log_x[needs_healing] = re_encoded
            else:
                healed_log_x[needs_healing] = projected

        post_energy: torch.Tensor = ood_score_fn(healed_log_x).detach()  # (B,)

        n_healed = int(needs_healing.sum().item())
        if n_healed > 0:
            pre_healed = pre_energy[needs_healing]
            post_healed = post_energy[needs_healing]
            energy_reduction = float((pre_healed - post_healed).mean().item())
            success_rate = float((post_healed < pre_healed).float().mean().item())
        else:
            energy_reduction = 0.0
            success_rate = 0.0

        return SimplexProjectResult(
            healed_log_x=healed_log_x,
            nearest_token_ids=nearest_token_ids,
            seq_mask=needs_healing,
            pre_energy=pre_energy,
            post_energy=post_energy,
            energy_reduction=energy_reduction,
            success_rate=success_rate,
        )


def _encode_token_ids_batch(
    token_ids: torch.Tensor,
    *,
    K: int,
    eps: float,
    label_smoothing: float,
    transform_mode: str,
    device: torch.device,
) -> torch.Tensor:
    """Encode a batch of token ID sequences as ILR/CLR features.

    Args:
        token_ids: ``(M, L)`` integer token IDs on any device.
        K: Vocabulary size.
        eps / label_smoothing / transform_mode: encoding kwargs.
        device: Target device for the output tensor.

    Returns:
        ``(M, L, D)`` feature tensor on ``device``.
    """
    from aitchinson_flow.data.transforms.discrete import token_ids_to_features  # noqa: PLC0415

    M, L = token_ids.shape
    rows = [
        token_ids_to_features(
            token_ids[m].cpu(),
            K=K,
            eps=eps,
            label_smoothing=label_smoothing,
            transform_mode=transform_mode,
        )
        for m in range(M)
    ]
    return torch.stack(rows, dim=0).to(device=device)
