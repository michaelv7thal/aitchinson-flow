"""Spilled Energy — training-free hallucination detection baseline.

Reference: Minut, Dewidar & Masi, "Spilled Energy in Large Language Models",
           ICLR 2026. https://openreview.net/forum?id=EXFKk4Y3yc

Definition (Eq. 8):
    ΔE(x_i) = −log Σ_k exp(logits[i−1, k])  ← marginal energy at step i
             + logits[i−1, token_id[i]]        ← logit of actual token at i−1

Convention: lower (more negative) spilled energy = more anomalous.
Anomaly score for a sequence: −spilled.mean()
"""

from __future__ import annotations

import torch
from torch import Tensor


def marginal_energy(logits: Tensor) -> Tensor:
    """Per-position marginal energy: −logsumexp(logits).

    Parameters
    ----------
    logits : Tensor, shape (L, vocab)

    Returns
    -------
    Tensor, shape (L,)
    """
    return -torch.logsumexp(logits, dim=-1)


def compute_spilled_energy(
    logits_cpu: Tensor,
    token_ids: list[int],
) -> Tensor:
    """Per-token spilled energy (Minut et al. ICLR 2026).

    Parameters
    ----------
    logits_cpu : Tensor, shape (L, vocab)
        Raw LM logits (float32) for each position.
    token_ids : list of int, length L

    Returns
    -------
    Tensor, shape (L-1,)
        spilled[i] = −logsumexp(logits[i]) + logits[i, token_ids[i+1]]
    """
    L = logits_cpu.shape[0]
    ids = torch.tensor(token_ids[1:], dtype=torch.long)
    marg = marginal_energy(logits_cpu)  # (L,)
    logit_e = logits_cpu[torch.arange(L - 1), ids]  # (L-1,)
    return marg[:-1] + logit_e


def sequence_anomaly_score(spilled: Tensor) -> float:
    """Anomaly score for a sequence: −mean(spilled). Higher = more anomalous."""
    return -float(spilled.mean())


def compute_spilled_energy_batch(
    logits_batch: Tensor,
    token_ids_batch: Tensor,
) -> Tensor:
    """Batched spilled energy for benchmark throughput.

    Parameters
    ----------
    logits_batch : Tensor, shape (B, L, vocab)
    token_ids_batch : Tensor, shape (B, L), dtype long

    Returns
    -------
    Tensor, shape (B, L-1)
    """
    marg = -torch.logsumexp(logits_batch, dim=-1)  # (B, L)
    ids_next = token_ids_batch[:, 1:]  # (B, L-1)
    logit_e = logits_batch[:, :-1, :].gather(
        dim=-1, index=ids_next.unsqueeze(-1)
    ).squeeze(-1)  # (B, L-1)
    return marg[:, :-1] + logit_e


def hard_negative_ids(
    logits_cpu: Tensor,
    token_ids: list[int],
    top_k: int = 5,
) -> list[int]:
    """Hard-negative mining: replace each token with its hardest LM confusor.

    Position 0 is unchanged (no preceding logits).  For positions 1..L-1, the
    hard negative is the highest-probability non-true token predicted by the
    preceding logit vector.
    """
    L = logits_cpu.shape[0]
    hard = list(token_ids)
    for i in range(1, L):
        top_ids = logits_cpu[i - 1].topk(top_k + 1).indices.tolist()
        for cand in top_ids:
            if cand != token_ids[i]:
                hard[i] = cand
                break
    return hard
