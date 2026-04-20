"""Random token corruption utilities shared by datamodules and benchmark tasks.

Originally lived under ``benchmarks/corruption.py`` which created a reverse
dependency: core datamodules in :mod:`aitchinson_flow.data` imported back into
the benchmark package. The logic belongs in :mod:`aitchinson_flow.data` because
the text8 datamodule and several other datamodules apply the same corruption
during training; benchmarks just consume the same helpers.
"""

from __future__ import annotations

import torch

from aitchinson_flow.data.transforms.discrete import token_ids_to_features, token_logits_to_features


def corrupt_token_ids(
    token_ids: torch.Tensor,
    *,
    vocab_size: int,
    corrupt_rate: float = 0.15,
    seed: int | None = None,
) -> torch.Tensor:
    """Replace a random subset of tokens with uniform random vocab ids.

    Parameters
    ----------
    token_ids : Tensor, shape (B, L), dtype long
    vocab_size : int
    corrupt_rate : float in (0, 1]
    seed : optional RNG seed for reproducibility

    Returns
    -------
    Tensor, shape (B, L), dtype long — corrupted copy
    """
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    mask = torch.rand(token_ids.shape, generator=gen) < corrupt_rate
    replacements = torch.randint(0, vocab_size, token_ids.shape, generator=gen)
    # Avoid replacing with the same token (best-effort: re-draw once)
    same = replacements == token_ids
    replacements[same] = (replacements[same] + 1) % vocab_size
    return torch.where(mask, replacements, token_ids)


def partially_shuffle_token_ids(
    token_ids: torch.Tensor,
    *,
    shuffle_rate: float = 0.15,
    seed: int | None = None,
) -> torch.Tensor:
    """Shuffle a random subset of positions per sequence, preserving token multiset."""
    if shuffle_rate <= 0.0:
        return token_ids.clone()

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    out = token_ids.clone()
    bsz, seq_len = out.shape
    for b in range(bsz):
        mask = torch.rand(seq_len, generator=gen) < shuffle_rate
        idx = mask.nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() <= 1:
            continue
        perm = idx[torch.randperm(idx.numel(), generator=gen)]
        out[b, idx] = out[b, perm]
    return out


def build_invalid_batch(
    batch: dict[str, torch.Tensor],
    *,
    K: int,
    corrupt_rate: float = 0.15,
    order_mix_rate: float = 0.15,
    order_mix_prob: float = 0.5,
    eps: float = 1e-8,
    label_smoothing: float = 0.0,
    transform_mode: str = "ilr",
    feature_mode: str = "token_ids",
    seed: int | None = None,
) -> dict[str, torch.Tensor]:
    """Augment a batch with corrupted invalid samples (in-place + returned).

    Expects ``batch["token_ids"]`` (B, L) and ``batch["logits"]`` (B, L, vocab).
    Adds ``batch["log_x_invalid"]``, ``batch["token_ids_invalid"]``, and
    ``batch["logits_invalid"]`` (logits unchanged — still original LM scores).

    The continuous representation honors ``label_smoothing`` and
    ``transform_mode`` so the ablation switches set on the valid path also
    apply to invalid samples (otherwise distances and AUROC would be biased
    by mismatched feature pipelines).
    """
    token_ids = batch["token_ids"]
    vocab_size = batch["logits"].shape[-1]

    bad_ids = token_ids.clone()
    if corrupt_rate > 0.0:
        bad_ids = corrupt_token_ids(
            bad_ids,
            vocab_size=vocab_size,
            corrupt_rate=corrupt_rate,
            seed=seed,
        )
    if order_mix_rate > 0.0 and order_mix_prob > 0.0:
        apply_order_mix = True
        if seed is not None:
            apply_order_mix = bool(
                (torch.rand(1, generator=torch.Generator().manual_seed(seed + 1)) < order_mix_prob).item()
            )
        if apply_order_mix:
            bad_ids = partially_shuffle_token_ids(
                bad_ids,
                shuffle_rate=order_mix_rate,
                seed=None if seed is None else seed + 2,
            )

    if feature_mode == "token_probs":
        logits_invalid = torch.full(
            (bad_ids.shape[0], bad_ids.shape[1], vocab_size),
            fill_value=float(torch.log(torch.tensor(eps))),
            dtype=torch.float32,
        )
        logits_invalid.scatter_(dim=-1, index=bad_ids.unsqueeze(-1), value=0.0)
        log_x_invalid = token_logits_to_features(
            logits_invalid,
            K=K,
            eps=eps,
            transform_mode=transform_mode,
        )
    else:
        rows = [
            token_ids_to_features(
                row,
                K=K,
                eps=eps,
                label_smoothing=label_smoothing,
                transform_mode=transform_mode,
            )
            for row in bad_ids
        ]
        log_x_invalid = torch.stack(rows, dim=0)
        logits_invalid = batch["logits"].clone()

    batch["token_ids_invalid"] = bad_ids
    batch["log_x_invalid"] = log_x_invalid
    batch["logits_invalid"] = logits_invalid
    return batch


__all__ = [
    "build_invalid_batch",
    "corrupt_token_ids",
    "partially_shuffle_token_ids",
]
