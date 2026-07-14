from __future__ import annotations

import re
from collections import defaultdict

import torch

from aitchinson_flow.data.transforms import token_ids_to_features

# Canonical text8 alphabet (K=27: a-z + space). Kept local so this module has no
# dependency on char_window_dataset / scripts (avoids any import-order fragility).
_ALPHABET = "abcdefghijklmnopqrstuvwxyz "
_CHAR2ID: dict[str, int] = {c: i for i, c in enumerate(_ALPHABET)}


def _decode_ids(ids) -> str:
    """Decode a 1-D iterable of token ids to a text8 char string ('?' for OOB)."""
    return "".join(_ALPHABET[int(i)] if int(i) < len(_ALPHABET) else "?" for i in ids)


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
    gen: torch.Generator | None = None
    if seed is not None:
        gen = torch.Generator(device=token_ids.device)
        gen.manual_seed(seed)

    mask = (
        torch.rand(token_ids.shape, device=token_ids.device, generator=gen)
        < corrupt_rate
    )
    replacements = torch.randint(
        0,
        vocab_size,
        token_ids.shape,
        device=token_ids.device,
        generator=gen,
    )
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

    gen: torch.Generator | None = None
    if seed is not None:
        gen = torch.Generator(device=token_ids.device)
        gen.manual_seed(seed)

    out = token_ids.clone()
    bsz, seq_len = out.shape
    for b in range(bsz):
        mask = torch.rand(seq_len, device=out.device, generator=gen) < shuffle_rate
        idx = mask.nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() <= 1:
            continue
        perm = idx[torch.randperm(idx.numel(), device=idx.device, generator=gen)]
        out[b, idx] = out[b, perm]
    return out


def build_invalid_batch(
    batch: dict[str, torch.Tensor],
    *,
    K: int,
    corrupt_rate: float = 0.15,
    order_mix_rate: float = 0.15,
    order_mix_prob: float = 0.5,
    label_smoothing: float = 1e-4,
    seed: int | None = None,
) -> dict[str, torch.Tensor]:
    """Augment a batch with corrupted invalid samples (in-place + returned).

    Expects ``batch["token_ids"]`` (B, L).
    Adds ``batch["x_invalid"]`` and ``batch["token_ids_invalid"]``.
    """
    token_ids = batch["token_ids"]

    bad_ids = token_ids.clone()
    if corrupt_rate > 0.0:
        bad_ids = corrupt_token_ids(
            bad_ids,
            vocab_size=K,
            corrupt_rate=corrupt_rate,
            seed=seed,
        )
    if order_mix_rate > 0.0 and order_mix_prob > 0.0:
        apply_order_mix = True
        if seed is not None:
            apply_order_mix = bool(
                (
                    torch.rand(1, generator=torch.Generator().manual_seed(seed + 1))
                    < order_mix_prob
                ).item()
            )
        if apply_order_mix:
            bad_ids = partially_shuffle_token_ids(
                bad_ids,
                shuffle_rate=order_mix_rate,
                seed=None if seed is None else seed + 2,
            )

    batch["token_ids_invalid"] = bad_ids
    batch["x_invalid"] = token_ids_to_features(bad_ids, K, label_smoothing=label_smoothing)
    return batch


def build_vocab_by_len(
    train_txt: str, *, min_len: int = 2, max_len: int = 18, cap: int = 6000
) -> dict[int, list[str]]:
    """First-seen-order (≈frequency-ranked) real words from text8, bucketed by length.

    Used to draw lexically-valid same-length replacement words for the "false info"
    corruption (see ``corrupt_false_info``).
    """
    by_len: dict[int, list[str]] = defaultdict(list)
    seen: dict[int, set[str]] = defaultdict(set)
    for w in re.findall(r"[a-z]+", train_txt):
        Lw = len(w)
        if min_len <= Lw <= max_len and w not in seen[Lw] and len(by_len[Lw]) < cap:
            seen[Lw].add(w)
            by_len[Lw].append(w)
    return by_len


def corrupt_false_info(
    windows: torch.Tensor,
    rate: float,
    by_len: dict[int, list[str]],
    *,
    seed: int,
) -> torch.Tensor:
    """Replace ``rate`` of fully-contained words in each window with a DIFFERENT real
    same-length word (lexically valid, semantically wrong).

    Positions and length are preserved and the text stays lexically valid, so a
    *local* character-level surprise (denoiser NLL) barely moves — this is the hard
    "false information" axis. ``by_len`` must come from ``build_vocab_by_len``.

    Parameters
    ----------
    windows : Tensor, shape (B, L), dtype long — char-level token ids.
    rate : float in (0, 1] — fraction of eligible words per window to swap.
    by_len : dict[int, list[str]] — length-bucketed replacement vocabulary.
    seed : RNG seed for reproducibility.

    Returns
    -------
    Tensor, shape (B, L), dtype long — corrupted copy.
    """
    g = torch.Generator().manual_seed(seed)
    out = windows.clone()

    def ri(n: int) -> int:
        return int(torch.randint(0, n, (1,), generator=g).item())

    for i in range(windows.shape[0]):
        s = _decode_ids(windows[i])
        spans = [(m.start(), m.end()) for m in re.finditer(r"[a-z]+", s)]
        spans = [(a, b) for (a, b) in spans if a > 0 and b < len(s)]  # drop partial edge words
        if not spans:
            continue
        n_sub = max(1, round(rate * len(spans)))
        order = torch.randperm(len(spans), generator=g).tolist()[:n_sub]
        for idx in order:
            a, b = spans[idx]
            cands = by_len.get(b - a)
            if not cands:
                continue
            orig = s[a:b]
            repl = None
            for _ in range(8):
                w = cands[ri(len(cands))]
                if w != orig:
                    repl = w
                    break
            if repl is None:
                continue
            for j, ch in enumerate(repl):
                out[i, a + j] = _CHAR2ID[ch]
    return out


__all__ = [
    "build_invalid_batch",
    "build_vocab_by_len",
    "corrupt_false_info",
    "corrupt_token_ids",
    "partially_shuffle_token_ids",
]
