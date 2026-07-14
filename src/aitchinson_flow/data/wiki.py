"""WikiText-2 auditor data — cache LM features and serve clean/invalid pairs.

Phase F (TRAINING_PROTOCOL.md §6) caches GPT-2 (or Qwen2.5) outputs for a
WikiText-2 subset and stores both the clean and span-corrupted variants.
The cache is a single ``torch.load``-able dict so the training loop reads
*nothing* from HuggingFace at runtime — the LM is run once at cache time
and is never instantiated during EqM auditor training.

Cache schema (all tensors float32 on CPU; each entry has B chunks):

* ``clean_ids``      (B, L) long  — BPE token ids for the clean chunk
* ``invalid_ids``    (B, L) long  — same with 25 % positions replaced
* ``mask_corrupt``   (B, L) bool  — True where invalid_ids != clean_ids
* ``clean_clr``      (B, L, K) float — per-position top-K log-simplex
                                       (centered log-ratio of top-K logits)
* ``invalid_clr``    (B, L, K) float — same on the invalid sequence
* ``clean_topk_idx`` (B, L, K) long  — vocab ids of the top-K slots
* ``invalid_topk_idx`` (B, L, K) long
* ``clean_h``        (B, L, H) float — last-hidden-state per position
* ``invalid_h``      (B, L, H) float
* ``clean_SE_pos``   (B, L) float — the LM's per-position **NLL**:
                                     ``logsumexp(logits) - logits[token_id]``.
                                     (Field name kept for cache compatibility; it is
                                     an NLL, NOT the spilled energy of Minut et al.
                                     ICLR 2026 — see ``_per_position_nll``.)
* ``invalid_SE_pos`` (B, L) float

Plus a small metadata blob: ``{"lm": "gpt2", "L": 64, "K": 64, "V": 50257,
"H": 768, "n": 300, "corrupt_rate": 0.25}``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


def load_wiki_cache(cache_path: str | Path) -> dict[str, Any]:
    """Load a wiki cache produced by ``scripts/cache_wiki.py``."""
    return torch.load(cache_path, map_location="cpu", weights_only=False)


class WikiAuditorDataset(Dataset[dict[str, torch.Tensor]]):
    """Serve (clean, invalid) feature pairs from a pre-computed cache.

    Each item is a dict ready to drop into the EqM auditor training step::

        {
          "x":                  (L, K) float — clean CLR feature
          "x_invalid":          (L, K) float — invalid CLR feature
          "token_ids":          (L,)   long  — clean token ids
          "token_ids_invalid":  (L,)   long
          "mask_corrupt":       (L,)   bool
          "SE_pos_clean":       (L,)   float
          "SE_pos_invalid":     (L,)   float
          "h_clean":            (L, H) float (only present if cached)
          "h_invalid":          (L, H) float (only present if cached)
        }
    """

    def __init__(self, cache: dict[str, Any], *, with_hidden: bool = True) -> None:
        self.cache = cache
        self.with_hidden = with_hidden and "clean_h" in cache
        self.n = int(cache["clean_clr"].shape[0])
        self.L = int(cache["clean_clr"].shape[1])
        self.K = int(cache["clean_clr"].shape[2])
        self.H = int(cache["clean_h"].shape[2]) if "clean_h" in cache else 0

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {
            "x": self.cache["clean_clr"][i].float(),
            "x_invalid": self.cache["invalid_clr"][i].float(),
            "token_ids": self.cache["clean_ids"][i].long(),
            "token_ids_invalid": self.cache["invalid_ids"][i].long(),
            "mask_corrupt": self.cache["mask_corrupt"][i].bool(),
            "SE_pos_clean": self.cache["clean_SE_pos"][i].float(),
            "SE_pos_invalid": self.cache["invalid_SE_pos"][i].float(),
        }
        if self.with_hidden:
            out["h_clean"] = self.cache["clean_h"][i].float()
            out["h_invalid"] = self.cache["invalid_h"][i].float()
        return out


@torch.no_grad()
def _topk_clr(logits: torch.Tensor, K: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-K logits → (centered log-ratio values, vocab indices).

    logits: (..., V). Returns (clr (..., K), idx (..., K)) where clr is
    ``topk_vals − topk_vals.mean(dim=-1, keepdim=True)`` and idx are the
    top-K vocab IDs sorted by descending logit.
    """
    vals, idx = logits.topk(K, dim=-1)
    clr = vals - vals.mean(dim=-1, keepdim=True)
    return clr, idx


@torch.no_grad()
def _per_position_nll(
    logits: torch.Tensor, token_ids: torch.Tensor
) -> torch.Tensor:
    """Per-position NLL for an autoregressive LM.

    RENAMED (2026-07) from ``_spilled_energy_per_pos``, which was a misnomer: this
    is the SAME-STEP negative log-likelihood ``−log p(x_i | x_<i)``, **not** the
    spilled energy of Minut, Dewidar & Masi (ICLR 2026, arXiv:2602.18671). Theirs is
    the CROSS-STEP discrepancy ``logsumexp(logits_i) − logits_{i-1}[x_i]``, pairing
    the logit energy at step i-1 with the marginal energy at step i; the two differ
    by the log-partition drift ``logsumexp(logits_i) − logsumexp(logits_{i-1})``,
    which is exactly the signal spilled energy is about. The real thing lives in
    ``scripts/bench_sflm_ebm._gpt2_bpe_scores(score="spilled")``.

    For GPT-2-style decoders ``logits[..., i, :]`` is a distribution over
    the *next* token (position i+1), so the per-position NLL of the
    actually-placed token at position i+1 is::

        NLL(i+1) = logsumexp(logits[..., i, :]) − logits[..., i, token_ids[..., i+1]]

    We return a tensor of the same shape as ``token_ids`` (..., L). Position
    0 has no prior context, so ``NLL(0) = 0``. The output is shifted-and-
    aligned so that ``out[..., k]`` is the NLL of the token at
    position k.

    logits: (..., L, V), token_ids: (..., L) → (..., L)
    """
    L = token_ids.shape[-1]
    if L < 2:
        return torch.zeros_like(token_ids, dtype=logits.dtype)
    # logits at positions [0, L-2] predict tokens at positions [1, L-1].
    pred_logits = logits[..., : L - 1, :]
    target_ids = token_ids[..., 1:].unsqueeze(-1)
    lse = torch.logsumexp(pred_logits, dim=-1)
    picked = pred_logits.gather(-1, target_ids).squeeze(-1)
    nll_shifted = lse - picked  # (..., L-1) — NLL for positions 1..L-1
    out = torch.zeros_like(token_ids, dtype=logits.dtype)
    out[..., 1:] = nll_shifted
    return out


def span_corrupt(
    token_ids: torch.Tensor,
    *,
    vocab_size: int,
    corrupt_rate: float,
    avoid_special_below: int = 5,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace ~corrupt_rate fraction of tokens with random vocab IDs.

    Avoids the lowest few special tokens (``<|endoftext|>`` etc.) by drawing
    from ``[avoid_special_below, vocab_size)``. Returns (invalid_ids, mask)
    where mask is True at corrupted positions. Best-effort: if the random
    draw would replace a token with itself we re-roll once, so the mask
    reliably indicates "this position changed".
    """
    gen = None
    if seed is not None:
        gen = torch.Generator(device=token_ids.device).manual_seed(seed)
    L = token_ids.shape[-1]
    mask = (
        torch.rand(token_ids.shape, device=token_ids.device, generator=gen)
        < corrupt_rate
    )
    if mask.sum() == 0:
        # ensure at least one corruption per sequence to avoid silent zeros
        idx = torch.randint(0, L, (token_ids.shape[0], 1), generator=gen)
        mask.scatter_(-1, idx, True)
    repl = avoid_special_below + torch.randint(
        0,
        vocab_size - avoid_special_below,
        token_ids.shape,
        device=token_ids.device,
        generator=gen,
    )
    same = repl == token_ids
    repl[same] = avoid_special_below + (
        (repl[same] - avoid_special_below + 1) % (vocab_size - avoid_special_below)
    )
    invalid = torch.where(mask, repl, token_ids)
    return invalid, mask


__all__ = [
    "WikiAuditorDataset",
    "load_wiki_cache",
    "span_corrupt",
    "_topk_clr",
    "_per_position_nll",
]
