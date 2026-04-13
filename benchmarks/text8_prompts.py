"""Load text8 chunks and tokenize them as GPT-2 prompts for benchmark generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from aitchinson_flow.llms.types import CausalLMForInference


def load_text8_prompts(
    lm: CausalLMForInference,
    *,
    n_prompts: int,
    prompt_length: int,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load text8, chunk into prompt_length-token windows, return (ids, mask).

    Returns
    -------
    prompt_ids : Tensor, shape (n_prompts, prompt_length)
    prompt_mask : Tensor, shape (n_prompts, prompt_length)
    """
    import datasets  # noqa: PLC0415

    ds = datasets.load_dataset("afm-intelligence/text8", split="train")
    full_text = " ".join(ds["text"])

    ids_1d, mask_1d = lm.encode_text(full_text, max_length=n_prompts * prompt_length + prompt_length)
    ids_1d = ids_1d.squeeze(0)  # (total_tokens,)
    mask_1d = mask_1d.squeeze(0)

    gen = torch.Generator().manual_seed(seed)
    max_start = ids_1d.shape[0] - prompt_length
    starts = torch.randint(0, max(max_start, 1), (n_prompts,), generator=gen)

    prompt_ids = torch.stack([ids_1d[s : s + prompt_length] for s in starts])
    prompt_mask = torch.stack([mask_1d[s : s + prompt_length] for s in starts])

    return prompt_ids, prompt_mask
