"""Load text8 chunks and tokenize them as prompts for LM generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from aitchinson_flow.config import Config, RawTextDatasetConfig
from aitchinson_flow.data.hf_hub import TEXT8_DATASET_CANDIDATES, load_raw_text_column

if TYPE_CHECKING:
    from aitchinson_flow.llms.types import CausalLMForInference


def load_text8_prompts(
    lm: "CausalLMForInference",
    *,
    n_prompts: int,
    prompt_length: int,
    seed: int = 0,
    cfg: Config | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load text8 and return ``(prompt_ids, prompt_mask)``."""
    raw_cfg = cfg.raw_text_dataset if cfg is not None else RawTextDatasetConfig()
    fallback_paths: tuple[str, ...] = ()
    if raw_cfg.source_ref == TEXT8_DATASET_CANDIDATES[0]:
        fallback_paths = TEXT8_DATASET_CANDIDATES[1:]
    full_text = load_raw_text_column(
        raw_cfg,
        split=raw_cfg.split_train,
        cache_dir=raw_cfg.cache_dir,
        fallback_paths=fallback_paths,
    )

    ids_1d, mask_1d = lm.encode_text(
        full_text,
        max_length=n_prompts * prompt_length + prompt_length,
    )
    ids_1d = ids_1d.squeeze(0)
    mask_1d = mask_1d.squeeze(0)

    gen = torch.Generator().manual_seed(seed)
    max_start = ids_1d.shape[0] - prompt_length
    starts = torch.randint(0, max(max_start, 1), (n_prompts,), generator=gen)

    prompt_ids = torch.stack([ids_1d[s : s + prompt_length] for s in starts])
    prompt_mask = torch.stack([mask_1d[s : s + prompt_length] for s in starts])
    return prompt_ids, prompt_mask


__all__ = ["load_text8_prompts"]
