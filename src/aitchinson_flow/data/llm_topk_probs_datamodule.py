"""Component 2 datamodule: raw text → LLM top-K softmax probabilities → ILR.

Data flow per batch:

1. Sample ``B`` char-level windows of length ``char_window_length`` from the
   text8 corpus.
2. Decode to strings and tokenize with the LLM tokenizer (truncate/pad to
   ``cfg.dataset.L`` LLM tokens) to obtain ``clean_ids``.
3. Build an invalid copy by corrupting exactly ``corrupt_rate`` fraction of
   the ``L`` LLM token positions with random replacement
   (:func:`aitchinson_flow.data.corruption.corrupt_token_ids`). Corruption at
   LLM-token level gives precise control; character-level corruption caused
   BPE re-segmentation cascades that affected nearly all token distributions.
4. Run the frozen LLM's forward pass on both the clean and corrupted token ids
   to obtain per-position logits, then extract the top-K softmax probabilities
   (shape ``(B, L, K)``) and re-normalize within the K slots.
5. Apply the ILR transform to yield ``(B, L, K-1)`` Aitchison coordinates.

The emitted batch keys are ``log_x`` (valid) and ``log_x_invalid`` (corrupted).
Both shapes are ``(B, L, K-1)``, matching what Stage 1 and Stage 2 expect from
the char-level text8 datamodule.

Key geometric insight: valid text produces exponential-decay-like top-K
distributions (one token strongly dominates); corrupted text produces flatter,
more uniform top-K distributions. The ILR transform makes this geometric
difference learnable by the flow-matching backbone and the GP head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from aitchinson_flow.config import Config
from aitchinson_flow.data.corruption import corrupt_token_ids
from aitchinson_flow.data.llm_embedding_datamodule import (
    _batch_encode,
    _char_ids_to_text,
)
from aitchinson_flow.data.text8_datamodule import _load_text8_splits_cfg
from aitchinson_flow.data.transforms.discrete import project_log_to_simplex_features
from aitchinson_flow.llms.types import CausalLMForInference
from aitchinson_flow.training.datamodule import DataModule


def _top_k_probs_to_ilr(
    lm: CausalLMForInference,
    input_ids: Tensor,
    K: int,
    *,
    renormalize: bool,
    eps: float,
    transform_mode: str,
) -> Tensor:
    """Run ``input_ids`` through ``lm``, extract top-K softmax probs, apply ILR.

    Args:
        lm: Frozen causal LM implementing :class:`~aitchinson_flow.llms.types.CausalLMForInference`.
        input_ids: ``(B, L)`` integer token ids.
        K: Number of top vocabulary slots to retain per position.
        renormalize: Re-normalize top-K probabilities to sum to 1.
        eps: Small constant added before log to avoid ``-inf``.
        transform_mode: ``"ilr"`` or ``"clr"``; passed to
            :func:`~aitchinson_flow.data.transforms.discrete.project_log_to_simplex_features`.

    Returns:
        ``(B, L, K-1)`` ILR coordinates (or ``(B, L, K)`` for CLR mode).
    """
    with torch.no_grad():
        logits = lm.forward_logits(input_ids=input_ids)  # (B, L, V)
    probs_full = torch.softmax(logits.to(dtype=torch.float32), dim=-1)
    top_probs, _ = torch.topk(probs_full, K, dim=-1, sorted=True)  # (B, L, K)
    if renormalize:
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)
    top_probs = top_probs.clamp_min(eps)
    log_probs = top_probs.log()
    return project_log_to_simplex_features(log_probs, mode=transform_mode)


@dataclass(frozen=True)
class _Splits:
    train: Tensor
    val: Tensor
    test: Tensor


class _RawCharWindowDataset(Dataset[dict[str, Tensor]]):
    def __init__(self, windows: Tensor) -> None:
        self._windows = windows

    def __len__(self) -> int:
        return self._windows.shape[0]

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return {"char_ids": self._windows[idx]}


class _LLMTopKProbsCollate:
    """Collate char windows, run clean+corrupt through the LLM, extract top-K ILR features."""

    def __init__(
        self,
        cfg: Config,
        lm: CausalLMForInference,
        *,
        corrupt_rate: float,
        seed: int,
    ) -> None:
        self._cfg = cfg
        self._lm = lm
        self._corrupt_rate = corrupt_rate
        self._seed = seed
        self._n_calls = 0
        self._lm_vocab_size = lm.vocab_size

    def __call__(self, samples: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        K = self._cfg.dataset.K
        L = self._cfg.dataset.L
        eps = self._cfg.hf_dataset.log_simplex_eps
        transform_mode = self._cfg.hf_dataset.transform_mode
        renormalize = self._cfg.llm_topk_probs.renormalize

        clean_char_ids = torch.stack([s["char_ids"] for s in samples], dim=0)
        seed = self._seed + self._n_calls
        self._n_calls += 1

        clean_texts = [_char_ids_to_text(row) for row in clean_char_ids]

        with torch.no_grad():
            clean_ids, _ = _batch_encode(self._lm, clean_texts, max_length=L)

        # Corrupt at LLM-token level so exactly corrupt_rate fraction of the L
        # token positions are replaced. Character-level corruption caused BPE
        # re-segmentation cascades that affected nearly all token distributions.
        corrupt_ids = corrupt_token_ids(
            clean_ids,
            vocab_size=self._lm_vocab_size,
            corrupt_rate=self._corrupt_rate,
            seed=seed,
        )

        log_x = _top_k_probs_to_ilr(
            self._lm, clean_ids, K,
            renormalize=renormalize, eps=eps, transform_mode=transform_mode,
        )
        log_x_invalid = _top_k_probs_to_ilr(
            self._lm, corrupt_ids, K,
            renormalize=renormalize, eps=eps, transform_mode=transform_mode,
        )

        with torch.inference_mode(False):
            return {
                "log_x": log_x.detach().clone(),
                "log_x_invalid": log_x_invalid.detach().clone(),
                "token_ids": clean_ids.cpu().detach().clone(),
                "token_ids_invalid": corrupt_ids.cpu().detach().clone(),
            }


class LLMTopKProbsDataModule(DataModule):
    """Component 2 datamodule: LLM top-K probability windows with char-corrupted negatives.

    Produces ``log_x`` and ``log_x_invalid`` tensors of shape ``(B, L, K-1)``
    (ILR coordinates) that are consumed directly by Stage 1 and Stage 2 without
    any additional projection step — unlike Path B's embedding datamodule, which
    requires a learned :class:`~aitchinson_flow.models.llm_projection.TokenEmbeddingToSimplex`
    head.
    """

    def __init__(self, cfg: Config, lm: CausalLMForInference) -> None:
        topk_cfg = cfg.llm_topk_probs
        if topk_cfg.raw_text_backend != "text8":
            raise NotImplementedError(
                f"LLMTopKProbsDataModule currently supports raw_text_backend='text8' only; "
                f"got {topk_cfg.raw_text_backend!r}."
            )
        self._cfg = cfg
        self._lm = lm

        W = topk_cfg.char_window_length
        train_by_chars, val_by_chars, test_by_chars = _load_text8_splits_cfg(
            cfg, cfg.text8_dataset.cache_dir, W
        )

        tcfg = cfg.text8_dataset
        if tcfg.max_train_windows is not None:
            train_by_chars = train_by_chars[: tcfg.max_train_windows]
        if tcfg.max_eval_windows is not None:
            val_by_chars = val_by_chars[: tcfg.max_eval_windows]
            test_by_chars = test_by_chars[: tcfg.max_eval_windows]

        self._splits = _Splits(train=train_by_chars, val=val_by_chars, test=test_by_chars)
        self._train_ds = _RawCharWindowDataset(train_by_chars)
        self._val_ds = _RawCharWindowDataset(val_by_chars)
        self._test_ds = _RawCharWindowDataset(test_by_chars)

        corrupt_rate = (
            topk_cfg.corrupt_rate
            if topk_cfg.corrupt_rate is not None
            else tcfg.train_corrupt_rate
        )
        self._train_collate = _LLMTopKProbsCollate(
            cfg, lm, corrupt_rate=corrupt_rate, seed=topk_cfg.generation_seed
        )
        self._eval_collate = _LLMTopKProbsCollate(
            cfg, lm, corrupt_rate=corrupt_rate, seed=topk_cfg.generation_seed + 10_000
        )

    @property
    def splits(self) -> _Splits:
        return self._splits

    def _loader(
        self, ds: Dataset[Any], *, shuffle: bool, collate: Any
    ) -> DataLoader[Any]:
        return DataLoader(
            ds,
            batch_size=self._cfg.training.B,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=collate,
            pin_memory=False,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        return self._loader(self._train_ds, shuffle=True, collate=self._train_collate)

    def val_dataloader(self) -> DataLoader[Any] | None:
        return self._loader(self._val_ds, shuffle=False, collate=self._eval_collate)

    def test_dataloader(self) -> DataLoader[Any] | None:
        return self._loader(self._test_ds, shuffle=False, collate=self._eval_collate)

    def num_train_samples(self) -> int | None:
        return len(self._train_ds)
