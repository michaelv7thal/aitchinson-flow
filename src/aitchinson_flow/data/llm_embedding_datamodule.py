"""Path B datamodule: raw text -> LLM tokenize -> frozen-embedding lookup -> auditor.

Mirrors Path A's map-style text8 pipeline but replaces the one-hot-on-char-vocab
row with the pretrained LLM's **input-embedding** at each tokenized position.
The datamodule is intentionally the only part of the pipeline that touches the
LLM — the subsequent learned ``Linear(d_embed, K)`` head
(:class:`~aitchinson_flow.models.llm_projection.TokenEmbeddingToSimplex`) lives
on the auditor model and is trained jointly with the backbone.

Data flow per batch:

1. Sample ``B`` char-level windows of length ``char_window_length`` from the
   text8 corpus.
2. Build an "invalid" copy by applying Path A's char-level corruption
   (:func:`aitchinson_flow.data.corruption.corrupt_token_ids`) to the same
   windows.
3. Decode both to strings, tokenize with the LLM tokenizer (truncate/pad to
   ``cfg.dataset.L`` LLM tokens).
4. Look up the LLM's frozen input embeddings for the clean and corrupted ids
   and return ``(B, L, d_embed)`` float tensors.

The emitted batch replaces Path A's ``log_x`` with ``embeddings``:
``embeddings``, ``embeddings_invalid``, ``token_ids``, ``token_ids_invalid``.
Auditor models detect ``embeddings`` in :meth:`prepare_batch` and run the
learned projection before training/evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from aitchinson_flow.config import Config
from aitchinson_flow.data.corruption import corrupt_token_ids
from aitchinson_flow.data.text8_datamodule import (
    VOCAB_SIZE as TEXT8_VOCAB_SIZE,
    _ALPHABET,
    _load_text8_splits_cfg,
)
from aitchinson_flow.llms.types import CausalLMForInference
from aitchinson_flow.training.datamodule import DataModule


def _char_ids_to_text(ids: Tensor) -> str:
    """Inverse of the text8 char-id map for a single window."""
    return "".join(_ALPHABET[int(i)] for i in ids.tolist())


def _batch_encode(
    lm: CausalLMForInference, texts: list[str], *, max_length: int
) -> tuple[Tensor, Tensor]:
    """Encode a list of strings to ``(B, L)`` input_ids / attention_mask tensors."""
    all_ids: list[Tensor] = []
    all_masks: list[Tensor] = []
    for t in texts:
        ids, mask = lm.encode_text(t, max_length=max_length)
        all_ids.append(ids[0])
        all_masks.append(mask[0] if mask is not None else torch.ones_like(ids[0]))
    input_ids = torch.stack(all_ids, dim=0)
    attention_mask = torch.stack(all_masks, dim=0)
    return input_ids, attention_mask


@dataclass(frozen=True)
class _Splits:
    train: Tensor
    val: Tensor
    test: Tensor


class _RawCharWindowDataset(Dataset[dict[str, Tensor]]):
    """Map-style dataset of length-``W`` char-id windows."""

    def __init__(self, windows: Tensor) -> None:
        self._windows = windows

    def __len__(self) -> int:
        return self._windows.shape[0]

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return {"char_ids": self._windows[idx]}


class _LLMEmbeddingCollate:
    """Collate windows, run clean+corrupt through the LLM tokenizer + embedding."""

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

    def __call__(self, samples: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        L = self._cfg.dataset.L

        clean_char_ids = torch.stack([s["char_ids"] for s in samples], dim=0)
        seed = self._seed + self._n_calls
        self._n_calls += 1
        corrupt_char_ids = corrupt_token_ids(
            clean_char_ids,
            vocab_size=TEXT8_VOCAB_SIZE,
            corrupt_rate=self._corrupt_rate,
            seed=seed,
        )

        clean_texts = [_char_ids_to_text(row) for row in clean_char_ids]
        corrupt_texts = [_char_ids_to_text(row) for row in corrupt_char_ids]

        with torch.no_grad():
            clean_ids, _ = _batch_encode(self._lm, clean_texts, max_length=L)
            corrupt_ids, _ = _batch_encode(self._lm, corrupt_texts, max_length=L)

            clean_emb = self._lm.embed_tokens(clean_ids)
            corrupt_emb = self._lm.embed_tokens(corrupt_ids)

        # ``embed_tokens`` already strips the inference-mode flag, but cloning
        # under ``inference_mode(False)`` keeps this collate robust against
        # CausalLMForInference implementations that skip that courtesy.
        with torch.inference_mode(False):
            return {
                "embeddings": clean_emb.detach().clone(),
                "embeddings_invalid": corrupt_emb.detach().clone(),
                "token_ids": clean_ids.cpu().detach().clone(),
                "token_ids_invalid": corrupt_ids.cpu().detach().clone(),
            }


class LLMEmbeddingDataModule(DataModule):
    """Path B datamodule: LLM-embedding windows with char-corrupted negatives."""

    def __init__(self, cfg: Config, lm: CausalLMForInference) -> None:
        topk_cfg = cfg.llm_embedding_dataset
        if topk_cfg.raw_text_backend != "text8":
            raise NotImplementedError(
                f"LLMEmbeddingDataModule currently supports raw_text_backend='text8' only; "
                f"got {topk_cfg.raw_text_backend!r}."
            )
        self._cfg = cfg
        self._lm = lm
        self._llm_embed_dim = int(lm.embed_dim)

        # Reuse text8's native split loader, but chunk by char_window_length
        # (not cfg.dataset.L — that's the LLM-token length for the auditor).
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

        self._splits = _Splits(
            train=train_by_chars, val=val_by_chars, test=test_by_chars
        )

        self._train_ds = _RawCharWindowDataset(train_by_chars)
        self._val_ds = _RawCharWindowDataset(val_by_chars)
        self._test_ds = _RawCharWindowDataset(test_by_chars)

        corrupt_rate = (
            topk_cfg.corrupt_rate
            if topk_cfg.corrupt_rate is not None
            else tcfg.train_corrupt_rate
        )
        self._train_collate = _LLMEmbeddingCollate(
            cfg, lm, corrupt_rate=corrupt_rate, seed=topk_cfg.generation_seed
        )
        self._eval_collate = _LLMEmbeddingCollate(
            cfg,
            lm,
            corrupt_rate=corrupt_rate,
            seed=topk_cfg.generation_seed + 10_000,
        )

    @property
    def splits(self) -> _Splits:
        return self._splits

    @property
    def llm_embed_dim(self) -> int:
        """Input-embedding dim ``d_embed`` reported by the loaded LLM."""
        return self._llm_embed_dim

    def _loader(
        self, ds: Dataset[Any], *, shuffle: bool, collate: Any
    ) -> DataLoader[Any]:
        # num_workers=0 because the collate fn drives GPU LLM inference;
        # forked workers cannot share the model cleanly.
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


# Backwards-compatible alias so any external code still importing the old
# symbol keeps working. New code should use ``LLMEmbeddingDataModule``.
LLMTopKDataModule = LLMEmbeddingDataModule
