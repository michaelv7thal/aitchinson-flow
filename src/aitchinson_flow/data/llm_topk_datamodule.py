"""Path B datamodule: raw text -> pretrained LLM -> per-position top-K -> ILR.

Mirrors Path A's map-style text8 pipeline but replaces the one-hot-on-char-vocab
row with the pretrained LLM's top-``K`` softmax probabilities at each position.

Data flow per batch:

1. Sample ``B`` char-level windows of length ``char_window_length`` from the
   text8 corpus.
2. Build an "invalid" copy by applying Path A's char-level corruption
   (:func:`aitchinson_flow.data.corruption.corrupt_token_ids`) to the same
   windows.
3. Decode both to strings, tokenize with the LLM tokenizer (truncate/pad to
   ``cfg.dataset.L`` LLM tokens), run ``forward_logits``.
4. Project logits through :func:`sorted_topk_logits_to_features` to yield
   ``(B, L, K-1)`` ILR coordinates for each of the clean and corrupted paths.

The emitted batch has the same keys Stage 1/Stage 2 already consume:
``log_x`` and ``log_x_invalid`` (plus ``token_ids`` / ``token_ids_invalid``
for downstream debugging).
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
from aitchinson_flow.data.transforms.discrete import sorted_topk_logits_to_features
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


class _LLMTopKCollate:
    """Collate windows, run clean+corrupt through the LLM, produce ILR features."""

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
        K = self._cfg.dataset.K
        L = self._cfg.dataset.L
        eps = self._cfg.hf_dataset.log_simplex_eps
        transform_mode = self._cfg.hf_dataset.transform_mode

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
            clean_ids, clean_mask = _batch_encode(self._lm, clean_texts, max_length=L)
            corrupt_ids, corrupt_mask = _batch_encode(self._lm, corrupt_texts, max_length=L)

            clean_logits = self._lm.forward_logits(clean_ids, clean_mask).cpu()
            corrupt_logits = self._lm.forward_logits(corrupt_ids, corrupt_mask).cpu()

            log_x = sorted_topk_logits_to_features(
                clean_logits, K=K, eps=eps, transform_mode=transform_mode
            )
            log_x_invalid = sorted_topk_logits_to_features(
                corrupt_logits, K=K, eps=eps, transform_mode=transform_mode
            )

        # Strip the inference-mode tag so tensors feeding Stage 1 / Stage 2
        # training can participate in autograd. The LM itself wraps
        # ``forward_logits`` in ``@torch.inference_mode()``, which propagates
        # that flag through ``.cpu()`` and subsequent ops — cloning under
        # ``inference_mode(False)`` is the supported way to promote the result
        # back to a normal tensor.
        with torch.inference_mode(False):
            return {
                "log_x": log_x.detach().clone(),
                "log_x_invalid": log_x_invalid.detach().clone(),
                "token_ids": clean_ids.cpu().detach().clone(),
                "token_ids_invalid": corrupt_ids.cpu().detach().clone(),
            }


class LLMTopKDataModule(DataModule):
    """Path B datamodule: pretrained LLM top-K simplex rows with char-corrupted negatives."""

    def __init__(self, cfg: Config, lm: CausalLMForInference) -> None:
        topk_cfg = cfg.llm_topk_dataset
        if topk_cfg.raw_text_backend != "text8":
            raise NotImplementedError(
                f"LLMTopKDataModule currently supports raw_text_backend='text8' only; "
                f"got {topk_cfg.raw_text_backend!r}."
            )
        self._cfg = cfg
        self._lm = lm

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
        self._train_collate = _LLMTopKCollate(
            cfg, lm, corrupt_rate=corrupt_rate, seed=topk_cfg.generation_seed
        )
        self._eval_collate = _LLMTopKCollate(
            cfg,
            lm,
            corrupt_rate=corrupt_rate,
            seed=topk_cfg.generation_seed + 10_000,
        )

    @property
    def splits(self) -> _Splits:
        return self._splits

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
