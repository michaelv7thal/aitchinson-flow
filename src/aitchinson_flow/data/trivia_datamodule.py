"""Trivia Q&A datamodule for one-class GP OOD detection.

Train split: correct answers only (no invalid samples).
Val/test split: interleaved correct (label=0) and incorrect (label=1) answers.

Incorrect answers are built by sampling the correct answer of a *different* question
from the same dataset (cross-question negatives). This creates plausible but factually
wrong answers — a harder OOD case than random character noise.

The datamodule reuses the same char-level 27-token vocabulary as Text8 so that the
same BayesianGenerator / BayesianAuditor backbone can be used without changes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from aitchinson_flow.config import Config
from aitchinson_flow.data.text8_datamodule import CHAR2ID, VOCAB_SIZE
from aitchinson_flow.data.transforms.discrete import token_ids_to_ilr_x
from aitchinson_flow.training.datamodule import DataModule


def _normalize_answer(text: str, max_len: int) -> str:
    """Lowercase, keep only a-z and space, truncate to max_len."""
    cleaned = "".join(c if c in CHAR2ID else " " for c in text.lower())
    return cleaned[:max_len]


def _text_to_ids(text: str, L: int) -> Tensor:
    """Char-level encode and pad/truncate to length L."""
    ids = [CHAR2ID.get(c, CHAR2ID[" "]) for c in text]
    if len(ids) >= L:
        ids = ids[:L]
    else:
        ids = ids + [CHAR2ID[" "]] * (L - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def _load_trivia_answers(
    split: str,
    max_samples: int | None = None,
    seed: int = 0,
) -> list[str]:
    """Load a flat list of correct answer strings from TriviaQA (rc.nocontext)."""
    import datasets  # noqa: PLC0415

    ds = datasets.load_dataset("trivia_qa", "rc.nocontext", split=split, trust_remote_code=True)
    if max_samples is not None:
        ds = ds.shuffle(seed=seed).select(range(min(max_samples, len(ds))))

    answers: list[str] = []
    for row in ds:
        aliases: list[str] = row["answer"]["aliases"]
        if aliases:
            answers.append(aliases[0])
        elif row["answer"]["value"]:
            answers.append(row["answer"]["value"])

    return answers


class _TriviaTrainDataset(Dataset[dict[str, Tensor]]):
    """Valid (correct) answers only — no invalid samples."""

    def __init__(self, answers: list[str], L: int, K: int, eps: float = 1e-8) -> None:
        self.answers = answers
        self.L = L
        self.K = K
        self.eps = eps

    def __len__(self) -> int:
        return len(self.answers)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        ids = _text_to_ids(_normalize_answer(self.answers[idx], self.L), self.L)
        log_x = token_ids_to_ilr_x(ids, self.K, eps=self.eps)
        return {"log_x": log_x, "token_ids": ids}


class _TriviaEvalDataset(Dataset[dict[str, Tensor]]):
    """Interleaved correct (label=0) + incorrect (label=1) answers.

    Incorrect answers are sampled from *other* questions in the same set.
    The even indices are correct, the odd are incorrect, so every pair
    (2i, 2i+1) shares the same query slot.
    """

    def __init__(self, answers: list[str], L: int, K: int, eps: float = 1e-8, seed: int = 0) -> None:
        rng = random.Random(seed)
        n = len(answers)
        # Build incorrect answers: for position i, sample j != i
        shuffled = list(range(n))
        rng.shuffle(shuffled)
        # Ensure no index maps to itself
        incorrect_idx = [(j if j != i else (j + 1) % n) for i, j in enumerate(shuffled)]

        self.correct = answers
        self.incorrect = [answers[j] for j in incorrect_idx]
        self.L = L
        self.K = K
        self.eps = eps

    def __len__(self) -> int:
        return 2 * len(self.correct)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        is_incorrect = idx % 2 == 1
        answer_idx = idx // 2
        text = self.incorrect[answer_idx] if is_incorrect else self.correct[answer_idx]
        ids = _text_to_ids(_normalize_answer(text, self.L), self.L)
        log_x = token_ids_to_ilr_x(ids, self.K, eps=self.eps)
        label = torch.tensor(1 if is_incorrect else 0, dtype=torch.long)
        return {"log_x": log_x, "token_ids": ids, "label": label}


@dataclass
class TriviaDataModuleConfig:
    max_train_samples: int | None = 10_000
    max_val_samples: int | None = 2_000
    max_test_samples: int | None = 2_000
    shuffle_seed: int = 0
    log_simplex_eps: float = 1e-8


class TriviaDataModule(DataModule):
    """Trivia Q&A datamodule (char-level, 27-token vocabulary).

    Uses ``cfg.dataset.L`` and ``cfg.dataset.K`` (should be 27 for char-level).
    Train returns correct answers only; val/test return labeled correct+incorrect pairs.
    """

    def __init__(self, cfg: Config, *, trivia_cfg: TriviaDataModuleConfig | None = None) -> None:
        self.cfg = cfg
        self.tcfg = trivia_cfg or TriviaDataModuleConfig()
        self._train_ds: _TriviaTrainDataset | None = None
        self._val_ds: _TriviaEvalDataset | None = None
        self._test_ds: _TriviaEvalDataset | None = None

    def _ensure_loaded(self) -> None:
        if self._train_ds is not None:
            return
        L = self.cfg.dataset.L
        K = self.cfg.dataset.K
        eps = self.tcfg.log_simplex_eps

        train_answers = _load_trivia_answers(
            "train",
            max_samples=self.tcfg.max_train_samples,
            seed=self.tcfg.shuffle_seed,
        )
        val_answers = _load_trivia_answers(
            "validation",
            max_samples=self.tcfg.max_val_samples,
            seed=self.tcfg.shuffle_seed,
        )

        self._train_ds = _TriviaTrainDataset(train_answers, L, K, eps)
        self._val_ds = _TriviaEvalDataset(val_answers, L, K, eps, seed=self.tcfg.shuffle_seed)
        self._test_ds = _TriviaEvalDataset(val_answers, L, K, eps, seed=self.tcfg.shuffle_seed + 1)

    def train_dataloader(self) -> DataLoader:  # type: ignore[override]
        self._ensure_loaded()
        assert self._train_ds is not None
        return DataLoader(
            self._train_ds,
            batch_size=self.cfg.training.B,
            shuffle=True,
            num_workers=self.cfg.training.num_workers,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader | None:  # type: ignore[override]
        self._ensure_loaded()
        assert self._val_ds is not None
        return DataLoader(
            self._val_ds,
            batch_size=self.cfg.training.B,
            shuffle=False,
            num_workers=self.cfg.training.num_workers,
        )

    def test_dataloader(self) -> DataLoader | None:  # type: ignore[override]
        self._ensure_loaded()
        assert self._test_ds is not None
        return DataLoader(
            self._test_ds,
            batch_size=self.cfg.training.B,
            shuffle=False,
            num_workers=self.cfg.training.num_workers,
        )
