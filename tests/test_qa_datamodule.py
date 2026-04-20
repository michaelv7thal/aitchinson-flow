"""Offline tests for Path B Q+A negatives and the QA datamodule.

No HuggingFace network calls — we construct the datamodule's internal
state directly from synthetic rows so these tests stay hermetic.
"""

from __future__ import annotations

import math
import random
from dataclasses import replace

import torch

from aitchinson_flow.config import Config, DatasetConfig
from aitchinson_flow.data.byte_vocab import ETX, EOT, STX
from aitchinson_flow.data.qa_datamodule import (
    QAPairsDataModule,
    _QAEvalDataset,
    _QATrainDataset,
    _make_train_collate,
)
from aitchinson_flow.data.qa_negatives import cross_question_swap


def _byte_cfg(L: int = 64, B: int = 4) -> Config:
    base = Config()
    ds = replace(base.dataset, K=256, L=L)
    tr = replace(base.training, B=B, num_workers=0)
    return replace(base, dataset=ds, training=tr)


def _sample_rows(n: int) -> list[dict]:
    rng = random.Random(0)
    qs = [f"question number {i} about topic {rng.randrange(1000)}" for i in range(n)]
    ans = [f"answer_{i}" for i in range(n)]
    return [
        {"question": q, "answer": a, "aliases": [a]}
        for q, a in zip(qs, ans, strict=True)
    ]


class TestCrossQuestionSwap:
    def test_shape_and_no_self_pair(self) -> None:
        rows = _sample_rows(6)
        qs = [r["question"] for r in rows]
        ans = [r["answer"] for r in rows]
        log_x, ids, mask = cross_question_swap(
            qs, ans,
            max_question_bytes=48, max_answer_bytes=16, L=64,
            K=256, eps=1e-8, label_smoothing=0.0, transform_mode="ilr",
            seed=42,
        )
        assert log_x.shape == (6, 64, 255)
        assert ids.shape == (6, 64)
        assert mask.shape == (6, 64)
        # Each row's answer-span bytes must come from a DIFFERENT row's answer.
        for i in range(6):
            answer_bytes = ids[i][mask[i]].tolist()
            # Decode to string; must not equal the original i-th answer.
            recovered = bytes(answer_bytes).decode("utf-8", errors="replace")
            assert recovered != ans[i]
            assert recovered in ans

    def test_requires_batch_ge_2(self) -> None:
        import pytest

        with pytest.raises(ValueError):
            cross_question_swap(
                ["only one q"], ["only one a"],
                max_question_bytes=32, max_answer_bytes=16, L=64,
                K=256, eps=1e-8, label_smoothing=0.0, transform_mode="ilr",
                seed=0,
            )


class TestTrainCollate:
    def test_batch_keys_and_shapes(self) -> None:
        cfg = _byte_cfg(L=64, B=4)
        rows = _sample_rows(16)
        ds = _QATrainDataset(rows, cfg)
        collate = _make_train_collate(cfg)
        batch = collate([ds[i] for i in range(4)])
        assert set(batch.keys()) == {"log_x", "token_ids", "answer_mask", "log_x_invalid"}
        assert batch["log_x"].shape == (4, 64, 255)
        assert batch["token_ids"].shape == (4, 64)
        assert batch["answer_mask"].shape == (4, 64)
        assert batch["log_x_invalid"].shape == (4, 64, 255)
        for key in ("log_x", "log_x_invalid"):
            assert torch.isfinite(batch[key]).all()

    def test_role_markers_in_positive_tokens(self) -> None:
        cfg = _byte_cfg(L=64, B=4)
        rows = _sample_rows(8)
        ds = _QATrainDataset(rows, cfg)
        collate = _make_train_collate(cfg)
        batch = collate([ds[i] for i in range(4)])
        ids = batch["token_ids"]
        # Every row must start with STX and contain ETX and EOT.
        assert (ids[:, 0] == STX).all()
        for b in range(4):
            row_ids = ids[b].tolist()
            assert ETX in row_ids
            assert EOT in row_ids

    def test_invalid_differs_from_positive(self) -> None:
        cfg = _byte_cfg(L=64, B=4)
        rows = _sample_rows(8)
        ds = _QATrainDataset(rows, cfg)
        collate = _make_train_collate(cfg)
        batch = collate([ds[i] for i in range(4)])
        diff = (batch["log_x"] - batch["log_x_invalid"]).abs().sum().item()
        assert diff > 0.0


class TestEvalDataset:
    def test_interleaves_labels(self) -> None:
        cfg = _byte_cfg(L=64, B=4)
        rows = _sample_rows(4)
        llm = [f"wrong_{i}" for i in range(4)]
        ds = _QAEvalDataset(rows, llm, cfg)
        assert len(ds) == 8
        for i, item in enumerate(ds):
            assert item["log_x"].shape == (64, 255)
            assert item["answer_mask"].shape == (64,)
            expected = 1 if i % 2 == 1 else 0
            assert int(item["label"]) == expected


class TestQAPairsDataModulePlaceholder:
    """End-to-end shape check using synthetic rows (no HF load)."""

    def test_val_loader_with_placeholder_llm(self) -> None:
        cfg = _byte_cfg(L=64, B=4)
        dm = QAPairsDataModule(cfg, skip_llm=True)
        rows = _sample_rows(8)
        dm._train_rows = rows
        dm._val_rows = rows
        dm._test_rows = []
        val_loader = dm.val_dataloader()
        assert val_loader is not None
        batch = next(iter(val_loader))
        assert batch["log_x"].shape[0] == 4
        assert batch["log_x"].shape[-1] == 255
        assert batch["answer_mask"].shape == (4, 64)
        assert set(batch.keys()) >= {"log_x", "token_ids", "answer_mask", "label"}
        assert torch.isfinite(batch["log_x"]).all()

    def test_train_loader_produces_log_x_invalid(self) -> None:
        cfg = _byte_cfg(L=64, B=4)
        dm = QAPairsDataModule(cfg, skip_llm=True)
        rows = _sample_rows(16)
        dm._train_rows = rows
        dm._val_rows = []
        dm._test_rows = []
        loader = dm.train_dataloader()
        batch = next(iter(loader))
        assert "log_x_invalid" in batch
        assert batch["log_x_invalid"].shape == batch["log_x"].shape

    def test_rejects_non_byte_K(self) -> None:
        import pytest

        base = Config()
        bad = replace(base, dataset=replace(base.dataset, K=27, L=64))
        with pytest.raises(ValueError):
            QAPairsDataModule(bad)
