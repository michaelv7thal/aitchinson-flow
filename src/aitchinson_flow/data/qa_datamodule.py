"""Byte-level Q+A datamodule (Path B).

Train split emits ``{log_x, token_ids, answer_mask, log_x_invalid}`` with
cross-question-swap in-batch negatives. Val/test splits emit
``{log_x, token_ids, answer_mask, label}`` with interleaved correct (label=0)
and LLM-generated (label=1) answers.

Questions and answers are byte-encoded (UTF-8) and concatenated via
:func:`~aitchinson_flow.data.byte_vocab.concat_qa_bytes` as
``[STX] question [ETX] answer [EOT]`` padded to ``cfg.dataset.L``.

The LLM answer generator is invoked lazily in :meth:`setup_eval` and its
output is cached under ``cfg.answer_generator.cache_dir`` (if set) so
repeated eval runs skip regeneration.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from aitchinson_flow.config import Config
from aitchinson_flow.data.byte_vocab import concat_qa_bytes
from aitchinson_flow.data.qa_negatives import cross_question_swap
from aitchinson_flow.data.transforms.discrete import token_ids_to_features
from aitchinson_flow.training.datamodule import DataModule


def _dotted_get(row: dict[str, Any], path: str) -> Any:
    """Walk a dotted-path through nested dicts, e.g. ``"answer.value"``."""
    node: Any = row
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _coerce_answer(value: Any) -> str | None:
    """Normalize a dataset answer cell into a non-empty string, or None."""
    if value is None:
        return None
    if isinstance(value, str):
        return value if value else None
    if isinstance(value, (list, tuple)):
        for item in value:
            s = _coerce_answer(item)
            if s is not None:
                return s
        return None
    return str(value)


def _coerce_aliases(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple)):
        return [s for s in (str(v) for v in value) if s]
    return [str(value)]


def _load_qa_rows(
    hf_path: str,
    *,
    name: str | None,
    split: str,
    question_col: str,
    answer_col: str,
    aliases_col: str | None,
    max_samples: int | None,
    seed: int,
    trust_remote_code: bool,
) -> list[dict[str, Any]]:
    """Load ``(question, answer, aliases)`` rows from a HuggingFace dataset."""
    import datasets  # noqa: PLC0415

    ds = datasets.load_dataset(
        hf_path, name, split=split, trust_remote_code=trust_remote_code
    )
    if max_samples is not None:
        ds = ds.shuffle(seed=seed).select(range(min(max_samples, len(ds))))

    rows: list[dict[str, Any]] = []
    for raw in ds:
        q = _dotted_get(raw, question_col)
        if not isinstance(q, str) or not q:
            continue
        a = _coerce_answer(_dotted_get(raw, answer_col))
        if a is None:
            continue
        aliases = _coerce_aliases(_dotted_get(raw, aliases_col)) if aliases_col else []
        rows.append({"question": q, "answer": a, "aliases": aliases})
    return rows


@dataclass
class _QACollatedBatch:
    """Dict keys returned by the QA datamodule collate function.

    Train: ``log_x, token_ids, answer_mask, log_x_invalid``.
    Eval:  ``log_x, token_ids, answer_mask, label``.
    """


def _encode_pair(
    question: str,
    answer: str,
    *,
    cfg: Config,
) -> tuple[Tensor, Tensor, Tensor]:
    """Encode one ``(question, answer)`` into ``(log_x, token_ids, answer_mask)``."""
    qa = cfg.qa_dataset
    hf = cfg.hf_dataset
    ids, mask = concat_qa_bytes(
        question,
        answer,
        max_question_bytes=qa.max_question_bytes,
        max_answer_bytes=qa.max_answer_bytes,
        L=cfg.dataset.L,
    )
    feats = token_ids_to_features(
        ids,
        cfg.dataset.K,
        eps=qa.log_simplex_eps,
        label_smoothing=hf.label_smoothing,
        transform_mode=hf.transform_mode,
    )
    return feats, ids, mask


class _QATrainDataset(Dataset[dict[str, Any]]):
    """Emits raw ``(question, answer)`` plus the positive encoding.

    Invalid (negative) pairs are assembled by a custom collate function so
    cross-question swap can select a distinct partner from within the batch.
    """

    def __init__(self, rows: list[dict[str, Any]], cfg: Config) -> None:
        self.rows = rows
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        feats, ids, mask = _encode_pair(row["question"], row["answer"], cfg=self.cfg)
        return {
            "question": row["question"],
            "answer": row["answer"],
            "log_x": feats,
            "token_ids": ids,
            "answer_mask": mask,
        }


def _make_train_collate(cfg: Config) -> Any:
    qa = cfg.qa_dataset
    hf = cfg.hf_dataset

    def collate(items: list[dict[str, Any]]) -> dict[str, Tensor]:
        questions = [it["question"] for it in items]
        answers = [it["answer"] for it in items]
        log_x = torch.stack([it["log_x"] for it in items], dim=0)
        token_ids = torch.stack([it["token_ids"] for it in items], dim=0)
        answer_mask = torch.stack([it["answer_mask"] for it in items], dim=0)

        seed = qa.shuffle_seed + int(torch.randint(0, 2**30, (1,)).item())
        log_x_invalid, _, _ = cross_question_swap(
            questions,
            answers,
            max_question_bytes=qa.max_question_bytes,
            max_answer_bytes=qa.max_answer_bytes,
            L=cfg.dataset.L,
            K=cfg.dataset.K,
            eps=qa.log_simplex_eps,
            label_smoothing=hf.label_smoothing,
            transform_mode=hf.transform_mode,
            seed=seed,
        )
        return {
            "log_x": log_x,
            "token_ids": token_ids,
            "answer_mask": answer_mask,
            "log_x_invalid": log_x_invalid,
        }

    return collate


class _QAEvalDataset(Dataset[dict[str, Tensor]]):
    """Interleaved correct (label=0) + LLM-generated (label=1) pairs.

    Even indices hold the correct answer; odd indices hold the LLM answer for
    the same question. Each call to ``__getitem__`` returns a single item.
    """

    def __init__(
        self,
        rows: list[dict[str, Any]],
        llm_answers: list[str],
        cfg: Config,
    ) -> None:
        if len(llm_answers) != len(rows):
            raise ValueError(
                f"llm_answers length {len(llm_answers)} does not match rows length {len(rows)}"
            )
        self.rows = rows
        self.llm_answers = llm_answers
        self.cfg = cfg

    def __len__(self) -> int:
        return 2 * len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        row_idx = idx // 2
        is_hallucinated = idx % 2 == 1
        row = self.rows[row_idx]
        answer = self.llm_answers[row_idx] if is_hallucinated else row["answer"]
        feats, ids, mask = _encode_pair(row["question"], answer, cfg=self.cfg)
        return {
            "log_x": feats,
            "token_ids": ids,
            "answer_mask": mask,
            "label": torch.tensor(1 if is_hallucinated else 0, dtype=torch.long),
        }


def _cache_key(
    hf_path: str,
    name: str | None,
    split: str,
    model_id: str,
    revision: str | None,
    prompt_template: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    generation_seed: int,
    max_samples: int | None,
    shuffle_seed: int,
) -> str:
    payload = json.dumps(
        {
            "hf_path": hf_path,
            "name": name,
            "split": split,
            "model_id": model_id,
            "revision": revision,
            "prompt_template": prompt_template,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "generation_seed": generation_seed,
            "max_samples": max_samples,
            "shuffle_seed": shuffle_seed,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _load_cached_answers(cache_path: Path) -> list[str] | None:
    if not cache_path.exists():
        return None
    with cache_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_cached_answers(cache_path: Path, answers: list[str]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as fh:
        json.dump(answers, fh)


class QAPairsDataModule(DataModule):
    """Byte-level Q+A datamodule (Path B). Requires ``cfg.dataset.K == 256``.

    The LLM-answer generator runs lazily when :meth:`val_dataloader` /
    :meth:`test_dataloader` is first called. Pass ``skip_llm=True`` to build
    the datamodule without eval loaders (e.g. for training-only smoke runs
    that don't require LLM-generated negatives).
    """

    def __init__(self, cfg: Config, *, skip_llm: bool = False) -> None:
        if cfg.dataset.K != 256:
            raise ValueError(
                f"QAPairsDataModule requires cfg.dataset.K == 256 (byte-level); got {cfg.dataset.K}"
            )
        self.cfg = cfg
        self.skip_llm = skip_llm
        self._train_rows: list[dict[str, Any]] | None = None
        self._val_rows: list[dict[str, Any]] | None = None
        self._test_rows: list[dict[str, Any]] | None = None
        self._val_llm_answers: list[str] | None = None
        self._test_llm_answers: list[str] | None = None

    def _ensure_rows_loaded(self) -> None:
        if self._train_rows is not None:
            return
        qa = self.cfg.qa_dataset
        self._train_rows = _load_qa_rows(
            qa.hf_path,
            name=qa.name,
            split=qa.split_train,
            question_col=qa.question_col,
            answer_col=qa.answer_col,
            aliases_col=qa.aliases_col,
            max_samples=qa.max_train_samples,
            seed=qa.shuffle_seed,
            trust_remote_code=qa.trust_remote_code,
        )
        self._val_rows = (
            _load_qa_rows(
                qa.hf_path,
                name=qa.name,
                split=qa.split_val,
                question_col=qa.question_col,
                answer_col=qa.answer_col,
                aliases_col=qa.aliases_col,
                max_samples=qa.max_val_samples,
                seed=qa.shuffle_seed + 1,
                trust_remote_code=qa.trust_remote_code,
            )
            if qa.split_val is not None
            else []
        )
        self._test_rows = (
            _load_qa_rows(
                qa.hf_path,
                name=qa.name,
                split=qa.split_test,
                question_col=qa.question_col,
                answer_col=qa.answer_col,
                aliases_col=qa.aliases_col,
                max_samples=qa.max_test_samples,
                seed=qa.shuffle_seed + 2,
                trust_remote_code=qa.trust_remote_code,
            )
            if qa.split_test is not None
            else []
        )

    def _placeholder_llm_answers(self, rows: list[dict[str, Any]]) -> list[str]:
        """Deterministic cross-question swap fallback used when ``skip_llm=True``.

        Substitutes each correct answer with another row's correct answer so
        eval pipelines that don't want to instantiate an LLM still see
        label=1 negatives.
        """
        if not rows:
            return []
        rng = random.Random(self.cfg.qa_dataset.shuffle_seed + 17)
        n = len(rows)
        perm = list(range(n))
        rng.shuffle(perm)
        for i in range(n):
            if perm[i] == i:
                perm[i], perm[(i + 1) % n] = perm[(i + 1) % n], perm[i]
        return [rows[j]["answer"] for j in perm]

    def _generate_llm_answers(
        self,
        rows: list[dict[str, Any]],
        *,
        split: str,
    ) -> list[str]:
        if self.skip_llm or not rows:
            return self._placeholder_llm_answers(rows)

        gen = self.cfg.answer_generator
        qa = self.cfg.qa_dataset
        cache_path: Path | None = None
        if gen.cache_dir is not None:
            key = _cache_key(
                hf_path=qa.hf_path,
                name=qa.name,
                split=split,
                model_id=gen.model_id,
                revision=gen.revision,
                prompt_template=gen.prompt_template,
                temperature=gen.temperature,
                top_p=gen.top_p,
                max_new_tokens=gen.max_new_tokens,
                generation_seed=gen.generation_seed,
                max_samples=(
                    qa.max_val_samples if split == qa.split_val else qa.max_test_samples
                ),
                shuffle_seed=qa.shuffle_seed,
            )
            cache_path = Path(gen.cache_dir) / f"qa_answers_{split}_{key}.json"
            cached = _load_cached_answers(cache_path)
            if cached is not None and len(cached) == len(rows):
                return cached

        from aitchinson_flow.config import TeacherConfig  # noqa: PLC0415
        from aitchinson_flow.llms.registry import build_lm  # noqa: PLC0415

        teacher_cfg = TeacherConfig(
            model_id=gen.model_id,
            revision=gen.revision,
            trust_remote_code=gen.trust_remote_code,
            dtype=gen.dtype,
            device=gen.device,
        )
        lm = build_lm(gen.lm_key, teacher_cfg)

        # Cap prompt length generously; covers "Q: <question>\nA:".
        prompt_max_len = qa.max_question_bytes + len(gen.prompt_template) + 16
        if torch.manual_seed is not None:
            torch.manual_seed(gen.generation_seed)

        answers: list[str] = []
        for row in rows:
            prompt = gen.prompt_template.format(q=row["question"])
            prompt_ids, prompt_mask = lm.encode_text(prompt, max_length=prompt_max_len)
            gen_ids = lm.generate_ids(
                batch_size=prompt_ids.shape[0],
                max_new_tokens=gen.max_new_tokens,
                prompt_ids=prompt_ids,
                prompt_attention_mask=prompt_mask,
                temperature=gen.temperature,
                top_p=gen.top_p,
            )
            new_ids = gen_ids[0, prompt_ids.shape[1]:]
            decoded = lm.decode(new_ids, skip_special_tokens=True)
            text = decoded[0] if decoded else ""
            first_line = text.strip().splitlines()[0] if text.strip() else ""
            answers.append(first_line)

        if cache_path is not None:
            _save_cached_answers(cache_path, answers)
        return answers

    def train_dataloader(self) -> DataLoader:  # type: ignore[override]
        self._ensure_rows_loaded()
        assert self._train_rows is not None
        ds = _QATrainDataset(self._train_rows, self.cfg)
        return DataLoader(
            ds,
            batch_size=self.cfg.training.B,
            shuffle=True,
            num_workers=self.cfg.training.num_workers,
            drop_last=True,
            collate_fn=_make_train_collate(self.cfg),
        )

    def val_dataloader(self) -> DataLoader | None:  # type: ignore[override]
        self._ensure_rows_loaded()
        assert self._val_rows is not None
        if not self._val_rows:
            return None
        if self._val_llm_answers is None:
            self._val_llm_answers = self._generate_llm_answers(
                self._val_rows, split=self.cfg.qa_dataset.split_val or "validation"
            )
        ds = _QAEvalDataset(self._val_rows, self._val_llm_answers, self.cfg)
        return DataLoader(
            ds,
            batch_size=self.cfg.training.B,
            shuffle=False,
            num_workers=self.cfg.training.num_workers,
        )

    def test_dataloader(self) -> DataLoader | None:  # type: ignore[override]
        self._ensure_rows_loaded()
        assert self._test_rows is not None
        if not self._test_rows:
            return None
        if self._test_llm_answers is None:
            self._test_llm_answers = self._generate_llm_answers(
                self._test_rows, split=self.cfg.qa_dataset.split_test or "test"
            )
        ds = _QAEvalDataset(self._test_rows, self._test_llm_answers, self.cfg)
        return DataLoader(
            ds,
            batch_size=self.cfg.training.B,
            shuffle=False,
            num_workers=self.cfg.training.num_workers,
        )

    def num_train_samples(self) -> int | None:  # pragma: no cover - trivial
        self._ensure_rows_loaded()
        assert self._train_rows is not None
        return len(self._train_rows)
