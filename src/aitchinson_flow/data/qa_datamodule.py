"""Byte-level Q+A datamodule (Path B).

Train split emits ``{log_x, token_ids, answer_mask, log_x_invalid}`` with
cross-question-swap in-batch negatives. When contextual batching is enabled
(``cfg.qa_dataset.emit_question_context``), train/eval batches also include
``log_x_question`` and ``question_mask``.

Val/test splits emit ``{log_x, token_ids, answer_mask, label}`` with
interleaved correct (label=0) and LLM-generated (label=1) answers. Optional
``final_decision`` metadata (yes/no/maybe) is propagated for uncertainty-only
slice analysis.

Questions and answers are byte-encoded (UTF-8) and concatenated via
:func:`~aitchinson_flow.data.byte_vocab.concat_qa_bytes` as
``[STX] question [ETX] answer [EOT]`` padded to ``cfg.dataset.L``.

The LLM answer generator is invoked lazily in :meth:`setup_eval` and its
output is cached under ``cfg.answer_generator.cache_dir`` (if set) so
repeated eval runs skip regeneration.
"""

from __future__ import annotations

import csv
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
from aitchinson_flow.data.byte_vocab import concat_qa_bytes, encode_text_bytes
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


def _coerce_context(value: Any, *, joiner: str) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        chunks = [str(v).strip() for v in value if str(v).strip()]
        return joiner.join(chunks)
    return str(value).strip()


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _coerce_label(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value != 0)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "hallucinated", "invalid", "incorrect"}:
            return 1
        if token in {"0", "false", "no", "correct", "valid"}:
            return 0
    return None


def _coerce_final_decision(value: Any) -> str | None:
    if value is None:
        return None
    token = str(value).strip().lower()
    if token in {"yes", "no", "maybe"}:
        return token
    return None


def _final_decision_id(value: str | None) -> int:
    if value == "yes":
        return 0
    if value == "no":
        return 1
    if value == "maybe":
        return 2
    return -1


def _map_maybe_to_binary(decision: str | None, *, maybe_policy: str) -> int | None:
    if decision is None:
        return None
    if decision == "yes":
        return 0
    if decision == "no":
        return 1
    if decision != "maybe":
        return None
    if maybe_policy == "drop_maybe":
        return None
    if maybe_policy == "treat_maybe_incorrect":
        return 1
    if maybe_policy == "treat_maybe_correct":
        return 0
    if maybe_policy == "separate_split":
        return None
    return None


def _load_qa_rows(
    hf_path: str,
    *,
    name: str | None,
    split: str,
    question_col: str,
    answer_col: str,
    aliases_col: str | None,
    context_col: str | None,
    include_context_in_question: bool,
    context_joiner: str,
    final_decision_col: str | None,
    maybe_policy: str,
    max_samples: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    """Load ``(question, answer, aliases)`` rows from a HuggingFace dataset."""
    import datasets  # noqa: PLC0415

    ds = datasets.load_dataset(hf_path, name, split=split)
    if max_samples is not None:
        ds = ds.shuffle(seed=seed).select(range(min(max_samples, len(ds))))

    rows: list[dict[str, Any]] = []
    for raw in ds:
        q = _dotted_get(raw, question_col)
        if not isinstance(q, str) or not q:
            continue
        context_text = (
            _coerce_context(_dotted_get(raw, context_col), joiner=context_joiner)
            if context_col
            else ""
        )
        question_context = q
        if include_context_in_question and context_text:
            question_context = f"{q.strip()}\n\n{context_text}"
        a = _coerce_answer(_dotted_get(raw, answer_col))
        if a is None:
            continue
        aliases = _coerce_aliases(_dotted_get(raw, aliases_col)) if aliases_col else []
        final_decision = (
            _coerce_final_decision(_dotted_get(raw, final_decision_col))
            if final_decision_col is not None
            else None
        )
        if final_decision == "maybe" and maybe_policy == "drop_maybe":
            continue
        rows.append(
            {
                "question": q,
                "question_context": question_context,
                "answer": a,
                "aliases": aliases,
                "final_decision": final_decision,
                "binary_decision_label": _map_maybe_to_binary(
                    final_decision, maybe_policy=maybe_policy
                ),
            }
        )
    return rows


def _load_external_eval_rows(
    path: str,
    *,
    question_col: str,
    answer_col: str,
    label_col: str | None,
    reference_answer_col: str | None,
    max_samples: int | None,
) -> list[dict[str, Any]]:
    """Load external eval rows as ``{question, answer, label}``.

    Supports JSON (list), JSONL, and CSV. Labels are read from ``label_col``
    when available; otherwise, if ``reference_answer_col`` is supplied, labels
    are inferred via normalized exact match between answer/reference.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"External eval file not found: {p}")

    suffix = p.suffix.lower()
    raw_rows: list[dict[str, Any]] = []
    if suffix == ".json":
        payload = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("External JSON eval input must be a list of objects.")
        raw_rows = [r for r in payload if isinstance(r, dict)]
    elif suffix == ".jsonl":
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                raw_rows.append(row)
    elif suffix == ".csv":
        with p.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            raw_rows = [dict(r) for r in reader]
    else:
        raise ValueError(f"Unsupported external eval format: {suffix}. Use .json, .jsonl, or .csv.")

    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        question = _dotted_get(raw, question_col)
        answer = _dotted_get(raw, answer_col)
        if not isinstance(question, str) or not question.strip():
            continue
        answer_text = _coerce_answer(answer)
        if answer_text is None:
            continue

        label = _coerce_label(_dotted_get(raw, label_col)) if label_col else None
        if label is None and reference_answer_col:
            ref = _coerce_answer(_dotted_get(raw, reference_answer_col))
            if ref is not None:
                label = int(_normalize_text(answer_text) != _normalize_text(ref))
        if label is None:
            continue

        rows.append(
            {
                "question": question,
                "question_context": question,
                "answer": answer_text,
                "label": int(label),
                "final_decision": _coerce_final_decision(_dotted_get(raw, "final_decision")),
            }
        )
        if max_samples is not None and len(rows) >= max_samples:
            break
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


def _encode_contextual_pair(
    question_context: str,
    answer: str,
    *,
    cfg: Config,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Encode one contextual QA pair into answer/question feature streams.

    Returns ``(answer_feats, answer_ids, answer_mask, question_feats, question_pad_mask)``
    where ``question_pad_mask`` uses MultiheadAttention convention
    (True means padding position should be ignored).
    """
    qa = cfg.qa_dataset
    hf = cfg.hf_dataset

    answer_ids, answer_mask = encode_text_bytes(
        answer,
        max_text_bytes=qa.max_answer_bytes,
        L=cfg.dataset.L,
    )
    question_ids, question_content_mask = encode_text_bytes(
        question_context,
        max_text_bytes=qa.max_question_bytes + qa.max_context_bytes,
        L=cfg.dataset.L,
    )
    answer_feats = token_ids_to_features(
        answer_ids,
        cfg.dataset.K,
        eps=qa.log_simplex_eps,
        label_smoothing=hf.label_smoothing,
        transform_mode=hf.transform_mode,
    )
    question_feats = token_ids_to_features(
        question_ids,
        cfg.dataset.K,
        eps=qa.log_simplex_eps,
        label_smoothing=hf.label_smoothing,
        transform_mode=hf.transform_mode,
    )
    question_pad_mask = ~question_content_mask
    return answer_feats, answer_ids, answer_mask, question_feats, question_pad_mask


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
        if self.cfg.qa_dataset.emit_question_context:
            feats, ids, mask, q_feats, q_mask = _encode_contextual_pair(
                row["question_context"], row["answer"], cfg=self.cfg
            )
            return {
                "question": row["question"],
                "question_context": row["question_context"],
                "answer": row["answer"],
                "log_x": feats,
                "token_ids": ids,
                "answer_mask": mask,
                "log_x_question": q_feats,
                "question_mask": q_mask,
                "final_decision": torch.tensor(
                    _final_decision_id(row.get("final_decision")), dtype=torch.long
                ),
            }

        feats, ids, mask = _encode_pair(row["question"], row["answer"], cfg=self.cfg)
        return {
            "question": row["question"],
            "question_context": row["question_context"],
            "answer": row["answer"],
            "log_x": feats,
            "token_ids": ids,
            "answer_mask": mask,
            "final_decision": torch.tensor(
                _final_decision_id(row.get("final_decision")), dtype=torch.long
            ),
        }


def _make_train_collate(cfg: Config) -> Any:
    qa = cfg.qa_dataset
    hf = cfg.hf_dataset

    def collate(items: list[dict[str, Any]]) -> dict[str, Tensor]:
        questions = [it["question"] for it in items]
        question_contexts = [it["question_context"] for it in items]
        answers = [it["answer"] for it in items]
        log_x = torch.stack([it["log_x"] for it in items], dim=0)
        token_ids = torch.stack([it["token_ids"] for it in items], dim=0)
        answer_mask = torch.stack([it["answer_mask"] for it in items], dim=0)
        final_decision = torch.stack([it["final_decision"] for it in items], dim=0)

        if cfg.qa_dataset.emit_question_context:
            log_x_question = torch.stack([it["log_x_question"] for it in items], dim=0)
            question_mask = torch.stack([it["question_mask"] for it in items], dim=0)

            n = len(items)
            perm_list = torch.randperm(n).tolist()
            if n > 1:
                fixed = [i for i, j in enumerate(perm_list) if i == j]
                for i in fixed:
                    j = (i + 1) % n
                    perm_list[i], perm_list[j] = perm_list[j], perm_list[i]
            invalid_answers = [answers[j] for j in perm_list]
            invalid_rows = [
                _encode_contextual_pair(question_contexts[i], invalid_answers[i], cfg=cfg)[0]
                for i in range(n)
            ]
            log_x_invalid = torch.stack(invalid_rows, dim=0)
            return {
                "log_x": log_x,
                "token_ids": token_ids,
                "answer_mask": answer_mask,
                "log_x_question": log_x_question,
                "question_mask": question_mask,
                "log_x_invalid": log_x_invalid,
                "final_decision": final_decision,
            }

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
            "final_decision": final_decision,
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
        final_decision = torch.tensor(
            _final_decision_id(row.get("final_decision")), dtype=torch.long
        )
        if self.cfg.qa_dataset.emit_question_context:
            feats, ids, mask, q_feats, q_mask = _encode_contextual_pair(
                row["question_context"], answer, cfg=self.cfg
            )
            return {
                "log_x": feats,
                "token_ids": ids,
                "answer_mask": mask,
                "log_x_question": q_feats,
                "question_mask": q_mask,
                "label": torch.tensor(1 if is_hallucinated else 0, dtype=torch.long),
                "final_decision": final_decision,
            }
        feats, ids, mask = _encode_pair(row["question"], answer, cfg=self.cfg)
        return {
            "log_x": feats,
            "token_ids": ids,
            "answer_mask": mask,
            "label": torch.tensor(1 if is_hallucinated else 0, dtype=torch.long),
            "final_decision": final_decision,
        }


class _QAExternalEvalDataset(Dataset[dict[str, Tensor]]):
    """Eval dataset from externally supplied `(question, answer, label)` rows."""

    def __init__(self, rows: list[dict[str, Any]], cfg: Config) -> None:
        self.rows = rows
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        row = self.rows[idx]
        final_decision = torch.tensor(
            _final_decision_id(row.get("final_decision")), dtype=torch.long
        )
        if self.cfg.qa_dataset.emit_question_context:
            feats, ids, mask, q_feats, q_mask = _encode_contextual_pair(
                row["question_context"], row["answer"], cfg=self.cfg
            )
            return {
                "log_x": feats,
                "token_ids": ids,
                "answer_mask": mask,
                "log_x_question": q_feats,
                "question_mask": q_mask,
                "label": torch.tensor(int(row["label"]), dtype=torch.long),
                "final_decision": final_decision,
            }
        feats, ids, mask = _encode_pair(row["question"], row["answer"], cfg=self.cfg)
        return {
            "log_x": feats,
            "token_ids": ids,
            "answer_mask": mask,
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "final_decision": final_decision,
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

    def _ensure_train_rows_loaded(self) -> None:
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
            context_col=qa.context_col,
            include_context_in_question=qa.include_context_in_question,
            context_joiner=qa.context_joiner,
            final_decision_col=qa.final_decision_col,
            maybe_policy=qa.maybe_policy,
            max_samples=qa.max_train_samples,
            seed=qa.shuffle_seed,
        )

    def _ensure_val_rows_loaded(self) -> None:
        if self._val_rows is not None:
            return
        qa = self.cfg.qa_dataset
        self._val_rows = (
            _load_qa_rows(
                qa.hf_path,
                name=qa.name,
                split=qa.split_val,
                question_col=qa.question_col,
                answer_col=qa.answer_col,
                aliases_col=qa.aliases_col,
                context_col=qa.context_col,
                include_context_in_question=qa.include_context_in_question,
                context_joiner=qa.context_joiner,
                final_decision_col=qa.final_decision_col,
                maybe_policy=qa.maybe_policy,
                max_samples=qa.max_val_samples,
                seed=qa.shuffle_seed + 1,
            )
            if qa.split_val is not None
            else []
        )

    def _ensure_test_rows_loaded(self) -> None:
        if self._test_rows is not None:
            return
        qa = self.cfg.qa_dataset
        self._test_rows = (
            _load_qa_rows(
                qa.hf_path,
                name=qa.name,
                split=qa.split_test,
                question_col=qa.question_col,
                answer_col=qa.answer_col,
                aliases_col=qa.aliases_col,
                context_col=qa.context_col,
                include_context_in_question=qa.include_context_in_question,
                context_joiner=qa.context_joiner,
                final_decision_col=qa.final_decision_col,
                maybe_policy=qa.maybe_policy,
                max_samples=qa.max_test_samples,
                seed=qa.shuffle_seed + 2,
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
                max_samples=(qa.max_val_samples if split == qa.split_val else qa.max_test_samples),
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
            new_ids = gen_ids[0, prompt_ids.shape[1] :]
            decoded = lm.decode(new_ids, skip_special_tokens=True)
            text = decoded[0] if decoded else ""
            first_line = text.strip().splitlines()[0] if text.strip() else ""
            answers.append(first_line)

        if cache_path is not None:
            _save_cached_answers(cache_path, answers)
        return answers

    def train_dataloader(self) -> DataLoader:  # type: ignore[override]
        self._ensure_train_rows_loaded()
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
        self._ensure_val_rows_loaded()
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
        self._ensure_test_rows_loaded()
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
        self._ensure_train_rows_loaded()
        assert self._train_rows is not None
        return len(self._train_rows)


class QAExternalEvalDataModule(DataModule):
    """Datamodule for external target-domain QA hallucination eval batches."""

    def __init__(
        self,
        cfg: Config,
        *,
        external_path: str,
        question_col: str = "question",
        answer_col: str = "answer",
        label_col: str | None = "label",
        reference_answer_col: str | None = None,
        max_samples: int | None = None,
    ) -> None:
        if cfg.dataset.K != 256:
            raise ValueError(
                f"QAExternalEvalDataModule requires cfg.dataset.K == 256 (byte-level); got {cfg.dataset.K}"
            )
        self.cfg = cfg
        self.rows = _load_external_eval_rows(
            external_path,
            question_col=question_col,
            answer_col=answer_col,
            label_col=label_col,
            reference_answer_col=reference_answer_col,
            max_samples=max_samples,
        )
        if not self.rows:
            raise ValueError(
                "No valid external eval rows loaded; ensure question/answer and label "
                "or reference-answer columns are present."
            )
        self._ds = _QAExternalEvalDataset(self.rows, cfg)

    def train_dataloader(self) -> DataLoader:  # type: ignore[override]
        return DataLoader(
            self._ds,
            batch_size=self.cfg.training.B,
            shuffle=False,
            num_workers=self.cfg.training.num_workers,
        )

    def val_dataloader(self) -> DataLoader | None:  # type: ignore[override]
        return None

    def test_dataloader(self) -> DataLoader | None:  # type: ignore[override]
        return DataLoader(
            self._ds,
            batch_size=self.cfg.training.B,
            shuffle=False,
            num_workers=self.cfg.training.num_workers,
        )
