"""Cross-question-swap negatives for the Path B Q+A hallucination auditor.

For each row ``i`` in a batch of ``(question_i, answer_i)`` pairs, pair the
question with the answer from a different row ``j != i``. The resulting
``[Q_i][A_j]`` concatenation is a factually plausible but incorrect pair —
a cheaper training signal than invoking an LLM to generate hallucinations.
"""

from __future__ import annotations

import random

import torch
from torch import Tensor

from aitchinson_flow.data.byte_vocab import concat_qa_bytes
from aitchinson_flow.data.transforms.discrete import token_ids_to_features


def cross_question_swap(
    questions: list[str],
    answers: list[str],
    *,
    max_question_bytes: int,
    max_answer_bytes: int,
    L: int,
    K: int,
    eps: float,
    label_smoothing: float,
    transform_mode: str,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build ``(log_x_invalid, token_ids_invalid, answer_mask_invalid)``.

    Args:
        questions: Batch of question strings, length ``B``.
        answers: Batch of correct answer strings, length ``B``.
        max_question_bytes: Max UTF-8 bytes kept for each question.
        max_answer_bytes: Max UTF-8 bytes kept for each answer.
        L: Total concatenated sequence length after padding.
        K: Vocabulary size (256 for byte-level).
        eps, label_smoothing, transform_mode: Forwarded to
            :func:`token_ids_to_features`.
        seed: RNG seed for the swap permutation (deterministic per batch).

    Returns:
        ``(log_x (B, L, F), token_ids (B, L), answer_mask (B, L))``.
    """
    n = len(questions)
    if len(answers) != n:
        raise ValueError(
            f"questions and answers must have the same length; got {len(questions)} and {len(answers)}"
        )
    if n < 2:
        raise ValueError(
            f"cross_question_swap requires batch >= 2 for distinct swap targets; got {n}"
        )

    rng = random.Random(seed)
    perm = list(range(n))
    rng.shuffle(perm)
    for i in range(n):
        if perm[i] == i:
            j = (i + 1) % n
            perm[i], perm[j] = perm[j], perm[i]

    log_x_rows: list[Tensor] = []
    id_rows: list[Tensor] = []
    mask_rows: list[Tensor] = []
    for i, j in enumerate(perm):
        ids, mask = concat_qa_bytes(
            questions[i],
            answers[j],
            max_question_bytes=max_question_bytes,
            max_answer_bytes=max_answer_bytes,
            L=L,
        )
        feats = token_ids_to_features(
            ids,
            K,
            eps=eps,
            label_smoothing=label_smoothing,
            transform_mode=transform_mode,
        )
        log_x_rows.append(feats)
        id_rows.append(ids)
        mask_rows.append(mask)

    return (
        torch.stack(log_x_rows, dim=0),
        torch.stack(id_rows, dim=0),
        torch.stack(mask_rows, dim=0),
    )
