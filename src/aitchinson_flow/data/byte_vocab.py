"""UTF-8 byte-level vocabulary for Path B (Q+A hallucination auditor).

A vocab of 256 bytes lets any HuggingFace LM output be losslessly re-encoded
via ``tokenizer.decode(...).encode("utf-8")``. Reserved ASCII control bytes
are used as role markers since they almost never appear in natural text:

- ``STX (0x02)`` — question-start marker
- ``ETX (0x03)`` — answer-start marker
- ``EOT (0x04)`` — end marker
- ``PAD (0x00)`` — right-pad filler
"""

from __future__ import annotations

import torch
from torch import Tensor


VOCAB_SIZE: int = 256

PAD: int = 0x00
STX: int = 0x02
ETX: int = 0x03
EOT: int = 0x04

RESERVED: frozenset[int] = frozenset({PAD, STX, ETX, EOT})


def text_to_byte_ids(text: str, max_len: int) -> list[int]:
    """UTF-8 encode ``text``, strip reserved role bytes, truncate to ``max_len``."""
    raw = text.encode("utf-8", errors="replace")
    ids = [b for b in raw if b not in RESERVED]
    return ids[:max_len]


def byte_ids_to_text(ids: Tensor | list[int]) -> str:
    """Decode a sequence of byte ids back to text, dropping pad/role markers."""
    if isinstance(ids, Tensor):
        ids = ids.tolist()
    payload = bytes(b for b in ids if b not in RESERVED)
    return payload.decode("utf-8", errors="replace")


def concat_qa_bytes(
    question: str,
    answer: str,
    *,
    max_question_bytes: int,
    max_answer_bytes: int,
    L: int,
) -> tuple[Tensor, Tensor]:
    """Encode ``[STX] q_bytes [ETX] a_bytes [EOT]`` padded to length ``L``.

    Returns ``(token_ids (L,), answer_mask (L,))`` where ``answer_mask`` is
    True exactly on the answer-span positions (the bytes between ``ETX`` and
    ``EOT``, exclusive of both markers). The caller uses this to restrict the
    Stage 2 GP loss to answer tokens.
    """
    q_bytes = text_to_byte_ids(question, max_question_bytes)
    a_bytes = text_to_byte_ids(answer, max_answer_bytes)

    tokens: list[int] = [STX, *q_bytes, ETX, *a_bytes, EOT]
    mask: list[bool] = (
        [False] * (1 + len(q_bytes) + 1)  # STX, question, ETX
        + [True] * len(a_bytes)
        + [False]  # EOT
    )

    if len(tokens) >= L:
        tokens = tokens[:L]
        mask = mask[:L]
    else:
        pad = L - len(tokens)
        tokens = tokens + [PAD] * pad
        mask = mask + [False] * pad

    return (
        torch.tensor(tokens, dtype=torch.long),
        torch.tensor(mask, dtype=torch.bool),
    )
