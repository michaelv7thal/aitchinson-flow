"""Unit tests for the byte-level codec used by Path B (Q+A auditor)."""

from __future__ import annotations

import torch

from aitchinson_flow.data.byte_vocab import (
    EOT,
    ETX,
    PAD,
    STX,
    VOCAB_SIZE,
    byte_ids_to_text,
    concat_qa_bytes,
    text_to_byte_ids,
)


class TestByteCodec:
    def test_vocab_size_is_256(self) -> None:
        assert VOCAB_SIZE == 256

    def test_ascii_roundtrip(self) -> None:
        s = "the capital of france is paris"
        assert byte_ids_to_text(text_to_byte_ids(s, max_len=64)) == s

    def test_utf8_roundtrip_with_accents(self) -> None:
        s = "café naïve résumé"
        assert byte_ids_to_text(text_to_byte_ids(s, max_len=64)) == s

    def test_digits_and_punctuation_preserved(self) -> None:
        s = "In 1492, Columbus sailed; he was 41 y/o."
        assert byte_ids_to_text(text_to_byte_ids(s, max_len=128)) == s

    def test_reserved_bytes_stripped_from_input(self) -> None:
        raw = "abc" + chr(STX) + chr(ETX) + chr(EOT) + chr(PAD) + "xyz"
        ids = text_to_byte_ids(raw, max_len=64)
        assert STX not in ids and ETX not in ids and EOT not in ids and PAD not in ids
        assert byte_ids_to_text(ids) == "abcxyz"

    def test_truncation_respects_max_len(self) -> None:
        s = "a" * 100
        ids = text_to_byte_ids(s, max_len=10)
        assert len(ids) == 10


class TestConcatQA:
    def test_shapes(self) -> None:
        ids, mask = concat_qa_bytes(
            "capital of france",
            "paris",
            max_question_bytes=32,
            max_answer_bytes=16,
            L=64,
        )
        assert ids.shape == (64,)
        assert mask.shape == (64,)
        assert ids.dtype == torch.long
        assert mask.dtype == torch.bool

    def test_role_markers_present(self) -> None:
        ids, _ = concat_qa_bytes(
            "q",
            "a",
            max_question_bytes=32,
            max_answer_bytes=16,
            L=64,
        )
        ids_list = ids.tolist()
        assert ids_list[0] == STX
        assert ETX in ids_list
        assert EOT in ids_list

    def test_answer_mask_marks_only_answer_bytes(self) -> None:
        q = "what is the capital of france"
        a = "paris"
        ids, mask = concat_qa_bytes(
            q, a, max_question_bytes=64, max_answer_bytes=16, L=128
        )
        # Decode only the masked positions — should equal the answer bytes
        masked_ids = ids[mask].tolist()
        assert bytes(masked_ids).decode("utf-8") == a

    def test_padding_bytes_past_content(self) -> None:
        ids, mask = concat_qa_bytes(
            "q", "a", max_question_bytes=8, max_answer_bytes=8, L=32
        )
        content_len = 1 + 1 + 1 + 1 + 1  # STX + q + ETX + a + EOT
        assert (ids[content_len:] == PAD).all()
        assert not mask[content_len:].any()

    def test_truncation_when_L_too_small(self) -> None:
        ids, mask = concat_qa_bytes(
            "a long question that will be truncated",
            "short",
            max_question_bytes=64,
            max_answer_bytes=16,
            L=8,
        )
        assert ids.shape == (8,)
        assert mask.shape == (8,)
