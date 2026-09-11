"""HaluEval-QA loader and feature-cache utilities.

Phase K (TRAINING_PROTOCOL_v2.md §5) caches GPT-2 (or larger LM)
outputs on HaluEval-QA's (question, right_answer, hallucinated_answer)
triples. The cache lets the UQ classifier train without ever loading
the LM at training time.

Cache schema (all tensors on CPU, float32 unless noted):

* ``n_pairs``       : int — number of (question, right, hallucinated) triples
* ``lm``            : str — LM identifier, e.g. "gpt2"
* ``L``             : int — padded sequence length
* ``H``             : int — LM hidden dim (768 for gpt2)
* ``V``             : int — vocab size (50257 for gpt2)
* ``full_ids``      : (2*n_pairs, L) long — BPE token ids; even rows are
                      clean (right_answer), odd rows are hallucinated
* ``attn_mask``     : (2*n_pairs, L) bool — True at real tokens, False at pads
* ``hidden_states`` : (2*n_pairs, L, H) float — last-hidden state
* ``SE_pos``        : (2*n_pairs, L) float — per-position NLL of the
                      actually-placed token under the LM (AR-shifted; SE[0]=0)
* ``answer_mask``   : (2*n_pairs, L) bool — True only at *answer-span*
                      positions (i.e. tokens after the prompt prefix)
* ``label``         : (2*n_pairs,) bool — True if the row is hallucinated
* ``pair_id``       : (2*n_pairs,) long — same value for clean[i] and invalid[i]
* ``prompts``       : list[str] of length 2*n_pairs — for inspection

The prompt template is::

    "Question: {question}\\nAnswer: {answer}"

and the answer span starts at the first token *after* ``"\\nAnswer: "``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


def load_hallueval_qa(split: str = "data", max_n: int | None = None) -> list[dict[str, str]]:
    """Fetch HaluEval-QA triples from HuggingFace.

    Returns a list of dicts with keys: knowledge, question, right_answer,
    hallucinated_answer.
    """
    from datasets import load_dataset

    ds = load_dataset("pminervini/HaluEval", "qa", split=split)
    if max_n is not None and max_n < len(ds):
        ds = ds.select(range(max_n))
    rows = []
    for row in ds:
        rows.append({
            "knowledge": row.get("knowledge", "") or "",
            "question": row["question"],
            "right_answer": row["right_answer"],
            "hallucinated_answer": row["hallucinated_answer"],
        })
    return rows


def build_prompts(
    rows: list[dict[str, str]],
    *,
    include_knowledge: bool = False,
) -> tuple[list[str], list[str]]:
    """Build (clean_prompt, invalid_prompt) pairs.

    The prefix shared by both is ``"Question: {q}\\nAnswer: "``; the
    suffix is the answer text. The two prompts in a pair share the same
    prefix and differ only in the suffix.

    If ``include_knowledge`` is True, prepend ``"Knowledge: {k}\\n"`` —
    document this in the cache metadata so downstream eval knows the
    setup.
    """
    clean = []
    invalid = []
    for r in rows:
        prefix = ""
        if include_knowledge and r["knowledge"]:
            prefix = f"Knowledge: {r['knowledge']}\n"
        q_prefix = f"{prefix}Question: {r['question']}\nAnswer: "
        clean.append(q_prefix + r["right_answer"])
        invalid.append(q_prefix + r["hallucinated_answer"])
    return clean, invalid


def load_cache(cache_path: str | Path) -> dict[str, Any]:
    """Load a cache produced by ``scripts/cache_hallueval.py``."""
    return torch.load(cache_path, map_location="cpu", weights_only=False)


class HalluEvalUQDataset(Dataset[dict[str, torch.Tensor]]):
    """Per-row view of the cache.

    Each item is a dict with keys: ``ids``, ``attn``, ``h``, ``SE_pos``,
    ``answer_mask``, ``label``, ``pair_id``. Useful for batched UQ
    eval scripts; for the closed-form classifiers in
    ``scripts/eval_uq.py`` we pull tensors from the cache directly.
    """

    def __init__(self, cache: dict[str, Any]) -> None:
        self.cache = cache
        self.n = int(cache["full_ids"].shape[0])

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "ids":         self.cache["full_ids"][i].long(),
            "attn":        self.cache["attn_mask"][i].bool(),
            "h":           self.cache["hidden_states"][i].float(),
            "SE_pos":      self.cache["SE_pos"][i].float(),
            "answer_mask": self.cache["answer_mask"][i].bool(),
            "label":       bool(self.cache["label"][i].item()),
            "pair_id":     int(self.cache["pair_id"][i].item()),
        }


__all__ = [
    "load_hallueval_qa",
    "build_prompts",
    "load_cache",
    "HalluEvalUQDataset",
]
