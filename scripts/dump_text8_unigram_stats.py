#!/usr/bin/env python3
"""Persist the text8 unigram statistics the paper quotes but never recorded.

The paper prints two constants that depend on the corpus unigram distribution
(unigram chance sum_k p_k^2 = 0.075 at results.tex:279/363, and the unigram-mean
norm ||mu_1|| = 2.43 at appendix:35) and several whole-word train counts
(insulin 347, ping 67, cute 23, semaglutide 0). None of these was persisted
anywhere: the corpus is fetched from HuggingFace at run time and the constants
lived in docstrings. This script writes them all to one small tracked JSON so
`check_claims.py` can verify them forever (see docs/paper-map.md).

    python scripts/dump_text8_unigram_stats.py [--out runs/text8_unigram_stats.json]

CPU-only, ~1 minute. Whole-word counts use the regex (?<![a-z])w(?![a-z]),
which is what makes insulin=347 (substring count is 399: insulinoma etc.).
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset

ALPHABET = "abcdefghijklmnopqrstuvwxyz "  # K = 27, the pipeline's vocabulary
WORDS = ["insulin", "semaglutide", "ping", "cute", "hemoglobin", "collagen"]
EPS = 1e-4  # label smoothing of the clr features (tab:config)


def whole_word_count(text: str, word: str) -> int:
    return len(re.findall(rf"(?<![a-z]){re.escape(word)}(?![a-z])", text))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent / "runs" / "text8_unigram_stats.json")
    args = ap.parse_args()

    ds = load_dataset("afmck/text8")
    splits = {name: ds[name]["text"][0] for name in ("train", "validation", "test")}

    train = splits["train"]
    counts = Counter(train)
    total = sum(counts[c] for c in ALPHABET)
    assert total == len(train), "train split contains characters outside the K=27 alphabet"
    probs = {c: counts[c] / total for c in ALPHABET}
    p = list(probs.values())

    entropy_nats = -sum(q * math.log(q) for q in p if q > 0)
    sum_p_squared = sum(q * q for q in p)

    # ||mu_1||: mu_1 = E[clr(smoothed one-hot)] = (a - b) p + b 1, with the clr
    # coordinates of a label-smoothed one-hot (hot 1-EPS+EPS/K, cold EPS/K).
    k = len(ALPHABET)
    hot, cold = 1.0 - EPS + EPS / k, EPS / k
    mean_ln = (math.log(hot) + (k - 1) * math.log(cold)) / k
    a, b = math.log(hot) - mean_ln, math.log(cold) - mean_ln
    mu1 = [(a - b) * q + b for q in p]
    mu1_norm = math.sqrt(sum(x * x for x in mu1))
    data_radius = math.sqrt(a * a + (k - 1) * b * b)  # ||x_1|| of any one window position

    out = {
        "source": "HF afmck/text8, train split",
        "produced_by": "scripts/dump_text8_unigram_stats.py",
        "alphabet": ALPHABET,
        "label_smoothing_eps": EPS,
        "split_chars": {name: len(text) for name, text in splits.items()},
        "unigram_probs": probs,
        "entropy_nats": entropy_nats,
        "sum_p_squared": sum_p_squared,
        "mu1_norm": mu1_norm,
        "data_radius": data_radius,
        "word_counts_train": {w: whole_word_count(train, w) for w in WORDS},
        "word_counts_validation": {w: whole_word_count(splits["validation"], w) for w in WORDS},
        "word_counts_test": {w: whole_word_count(splits["test"], w) for w in WORDS},
    }
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}")
    print(f"  entropy {entropy_nats:.4f} nats | sum p^2 {sum_p_squared:.6f} | "
          f"||mu1|| {mu1_norm:.4f} | data radius {data_radius:.4f}")
    print(f"  train counts: {out['word_counts_train']}")


if __name__ == "__main__":
    main()
