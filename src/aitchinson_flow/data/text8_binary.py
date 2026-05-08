"""Phase S — binary alphabet variant of text8 (vowel/consonant collapse).

Maps each character of the K=27 text8 alphabet to {0, 1}:
* vowels ``aeiou`` → class 0
* every other letter, plus the space character, → class 1

The result is a K=2 sequence of the same length L. Plumbed through the
existing ``CharWindowDataset`` so the training/eval pipeline doesn't
change — only the K and the windows do.

The cache key includes ``alphabet=binary`` so a binary K=2 cache and a
full K=27 cache can coexist. The Phase S sweep flips
``cfg.text8_dataset.alphabet`` to ``"binary"`` and ``cfg.training.K`` /
``cfg.text8_dataset.K`` to 2.
"""

from __future__ import annotations

from typing import Iterable

import torch

from aitchinson_flow.data.char_window_dataset import CHAR2ID

_VOWELS = set("aeiou")


def _binary_class_of(char: str) -> int:
    """Return 0 for vowels, 1 for everything else (consonants + space)."""
    if char in _VOWELS:
        return 0
    return 1


def char_id_to_binary_class(char_ids: torch.Tensor) -> torch.Tensor:
    """Map a tensor of K=27 character ids to K=2 vowel/consonant classes.

    Character order is the one in :data:`CHAR2ID`
    (``"abcdefghijklmnopqrstuvwxyz "``). The mapping is fixed at module
    import time.
    """
    table = torch.tensor(
        [
            _binary_class_of(c)
            for c in sorted(CHAR2ID, key=CHAR2ID.__getitem__)
        ],
        dtype=torch.long,
        device=char_ids.device,
    )
    return table[char_ids.long()]


def text_to_binary_windows(text: str, L: int) -> torch.Tensor:
    """Tokenize a raw string into binary K=2 length-L windows."""
    ids = [CHAR2ID.get(c, CHAR2ID[" "]) for c in text]
    n = (len(ids) // L) * L
    char_ids = torch.tensor(ids[:n], dtype=torch.long).view(-1, L)
    return char_id_to_binary_class(char_ids)


def windows_to_binary(windows: torch.Tensor) -> torch.Tensor:
    """Convert a (N, L) tensor of K=27 char ids to K=2 binary classes."""
    return char_id_to_binary_class(windows)
