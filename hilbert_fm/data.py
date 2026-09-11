"""Character-level text data: text8 if available, tiny-shakespeare fallback.

Vocab is fixed to 26 lowercase letters plus space (K = 27). Anything outside
that set is dropped or normalised to a single space. Whatever's quickest — no
dataset infrastructure beyond this module.
"""
from __future__ import annotations

import os
import urllib.request

import torch

CHARS = "abcdefghijklmnopqrstuvwxyz "  # K = 27
CHAR2ID = {c: i for i, c in enumerate(CHARS)}
K = len(CHARS)

DEFAULT_CACHE = os.path.join(os.path.dirname(__file__), "cache")
TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)


def _normalise(text: str) -> str:
    out = []
    for ch in text.lower():
        if ch in CHAR2ID:
            out.append(ch)
        elif ch.isspace():
            out.append(" ")
    return "".join(out)


def _try_load_text8(cache_dir: str) -> str:
    path = os.path.join(cache_dir, "text8")
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read()
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset("afmck/text8", split="train")
        text = ds[0]["text"]
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, "w") as f:
            f.write(text)
        return text
    except Exception as e:
        print(f"[data] text8 unavailable ({e!r}); falling back to tiny-shakespeare.")
        return ""


def _load_tiny_shakespeare(cache_dir: str) -> str:
    path = os.path.join(cache_dir, "tiny_shakespeare.txt")
    if not os.path.exists(path):
        os.makedirs(cache_dir, exist_ok=True)
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)
    with open(path, "r") as f:
        return f.read()


def load_corpus(cache_dir: str = DEFAULT_CACHE) -> str:
    """Return the normalised character corpus as a single string."""
    text = _try_load_text8(cache_dir)
    if not text:
        text = _load_tiny_shakespeare(cache_dir)
    return _normalise(text)


def encode(text: str) -> torch.Tensor:
    return torch.tensor([CHAR2ID[c] for c in text if c in CHAR2ID], dtype=torch.long)


def decode(ids: torch.Tensor) -> str:
    return "".join(CHARS[int(i)] for i in ids.tolist())


def split_train_val(ids: torch.Tensor, val_frac: float = 0.005):
    n_val = max(int(len(ids) * val_frac), 10_000)
    n_val = min(n_val, len(ids) // 10)
    return ids[:-n_val], ids[-n_val:]


def get_batch(ids: torch.Tensor, B: int, L: int, device: str | torch.device = "cpu") -> torch.Tensor:
    """Return a ``(B, L)`` long tensor of randomly sampled windows."""
    starts = torch.randint(0, len(ids) - L - 1, (B,))
    batch = torch.stack([ids[s : s + L] for s in starts])
    return batch.to(device)
