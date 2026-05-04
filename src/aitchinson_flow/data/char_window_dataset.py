from __future__ import annotations

import torch
from torch.utils.data import Dataset

from aitchinson_flow.data.transforms import token_ids_to_features

_ALPHABET = "abcdefghijklmnopqrstuvwxyz "
CHAR2ID: dict[str, int] = {c: i for i, c in enumerate(_ALPHABET)}
VOCAB_SIZE: int = len(_ALPHABET)  # 27


def text_to_windows(text: str, L: int) -> torch.Tensor:
    ids = [CHAR2ID.get(c, CHAR2ID[" "]) for c in text]
    n = (len(ids) // L) * L
    return torch.tensor(ids[:n], dtype=torch.long).view(-1, L)


class CharWindowDataset(Dataset):
    """Map-style dataset of length-L char windows → {'x', 'token_ids'}."""

    def __init__(
        self,
        windows: torch.Tensor,
        *,
        K: int,
        label_smoothing: float = 1e-4,
    ) -> None:
        self._windows = windows
        self._features = token_ids_to_features(windows, K, label_smoothing=label_smoothing)

    def __len__(self) -> int:
        return self._windows.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"x": self._features[idx], "token_ids": self._windows[idx]}
