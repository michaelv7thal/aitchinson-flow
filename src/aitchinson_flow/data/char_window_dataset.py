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


class VariableLengthCharWindowDataset(Dataset):
    """Variable-L windows for AE/EqMAE training.

    Each ``__getitem__`` returns a 1-D ``token_ids`` tensor of length
    ``L_i`` drawn uniformly from ``[L_min, L_max]`` with a random offset
    into the flattened source corpus. The collate (``variable_length_collate``)
    pads to the batch's max L and emits a ``pad_mask`` for the transformer's
    ``src_key_padding_mask``.

    The (offset, L) sequence is sampled once at construction with a fixed
    seed, so the dataset is iterable-deterministic and supports
    DataLoader workers.

    No CLR features are precomputed — AE/EqMAE only need token_ids.
    """

    def __init__(
        self,
        windows: torch.Tensor,
        *,
        L_min: int,
        L_max: int,
        num_windows: int | None = None,
        seed: int = 0,
    ) -> None:
        if L_min < 1 or L_max < L_min:
            raise ValueError(f"need 1 <= L_min ({L_min}) <= L_max ({L_max})")
        # Flatten the fixed-L windows back into a 1-D token stream we can
        # slice at arbitrary (offset, L). This way we share the cache with
        # the deterministic CharWindowDataset and avoid touching the data
        # loader's text source.
        if windows.dim() != 2:
            raise ValueError(f"expected 2-D windows tensor, got {tuple(windows.shape)}")
        self._text = windows.reshape(-1).contiguous()
        n_tokens = int(self._text.numel())
        if n_tokens < L_max + 1:
            raise ValueError(
                f"text too short for L_max={L_max}: only {n_tokens} tokens available"
            )

        self.L_min = int(L_min)
        self.L_max = int(L_max)
        if num_windows is None:
            num_windows = n_tokens // L_max
        self._num = int(num_windows)

        g = torch.Generator().manual_seed(int(seed))
        self._Ls = torch.randint(L_min, L_max + 1, (self._num,), generator=g)
        # Max valid offset for each window = n_tokens - L_i
        max_offsets = n_tokens - self._Ls
        # Sample offsets via uniform float * range (avoids per-row randint loop)
        u = torch.rand(self._num, generator=g)
        self._offsets = (u * max_offsets.float()).long()

    def __len__(self) -> int:
        return self._num

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        L = int(self._Ls[idx])
        off = int(self._offsets[idx])
        return {"token_ids": self._text[off : off + L], "L": L}


def variable_length_collate(samples: list[dict], pad_token: int = 26) -> dict[str, torch.Tensor]:
    """Pad variable-length samples to the batch's max L; emit pad_mask.

    Returns:
        token_ids: (B, L_max_in_batch) long, padded with ``pad_token`` (default
            CHAR2ID[' ']=26).
        pad_mask:  (B, L_max_in_batch) bool, True = padding (drop from
            attention and loss).
        valid_lengths: (B,) long, the actual L_i per sample.

    The padding token is space (id 26) by default — same as the
    out-of-vocab fallback in ``text_to_windows``, so even un-masked uses
    won't catastrophically corrupt the model.
    """
    B = len(samples)
    L_max = max(int(s["L"]) for s in samples)
    token_ids = torch.full((B, L_max), pad_token, dtype=torch.long)
    pad_mask = torch.ones((B, L_max), dtype=torch.bool)
    valid_lengths = torch.zeros(B, dtype=torch.long)
    for i, s in enumerate(samples):
        L = int(s["L"])
        token_ids[i, :L] = s["token_ids"].long()
        pad_mask[i, :L] = False
        valid_lengths[i] = L
    return {
        "token_ids": token_ids,
        "pad_mask": pad_mask,
        "valid_lengths": valid_lengths,
    }
