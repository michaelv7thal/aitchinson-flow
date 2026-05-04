from __future__ import annotations

import torch

from aitchinson_flow.data import build_invalid_batch


class CorruptingCollate:
    """Collate window dicts and add x_invalid / token_ids_invalid when corruption is enabled."""

    def __init__(
        self,
        *,
        K: int,
        corrupt_rate: float,
        order_mix_rate: float,
        order_mix_prob: float,
        seed: int,
        label_smoothing: float = 1e-4,
    ) -> None:
        self._K = K
        self._rate = corrupt_rate
        self._order_mix_rate = order_mix_rate
        self._order_mix_prob = order_mix_prob
        self._seed = seed
        self._label_smoothing = label_smoothing
        self._n_calls = 0

    def __call__(
        self, samples: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        x = torch.stack([s["x"] for s in samples], dim=0)
        token_ids = torch.stack([s["token_ids"] for s in samples], dim=0)
        batch: dict[str, torch.Tensor] = {"x": x, "token_ids": token_ids}

        if self._rate > 0.0 or self._order_mix_rate > 0.0:
            seed = self._seed + self._n_calls
            build_invalid_batch(
                batch,
                K=self._K,
                corrupt_rate=self._rate,
                order_mix_rate=self._order_mix_rate,
                order_mix_prob=self._order_mix_prob,
                label_smoothing=self._label_smoothing,
                seed=seed,
            )

        self._n_calls += 1
        return batch
