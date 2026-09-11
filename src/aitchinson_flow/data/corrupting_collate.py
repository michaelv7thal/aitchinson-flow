from __future__ import annotations

import torch

from aitchinson_flow.data import build_invalid_batch
from aitchinson_flow.data.transforms import token_ids_to_features_dirichlet


class CorruptingCollate:
    """Collate window dicts and add x_invalid / token_ids_invalid when corruption is enabled.

    When ``dirichlet_sampling=True``, ``batch["x"]`` is *replaced* with a
    fresh Dirichlet draw → CLR each call. The dataset's precomputed
    deterministic features are discarded for that step — this is intentional;
    the whole point is a different x_1 per batch for the same token.
    """

    def __init__(
        self,
        *,
        K: int,
        corrupt_rate: float,
        order_mix_rate: float,
        order_mix_prob: float,
        seed: int,
        label_smoothing: float = 1e-4,
        dirichlet_sampling: bool = False,
        dirichlet_alpha_peak: float = 50.0,
        dirichlet_alpha_base: float = 0.1,
    ) -> None:
        self._K = K
        self._rate = corrupt_rate
        self._order_mix_rate = order_mix_rate
        self._order_mix_prob = order_mix_prob
        self._seed = seed
        self._label_smoothing = label_smoothing
        self._dirichlet = dirichlet_sampling
        self._alpha_peak = dirichlet_alpha_peak
        self._alpha_base = dirichlet_alpha_base
        self._n_calls = 0

    def __call__(
        self, samples: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        token_ids = torch.stack([s["token_ids"] for s in samples], dim=0)
        if self._dirichlet:
            x = token_ids_to_features_dirichlet(
                token_ids,
                K=self._K,
                alpha_peak=self._alpha_peak,
                alpha_base=self._alpha_base,
            )
        else:
            x = torch.stack([s["x"] for s in samples], dim=0)
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
