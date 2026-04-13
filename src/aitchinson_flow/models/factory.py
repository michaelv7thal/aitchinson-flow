"""Register and construct generative models by name (for training / benchmarks)."""

from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn

from aitchinson_flow.config import Config


ModelBuilder = Callable[[Config], nn.Module]

REGISTRY: dict[str, ModelBuilder] = {}


def register(name: str) -> Callable[[ModelBuilder], ModelBuilder]:
    """Decorator to register a builder under a stable string key."""

    def deco(fn: ModelBuilder) -> ModelBuilder:
        if name in REGISTRY:
            raise ValueError(f"Duplicate model name: {name!r}")

        REGISTRY[name] = fn
        return fn

    return deco


def build_model(cfg: Config) -> nn.Module:
    """Instantiate the model selected by `cfg.training.model_name`."""
    key = cfg.training.model_name
    try:
        builder = REGISTRY[key]
    except KeyError as e:
        raise KeyError(f"Unknown model {key!r}. Registered: {sorted(REGISTRY)}") from e

    return builder(cfg)


def registered_model_names() -> tuple[str, ...]:
    """List all registered model names."""
    return tuple(sorted(REGISTRY))
