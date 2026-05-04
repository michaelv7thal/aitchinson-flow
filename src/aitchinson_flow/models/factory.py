from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn

from aitchinson_flow.config import Config
from aitchinson_flow.utils import Registry


ModelBuilder = Callable[[Config], nn.Module]


_REGISTRY: Registry[ModelBuilder] = Registry("model")
REGISTRY = _REGISTRY.builders


def register(name: str) -> Callable[[ModelBuilder], ModelBuilder]:
    """Decorators to register a builder under a stable string key."""
    return _REGISTRY.register(name)


def build_model(cfg: Config) -> nn.Module:
    """Instantiate the model selected by `cfg.training.model_name`."""
    return _REGISTRY.get(cfg.training.model_name)(cfg)


def registered_model_names() -> tuple[str, ...]:
    """List all registered model names."""
    return _REGISTRY.keys()
