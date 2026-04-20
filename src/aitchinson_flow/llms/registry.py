from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from aitchinson_flow.config import TeacherConfig
from aitchinson_flow.utils.registry import Registry

if TYPE_CHECKING:
    from aitchinson_flow.llms.types import CausalLMForInference

LMBuilder = Callable[[TeacherConfig], "CausalLMForInference"]

_REGISTRY: Registry[LMBuilder] = Registry("lm")
REGISTRY = _REGISTRY.builders


def register(name: str) -> Callable[[LMBuilder], LMBuilder]:
    return _REGISTRY.register(name)


def build_lm(lm_key: str, cfg: TeacherConfig) -> "CausalLMForInference":
    return _REGISTRY.get(lm_key)(cfg)


def registered_lm_keys() -> tuple[str, ...]:
    return _REGISTRY.keys()
