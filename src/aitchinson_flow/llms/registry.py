from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from aitchinson_flow.config import TeacherConfig

if TYPE_CHECKING:
    from aitchinson_flow.llms.types import CausalLMForInference

LMBuilder = Callable[[TeacherConfig], "CausalLMForInference"]

REGISTRY: dict[str, LMBuilder] = {}


def register(name: str) -> Callable[[LMBuilder], LMBuilder]:
    def deco(fn: LMBuilder) -> LMBuilder:
        if name in REGISTRY:
            raise ValueError(f"Duplicate lm registry key: {name!r}")
        REGISTRY[name] = fn
        return fn

    return deco


def build_lm(lm_key: str, cfg: TeacherConfig) -> CausalLMForInference:
    try:
        builder = REGISTRY[lm_key]
    except KeyError as e:
        raise KeyError(f"Unknown lm_key={lm_key!r}. Registered: {sorted(REGISTRY)}") from e
    return builder(cfg)


def registered_lm_keys() -> tuple[str, ...]:
    return tuple(sorted(REGISTRY))
