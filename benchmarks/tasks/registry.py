from __future__ import annotations

from collections.abc import Callable

from aitchinson_flow.utils.registry import Registry
from benchmarks.tasks.base import BenchmarkTask

TaskBuilder = Callable[[], BenchmarkTask]

_REGISTRY: Registry[TaskBuilder] = Registry("benchmark task")
REGISTRY = _REGISTRY.builders


def register(name: str) -> Callable[[TaskBuilder], TaskBuilder]:
    return _REGISTRY.register(name)


def build_task(name: str) -> BenchmarkTask:
    return _REGISTRY.get(name)()


def registered_task_keys() -> tuple[str, ...]:
    return _REGISTRY.keys()
