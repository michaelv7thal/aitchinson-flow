from __future__ import annotations

from collections.abc import Callable

from benchmarks.tasks.base import BenchmarkTask

TaskBuilder = Callable[[], BenchmarkTask]

REGISTRY: dict[str, TaskBuilder] = {}


def register(name: str) -> Callable[[TaskBuilder], TaskBuilder]:
    def deco(fn: TaskBuilder) -> TaskBuilder:
        if name in REGISTRY:
            raise ValueError(f"Duplicate benchmark task key: {name!r}")
        REGISTRY[name] = fn
        return fn

    return deco


def build_task(name: str) -> BenchmarkTask:
    try:
        builder = REGISTRY[name]
    except KeyError as e:
        raise KeyError(f"Unknown benchmark task {name!r}. Registered: {sorted(REGISTRY)}") from e
    return builder()


def registered_task_keys() -> tuple[str, ...]:
    return tuple(sorted(REGISTRY))
