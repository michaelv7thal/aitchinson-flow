from __future__ import annotations

from typing import Protocol

import torch.nn as nn

from aitchinson_flow.config import Config
from aitchinson_flow.training.datamodule import DataModule


class BenchmarkTask(Protocol):
    """Runs a benchmark given a model, datamodule, and resolved config."""

    def run(self, model: nn.Module, datamodule: DataModule, cfg: Config) -> dict[str, float]:
        ...
