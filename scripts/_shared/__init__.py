"""Shared plumbing for objective-specific training entrypoints under ``scripts/``.

This package defines the conventions that every objective script (two-stage
auditor, single-stage baseline, future distillation objectives, ...) should
follow so adding a new entrypoint does not require duplicating argparse
wiring, smoke config, or data-source dispatch.

Modules:
    * ``bootstrap``  — put ``src/`` + repo root on ``sys.path`` for direct script
      execution (no editable install needed).
    * ``cli``        — shared argparse helpers (``positive_int``) and the
      canonical ``--training-data-*`` flag group that every objective script
      must expose so users can switch between raw-text corpora and LLM
      generated batches without touching the objective code.
    * ``smoke``      — a default tiny-CPU smoke ``Config`` used by all
      objective scripts behind ``--smoke``.
"""

from __future__ import annotations

__all__: list[str] = []
