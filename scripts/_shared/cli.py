"""Reusable CLI building blocks shared by objective-specific training scripts.

Objective scripts (e.g. ``scripts/two_stage_train.py``) wire their own
objective-specific flags (``--stage1-epochs`` etc.) but delegate everything
related to *training data source selection* to this module. That keeps the
raw-text / LLM-generated dispatch consistent across entrypoints and means
adding a new training objective does not require re-deriving argparse
wiring for the data source.

Typical usage inside an objective script::

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=str, default="checkpoints/my_objective")
    p.add_argument("--epochs", type=positive_int, default=25)
    add_training_data_args(p)
    args = p.parse_args(argv)

    cfg = make_smoke_config() if args.smoke else Config()
    apply_training_data_args(cfg, args)
    # ... run objective-specific training ...
"""

from __future__ import annotations

import argparse
from typing import Any

from aitchinson_flow.config import Config

__all__ = [
    "positive_int",
    "add_training_data_args",
    "apply_training_data_args",
]


def positive_int(raw: str) -> int:
    """argparse ``type`` for flags that must be >= 1."""
    try:
        value = int(raw)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"expected integer, got {raw!r}") from e
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer (>= 1), got {value}")
    return value


def add_training_data_args(parser: argparse.ArgumentParser) -> None:
    """Add the standard ``--training-data-*`` / generation flag group.

    All flags default to ``None`` (or ``False`` for booleans) so the
    corresponding ``Config`` defaults survive when the user does not pass
    them explicitly. :func:`apply_training_data_args` is responsible for
    translating the namespace into ``cfg.training_data`` mutations.
    """
    group = parser.add_argument_group("training data source")
    group.add_argument(
        "--training-data-source",
        type=str,
        choices=("raw_text", "llm_generated"),
        default=None,
        help="Training input source: raw text datamodule or LLM-generated stream.",
    )
    group.add_argument(
        "--raw-dataset",
        type=str,
        choices=("text8", "hf"),
        default=None,
        help="Raw dataset backend when --training-data-source=raw_text.",
    )
    group.add_argument(
        "--training-lm-key",
        type=str,
        default=None,
        help="LLM registry key when --training-data-source=llm_generated.",
    )
    group.add_argument(
        "--generated-batches",
        type=positive_int,
        default=None,
        help="Number of generated batches when using llm_generated training data.",
    )
    group.add_argument(
        "--generation-seed",
        type=int,
        default=None,
        help="Base seed for reproducible LLM generation batches.",
    )
    group.add_argument(
        "--generation-temperature",
        type=float,
        default=None,
        help="Sampling temperature for LLM generation.",
    )
    group.add_argument(
        "--generation-top-p",
        type=float,
        default=None,
        help="Top-p nucleus sampling value for LLM generation.",
    )
    group.add_argument(
        "--use-text8-prompts",
        action="store_true",
        help="Condition LLM-generated training batches on text8 prompt windows.",
    )
    group.add_argument(
        "--text8-prompt-length",
        type=positive_int,
        default=None,
        help="Prompt length used with --use-text8-prompts.",
    )


def apply_training_data_args(cfg: Config, args: argparse.Namespace) -> None:
    """Apply an argparse namespace produced by :func:`add_training_data_args`.

    Overrides are only written when the user passed the corresponding flag —
    any ``None`` value leaves the existing ``Config`` default untouched.
    ``--use-text8-prompts`` is a boolean flag: we only set it to ``True`` when
    present (it never clears a config-level ``True``).
    """
    _maybe_set(cfg.training_data, "source", getattr(args, "training_data_source", None))
    _maybe_set(cfg.training_data, "raw_dataset", getattr(args, "raw_dataset", None))
    _maybe_set(cfg.training_data, "lm_key", getattr(args, "training_lm_key", None))
    _maybe_set(cfg.training_data, "n_batches", getattr(args, "generated_batches", None))
    _maybe_set(cfg.training_data, "generation_seed", getattr(args, "generation_seed", None))
    _maybe_set(
        cfg.training_data,
        "generation_temperature",
        getattr(args, "generation_temperature", None),
    )
    _maybe_set(cfg.training_data, "generation_top_p", getattr(args, "generation_top_p", None))
    _maybe_set(
        cfg.training_data,
        "text8_prompt_length",
        getattr(args, "text8_prompt_length", None),
    )
    if getattr(args, "use_text8_prompts", False):
        cfg.training_data.use_text8_prompts = True


def _maybe_set(obj: Any, attr: str, value: Any) -> None:
    if value is not None:
        setattr(obj, attr, value)
