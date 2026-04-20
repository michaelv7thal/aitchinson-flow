"""Training data-source builders (raw datasets or LLM-generated streams).

This module is the **single** entrypoint every training script should use to
turn a ``Config`` into a ``DataModule``. By funnelling dispatch through
:func:`build_training_datamodule`, objective scripts never branch on the
source kind themselves — they read ``cfg.training_data`` and run.

Two sources are first-class:

* ``cfg.training_data.source == "raw_text"``: dispatches to
  :class:`~aitchinson_flow.data.text8_datamodule.Text8DataModule` or
  :class:`~aitchinson_flow.data.transforms.hf_datamodule.HFDataModule`
  depending on ``cfg.training_data.raw_dataset``.
* ``cfg.training_data.source == "llm_generated"``: builds a frozen causal LM
  from ``cfg.training_data.lm_key`` via
  :func:`~aitchinson_flow.llms.registry.build_lm`, wraps it in a
  :class:`~aitchinson_flow.data.teachers.causal_lm.CausalLMTeacher`, and
  emits a reproducible seeded stream of ``(valid, invalid)`` batches.

Both branches return ``(datamodule, metadata)``. The metadata dict is
intended to be persisted verbatim next to checkpoints (training manifests
record it under ``training_data_source``) and carries enough information —
including the teacher ``model_id``, generation seed, temperature, top-p, and
prompt settings — to reproduce the dataset at a later date.
"""

from __future__ import annotations

from typing import Any

from aitchinson_flow.config import Config
from aitchinson_flow.training.datamodule import DataModule


def build_training_datamodule(cfg: Config) -> tuple[DataModule, dict[str, Any]]:
    """Build a training datamodule and metadata from ``cfg.training_data``.

    Returns:
        ``(datamodule, metadata)`` where ``metadata`` is a JSON-serialisable
        dict describing the source choice, the raw backend or LLM config,
        and — for ``llm_generated`` — the sampling knobs used. Persist this
        alongside your training manifest so runs are reproducible.
    """
    source = cfg.training_data.source
    if source == "raw_text":
        return _build_raw_text_datamodule(cfg)
    if source == "llm_generated":
        return _build_llm_generated_datamodule(cfg)
    raise ValueError(
        f"Unknown cfg.training_data.source={source!r}; expected 'raw_text' or 'llm_generated'."
    )


def _build_raw_text_datamodule(cfg: Config) -> tuple[DataModule, dict[str, Any]]:
    dataset = cfg.training_data.raw_dataset
    if dataset == "text8":
        from aitchinson_flow.data.text8_datamodule import Text8DataModule  # noqa: PLC0415

        return Text8DataModule(cfg), {
            "source": "raw_text",
            "raw_dataset": "text8",
            "seq_length": cfg.dataset.L,
            "vocab_size": cfg.dataset.K,
            "corrupt_rate": cfg.text8_dataset.train_corrupt_rate,
            "order_mix_rate": cfg.text8_dataset.train_order_mix_rate,
            "order_mix_prob": cfg.text8_dataset.order_mix_prob,
            "corruption_seed": cfg.text8_dataset.corruption_seed,
        }
    if dataset == "hf":
        from aitchinson_flow.data.transforms.hf_datamodule import HFDataModule  # noqa: PLC0415

        return HFDataModule(cfg), {
            "source": "raw_text",
            "raw_dataset": "hf",
            "hf_path": cfg.hf_dataset.path,
            "hf_name": cfg.hf_dataset.name,
            "hf_revision": cfg.hf_dataset.revision,
            "hf_split_train": cfg.hf_dataset.split_train,
            "hf_split_val": cfg.hf_dataset.split_val,
            "seq_length": cfg.dataset.L,
            "vocab_size": cfg.dataset.K,
        }
    raise ValueError(
        f"Unknown cfg.training_data.raw_dataset={dataset!r}; expected 'text8' or 'hf'."
    )


def _build_llm_generated_datamodule(cfg: Config) -> tuple[DataModule, dict[str, Any]]:
    from aitchinson_flow.data.teachers.causal_lm import (  # noqa: PLC0415
        CausalLMTeacher,
        CausalLMTeacherDataModule,
    )
    from aitchinson_flow.data.text8_prompts import load_text8_prompts  # noqa: PLC0415
    from aitchinson_flow.llms.registry import build_lm  # noqa: PLC0415

    lm_key = cfg.training_data.lm_key
    lm = build_lm(lm_key, cfg.teacher)
    teacher = CausalLMTeacher(lm, cfg)

    prompt_ids = None
    prompt_mask = None
    if cfg.training_data.use_text8_prompts:
        n_prompts = cfg.training_data.n_batches * cfg.training.B
        prompt_ids, prompt_mask = load_text8_prompts(
            lm,
            n_prompts=n_prompts,
            prompt_length=cfg.training_data.text8_prompt_length,
            seed=cfg.training_data.generation_seed,
        )

    dm = CausalLMTeacherDataModule(
        teacher,
        cfg,
        n_batches=cfg.training_data.n_batches,
        emit_logits=True,
        prompt_ids=prompt_ids,
        prompt_attention_mask=prompt_mask,
        temperature=cfg.training_data.generation_temperature,
        top_p=cfg.training_data.generation_top_p,
        generation_seed=cfg.training_data.generation_seed,
        add_invalid=True,
        invalid_corrupt_rate=cfg.text8_dataset.train_corrupt_rate,
        invalid_order_mix_rate=cfg.text8_dataset.train_order_mix_rate,
        invalid_order_mix_prob=cfg.text8_dataset.order_mix_prob,
    )
    return dm, {
        "source": "llm_generated",
        "lm_key": lm_key,
        "teacher": {
            "model_id": cfg.teacher.model_id,
            "revision": cfg.teacher.revision,
            "dtype": cfg.teacher.dtype,
            "device": cfg.teacher.device,
        },
        "n_batches": cfg.training_data.n_batches,
        "batch_size": cfg.training.B,
        "seq_length": cfg.dataset.L,
        "vocab_size": cfg.dataset.K,
        "use_text8_prompts": cfg.training_data.use_text8_prompts,
        "text8_prompt_length": cfg.training_data.text8_prompt_length,
        "generation_seed": cfg.training_data.generation_seed,
        "generation_temperature": cfg.training_data.generation_temperature,
        "generation_top_p": cfg.training_data.generation_top_p,
        "invalid_corrupt_rate": cfg.text8_dataset.train_corrupt_rate,
        "invalid_order_mix_rate": cfg.text8_dataset.train_order_mix_rate,
        "invalid_order_mix_prob": cfg.text8_dataset.order_mix_prob,
    }


__all__ = ["build_training_datamodule"]
