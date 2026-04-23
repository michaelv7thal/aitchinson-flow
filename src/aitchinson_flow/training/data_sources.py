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

from dataclasses import replace
from typing import Any

from aitchinson_flow.config import Config, HFDatasetConfig
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
    if source == "llm_topk":
        return _build_llm_topk_datamodule(cfg)
    if source == "llm_topk_probs":
        return _build_llm_topk_probs_datamodule(cfg)
    if source == "llm_generated":
        return _build_llm_generated_datamodule(cfg)
    if source == "qa_pairs":
        return _build_qa_pairs_datamodule(cfg)
    if source == "dna":
        return _build_dna_datamodule(cfg)
    if source == "medical":
        return _build_medical_datamodule(cfg)
    raise ValueError(
        f"Unknown cfg.training_data.source={source!r}; expected one of "
        f"'raw_text', 'llm_topk', 'llm_topk_probs', 'llm_generated', "
        f"'qa_pairs', 'dna', or 'medical'."
    )


def _build_raw_text_datamodule(cfg: Config) -> tuple[DataModule, dict[str, Any]]:
    dataset = cfg.training_data.raw_dataset
    raw_cfg = cfg.raw_text_dataset
    if (
        dataset == "hf"
        and raw_cfg.source_ref == "afmck/text8"
        and cfg.hf_dataset.path
    ):
        raw_cfg = replace(
            raw_cfg,
            provider="manual"
            if cfg.hf_dataset.path.startswith(("/", "./", "../"))
            else "huggingface",
            source_ref=cfg.hf_dataset.path,
            dataset_name=cfg.hf_dataset.name,
            revision=cfg.hf_dataset.revision,
            split_train=cfg.hf_dataset.split_train,
            split_val=cfg.hf_dataset.split_val,
            split_test=cfg.hf_dataset.split_test,
            trust_remote_code=cfg.hf_dataset.trust_remote_code,
            streaming=cfg.hf_dataset.streaming,
        )
    if dataset == "text8":
        from aitchinson_flow.data.text8_datamodule import Text8DataModule  # noqa: PLC0415

        return Text8DataModule(cfg), {
            "source": "raw_text",
            "raw_dataset": "text8",
            "raw_provider": raw_cfg.provider,
            "raw_source_ref": raw_cfg.source_ref,
            "raw_dataset_name": raw_cfg.dataset_name,
            "raw_split_train": raw_cfg.split_train,
            "raw_split_val": raw_cfg.split_val,
            "raw_split_test": raw_cfg.split_test,
            "seq_length": cfg.dataset.L,
            "vocab_size": cfg.dataset.K,
            "corrupt_rate": cfg.text8_dataset.train_corrupt_rate,
            "order_mix_rate": cfg.text8_dataset.train_order_mix_rate,
            "order_mix_prob": cfg.text8_dataset.order_mix_prob,
            "corruption_seed": cfg.text8_dataset.corruption_seed,
        }
    if dataset == "hf":
        from aitchinson_flow.data.transforms.hf_datamodule import HFDataModule  # noqa: PLC0415

        provider = raw_cfg.provider
        hf_path = raw_cfg.source_ref
        hf_cfg: HFDatasetConfig = replace(
            cfg.hf_dataset,
            enabled=True,
            path=hf_path,
            name=raw_cfg.dataset_name,
            revision=raw_cfg.revision,
            split_train=raw_cfg.split_train,
            split_val=raw_cfg.split_val,
            split_test=raw_cfg.split_test,
            streaming=raw_cfg.streaming,
            trust_remote_code=raw_cfg.trust_remote_code,
            row_input_key=cfg.hf_dataset.row_input_key,
        )
        runtime_cfg = replace(cfg, hf_dataset=hf_cfg)
        return HFDataModule(runtime_cfg), {
            "source": "raw_text",
            "raw_dataset": "hf",
            "raw_provider": provider,
            "raw_source_ref": raw_cfg.source_ref,
            "raw_dataset_name": raw_cfg.dataset_name,
            "hf_path": hf_cfg.path,
            "hf_name": hf_cfg.name,
            "hf_revision": hf_cfg.revision,
            "hf_split_train": hf_cfg.split_train,
            "hf_split_val": hf_cfg.split_val,
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
            cfg=cfg,
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
        "llm_feature_mode": cfg.training_data.llm_feature_mode,
        "invalid_corrupt_rate": cfg.text8_dataset.train_corrupt_rate,
        "invalid_order_mix_rate": cfg.text8_dataset.train_order_mix_rate,
        "invalid_order_mix_prob": cfg.text8_dataset.order_mix_prob,
    }


def _build_llm_topk_datamodule(cfg: Config) -> tuple[DataModule, dict[str, Any]]:
    from aitchinson_flow.data.llm_embedding_datamodule import (  # noqa: PLC0415
        LLMEmbeddingDataModule,
    )
    from aitchinson_flow.llms.registry import build_lm  # noqa: PLC0415

    emb_cfg = cfg.llm_embedding_dataset
    lm = build_lm(emb_cfg.lm_key, cfg.teacher)
    dm = LLMEmbeddingDataModule(cfg, lm)

    # Surface the LLM's embedding dim back onto the config so model builders
    # (Stage 1 / Stage 2 / fused) can size their ``TokenEmbeddingToSimplex``
    # head without re-loading the LLM.
    cfg.llm_embedding_dataset.llm_embed_dim = dm.llm_embed_dim

    corrupt_rate = (
        emb_cfg.corrupt_rate
        if emb_cfg.corrupt_rate is not None
        else cfg.text8_dataset.train_corrupt_rate
    )
    return dm, {
        "source": "llm_topk",
        "lm_key": emb_cfg.lm_key,
        "teacher": {
            "model_id": cfg.teacher.model_id,
            "revision": cfg.teacher.revision,
            "dtype": cfg.teacher.dtype,
            "device": cfg.teacher.device,
        },
        "raw_text_backend": emb_cfg.raw_text_backend,
        "char_window_length": emb_cfg.char_window_length,
        "llm_token_length": cfg.dataset.L,
        "simplex_dim": cfg.dataset.K,
        "llm_embed_dim": dm.llm_embed_dim,
        "corrupt_rate": corrupt_rate,
        "generation_seed": emb_cfg.generation_seed,
    }


def _build_llm_topk_probs_datamodule(cfg: Config) -> tuple[DataModule, dict[str, Any]]:
    """Build the Component 2 (top-K probability) datamodule.

    Tokenizes raw text with a frozen LLM, extracts top-K softmax probabilities
    per token position, re-normalizes, and applies ILR. The resulting batch keys
    (``log_x``, ``log_x_invalid``) are compatible with Stage 1 and Stage 2.
    """
    from aitchinson_flow.data.llm_topk_probs_datamodule import (  # noqa: PLC0415
        LLMTopKProbsDataModule,
    )
    from aitchinson_flow.llms.registry import build_lm  # noqa: PLC0415

    topk_cfg = cfg.llm_topk_probs
    lm = build_lm(topk_cfg.lm_key, cfg.teacher)
    dm = LLMTopKProbsDataModule(cfg, lm)

    corrupt_rate = (
        topk_cfg.corrupt_rate
        if topk_cfg.corrupt_rate is not None
        else cfg.text8_dataset.train_corrupt_rate
    )
    return dm, {
        "source": "llm_topk_probs",
        "lm_key": topk_cfg.lm_key,
        "teacher": {
            "model_id": cfg.teacher.model_id,
            "revision": cfg.teacher.revision,
            "dtype": cfg.teacher.dtype,
            "device": cfg.teacher.device,
        },
        "raw_text_backend": topk_cfg.raw_text_backend,
        "char_window_length": topk_cfg.char_window_length,
        "llm_token_length": cfg.dataset.L,
        "top_k": cfg.dataset.K,
        "renormalize": topk_cfg.renormalize,
        "corrupt_rate": corrupt_rate,
        "generation_seed": topk_cfg.generation_seed,
    }


def _build_qa_pairs_datamodule(cfg: Config) -> tuple[DataModule, dict[str, Any]]:
    from aitchinson_flow.data.qa_datamodule import QAPairsDataModule  # noqa: PLC0415

    qa = cfg.qa_dataset
    dm = QAPairsDataModule(cfg, skip_llm=cfg.training_data.qa_skip_llm_eval)
    return dm, {
        "source": "qa_pairs",
        "hf_path": qa.hf_path,
        "hf_name": qa.name,
        "hf_revision": qa.revision,
        "split_train": qa.split_train,
        "split_val": qa.split_val,
        "split_test": qa.split_test,
        "question_col": qa.question_col,
        "answer_col": qa.answer_col,
        "aliases_col": qa.aliases_col,
        "max_train_samples": qa.max_train_samples,
        "max_val_samples": qa.max_val_samples,
        "max_test_samples": qa.max_test_samples,
        "max_question_bytes": qa.max_question_bytes,
        "max_answer_bytes": qa.max_answer_bytes,
        "shuffle_seed": qa.shuffle_seed,
        "seq_length": cfg.dataset.L,
        "vocab_size": cfg.dataset.K,
        "skip_llm_eval": cfg.training_data.qa_skip_llm_eval,
        "answer_generator": {
            "lm_key": cfg.answer_generator.lm_key,
            "model_id": cfg.answer_generator.model_id,
            "revision": cfg.answer_generator.revision,
            "temperature": cfg.answer_generator.temperature,
            "top_p": cfg.answer_generator.top_p,
            "max_new_tokens": cfg.answer_generator.max_new_tokens,
            "prompt_template": cfg.answer_generator.prompt_template,
            "generation_seed": cfg.answer_generator.generation_seed,
        },
    }


def _build_dna_datamodule(cfg: Config) -> tuple[DataModule, dict[str, Any]]:
    """Build the DNA nucleotide sequence datamodule (Component 1 structural UQ)."""
    from aitchinson_flow.data.dna_datamodule import DNADataModule  # noqa: PLC0415

    dcfg = cfg.dna_dataset
    dm = DNADataModule(cfg)
    return dm, {
        "source": "dna",
        "hf_path": dcfg.hf_path,
        "hf_name": dcfg.hf_name,
        "use_n_base": dcfg.use_n_base,
        "seq_length": cfg.dataset.L,
        "vocab_size": cfg.dataset.K,
        "gc_content": dcfg.gc_content,
        "train_corrupt_rate": dcfg.train_corrupt_rate,
        "eval_corrupt_rate": dcfg.eval_corrupt_rate,
        "corruption_seed": dcfg.corruption_seed,
        "synthetic_seed": dcfg.synthetic_seed,
    }


def _build_medical_datamodule(cfg: Config) -> tuple[DataModule, dict[str, Any]]:
    """Build the clinical text datamodule (Component 1+2 structural + contextual UQ)."""
    from aitchinson_flow.data.medical_datamodule import MedicalDataModule  # noqa: PLC0415

    mcfg = cfg.medical_dataset
    dm = MedicalDataModule(cfg)
    return dm, {
        "source": "medical",
        "hf_path": mcfg.hf_path,
        "hf_name": mcfg.hf_name,
        "seq_length": cfg.dataset.L,
        "vocab_size": cfg.dataset.K,
        "train_corrupt_rate": mcfg.train_corrupt_rate,
        "eval_corrupt_rate": mcfg.eval_corrupt_rate,
        "corruption_seed": mcfg.corruption_seed,
        "synthetic_seed": mcfg.synthetic_seed,
    }


__all__ = ["build_training_datamodule"]
