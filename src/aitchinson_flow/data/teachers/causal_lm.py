from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, IterableDataset

from aitchinson_flow.config import Config
from aitchinson_flow.data.teachers.base import TeacherBackend
from aitchinson_flow.data.transforms.discrete import token_ids_to_features, token_logits_to_features
from aitchinson_flow.llms.types import CausalLMForInference
from aitchinson_flow.training.datamodule import DataModule


class CausalLMTeacher(TeacherBackend):
    """Wrap an inference-only causal LM and emit benchmark-ready tensors."""

    def __init__(self, lm: CausalLMForInference, cfg: Config) -> None:
        self._lm = lm
        self._cfg = cfg

    @property
    def cfg(self) -> Config:
        return self._cfg

    def _feature_mode(self) -> str:
        return self._cfg.training_data.llm_feature_mode

    def _materialize(self, t: torch.Tensor) -> torch.Tensor:
        # Convert inference-mode tensors into ordinary tensors so downstream
        # training/eval code can request gradients when needed.
        with torch.inference_mode(False):
            return t.detach().clone()

    def _token_ids_to_features(self, ids: torch.Tensor) -> torch.Tensor:
        K = self._cfg.dataset.K
        eps = self._cfg.hf_dataset.log_simplex_eps
        ls = self._cfg.hf_dataset.label_smoothing
        tm = self._cfg.hf_dataset.transform_mode
        rows = [
            token_ids_to_features(row, K=K, eps=eps, label_smoothing=ls, transform_mode=tm)
            for row in ids
        ]
        return torch.stack(rows, dim=0)

    @torch.inference_mode()
    def sample_log_x(
        self,
        *,
        batch_size: int,
        prompt_ids: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        L = self._cfg.dataset.L
        ids = self._lm.generate_ids(
            batch_size=batch_size,
            max_new_tokens=L,
            prompt_ids=prompt_ids,
            prompt_attention_mask=prompt_attention_mask,
            temperature=temperature,
            top_p=top_p,
        )
        ids = ids[:, -L:]  # Force exact sequence length

        ids_cpu = ids.cpu()
        if self._feature_mode() == "token_probs":
            logits = self._lm.forward_logits(input_ids=ids).cpu()
            return self._materialize(
                token_logits_to_features(
                    logits,
                    K=self._cfg.dataset.K,
                    eps=self._cfg.hf_dataset.log_simplex_eps,
                    transform_mode=self._cfg.hf_dataset.transform_mode,
                )
            )
        return self._materialize(self._token_ids_to_features(ids_cpu))

    @torch.inference_mode()
    def sample_with_logits(
        self,
        *,
        batch_size: int,
        prompt_ids: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Generate tokens then re-score to get aligned logits for spilled-energy.

        Returns dict with keys:
            log_x        : (B, L, K-1)  — ILR feature representation
            token_ids    : (B, L)     — integer token ids
            logits       : (B, L, V)  — raw LM logits aligned to token_ids
        """
        L = self._cfg.dataset.L
        ids = self._lm.generate_ids(
            batch_size=batch_size,
            max_new_tokens=L,
            prompt_ids=prompt_ids,
            prompt_attention_mask=prompt_attention_mask,
            temperature=temperature,
            top_p=top_p,
        )
        ids = ids[:, -L:]  # (B, L)

        logits = self._lm.forward_logits(input_ids=ids)  # (B, L, vocab)

        ids_cpu = ids.cpu()
        logits_cpu = logits.cpu()
        if self._feature_mode() == "token_probs":
            log_x = token_logits_to_features(
                logits_cpu,
                K=self._cfg.dataset.K,
                eps=self._cfg.hf_dataset.log_simplex_eps,
                transform_mode=self._cfg.hf_dataset.transform_mode,
            )
        else:
            log_x = self._token_ids_to_features(ids_cpu)

        return {
            "log_x": self._materialize(log_x),
            "token_ids": self._materialize(ids_cpu),
            "logits": self._materialize(logits_cpu),
        }

    def __call__(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> dict[str, torch.Tensor]:
        # Optional teacher logits path if you need it for distillation later.
        logits = self._lm.forward_logits(input_ids=input_ids, attention_mask=attention_mask)

        return {"logits_teacher": logits}


@dataclass(frozen=True)
class TeacherStreamConfig:
    n_batches: int
    batch_size: int
    emit_logits: bool = False
    prompt_ids: torch.Tensor | None = None
    prompt_attention_mask: torch.Tensor | None = None
    temperature: float = 1.0
    top_p: float = 1.0
    generation_seed: int = 0
    add_invalid: bool = False
    invalid_corrupt_rate: float = 0.15
    invalid_order_mix_rate: float = 0.15
    invalid_order_mix_prob: float = 0.0


class CausalLMTeacherDataset(IterableDataset[dict[str, torch.Tensor]]):
    def __init__(self, teacher: CausalLMTeacher, cfg: TeacherStreamConfig) -> None:
        self._teacher = teacher
        self._cfg = cfg

    def _prompt_slice(self, batch_idx: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return the prompt slice for this batch, or (None, None) if no prompts."""
        if self._cfg.prompt_ids is None:
            return None, None
        B = self._cfg.batch_size
        start = batch_idx * B
        end = start + B
        total = self._cfg.prompt_ids.shape[0]
        if start >= total:
            start = start % total
            end = start + B
        ids = self._cfg.prompt_ids[start:end]
        mask = (
            self._cfg.prompt_attention_mask[start:end]
            if self._cfg.prompt_attention_mask is not None
            else None
        )
        return ids, mask

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        from aitchinson_flow.data.corruption import build_invalid_batch  # noqa: PLC0415

        for i in range(self._cfg.n_batches):
            seed = self._cfg.generation_seed + i
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            if self._cfg.emit_logits:
                p_ids, p_mask = self._prompt_slice(i)
                batch = self._teacher.sample_with_logits(
                    batch_size=self._cfg.batch_size,
                    prompt_ids=p_ids,
                    prompt_attention_mask=p_mask,
                    temperature=self._cfg.temperature,
                    top_p=self._cfg.top_p,
                )
                if self._cfg.add_invalid:
                    build_invalid_batch(
                        batch,
                        K=self._teacher.cfg.dataset.K,
                        corrupt_rate=self._cfg.invalid_corrupt_rate,
                        order_mix_rate=self._cfg.invalid_order_mix_rate,
                        order_mix_prob=self._cfg.invalid_order_mix_prob,
                        eps=self._teacher.cfg.hf_dataset.log_simplex_eps,
                        label_smoothing=self._teacher.cfg.hf_dataset.label_smoothing,
                        transform_mode=self._teacher.cfg.hf_dataset.transform_mode,
                        feature_mode=self._teacher.cfg.training_data.llm_feature_mode,
                        seed=seed,
                    )
                yield batch
            else:
                p_ids, p_mask = self._prompt_slice(i)
                yield {
                    "log_x": self._teacher.sample_log_x(
                        batch_size=self._cfg.batch_size,
                        prompt_ids=p_ids,
                        prompt_attention_mask=p_mask,
                        temperature=self._cfg.temperature,
                        top_p=self._cfg.top_p,
                    )
                }


class CausalLMTeacherDataModule(DataModule):
    """Synthetic datamodule backed by LM generation."""

    def __init__(
        self,
        teacher: CausalLMTeacher,
        cfg: Config,
        *,
        n_batches: int | None = None,
        emit_logits: bool = False,
        prompt_ids: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        generation_seed: int = 0,
        add_invalid: bool = False,
        invalid_corrupt_rate: float = 0.15,
        invalid_order_mix_rate: float = 0.15,
        invalid_order_mix_prob: float = 0.0,
    ) -> None:
        self._dataset = CausalLMTeacherDataset(
            teacher=teacher,
            cfg=TeacherStreamConfig(
                n_batches=cfg.benchmark.n_batches if n_batches is None else n_batches,
                batch_size=cfg.training.B,
                emit_logits=emit_logits,
                prompt_ids=prompt_ids,
                prompt_attention_mask=prompt_attention_mask,
                temperature=temperature,
                top_p=top_p,
                generation_seed=generation_seed,
                add_invalid=add_invalid,
                invalid_corrupt_rate=invalid_corrupt_rate,
                invalid_order_mix_rate=invalid_order_mix_rate,
                invalid_order_mix_prob=invalid_order_mix_prob,
            ),
        )

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(self._dataset, batch_size=None, num_workers=0)

    def val_dataloader(self) -> DataLoader[Any] | None:
        return None

    def test_dataloader(self) -> DataLoader[Any] | None:
        return None
