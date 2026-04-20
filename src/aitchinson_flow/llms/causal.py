"""Hugging Face AutoModelForCausalLM wrapper; registered as ``hf_causal``."""

from __future__ import annotations

from typing import Any, cast

import torch

from aitchinson_flow.config import TeacherConfig
from aitchinson_flow.llms.registry import register


def _dtype_from_string(s: str) -> torch.dtype:
    m = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }

    if s not in m:
        raise ValueError(f"Unknown dtype {s!r}; expected one of {sorted(m)}")
    return m[s]


def _resolve_device(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


class HFCausalLMInference:
    """Concrete ``CausalLMForInference`` using transformers."""

    def __init__(self, cfg: TeacherConfig) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        self._device = _resolve_device(cfg.device)
        dtype = _dtype_from_string(cfg.dtype)

        self._tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_id, revision=cfg.revision, trust_remote_code=cfg.trust_remote_code
        )

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            revision=cfg.revision,
            trust_remote_code=cfg.trust_remote_code,
            dtype=dtype,
            device_map={"": self._device},
        )

        self._model.eval()

        for p in self._model.parameters():
            p.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return self._device

    @torch.inference_mode()
    def forward_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_ids = input_ids.to(self._device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)
        out = self._model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits

        if not isinstance(logits, torch.Tensor):
            raise TypeError("Expected model output logits to be a Tensor")
        return logits

    def encode_text(self, text: str, *, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length"
        )
        input_ids = enc["input_ids"].to(self._device)
        attention_mask = enc["attention_mask"].to(self._device)
        return input_ids, attention_mask

    @torch.inference_mode()
    def generate_ids(
        self,
        *,
        batch_size: int,
        max_new_tokens: int,
        prompt_ids: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        if prompt_ids is None:
            # BOS fallback prompt
            bos = self._tokenizer.bos_token_id

            if bos is None:
                bos = self._tokenizer.eos_token_id

            if bos is None:
                raise ValueError("Tokenizer has neither bos_token_id nor eos_token_id")

            input_ids = torch.full((batch_size, 1), bos, device=self._device, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)

        else:
            input_ids = prompt_ids.to(self._device)
            attention_mask = (
                prompt_attention_mask.to(self._device)
                if prompt_attention_mask is not None
                else torch.ones_like(input_ids)
            )

        gen_model = cast(Any, self._model)

        out = gen_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
        )

        return out

    def decode(self, ids: torch.Tensor, *, skip_special_tokens: bool = True) -> list[str]:
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if ids.ndim != 2:
            raise ValueError(f"decode expects 1D or 2D token ids, got shape {tuple(ids.shape)}")
        return list(
            self._tokenizer.batch_decode(
                ids.to("cpu"),
                skip_special_tokens=skip_special_tokens,
            )
        )


@register("hf_causal")
def _build_hf_causal(cfg: TeacherConfig) -> HFCausalLMInference:
    return HFCausalLMInference(cfg)
