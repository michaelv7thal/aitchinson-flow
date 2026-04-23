"""Evaluate a trained Stage 2 QA auditor for hallucination detection.

Primary mode consumes externally supplied target-domain LLM responses
(``question, answer, label`` or ``question, answer, reference_answer``), encodes
them as byte-level ``[Q][A]`` pairs, and reports hallucination AUROC.

For backwards compatibility, the script can still fall back to the legacy
answer-generator path over the configured QA dataset.

Usage::

    python scripts/phase2_eval_hallucination.py \
        --stage2-ckpt checkpoints/phase2_qa/baseline/stage2.pt \
        --answer-model-id gpt2 --max-val-samples 200
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from _shared.bootstrap import bootstrap_repo_paths

_REPO_ROOT = bootstrap_repo_paths(Path(__file__))

import torch  # noqa: E402

import aitchinson_flow.models  # noqa: E402,F401 — populate model REGISTRY
import benchmarks.tasks  # noqa: E402,F401 — register hallucination_audit

from _shared.cli import positive_int  # noqa: E402
from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.data.qa_datamodule import (  # noqa: E402
    QAPairsDataModule,
    QAExternalEvalDataModule,
)
from aitchinson_flow.models.factory import build_model  # noqa: E402
from aitchinson_flow.training.checkpoint import load_checkpoint  # noqa: E402
from benchmarks.tasks.registry import build_task  # noqa: E402


def _build_cfg_from_checkpoint(ckpt_cfg: dict[str, Any]) -> Config:
    """Rebuild a Config from the saved state. We do a shallow reconstruction
    (re-instantiate defaults then patch K/L) — enough for inference."""
    cfg = Config()
    cfg.dataset = replace(
        cfg.dataset,
        K=int(ckpt_cfg["dataset"]["K"]),
        L=int(ckpt_cfg["dataset"]["L"]),
    )
    cfg.training = replace(
        cfg.training,
        model_name=ckpt_cfg["training"]["model_name"],
        device=torch.device(ckpt_cfg["training"].get("device", "cpu")),
        B=int(ckpt_cfg["training"].get("B", cfg.training.B)),
        num_workers=int(ckpt_cfg["training"].get("num_workers", 0)),
    )
    gp_block = ckpt_cfg.get("gp", {})
    cfg.gp = replace(
        cfg.gp,
        score_answer_tokens_only=bool(
            gp_block.get("score_answer_tokens_only", cfg.gp.score_answer_tokens_only)
        ),
        num_inducing=int(gp_block.get("num_inducing", cfg.gp.num_inducing)),
    )
    return cfg


def run_eval(
    *,
    stage2_ckpt: Path,
    out_path: Path | None,
    max_val_samples: int | None,
    answer_model_id: str | None,
    prompt_template: str | None,
    generation_seed: int | None,
    skip_llm: bool,
    external_eval_path: Path | None,
    external_question_col: str,
    external_answer_col: str,
    external_label_col: str | None,
    external_reference_answer_col: str | None,
    max_external_samples: int | None,
) -> dict[str, Any]:
    if not stage2_ckpt.is_file():
        raise FileNotFoundError(f"Stage 2 checkpoint not found: {stage2_ckpt}")

    model_skeleton = build_model(Config())
    ckpt = torch.load(stage2_ckpt, map_location="cpu", weights_only=False)
    cfg = _build_cfg_from_checkpoint(ckpt["cfg"])
    model = build_model(cfg)
    load_checkpoint(stage2_ckpt, model=model, map_location=cfg.training.device)
    _ = model_skeleton  # keep reference to avoid unused-import warnings

    if max_val_samples is not None:
        cfg.qa_dataset = replace(cfg.qa_dataset, max_val_samples=max_val_samples)
    if answer_model_id is not None:
        cfg.answer_generator = replace(cfg.answer_generator, model_id=answer_model_id)
    if prompt_template is not None:
        cfg.answer_generator = replace(cfg.answer_generator, prompt_template=prompt_template)
    if generation_seed is not None:
        cfg.answer_generator = replace(cfg.answer_generator, generation_seed=generation_seed)

    if external_eval_path is not None:
        datamodule = QAExternalEvalDataModule(
            cfg,
            external_path=str(external_eval_path),
            question_col=external_question_col,
            answer_col=external_answer_col,
            label_col=external_label_col,
            reference_answer_col=external_reference_answer_col,
            max_samples=max_external_samples,
        )
    else:
        datamodule = QAPairsDataModule(cfg, skip_llm=skip_llm)

    task = build_task("hallucination_audit")
    result: dict[str, Any] = task.run(model, datamodule, cfg)
    result.pop("_scores", None)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage2-ckpt", type=str, required=True)
    p.add_argument("--out", type=str, default=None)
    p.add_argument(
        "--external-eval-path",
        type=str,
        default=None,
        help=(
            "Primary eval mode: external .json/.jsonl/.csv with question+answer and "
            "either label or reference answer."
        ),
    )
    p.add_argument("--external-question-col", type=str, default="question")
    p.add_argument("--external-answer-col", type=str, default="answer")
    p.add_argument("--external-label-col", type=str, default="label")
    p.add_argument("--external-reference-answer-col", type=str, default=None)
    p.add_argument("--max-external-samples", type=positive_int, default=None)
    p.add_argument("--max-val-samples", type=positive_int, default=None)
    p.add_argument("--answer-model-id", type=str, default=None)
    p.add_argument("--prompt-template", type=str, default=None)
    p.add_argument("--generation-seed", type=int, default=None)
    p.add_argument(
        "--skip-llm",
        action="store_true",
        help="Use cross-question-swap placeholders instead of instantiating an LLM (for plumbing tests).",
    )
    return p


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _build_argparser().parse_args(argv)
    label_col = args.external_label_col.strip() if args.external_label_col is not None else ""
    result = run_eval(
        stage2_ckpt=Path(args.stage2_ckpt),
        out_path=Path(args.out) if args.out else None,
        max_val_samples=args.max_val_samples,
        answer_model_id=args.answer_model_id,
        prompt_template=args.prompt_template,
        generation_seed=args.generation_seed,
        skip_llm=args.skip_llm,
        external_eval_path=Path(args.external_eval_path) if args.external_eval_path else None,
        external_question_col=args.external_question_col,
        external_answer_col=args.external_answer_col,
        external_label_col=label_col or None,
        external_reference_answer_col=args.external_reference_answer_col,
        max_external_samples=args.max_external_samples,
    )
    print(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    main()
