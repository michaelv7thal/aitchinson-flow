"""Path B: evaluate a trained Stage 2 QA auditor against LLM-generated answers.

Instantiates the configured answer-generator LLM, has it answer val/test
questions, re-encodes each ``[Q][A_llm]`` pair at the byte level, scores
them with the trained Stage 2 auditor, and reports hallucination-detection
AUROC via :class:`~benchmarks.tasks.hallucination_audit.HallucinationAuditTask`.

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
from aitchinson_flow.data.qa_datamodule import QAPairsDataModule  # noqa: E402
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
    result = run_eval(
        stage2_ckpt=Path(args.stage2_ckpt),
        out_path=Path(args.out) if args.out else None,
        max_val_samples=args.max_val_samples,
        answer_model_id=args.answer_model_id,
        prompt_template=args.prompt_template,
        generation_seed=args.generation_seed,
        skip_llm=args.skip_llm,
    )
    print(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    main()
