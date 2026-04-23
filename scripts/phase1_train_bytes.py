"""Phase 1: train the byte-level (K=256) Stage 1 backbone on raw QA pairs.

This driver specializes :mod:`two_stage_train` to train *only* Stage 1 on
byte-encoded concatenated ``[Q][A]`` samples from
:class:`~aitchinson_flow.data.qa_datamodule.QAPairsDataModule`.

Unlike the old Path-B logit/embedding path, this run uses raw question/answer
text, with Stage 1 updates optionally restricted to answer tokens via
``cfg.training.stage1_answer_tokens_only``.

Usage::

    python scripts/phase1_train_bytes.py \
        --out-dir checkpoints/phase1_bytes/baseline \
        --epochs 10 --L 512

    python scripts/phase1_train_bytes.py --smoke  # tiny CPU run
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

from _shared.bootstrap import bootstrap_repo_paths

_REPO_ROOT = bootstrap_repo_paths(Path(__file__))

import torch  # noqa: E402

import aitchinson_flow.models  # noqa: E402,F401 — populate model REGISTRY

from _shared.cli import positive_int  # noqa: E402
from _shared.smoke import make_smoke_config  # noqa: E402
from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.data.qa_datamodule import QAPairsDataModule  # noqa: E402
from aitchinson_flow.models.factory import build_model  # noqa: E402
from aitchinson_flow.training.checkpoint import (  # noqa: E402
    config_checkpoint_dict,
    save_checkpoint,
)
from aitchinson_flow.training.runner import fit  # noqa: E402


def _qa_byte_level_config(base: Config, *, epochs: int, L: int, ckpt_dir: Path) -> Config:
    cfg = deepcopy(base)
    cfg.dataset = replace(cfg.dataset, K=256, L=L)
    cfg.training = replace(
        cfg.training,
        model_name="bayesian_auditor_stage1",
        epochs=epochs,
        checkpoint_dir=str(ckpt_dir),
        stage1_answer_tokens_only=True,
    )
    cfg.training_data = replace(cfg.training_data, source="qa_pairs", qa_skip_llm_eval=True)
    return cfg


def run_phase1_qa(
    cfg: Config,
    *,
    out_dir: Path,
    epochs: int,
    L: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_cfg = _qa_byte_level_config(cfg, epochs=epochs, L=L, ckpt_dir=out_dir / "stage1_ckpts")

    datamodule = QAPairsDataModule(run_cfg, skip_llm=True)
    model = build_model(run_cfg)

    history: list[dict[str, float]] = []
    model = fit(
        run_cfg,
        datamodule,
        model=model,
        history_out=history,
        wandb_run_name=f"{out_dir.name}-phase1-qa",
        wandb_group=run_cfg.training.wandb_group or out_dir.name,
        wandb_job_type="phase1_qa_train",
        wandb_tags=["phase1", "qa", "raw-bytes"],
        wandb_extra_config={
            "stage": "phase1_qa",
            "K": run_cfg.dataset.K,
            "L": run_cfg.dataset.L,
            "source": run_cfg.training_data.source,
            "stage1_answer_tokens_only": run_cfg.training.stage1_answer_tokens_only,
            "out_dir": str(out_dir),
        },
    )
    ckpt_path = out_dir / "stage1.pt"
    save_checkpoint(
        ckpt_path,
        model=model,
        cfg=run_cfg,
        optimizer=None,
        epoch=epochs,
        global_step=0,
    )

    manifest: dict[str, Any] = {
        "stage1_ckpt": str(ckpt_path),
        "epochs": epochs,
        "K": run_cfg.dataset.K,
        "L": run_cfg.dataset.L,
        "source": run_cfg.training_data.source,
        "stage1_answer_tokens_only": run_cfg.training.stage1_answer_tokens_only,
        "stage1_cfg": config_checkpoint_dict(run_cfg),
    }
    (out_dir / "phase1_qa_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    return manifest


def _smoke_qa_config() -> Config:
    cfg = make_smoke_config()
    # QA smoke: tiny sequence and tiny QA row budgets.
    cfg.dataset.L = 32
    cfg.training.epochs = 1
    cfg.qa_dataset.max_train_samples = 16
    cfg.qa_dataset.max_val_samples = 8
    cfg.qa_dataset.max_test_samples = 8
    cfg.qa_dataset.max_question_bytes = 24
    cfg.qa_dataset.max_answer_bytes = 8
    return cfg


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=str, default="checkpoints/phase1_bytes/baseline")
    p.add_argument("--epochs", type=positive_int, default=25)
    p.add_argument("--L", type=positive_int, default=256)
    p.add_argument("--smoke", action="store_true")
    return p


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _build_argparser().parse_args(argv)
    cfg = _smoke_qa_config() if args.smoke else Config()
    out_dir = Path(args.out_dir)
    epochs = 1 if args.smoke else args.epochs
    L = 32 if args.smoke else args.L
    manifest = run_phase1_qa(cfg, out_dir=out_dir, epochs=epochs, L=L)
    print(json.dumps(manifest, indent=2, default=str))
    return manifest


if __name__ == "__main__":
    main()
