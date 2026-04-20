"""Path B Phase 2: train the byte-level Q+A Stage 2 GP head.

Loads the Path B Phase 1 backbone (produced by
:mod:`scripts.phase1_train_bytes`) into a fresh ``BayesianAuditorStage2`` at
``K=256``, then trains the contrastive GP head on byte-encoded ``[Q][A]``
pairs from :class:`~aitchinson_flow.data.qa_datamodule.QAPairsDataModule`.

The training loss uses the QA datamodule's ``log_x_invalid`` (cross-question
swap) instead of ``randn_like`` negatives, and — when
``cfg.gp.score_answer_tokens_only`` is True — restricts the GP loss to
answer-span positions via ``answer_mask``.

Usage::

    python scripts/phase2_train_qa.py \
        --out-dir checkpoints/phase2_qa/baseline \
        --stage1-backbone-ckpt checkpoints/phase1_bytes/baseline/stage1.pt \
        --epochs 10
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

# Allow direct script execution: ``scripts/`` is not a package, so import the
# Path A two-stage helpers via the module name after bootstrap has added
# ``scripts/`` to ``sys.path``.
import sys  # noqa: E402
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import torch  # noqa: E402

import aitchinson_flow.models  # noqa: E402,F401 — populate REGISTRY

from _shared.cli import positive_int  # noqa: E402
from _shared.smoke import make_smoke_config  # noqa: E402
from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.data.qa_datamodule import QAPairsDataModule  # noqa: E402
from aitchinson_flow.models.bayesian_auditor import (  # noqa: E402
    compose_auditor_from_stages,
)
from aitchinson_flow.models.factory import build_model  # noqa: E402
from aitchinson_flow.training.checkpoint import (  # noqa: E402
    config_checkpoint_dict,
    save_checkpoint,
)
from aitchinson_flow.training.runner import fit  # noqa: E402
from two_stage_train import (  # noqa: E402
    _init_inducing_from_data,
    _load_backbone_into_stage2,
    _load_stage1_checkpoint_state,
)


def _phase2_qa_config(base: Config, *, epochs: int, L: int, ckpt_dir: Path) -> Config:
    cfg = deepcopy(base)
    cfg.dataset = replace(cfg.dataset, K=256, L=L)
    cfg.training = replace(
        cfg.training,
        model_name="bayesian_auditor_stage2",
        epochs=epochs,
        checkpoint_dir=str(ckpt_dir),
    )
    cfg.training_data = replace(cfg.training_data, source="qa_pairs")
    cfg.gp = replace(cfg.gp, score_answer_tokens_only=True)
    return cfg


def run_phase2_qa(
    cfg: Config,
    *,
    out_dir: Path,
    stage1_backbone_ckpt: Path,
    epochs: int,
    L: int,
    skip_llm_eval: bool = False,
) -> dict[str, Any]:
    if not stage1_backbone_ckpt.is_file():
        raise FileNotFoundError(
            f"Phase 1 (byte) backbone checkpoint not found: {stage1_backbone_ckpt}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = _phase2_qa_config(cfg, epochs=epochs, L=L, ckpt_dir=out_dir / "stage2_ckpts")
    run_cfg.training_data = replace(run_cfg.training_data, qa_skip_llm_eval=skip_llm_eval)

    datamodule = QAPairsDataModule(run_cfg, skip_llm=skip_llm_eval)

    stage1_state, _, _ = _load_stage1_checkpoint_state(
        stage1_backbone_ckpt, map_location=run_cfg.training.device
    )
    model = build_model(run_cfg)
    unexpected = _load_backbone_into_stage2(model, stage1_state)
    model._freeze_representations()  # type: ignore[attr-defined]
    _init_inducing_from_data(model, datamodule, run_cfg)

    history: list[dict[str, float]] = []
    model = fit(
        run_cfg,
        datamodule,
        model=model,
        history_out=history,
        wandb_run_name=f"{out_dir.name}-phase2-qa",
        wandb_group=run_cfg.training.wandb_group or out_dir.name,
        wandb_job_type="phase2_qa_train",
        wandb_tags=["path-b", "phase2", "qa"],
        wandb_extra_config={
            "stage": "phase2_qa",
            "K": run_cfg.dataset.K,
            "L": run_cfg.dataset.L,
            "skip_llm_eval": skip_llm_eval,
            "stage1_backbone_ckpt": str(stage1_backbone_ckpt),
            "out_dir": str(out_dir),
        },
    )

    s2_path = out_dir / "stage2.pt"
    save_checkpoint(
        s2_path,
        model=model,
        cfg=run_cfg,
        optimizer=None,
        epoch=epochs,
        global_step=0,
    )

    fused = compose_auditor_from_stages(
        run_cfg,
        stage1_state=stage1_state,
        stage2_state=dict(model.state_dict()),
        strict=False,
    )
    fused_path = out_dir / "fused.pt"
    save_checkpoint(
        fused_path,
        model=fused,
        cfg=run_cfg,
        optimizer=None,
        epoch=0,
        global_step=0,
        extra_metadata={
            "composed_from": {
                "stage1": str(stage1_backbone_ckpt),
                "stage2": str(s2_path),
            },
            "path": "B",
            "score_answer_tokens_only": run_cfg.gp.score_answer_tokens_only,
            "stage2_unexpected_backbone_keys": sorted(unexpected),
        },
    )

    manifest: dict[str, Any] = {
        "stage1_backbone_ckpt": str(stage1_backbone_ckpt),
        "stage2_ckpt": str(s2_path),
        "fused_ckpt": str(fused_path),
        "epochs": epochs,
        "K": run_cfg.dataset.K,
        "L": run_cfg.dataset.L,
        "skip_llm_eval": skip_llm_eval,
        "score_answer_tokens_only": run_cfg.gp.score_answer_tokens_only,
        "stage2_cfg": config_checkpoint_dict(run_cfg),
    }
    (out_dir / "phase2_qa_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    return manifest


def _smoke_qa_config() -> Config:
    cfg = make_smoke_config()
    cfg.dataset.L = 32
    cfg.training.epochs = 1
    cfg.qa_dataset.max_train_samples = 16
    cfg.qa_dataset.max_val_samples = 8
    cfg.qa_dataset.max_test_samples = None
    cfg.qa_dataset.max_question_bytes = 24
    cfg.qa_dataset.max_answer_bytes = 6
    return cfg


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=str, default="checkpoints/phase2_qa/baseline")
    p.add_argument(
        "--stage1-backbone-ckpt",
        type=str,
        required=False,
        default="checkpoints/phase1_bytes/baseline/stage1.pt",
    )
    p.add_argument("--epochs", type=positive_int, default=10)
    p.add_argument("--L", type=positive_int, default=128)
    p.add_argument(
        "--skip-llm-eval",
        action="store_true",
        help="Use cross-question-swap placeholders instead of instantiating an LLM.",
    )
    p.add_argument("--smoke", action="store_true")
    return p


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _build_argparser().parse_args(argv)
    cfg = _smoke_qa_config() if args.smoke else Config()
    epochs = 1 if args.smoke else args.epochs
    L = 32 if args.smoke else args.L
    manifest = run_phase2_qa(
        cfg,
        out_dir=Path(args.out_dir),
        stage1_backbone_ckpt=Path(args.stage1_backbone_ckpt),
        epochs=epochs,
        L=L,
        skip_llm_eval=args.skip_llm_eval,
    )
    print(json.dumps(manifest, indent=2, default=str))
    return manifest


if __name__ == "__main__":
    main()
