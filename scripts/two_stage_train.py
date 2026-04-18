"""Train Stage 1, then Stage 2, then compose a fused BayesianAuditor checkpoint.

This is the canonical workflow for the two-stage Bayesian Auditor (see plan
sections "Architecture: Two-Stage Design" and "Stage 2: Contrastive GP on
Frozen Backbone"):

    1. Train ``BayesianAuditorStage1`` on valid sequences with EqM + Hilbert
       backbone training. Writes ``stage1.pt``.
    2. Train ``BayesianAuditorStage2`` with the contrastive GP head on top of
       a frozen backbone. By default the Stage 2 backbone is initialized from
       the Stage 1 checkpoint; pass ``--random-stage2-backbone`` for the
       random-backbone ablation called out in the plan.
       Writes ``stage2.pt``.
    3. Compose Stage 1 backbone + Stage 2 latent_head/GP into a
       ``BayesianAuditor`` and save the fused checkpoint as ``fused.pt``.

Outputs (all under ``--out-dir``):
    * ``stage1.pt`` — Stage 1 model state dict.
    * ``stage2.pt`` — Stage 2 model state dict.
    * ``fused.pt`` — Fused ``BayesianAuditor`` state dict (inference-ready).
    * ``orchestration.json`` — Manifest with per-stage configs and paths.
    * ``plots/stage1/`` and ``plots/stage2/`` — stage-specific diagnostics
      (valid/invalid histograms, token heatmaps, latent density with inducing
      points, and train/val loss curves) produced by
      ``aitchinson_flow.plots.save_stage_plots`` unless ``--no-plots``.

Usage:
    python scripts/two_stage_train.py \
        --out-dir checkpoints/two_stage/baseline \
        --stage1-epochs 10 --stage2-epochs 5

Run with ``--smoke`` for a tiny end-to-end smoke run (used in tests).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

# Allow `python scripts/two_stage_train.py` without an editable install.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
# Repo root so ``import benchmarks.*`` resolves (plots + text_audit task).
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

import torch  # noqa: E402

import aitchinson_flow.models  # noqa: E402,F401  — populate model REGISTRY

from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.models.bayesian_auditor import (  # noqa: E402
    compose_auditor_from_stages,
)
from aitchinson_flow.models.factory import build_model  # noqa: E402
from aitchinson_flow.training.checkpoint import (  # noqa: E402
    config_checkpoint_dict,
    save_checkpoint,
)
from aitchinson_flow.training.datamodule import DataModule  # noqa: E402
from aitchinson_flow.training.runner import fit  # noqa: E402
from aitchinson_flow.training.wandb_logger import WandbLogger  # noqa: E402


def _numeric_stage_metrics(stage: str, audit: dict[str, Any] | None) -> dict[str, float]:
    if not audit:
        return {}
    out: dict[str, float] = {}
    for key, value in audit.items():
        if isinstance(value, bool):
            out[f"{stage}/{key}"] = float(value)
        elif isinstance(value, (int, float)):
            out[f"{stage}/{key}"] = float(value)
    return out


def _build_default_datamodule(cfg: Config) -> DataModule:
    """Default to the text8 datamodule (matches benchmark + plan defaults)."""
    from aitchinson_flow.data.text8_datamodule import Text8DataModule  # noqa: PLC0415

    return Text8DataModule(cfg)


def _stage1_config(base: Config, *, epochs: int, ckpt_dir: Path) -> Config:
    cfg = deepcopy(base)
    cfg.training = replace(
        cfg.training,
        model_name="bayesian_auditor_stage1",
        epochs=epochs,
        checkpoint_dir=str(ckpt_dir),
    )
    return cfg


def _stage2_config(base: Config, *, epochs: int, ckpt_dir: Path) -> Config:
    cfg = deepcopy(base)
    cfg.training = replace(
        cfg.training,
        model_name="bayesian_auditor_stage2",
        epochs=epochs,
        checkpoint_dir=str(ckpt_dir),
    )
    return cfg


def _filter_state_dict_by_prefix(
    state: dict[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    return {k: v for k, v in state.items() if k.startswith(prefix)}


def _load_backbone_into_stage2(
    stage2_model: torch.nn.Module,
    stage1_state: dict[str, torch.Tensor],
) -> list[str]:
    """Copy ``backbone.*`` weights from Stage 1 into a freshly built Stage 2 model.

    The Stage 2 model already enforces that the backbone is frozen (see
    ``BayesianAuditorStage2._freeze_representations``); copying weights here
    just gives Stage 2 a meaningful starting representation. ``latent_head``
    and ``gp`` are left at random init.

    Returns the list of unexpected backbone keys that were filtered out, so
    callers can surface them to logs (M7). An empty return value means the
    Stage 1 backbone and Stage 2 backbone state dicts matched exactly.
    """
    backbone_state = _filter_state_dict_by_prefix(stage1_state, "backbone.")
    if not backbone_state:
        raise ValueError("Stage 1 checkpoint contains no 'backbone.*' keys; cannot seed Stage 2.")
    missing, unexpected = stage2_model.backbone.load_state_dict(  # type: ignore[attr-defined]
        {k.removeprefix("backbone."): v for k, v in backbone_state.items()},
        strict=False,
    )
    if missing:
        # Hard error: a key the Stage 2 backbone needs was not provided.
        raise RuntimeError(
            f"Missing backbone keys when seeding Stage 2 from Stage 1: {sorted(missing)}"
        )
    # Unexpected keys (e.g. velocity head fragments) are tolerated but reported.
    return list(unexpected)


def run_two_stage(
    cfg: Config,
    *,
    out_dir: Path,
    stage1_epochs: int,
    stage2_epochs: int,
    datamodule: DataModule | None = None,
    random_stage2_backbone: bool = False,
    save_plots: bool = True,
) -> dict[str, Any]:
    """Execute the Stage 1 → Stage 2 → compose pipeline.

    Args:
        cfg: Base config used as the starting point for both stages. Per-stage
            overrides (``model_name``, ``epochs``, ``checkpoint_dir``) are
            applied internally without mutating the input.
        out_dir: Output directory for ``stage1.pt`` / ``stage2.pt`` /
            ``fused.pt`` / ``orchestration.json``.
        stage1_epochs / stage2_epochs: Per-stage epoch counts.
        datamodule: Optional pre-built datamodule. Both stages share the same
            datamodule by design (Stage 2 needs the contrastive ``log_x_invalid``
            field that the text8 datamodule already produces).
        random_stage2_backbone: If True, Stage 2 starts from a fresh random
            backbone instead of seeding from Stage 1 (random-backbone
            ablation).
        save_plots: If True, run ``text_audit`` on the Stage 1 and Stage 2
            models separately and write stage-specific diagnostics under
            ``out_dir/plots/stage1`` and ``out_dir/plots/stage2`` via
            ``aitchinson_flow.plots.save_stage_plots``.

    Returns:
        Manifest dictionary describing all produced artifacts.
    """
    if stage1_epochs < 1:
        raise ValueError(
            f"stage1_epochs must be >= 1 (Stage 1 backbone training must run at "
            f"least one epoch to produce a meaningful checkpoint), got {stage1_epochs}"
        )
    if stage2_epochs < 1:
        raise ValueError(
            f"stage2_epochs must be >= 1 (Stage 2 GP head training must run at "
            f"least one epoch), got {stage2_epochs}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    if datamodule is None:
        datamodule = _build_default_datamodule(cfg)
    run_group = cfg.training.wandb_group or out_dir.name

    # ---- Stage 1 ----
    s1_cfg = _stage1_config(cfg, epochs=stage1_epochs, ckpt_dir=out_dir / "stage1_ckpts")
    s1_model = build_model(s1_cfg)
    s1_history: list[dict[str, float]] = []
    s1_model = fit(
        s1_cfg,
        datamodule,
        model=s1_model,
        history_out=s1_history,
        wandb_run_name=f"{out_dir.name}-stage1",
        wandb_group=run_group,
        wandb_job_type="stage1_train",
        wandb_tags=["two-stage", "stage1"],
        wandb_extra_config={
            "stage": "stage1",
            "stage_epochs": stage1_epochs,
            "random_stage2_backbone": random_stage2_backbone,
            "out_dir": str(out_dir),
        },
    )
    s1_path = out_dir / "stage1.pt"
    save_checkpoint(
        s1_path,
        model=s1_model,
        cfg=s1_cfg,
        optimizer=None,
        epoch=stage1_epochs,
        global_step=0,
    )

    # ---- Stage 2 ----
    s2_cfg = _stage2_config(cfg, epochs=stage2_epochs, ckpt_dir=out_dir / "stage2_ckpts")
    s2_model = build_model(s2_cfg)
    if not random_stage2_backbone:
        unexpected_backbone_keys = _load_backbone_into_stage2(
            s2_model, dict(s1_model.state_dict())
        )
        if unexpected_backbone_keys:
            logging.getLogger(__name__).warning(
                "Stage 2 backbone load ignored %d unexpected key(s) from Stage 1: %s. "
                "These are typically Stage-1-only heads (e.g. velocity_head.*) that "
                "have no counterpart in the Stage 2 backbone; confirm none of these "
                "belong inside the shared backbone.",
                len(unexpected_backbone_keys),
                sorted(unexpected_backbone_keys),
            )
        # Re-pin the freeze policy now that we mutated backbone weights.
        s2_model._freeze_representations()  # type: ignore[attr-defined]
    s2_history: list[dict[str, float]] = []
    s2_model = fit(
        s2_cfg,
        datamodule,
        model=s2_model,
        history_out=s2_history,
        wandb_run_name=f"{out_dir.name}-stage2",
        wandb_group=run_group,
        wandb_job_type="stage2_train",
        wandb_tags=["two-stage", "stage2"],
        wandb_extra_config={
            "stage": "stage2",
            "stage_epochs": stage2_epochs,
            "random_stage2_backbone": random_stage2_backbone,
            "out_dir": str(out_dir),
        },
    )
    s2_path = out_dir / "stage2.pt"
    save_checkpoint(
        s2_path,
        model=s2_model,
        cfg=s2_cfg,
        optimizer=None,
        epoch=stage2_epochs,
        global_step=0,
    )

    # ---- Compose ----
    fused = compose_auditor_from_stages(
        cfg,
        stage1_state=dict(s1_model.state_dict()),
        stage2_state=dict(s2_model.state_dict()),
        strict=False,
    )
    fused_path = out_dir / "fused.pt"
    # The fused model is composed from stage checkpoints, not trained directly.
    # Record epoch=0/global_step=0 (accurate, no training steps were taken on
    # the composed model) and surface composition provenance via
    # ``extra_metadata`` so downstream tools do not misread the counters as
    # indicating stage1_epochs + stage2_epochs of training on the fused head.
    save_checkpoint(
        fused_path,
        model=fused,
        cfg=cfg,
        optimizer=None,
        epoch=0,
        global_step=0,
        extra_metadata={
            "composed_from": {"stage1": str(s1_path), "stage2": str(s2_path)},
            "stage1_epochs": stage1_epochs,
            "stage2_epochs": stage2_epochs,
            "random_stage2_backbone": random_stage2_backbone,
            "trained": False,
        },
    )

    manifest: dict[str, Any] = {
        "stage1_ckpt": str(s1_path),
        "stage2_ckpt": str(s2_path),
        "fused_ckpt": str(fused_path),
        "stage1_epochs": stage1_epochs,
        "stage2_epochs": stage2_epochs,
        "random_stage2_backbone": random_stage2_backbone,
        "fused_trained": False,
        "fused_composed_from": {"stage1": str(s1_path), "stage2": str(s2_path)},
        "stage1_cfg": config_checkpoint_dict(s1_cfg),
        "stage2_cfg": config_checkpoint_dict(s2_cfg),
        "fused_cfg": config_checkpoint_dict(cfg),
    }

    if save_plots:
        import numpy as np  # noqa: PLC0415

        import benchmarks.tasks  # noqa: F401 — register text_audit

        from aitchinson_flow.plots import StagePlotData, save_stage_plots  # noqa: E402
        from benchmarks.tasks.registry import build_task  # noqa: E402

        plots_dir = out_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        def _stage_data_from_scores(
            *,
            stage: str,
            scores: dict[str, np.ndarray | None],
            history: list[dict[str, float]] | None,
        ) -> StagePlotData:
            return StagePlotData(
                stage=stage,
                energy_token_valid=scores.get("auditor_energy_seq_valid"),
                energy_token_invalid=scores.get("auditor_energy_seq_invalid"),
                variance_token_valid=scores.get("auditor_var_seq_valid"),
                variance_token_invalid=scores.get("auditor_var_seq_invalid"),
                latent_tokens_valid=scores.get("latent_tokens_valid"),
                latent_tokens_invalid=scores.get("latent_tokens_invalid"),
                inducing_points=scores.get("inducing_points"),
                history=history,
            )

        task = build_task("text_audit")

        # Stage 1 audit (no GP; energy channel = Hilbert energy per token).
        stage1_audit = task.run(s1_model, datamodule, s1_cfg)
        stage1_scores = cast(dict[str, np.ndarray | None] | None, stage1_audit.pop("_scores", None))
        stage1_written: dict[str, str] = {}
        if stage1_scores is not None:
            stage1_data = _stage_data_from_scores(
                stage="stage1", scores=stage1_scores, history=s1_history or None
            )
            # Stage 1 "variance" is a velocity-norm surrogate; the user plots
            # requested only energy diagnostics for Stage 1, so drop it to
            # avoid misleading variance histograms/heatmaps.
            stage1_data.variance_token_valid = None
            stage1_data.variance_token_invalid = None
            stage1_written = save_stage_plots(plots_dir / "stage1", stage1_data)

        # Stage 2 audit (GP mean + epistemic variance + inducing points).
        stage2_audit = task.run(s2_model, datamodule, s2_cfg)
        stage2_scores = cast(dict[str, np.ndarray | None] | None, stage2_audit.pop("_scores", None))
        stage2_written: dict[str, str] = {}
        if stage2_scores is not None:
            stage2_data = _stage_data_from_scores(
                stage="stage2", scores=stage2_scores, history=s2_history or None
            )
            stage2_written = save_stage_plots(plots_dir / "stage2", stage2_data)

        manifest["plots_dir"] = str(plots_dir)
        manifest["stage1_plots"] = stage1_written
        manifest["stage2_plots"] = stage2_written
        manifest["stage1_audit"] = {
            k: v for k, v in stage1_audit.items() if not str(k).startswith("_")
        }
        manifest["stage2_audit"] = {
            k: v for k, v in stage2_audit.items() if not str(k).startswith("_")
        }

    orchestration_path = out_dir / "orchestration.json"
    orchestration_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    orchestration_logger = WandbLogger(
        cfg,
        run_name=f"{out_dir.name}-orchestration",
        group=run_group,
        job_type="two_stage_orchestration",
        tags=["two-stage", "orchestration"],
        extra_config={
            "stage1_epochs": stage1_epochs,
            "stage2_epochs": stage2_epochs,
            "random_stage2_backbone": random_stage2_backbone,
            "out_dir": str(out_dir),
        },
    )
    try:
        if orchestration_logger.active:
            orchestration_metrics: dict[str, float] = {
                "stage1_epochs": float(stage1_epochs),
                "stage2_epochs": float(stage2_epochs),
                "total_epochs": float(stage1_epochs + stage2_epochs),
                "random_stage2_backbone": float(random_stage2_backbone),
            }
            orchestration_metrics.update(
                _numeric_stage_metrics(
                    "stage1",
                    cast(dict[str, Any] | None, manifest.get("stage1_audit")),
                )
            )
            orchestration_metrics.update(
                _numeric_stage_metrics(
                    "stage2",
                    cast(dict[str, Any] | None, manifest.get("stage2_audit")),
                )
            )
            orchestration_logger.log_metrics(orchestration_metrics)
            if cfg.training.wandb_log_model:
                orchestration_logger.log_artifact(
                    s1_path,
                    name=f"{out_dir.name}-stage1-model",
                    artifact_type="model",
                    aliases=["latest", "stage1"],
                )
                orchestration_logger.log_artifact(
                    s2_path,
                    name=f"{out_dir.name}-stage2-model",
                    artifact_type="model",
                    aliases=["latest", "stage2"],
                )
                orchestration_logger.log_artifact(
                    fused_path,
                    name=f"{out_dir.name}-fused-model",
                    artifact_type="model",
                    aliases=["latest", "fused"],
                )
            orchestration_logger.log_artifact(
                orchestration_path,
                name=f"{out_dir.name}-orchestration-manifest",
                artifact_type="manifest",
                aliases=["latest"],
            )
            if save_plots and "plots_dir" in manifest:
                orchestration_logger.log_artifact(
                    manifest["plots_dir"],
                    name=f"{out_dir.name}-plots",
                    artifact_type="plots",
                    aliases=["latest"],
                )
    finally:
        orchestration_logger.finish()
    return manifest


def _smoke_config() -> Config:
    cfg = Config()
    cfg.training.device = torch.device("cpu")
    cfg.training.seed = 0
    cfg.training.B = 4
    cfg.training.lr = 1e-3
    cfg.training.checkpoint_every = 1
    cfg.training.lr_scheduler = None
    cfg.training.use_tqdm = False
    cfg.dataset.K = 27
    cfg.dataset.L = 8
    cfg.transformer.d_model = 16
    cfg.transformer.nhead = 2
    cfg.transformer.num_layers = 1
    cfg.transformer.d_latent = 16
    cfg.gp.num_inducing = 8
    cfg.benchmark.use_tqdm = False
    return cfg


def _positive_int(raw: str) -> int:
    """argparse type for flags that must be >= 1."""
    try:
        value = int(raw)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"expected integer, got {raw!r}") from e
    if value < 1:
        raise argparse.ArgumentTypeError(
            f"expected a positive integer (>= 1), got {value}"
        )
    return value


def main(argv: list[str] | None = None) -> dict[str, Any]:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=str, default="checkpoints/two_stage/baseline")
    p.add_argument("--stage1-epochs", type=_positive_int, default=25)
    p.add_argument("--stage2-epochs", type=_positive_int, default=25)
    p.add_argument(
        "--random-stage2-backbone",
        action="store_true",
        help="Skip seeding Stage 2 from Stage 1 (random-backbone ablation).",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Use a tiny CPU config for end-to-end smoke testing.",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip per-stage text_audit + plotting outputs under out-dir/plots.",
    )
    args = p.parse_args(argv)

    cfg = _smoke_config() if args.smoke else Config()
    out_dir = Path(args.out_dir)
    manifest = run_two_stage(
        cfg,
        out_dir=out_dir,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        random_stage2_backbone=args.random_stage2_backbone,
        save_plots=not args.no_plots,
    )
    print(json.dumps({k: v for k, v in manifest.items() if not k.endswith("_cfg")}, indent=2))
    return manifest


if __name__ == "__main__":
    main()
