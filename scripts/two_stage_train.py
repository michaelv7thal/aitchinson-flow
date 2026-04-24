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

Skip Stage 1 training and start from an existing Stage 1 checkpoint (frozen
backbone for Stage 2)::

    python scripts/two_stage_train.py --stage2-only \\
        --stage1-backbone-ckpt checkpoints/two_stage_baseline_stage1_ckpts.epoch_25.pt

Run with ``--smoke`` for a tiny end-to-end smoke run (used in tests).
"""

from __future__ import annotations

import argparse
import json
import logging
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, cast

from _shared.bootstrap import bootstrap_repo_paths

# Allow direct script execution without editable install.
_REPO_ROOT = bootstrap_repo_paths(Path(__file__))

import torch  # noqa: E402

import aitchinson_flow.models  # noqa: E402,F401  — populate model REGISTRY

from _shared.cli import (  # noqa: E402
    add_training_data_args,
    apply_training_data_args,
    positive_int,
)
from _shared.smoke import make_smoke_config  # noqa: E402
from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.models.bayesian_auditor import (  # noqa: E402
    compose_auditor_from_stages,
)
from aitchinson_flow.models.factory import build_model  # noqa: E402
from aitchinson_flow.training.data_sources import build_training_datamodule  # noqa: E402
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


def _build_default_datamodule(cfg: Config) -> tuple[DataModule, dict[str, Any]]:
    """Build the configured training data source (raw text or LLM-generated)."""
    return build_training_datamodule(cfg)


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


def _load_stage1_checkpoint_state(
    path: Path,
    *,
    map_location: str | torch.device | None = None,
) -> tuple[dict[str, torch.Tensor], int | None, int | None]:
    """Load ``model_state_dict`` and counters from a training checkpoint file."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(
            f"Checkpoint at {path} must be a dict with 'model_state_dict' (training format)."
        )
    state = ckpt["model_state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"model_state_dict at {path} is not a mapping.")
    epoch = ckpt.get("epoch")
    global_step = ckpt.get("global_step")
    return (
        dict(state),
        int(epoch) if epoch is not None else None,
        int(global_step) if global_step is not None else None,
    )


def _load_backbone_into_stage2(
    stage2_model: torch.nn.Module,
    stage1_state: dict[str, torch.Tensor],
) -> list[str]:
    """Copy ``backbone.*`` (and Path B ``llm_projection.*``) from Stage 1 into Stage 2.

    The Stage 2 model already enforces that the backbone is frozen (see
    ``BayesianAuditorStage2._freeze_representations``); copying weights here
    just gives Stage 2 a meaningful starting representation. ``latent_head``
    and ``gp`` are left at random init.

    For Path B runs, the ``llm_projection.*`` learned during Stage 1 is also
    copied so Stage 2 sees the same simplex features Stage 1 was trained on.
    It is frozen alongside the backbone by ``_freeze_representations`` — Stage
    2 never updates it.

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

    # Path B: seed llm_projection from Stage 1 when both sides carry one.
    stage1_proj = _filter_state_dict_by_prefix(stage1_state, "llm_projection.")
    if stage1_proj and getattr(stage2_model, "llm_projection", None) is not None:
        proj_missing, proj_unexpected = stage2_model.llm_projection.load_state_dict(  # type: ignore[attr-defined]
            {k.removeprefix("llm_projection."): v for k, v in stage1_proj.items()},
            strict=False,
        )
        if proj_missing:
            raise RuntimeError(
                f"Missing llm_projection keys when seeding Stage 2 from Stage 1: "
                f"{sorted(proj_missing)}"
            )
        unexpected = list(unexpected) + [f"llm_projection.{k}" for k in proj_unexpected]

    # Unexpected keys (e.g. velocity head fragments) are tolerated but reported.
    return list(unexpected)


def _init_inducing_from_data(
    model: torch.nn.Module,
    datamodule: DataModule,
    cfg: Config,
) -> None:
    """Initialise GP inducing points from actual training token latents."""
    model.eval()
    z_samples: list[torch.Tensor] = []
    prepare = getattr(model, "prepare_batch", None)
    with torch.no_grad():
        for batch in datamodule.train_dataloader():
            # Path B: translate ``embeddings`` → ``log_x`` via the (frozen)
            # projection so Path A and Path B take the same inducing-init path.
            if prepare is not None:
                batch = prepare(batch)
            log_x = batch["log_x"]
            z = model._extract_tokens(log_x)  # type: ignore[attr-defined]
            z_samples.append(z.reshape(-1, z.shape[-1]))
            if sum(t.shape[0] for t in z_samples) >= cfg.gp.num_inducing:
                break
    Z_all = torch.cat(z_samples)
    M = cfg.gp.num_inducing
    if len(Z_all) >= M:
        idx = torch.randperm(len(Z_all))[:M]
        Z_sel = Z_all[idx]
    else:
        idx = torch.randint(0, len(Z_all), (M,))
        Z_sel = Z_all[idx]
    model.gp.Z.data.copy_(Z_sel)  # type: ignore[attr-defined]
    model.train()


def run_two_stage(
    cfg: Config,
    *,
    out_dir: Path,
    stage1_epochs: int,
    stage2_epochs: int,
    datamodule: DataModule | None = None,
    random_stage2_backbone: bool = False,
    save_plots: bool = True,
    stage2_only: bool = False,
    stage1_backbone_ckpt: Path | None = None,
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
        stage2_only: If True, skip Stage 1 training and load the frozen-backbone
            weights from ``stage1_backbone_ckpt`` (unless
            ``random_stage2_backbone`` is True).
        stage1_backbone_ckpt: Checkpoint path whose ``backbone.*`` weights seed
            Stage 2 and fuse into ``BayesianAuditor``. Required when
            ``stage2_only`` and not ``random_stage2_backbone``.

    Returns:
        Manifest dictionary describing all produced artifacts.
    """
    if not stage2_only and stage1_epochs < 1:
        raise ValueError(
            f"stage1_epochs must be >= 1 (Stage 1 backbone training must run at "
            f"least one epoch to produce a meaningful checkpoint), got {stage1_epochs}"
        )
    if stage2_only and not random_stage2_backbone and stage1_backbone_ckpt is None:
        raise ValueError(
            "stage2_only requires --stage1-backbone-ckpt (or pass stage1_backbone_ckpt=...) "
            "unless --random-stage2-backbone is set."
        )
    if stage2_only and stage1_backbone_ckpt is not None and not stage1_backbone_ckpt.is_file():
        raise FileNotFoundError(f"Stage 1 backbone checkpoint not found: {stage1_backbone_ckpt}")
    if stage2_epochs < 1:
        raise ValueError(
            f"stage2_epochs must be >= 1 (Stage 2 GP head training must run at "
            f"least one epoch), got {stage2_epochs}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    data_source_meta: dict[str, Any] | None = None
    if datamodule is None:
        datamodule, data_source_meta = _build_default_datamodule(cfg)
    run_group = cfg.training.wandb_group or out_dir.name

    s1_history: list[dict[str, float]] = []
    s1_model: torch.nn.Module | None = None
    s1_cfg: Config
    s1_path: Path | None
    stage1_epochs_effective: int = stage1_epochs

    if stage2_only:
        if random_stage2_backbone:
            stage1_epochs_effective = 0
            s1_cfg = _stage1_config(cfg, epochs=1, ckpt_dir=out_dir / "stage1_ckpts")
            s1_path = None
        else:
            assert stage1_backbone_ckpt is not None
            s1_state_loaded, ep_loaded, _ = _load_stage1_checkpoint_state(
                stage1_backbone_ckpt,
                map_location=cfg.training.device,
            )
            stage1_epochs_effective = ep_loaded if ep_loaded is not None else stage1_epochs
            s1_cfg = _stage1_config(
                cfg, epochs=stage1_epochs_effective, ckpt_dir=out_dir / "stage1_ckpts"
            )
            s1_model = build_model(s1_cfg)
            s1_model.load_state_dict(s1_state_loaded, strict=False)
            s1_path = stage1_backbone_ckpt.resolve()
    else:
        # ---- Stage 1 ----
        s1_cfg = _stage1_config(cfg, epochs=stage1_epochs, ckpt_dir=out_dir / "stage1_ckpts")
        s1_model = build_model(s1_cfg)
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
        assert s1_model is not None
        unexpected_backbone_keys = _load_backbone_into_stage2(s2_model, dict(s1_model.state_dict()))
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

    s2_model = s2_model.to(s2_cfg.training.device)
    _init_inducing_from_data(s2_model, datamodule, s2_cfg)

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
            "stage2_only": stage2_only,
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
    stage1_state_for_fuse: Mapping[str, torch.Tensor] | None
    if stage2_only and random_stage2_backbone:
        stage1_state_for_fuse = None
    else:
        assert s1_model is not None
        stage1_state_for_fuse = dict(s1_model.state_dict())
    fused = compose_auditor_from_stages(
        cfg,
        stage1_state=stage1_state_for_fuse,
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
            "composed_from": {
                "stage1": str(s1_path) if s1_path is not None else None,
                "stage2": str(s2_path),
            },
            "stage1_epochs": stage1_epochs_effective,
            "stage2_epochs": stage2_epochs,
            "random_stage2_backbone": random_stage2_backbone,
            "stage2_only": stage2_only,
            "trained": False,
        },
    )

    manifest: dict[str, Any] = {
        "stage1_ckpt": str(s1_path) if s1_path is not None else None,
        "stage2_ckpt": str(s2_path),
        "fused_ckpt": str(fused_path),
        "stage1_epochs": stage1_epochs_effective,
        "stage2_epochs": stage2_epochs,
        "random_stage2_backbone": random_stage2_backbone,
        "stage2_only": stage2_only,
        "stage1_backbone_ckpt": str(stage1_backbone_ckpt.resolve())
        if stage2_only and stage1_backbone_ckpt is not None
        else None,
        "fused_trained": False,
        "fused_composed_from": {
            "stage1": str(s1_path) if s1_path is not None else None,
            "stage2": str(s2_path),
        },
        "stage1_cfg": config_checkpoint_dict(s1_cfg),
        "stage2_cfg": config_checkpoint_dict(s2_cfg),
        "fused_cfg": config_checkpoint_dict(cfg),
        "training_data_source": data_source_meta,
    }

    if save_plots:
        import numpy as np  # noqa: PLC0415

        import benchmarks.tasks  # noqa: F401 — register text_audit

        from aitchinson_flow.plots import (  # noqa: E402
            save_stage_plots,
            stage_plot_data_from_scores,
        )
        from benchmarks.tasks.registry import build_task  # noqa: E402

        plots_dir = out_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        task = build_task("text_audit")

        # Stage 1 audit (no GP; energy channel = Hilbert energy per token).
        stage1_audit: dict[str, Any] = {}
        stage1_written: dict[str, str] = {}
        if s1_model is not None:
            stage1_audit = task.run(s1_model, datamodule, s1_cfg)
            stage1_scores = cast(
                dict[str, np.ndarray | None] | None, stage1_audit.pop("_scores", None)
            )
            if stage1_scores is not None:
                stage1_data = stage_plot_data_from_scores(
                    stage1_scores, stage="stage1", history=s1_history or None
                )
                # Stage 1 "variance" is a velocity-norm surrogate; the user plots
                # requested only energy diagnostics for Stage 1, so drop it to
                # avoid misleading variance histograms/heatmaps.
                stage1_data.variance_token_valid = None
                stage1_data.variance_token_invalid = None
                stage1_data.variance_seq_valid = None
                stage1_data.variance_seq_invalid = None
                stage1_written = save_stage_plots(plots_dir / "stage1", stage1_data)

        # Stage 2 audit (GP mean + epistemic variance + inducing points).
        stage2_audit = task.run(s2_model, datamodule, s2_cfg)
        stage2_scores = cast(dict[str, np.ndarray | None] | None, stage2_audit.pop("_scores", None))
        stage2_written: dict[str, str] = {}
        if stage2_scores is not None:
            stage2_data = stage_plot_data_from_scores(
                stage2_scores, stage="stage2", history=s2_history or None
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
            "stage1_epochs": stage1_epochs_effective,
            "stage2_epochs": stage2_epochs,
            "random_stage2_backbone": random_stage2_backbone,
            "stage2_only": stage2_only,
            "training_data_source": (data_source_meta or {}).get("source", "external"),
            "out_dir": str(out_dir),
        },
    )
    try:
        if orchestration_logger.active:
            orchestration_metrics: dict[str, float] = {
                "stage1_epochs": float(stage1_epochs_effective),
                "stage2_epochs": float(stage2_epochs),
                "total_epochs": float(stage1_epochs_effective + stage2_epochs),
                "random_stage2_backbone": float(random_stage2_backbone),
                "stage2_only": float(stage2_only),
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
            if cfg.training.wandb_log_model and s1_path is not None:
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
    """Return the shared smoke ``Config`` (re-exported for backwards compat)."""
    return make_smoke_config()


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=str, default="checkpoints/two_stage/baseline")
    p.add_argument("--stage1-epochs", type=positive_int, default=25)
    p.add_argument("--stage2-epochs", type=positive_int, default=25)
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
    p.add_argument(
        "--stage2-only",
        action="store_true",
        help="Skip Stage 1 training; load frozen backbone from --stage1-backbone-ckpt.",
    )
    p.add_argument(
        "--stage1-backbone-ckpt",
        type=str,
        default=None,
        help=(
            "With --stage2-only: path to Stage 1 .pt (model_state_dict). "
            f"Default: {_REPO_ROOT / 'checkpoints/two_stage_baseline_stage1_ckpts.epoch_25.pt'}"
        ),
    )
    add_training_data_args(p)
    return p


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _build_argparser().parse_args(argv)

    cfg = make_smoke_config() if args.smoke else Config()
    apply_training_data_args(cfg, args)

    out_dir = Path(args.out_dir)
    stage1_backbone_path: Path | None = None
    if args.stage2_only and not args.random_stage2_backbone:
        default_s1 = _REPO_ROOT / "checkpoints/two_stage/baseline/stage1_ckpts/epoch_25.pt"
        stage1_backbone_path = (
            Path(args.stage1_backbone_ckpt) if args.stage1_backbone_ckpt else default_s1
        )
    manifest = run_two_stage(
        cfg,
        out_dir=out_dir,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        random_stage2_backbone=args.random_stage2_backbone,
        save_plots=not args.no_plots,
        stage2_only=args.stage2_only,
        stage1_backbone_ckpt=stage1_backbone_path,
    )
    print(json.dumps({k: v for k, v in manifest.items() if not k.endswith("_cfg")}, indent=2))
    return manifest


if __name__ == "__main__":
    main()
