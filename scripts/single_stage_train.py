"""Single-stage training entrypoint: train a registered model via ``fit()``.

This is the canonical thin objective entrypoint pattern. It does **no**
two-stage orchestration — it simply:

    1. Builds a ``Config`` (optionally the smoke baseline).
    2. Applies the shared ``--training-data-*`` CLI overrides (raw text vs
       LLM-generated).
    3. Builds the training datamodule via
       :func:`aitchinson_flow.training.data_sources.build_training_datamodule`.
    4. Calls :func:`aitchinson_flow.training.runner.fit` on the model named
       by ``--model-name`` (defaults to whatever ``Config`` declares).

Usage example::

    python scripts/single_stage_train.py \\
        --model-name bayesian_auditor \\
        --epochs 50 \\
        --training-data-source llm_generated \\
        --training-lm-key hf_causal

This file is intentionally tiny so it serves as a reference template for any
future training objective: copy it, swap ``fit(...)`` for the objective-specific
orchestration function, and you inherit the entire shared data-source pipeline.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from _shared.bootstrap import bootstrap_repo_paths

# Allow direct script execution without editable install.
bootstrap_repo_paths(Path(__file__))

import aitchinson_flow.models  # noqa: E402,F401  — populate model REGISTRY

from _shared.cli import (  # noqa: E402
    add_training_data_args,
    apply_training_data_args,
    positive_int,
)
from _shared.smoke import make_smoke_config  # noqa: E402
from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.training.data_sources import build_training_datamodule  # noqa: E402
from aitchinson_flow.training.runner import fit  # noqa: E402


def run_single_stage(
    cfg: Config,
    *,
    out_dir: Path,
    epochs: int,
    model_name: str | None = None,
    wandb_run_name: str | None = None,
) -> dict[str, Any]:
    """Run single-objective training via ``fit`` and return a manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if model_name is not None:
        cfg.training = replace(cfg.training, model_name=model_name)
    cfg.training = replace(
        cfg.training,
        epochs=epochs,
        checkpoint_dir=str(out_dir / "ckpts"),
    )

    datamodule, data_source_meta = build_training_datamodule(cfg)

    history: list[dict[str, float]] = []
    fit(
        cfg,
        datamodule,
        history_out=history,
        wandb_run_name=wandb_run_name or out_dir.name,
        wandb_job_type="single_stage_train",
        wandb_tags=["single-stage", cfg.training.model_name],
        wandb_extra_config={
            "objective": "single_stage",
            "model_name": cfg.training.model_name,
            "epochs": epochs,
            "out_dir": str(out_dir),
        },
    )

    manifest: dict[str, Any] = {
        "objective": "single_stage",
        "model_name": cfg.training.model_name,
        "epochs": epochs,
        "out_dir": str(out_dir),
        "training_data_source": data_source_meta,
        "history": history,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    return manifest


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=str, default="checkpoints/single_stage/baseline")
    p.add_argument("--epochs", type=positive_int, default=25)
    p.add_argument(
        "--model-name",
        type=str,
        default=None,
        help=(
            "Override ``cfg.training.model_name``. Must be a key registered in "
            "``aitchinson_flow.models.factory.REGISTRY`` (e.g. "
            "``bayesian_auditor``, ``flow_matching``, ``equilibrium``)."
        ),
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Use a tiny CPU config for end-to-end smoke testing.",
    )
    add_training_data_args(p)
    return p


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _build_argparser().parse_args(argv)
    cfg = make_smoke_config() if args.smoke else Config()
    apply_training_data_args(cfg, args)

    manifest = run_single_stage(
        cfg,
        out_dir=Path(args.out_dir),
        epochs=args.epochs,
        model_name=args.model_name,
    )
    print(
        json.dumps({k: v for k, v in manifest.items() if k != "history"}, indent=2, default=str)
    )
    return manifest


if __name__ == "__main__":
    main()
