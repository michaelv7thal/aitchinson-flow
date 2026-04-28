"""Train stage-1 models at several transformer sizes and benchmark each one.

For every entry in the scale grid this script:
  1. Trains a stage-1 model (``bayesian_auditor_stage1``) via :func:`run_single_stage`.
  2. Evaluates it with the unified benchmark runner on the configured task.
  3. Collects per-scale metrics (AUROC energy/variance/combined, param count,
     final training loss) into a consolidated JSON results file.

Usage examples::

    # Default run (3 sizes, llm_topk_probs data, text_audit task)
    python scripts/stage1_scale_benchmark.py

    # Smoke test (tiny CPU config, 1 epoch)
    python scripts/stage1_scale_benchmark.py --smoke

    # Custom grid and output location
    python scripts/stage1_scale_benchmark.py \\
        --out-dir results/scale_benchmark_v2 \\
        --epochs 30 \\
        --training-data-source llm_topk_probs \\
        --training-lm-key hf_causal

Results are written to ``<out-dir>/results.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

from _shared.bootstrap import bootstrap_repo_paths

bootstrap_repo_paths(Path(__file__))

import aitchinson_flow.models  # noqa: E402,F401 — populate model REGISTRY

from _shared.cli import add_training_data_args, apply_training_data_args, positive_int  # noqa: E402
from _shared.smoke import make_smoke_config  # noqa: E402
from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.models.factory import build_model  # noqa: E402
from aitchinson_flow.training.data_sources import build_training_datamodule  # noqa: E402
from aitchinson_flow.training.runner import fit  # noqa: E402
from benchmarks.runner import ComponentTaskSpec, run_unified_benchmark  # noqa: E402

# ---------------------------------------------------------------------------
# Default scale grid: (d_model, num_layers, nhead)
# nhead must divide d_model evenly.
# ---------------------------------------------------------------------------
DEFAULT_SCALE_GRID: list[tuple[int, int, int]] = [
    (128, 4, 4),
    (256, 6, 8),
    (512, 8, 8),
    (768, 10, 8),
    (1024, 12, 16),
    (1536, 16, 16),
    (2048, 24, 16),
]

# Batch sizes chosen so the largest models fit inside ~20 GB VRAM.
# Values are conservative; increase if your GPU has headroom.
_BATCH_SIZE_FOR_D_MODEL: dict[int, int] = {
    128: 84,
    256: 64,
    512: 32,
    768: 24,
    1024: 16,
    1536: 8,
    2048: 4,
}

MODEL_NAME = "bayesian_auditor_stage1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_params(cfg: Config) -> int:
    """Instantiate the model for *cfg* and return its trainable param count."""
    import torch  # noqa: PLC0415

    device = torch.device("cpu")
    cfg_cpu = deepcopy(cfg)
    cfg_cpu.training = replace(cfg_cpu.training, device=device)
    model = build_model(cfg_cpu)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _scale_tag(d_model: int, num_layers: int, nhead: int) -> str:
    return f"d{d_model}_L{num_layers}_h{nhead}"


def _make_scale_config(
    base: Config,
    *,
    d_model: int,
    num_layers: int,
    nhead: int,
    out_dir: Path,
    tag: str,
    batch_size: int | None = None,
) -> Config:
    """Return a deep copy of *base* with the given transformer dimensions.

    *batch_size* overrides auto-scaling; pass ``None`` to select a safe
    default from ``_BATCH_SIZE_FOR_D_MODEL`` (falling back to dividing the
    base batch by (d_model / 128) ** 2).
    """
    cfg = deepcopy(base)
    cfg.transformer = replace(
        cfg.transformer,
        d_model=d_model,
        num_layers=num_layers,
        nhead=nhead,
        d_latent=d_model,  # keep d_latent == d_model unless overridden
    )
    assert cfg.transformer.d_model % cfg.transformer.nhead == 0, (
        f"d_model={d_model} must be divisible by nhead={nhead}"
    )
    if batch_size is not None:
        effective_bs = batch_size
    elif d_model in _BATCH_SIZE_FOR_D_MODEL:
        effective_bs = _BATCH_SIZE_FOR_D_MODEL[d_model]
    else:
        # Approximate: memory scales ~quadratically with d_model.
        scale_factor = (d_model / 128) ** 2
        effective_bs = max(1, int(base.training.B / scale_factor))
    cfg.training = replace(cfg.training, model_name=MODEL_NAME, B=effective_bs)
    return cfg


# ---------------------------------------------------------------------------
# Per-scale train + eval
# ---------------------------------------------------------------------------


def _train_scale(
    cfg: Config,
    *,
    scale_out: Path,
    epochs: int,
    tag: str,
) -> dict[str, Any]:
    """Train one scale, checkpoint under *scale_out*, and return a manifest."""
    scale_out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = scale_out / "ckpts"

    cfg_run = deepcopy(cfg)
    cfg_run.training = replace(
        cfg_run.training,
        model_name=MODEL_NAME,
        epochs=epochs,
        checkpoint_dir=str(ckpt_dir),
    )

    datamodule, data_source_meta = build_training_datamodule(cfg_run)
    history: list[dict[str, float]] = []
    fit(
        cfg_run,
        datamodule,
        history_out=history,
        wandb_run_name=f"stage1_scale_{tag}",
        wandb_job_type="stage1_scale_benchmark",
        wandb_tags=["stage1", "scale-sweep", tag],
        wandb_extra_config={
            "tag": tag,
            "d_model": cfg_run.transformer.d_model,
            "num_layers": cfg_run.transformer.num_layers,
            "nhead": cfg_run.transformer.nhead,
            "epochs": epochs,
        },
    )

    manifest: dict[str, Any] = {
        "model_name": MODEL_NAME,
        "epochs": epochs,
        "scale_out": str(scale_out),
        "ckpt_dir": str(ckpt_dir),
        "training_data_source": data_source_meta,
        "history": history,
    }
    (scale_out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    return manifest


def _eval_scale(
    cfg: Config,
    *,
    task_name: str,
    data_source: str,
) -> dict[str, float | None]:
    """Run the benchmark on a trained cfg and return the key metrics."""
    spec = ComponentTaskSpec(
        component="scale_sweep",
        task_name=task_name,
        data_source=data_source,
        model_name=MODEL_NAME,
    )
    # Disable training inside the benchmark (model already trained).
    cfg_eval = deepcopy(cfg)
    cfg_eval.benchmark = replace(cfg_eval.benchmark, train_before_eval=False)
    results = run_unified_benchmark(cfg_eval, specs=[spec])
    if not results.rows:
        return {}
    row = results.rows[0]
    return {
        "auroc_energy": row.auroc_energy,
        "auroc_variance": row.auroc_variance,
        "auroc_combined": row.auroc_combined,
        "auroc_spilled": row.auroc_spilled,
        "energy_gap": row.energy_gap,
        "variance_ratio": row.variance_ratio,
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------


def run_stage1_scale_benchmark(
    cfg: Config,
    *,
    out_dir: Path,
    epochs: int,
    scale_grid: list[tuple[int, int, int]],
    task_name: str,
    data_source: str,
    batch_size: int | None = None,
) -> list[dict[str, Any]]:
    """Train stage-1 at each scale, evaluate, and return collected rows."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for d_model, num_layers, nhead in scale_grid:
        tag = _scale_tag(d_model, num_layers, nhead)
        print(f"\n{'=' * 60}")
        print(f"  Scale: {tag}  (d_model={d_model}, num_layers={num_layers}, nhead={nhead})")
        print(f"{'=' * 60}\n")

        scale_cfg = _make_scale_config(
            cfg,
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            out_dir=out_dir,
            tag=tag,
            batch_size=batch_size,
        )

        n_params = _count_params(scale_cfg)
        print(f"  Trainable parameters: {n_params:,}")
        print(f"  Batch size:           {scale_cfg.training.B}")

        t0 = time.monotonic()
        scale_out = out_dir / tag
        manifest = _train_scale(scale_cfg, scale_out=scale_out, epochs=epochs, tag=tag)
        train_seconds = time.monotonic() - t0

        # Extract final training loss from history (last epoch entry).
        history: list[dict[str, float]] = manifest.get("history", [])
        final_loss = history[-1].get("loss") if history else None

        print(f"\n  Evaluating benchmark: task={task_name}, data_source={data_source}")
        # Point the benchmark config at the trained checkpoint dir.
        scale_cfg.training = replace(
            scale_cfg.training,
            checkpoint_dir=str(scale_out / "ckpts"),
        )
        benchmark_metrics = _eval_scale(scale_cfg, task_name=task_name, data_source=data_source)

        row: dict[str, Any] = {
            "tag": tag,
            "d_model": d_model,
            "num_layers": num_layers,
            "nhead": nhead,
            "n_params": n_params,
            "batch_size": scale_cfg.training.B,
            "epochs": epochs,
            "train_seconds": round(train_seconds, 1),
            "final_loss": final_loss,
            **benchmark_metrics,
        }
        rows.append(row)

        # Write incremental results so a crashed run doesn't lose prior scales.
        (out_dir / "results.json").write_text(
            json.dumps(rows, indent=2, default=str), encoding="utf-8"
        )
        print(f"  Results so far written to {out_dir / 'results.json'}")

    return rows


def _print_summary(rows: list[dict[str, Any]]) -> None:
    """Print a compact summary table to stdout."""
    cols = [
        "tag",
        "n_params",
        "batch_size",
        "final_loss",
        "auroc_energy",
        "auroc_variance",
        "auroc_combined",
    ]
    header = "  ".join(f"{c:<24}" for c in cols)
    print("\n" + "=" * len(header))
    print("SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for row in rows:
        line = "  ".join(f"{str(row.get(c, 'N/A')):<24}" for c in cols)
        print(line)
    print("=" * len(header))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="results/stage1_scale_benchmark",
        help="Output directory for checkpoints and results.json.",
    )
    p.add_argument(
        "--epochs",
        type=positive_int,
        default=25,
        help="Training epochs per scale (default: 25).",
    )
    p.add_argument(
        "--task",
        type=str,
        default="text_audit",
        help="Benchmark task name (default: text_audit).",
    )
    p.add_argument(
        "--data-source",
        type=str,
        default="text8",
        help="Benchmark eval data source (default: text8).",
    )
    p.add_argument(
        "--scales",
        type=str,
        default=None,
        metavar="d_model:num_layers:nhead,...",
        help=(
            "Comma-separated scale specs, e.g. '128:4:4,256:6:8,512:8:8'. "
            "Overrides the default grid."
        ),
    )
    p.add_argument(
        "--batch-size",
        type=positive_int,
        default=None,
        help=(
            "Override the per-scale batch size. By default the script uses "
            "_BATCH_SIZE_FOR_D_MODEL to pick a VRAM-safe value for each scale."
        ),
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny CPU config for end-to-end smoke testing (1 epoch, small model).",
    )
    add_training_data_args(p)
    return p


def _parse_scales(raw: str) -> list[tuple[int, int, int]]:
    grid = []
    for spec in raw.split(","):
        parts = spec.strip().split(":")
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                f"Each scale must be d_model:num_layers:nhead, got {spec!r}"
            )
        d_model, num_layers, nhead = int(parts[0]), int(parts[1]), int(parts[2])
        grid.append((d_model, num_layers, nhead))
    return grid


def main(argv: list[str] | None = None) -> None:
    args = _build_argparser().parse_args(argv)

    if args.smoke:
        cfg = make_smoke_config()
        cfg.training = replace(cfg.training, model_name=MODEL_NAME)
        # Smoke: single tiny scale, 1 epoch
        scale_grid: list[tuple[int, int, int]] = [(16, 1, 2)]
        epochs = 1
    else:
        cfg = Config()
        cfg.training = replace(cfg.training, model_name=MODEL_NAME)
        apply_training_data_args(cfg, args)
        scale_grid = _parse_scales(args.scales) if args.scales else DEFAULT_SCALE_GRID
        epochs = args.epochs

    out_dir = Path(args.out_dir)
    rows = run_stage1_scale_benchmark(
        cfg,
        out_dir=out_dir,
        epochs=epochs,
        scale_grid=scale_grid,
        task_name=args.task,
        data_source=args.data_source,
        batch_size=getattr(args, "batch_size", None),
    )

    _print_summary(rows)
    print(f"\nFull results: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
