"""Scale sweep over ``cfg.benchmark.scale_grid`` with HF LM-generated batches and a benchmark task."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from tqdm.auto import tqdm

import aitchinson_flow.models  # noqa: F401 — populate model REGISTRY

from aitchinson_flow.config import Config
from aitchinson_flow.models.factory import build_model
from aitchinson_flow.training.datamodule import DataModule
from aitchinson_flow.training.data_sources import build_training_datamodule
from aitchinson_flow.training.runner import fit
from benchmarks.results_schema import BenchmarkResultRow, FullBenchmarkResults
from benchmarks.tasks.registry import build_task
from aitchinson_flow.data.text8_datamodule import Text8DataModule


_ABLATION_KEYS = {
    "velocity_loss",
    "transform_mode",
    "label_smoothing",
    "model_name",
}


@dataclass(frozen=True)
class ComponentTaskSpec:
    """One component+task benchmark sweep specification."""

    component: str
    task_name: str
    data_source: str
    model_name: str | None = None


def apply_transformer_scale(
    cfg: Config,
    *,
    d_model: int | None = None,
    num_layers: int | None = None,
    nhead: int | None = None,
    d_latent: int | None = None,
    pretrained_backbone: str | None = None,
    velocity_loss: str | None = None,
    transform_mode: str | None = None,
    label_smoothing: float | None = None,
    model_name: str | None = None,
) -> Config:
    """Return a deep-copied config with scaled transformer + ablation overrides.

    Backbone overrides (``d_model`` / ``num_layers`` / ``nhead`` /
    ``pretrained_backbone``) work as before. The ablation overrides apply Plan
    point 5 axes:

    * ``velocity_loss``: switches Stage 1 between Hilbert family
      (``soft_hilbert``/``hard_hilbert``) and MSE family
      (``clr_mse``/``ilr_mse``).
    * ``transform_mode``: switches the discrete→simplex pipeline between
      ``"ilr"`` (default; output dim ``K-1``) and ``"clr"`` (ILR-off ablation;
      output dim ``K``).
    * ``label_smoothing``: explicit simplex-interior smoothing α.
    * ``model_name``: registered model key, e.g. ``"bayesian_auditor_stage1"``
      vs ``"bayesian_auditor"``, for sequence-level (Stage 1 energy) vs
      token-level (Stage 2 GP variance) UQ comparisons.

    When ``pretrained_backbone`` is set (e.g. ``"gpt2-medium"``), the custom
    transformer dims are left unchanged and only the backbone ID is updated.
    """
    out = deepcopy(cfg)
    if pretrained_backbone is not None:
        out.transformer = replace(out.transformer, pretrained_backbone=pretrained_backbone)
    elif d_model is not None or num_layers is not None or nhead is not None:
        assert d_model is not None and num_layers is not None and nhead is not None
        dl = d_latent if d_latent is not None else d_model
        out.transformer = replace(
            out.transformer,
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            d_latent=dl,
        )
        assert out.transformer.d_model % out.transformer.nhead == 0, (
            f"d_model={out.transformer.d_model} must be divisible by nhead={out.transformer.nhead}"
        )

    if velocity_loss is not None:
        out.training = replace(out.training, velocity_loss=velocity_loss)
    if model_name is not None:
        out.training = replace(out.training, model_name=model_name)
    if transform_mode is not None:
        out.hf_dataset = replace(out.hf_dataset, transform_mode=transform_mode)
    if label_smoothing is not None:
        out.hf_dataset = replace(out.hf_dataset, label_smoothing=float(label_smoothing))
    return out


def _scale_tag(scale: dict[str, Any]) -> str:
    """Generate a unique tag for the scale."""
    parts = [f"{k}{scale[k]}" for k in sorted(scale)]
    return "_".join(parts) if parts else "baseline"


def _count_params(model: Any) -> int:
    """Total trainable parameter count."""
    import torch.nn as nn  # noqa: PLC0415
    if isinstance(model, nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return 0


def _build_datamodule(cfg: Config) -> DataModule:
    """Build the datamodule selected by ``cfg.benchmark.data_source``."""
    source = cfg.benchmark.data_source
    if source == "trivia":
        from aitchinson_flow.data.trivia_datamodule import TriviaDataModule  # noqa: PLC0415
        return TriviaDataModule(cfg)
    return Text8DataModule(cfg)


def _strip_scores(result: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any] | None]:
    """Strip the scores from the result."""
    scores = result.pop("_scores", None)
    return result, scores


def _cfg_for_data_source(cfg: Config, source: str) -> Config:
    """Map benchmark source labels to training-data source config."""
    out = deepcopy(cfg)
    out.benchmark = replace(out.benchmark, data_source=source)

    td = out.training_data
    if source == "text8":
        out.training_data = replace(td, source="raw_text", raw_dataset="text8")
    elif source == "trivia":
        out.training_data = replace(td, source="qa_pairs")
    elif source in {"dna", "medical", "llm_topk_probs"}:
        out.training_data = replace(td, source=source)
    else:
        raise ValueError(
            f"Unsupported benchmark data_source={source!r}; expected one of "
            "'text8', 'trivia', 'dna', 'medical', or 'llm_topk_probs'."
        )
    return out


def _normalize_to_result_row(
    *,
    spec: ComponentTaskSpec,
    scale_tag: str,
    metrics: dict[str, Any],
) -> BenchmarkResultRow:
    """Normalize a task result dict into the Phase 3 typed row."""

    def _pick(keys: list[str]) -> float | None:
        for key in keys:
            val = metrics.get(key)
            if val is None:
                continue
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
        return None

    auroc_energy = _pick(["auroc_energy"])
    auroc_variance = _pick(
        [
            "auroc_variance",
            "auroc_auditor",
            "auroc_trivia",
            "auroc_hallucination",
            "auroc_dna_overall",
            "auroc_medical_overall",
            "post_auroc",
            "pre_auroc",
        ]
    )
    auroc_spilled = _pick(["auroc_spilled"])
    auroc_combined = _pick(["auroc_combined"])
    if auroc_combined is None:
        combo_parts = [v for v in (auroc_energy, auroc_variance) if v is not None]
        auroc_combined = sum(combo_parts) / len(combo_parts) if combo_parts else None

    energy_gap = _pick(["energy_gap", "energy_separation"])
    variance_ratio = _pick(["variance_ratio"])
    per_token = _pick(
        [
            "per_token_auroc_at_corruption",
            "auroc_auditor_token_corrupted",
            "auroc_auditor_token_all",
            "auroc_spilled_token_all",
        ]
    )

    if variance_ratio is None:
        v_valid = _pick(["var_correct_mean"])
        v_invalid = _pick(["var_incorrect_mean", "var_hallucinated_mean"])
        if v_valid is not None and v_invalid is not None and abs(v_valid) > 1e-12:
            variance_ratio = v_invalid / v_valid

    return BenchmarkResultRow(
        component=spec.component,
        task=spec.task_name,
        scale=scale_tag,
        auroc_energy=auroc_energy,
        auroc_variance=auroc_variance,
        auroc_combined=auroc_combined,
        auroc_spilled=auroc_spilled,
        energy_gap=energy_gap,
        variance_ratio=variance_ratio,
        per_token_auroc_at_corruption=per_token,
        metadata={k: v for k, v in metrics.items() if k != "_scores"},
    )


def run_benchmark(cfg: Config) -> dict[str, dict[str, float]]:
    """Build LM teacher data, then for each scale entry build model and run the benchmark task."""
    import benchmarks.tasks  # noqa: F401 — register tasks (text_audit)

    datamodule = _build_datamodule(cfg)
    task = build_task(cfg.benchmark.task_name)

    grid: list[dict[str, int]] = cfg.benchmark.scale_grid or [{}]
    results: dict[str, dict[str, float]] = {}

    out_root = Path(cfg.benchmark.results_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    scale_iter = tqdm(grid, desc="benchmark scales", disable=not cfg.benchmark.use_tqdm)
    for scale in scale_iter:
        if scale:
            ablation_kwargs = {k: scale[k] for k in _ABLATION_KEYS if k in scale}
            if "pretrained_backbone" in scale:
                scfg = apply_transformer_scale(
                    cfg,
                    pretrained_backbone=scale["pretrained_backbone"],
                    **ablation_kwargs,
                )
            elif "d_model" in scale or "num_layers" in scale or "nhead" in scale:
                scfg = apply_transformer_scale(
                    cfg,
                    d_model=scale["d_model"],
                    num_layers=scale["num_layers"],
                    nhead=scale["nhead"],
                    d_latent=scale.get("d_latent"),
                    **ablation_kwargs,
                )
            else:
                scfg = apply_transformer_scale(cfg, **ablation_kwargs)
        else:
            scfg = deepcopy(cfg)

        if cfg.benchmark.train_epochs is not None:
            scfg.training = replace(scfg.training, epochs=cfg.benchmark.train_epochs)

        model = build_model(scfg)
        history: list[dict[str, float]] = []
        tag = _scale_tag(scale) if scale else "baseline"
        if cfg.benchmark.train_before_eval:
            model = fit(
                scfg,
                datamodule,
                model=model,
                history_out=history,
                wandb_run_name=f"{scfg.training.model_name}-{tag}",
                wandb_group=scfg.training.wandb_group or "benchmark",
                wandb_job_type="benchmark_train",
                wandb_tags=["benchmark", tag],
                wandb_extra_config={
                    "benchmark_tag": tag,
                    "scale": dict(scale),
                    "task_name": cfg.benchmark.task_name,
                    "data_source": cfg.benchmark.data_source,
                },
            )

        raw = task.run(model, datamodule, scfg)
        clean, scores = _strip_scores(raw)
        clean["num_params"] = _count_params(model)
        results[tag] = clean

        if cfg.benchmark.save_plots and scores is not None:
            from aitchinson_flow.plots import save_benchmark_plots  # noqa: PLC0415

            save_benchmark_plots(
                out_root / tag,
                scores=scores,
                loss_history=history,
                stage=tag or "benchmark",
            )

        if cfg.benchmark.use_tqdm:
            scale_iter.set_postfix(scale=tag)

    return results


def run_unified_benchmark(
    cfg: Config,
    *,
    specs: list[ComponentTaskSpec],
) -> FullBenchmarkResults:
    """Run Phase 3 unified benchmark over component-task-scale tuples."""
    import benchmarks.tasks  # noqa: F401 — register tasks

    rows: list[BenchmarkResultRow] = []
    base_grid: list[dict[str, int]] = cfg.benchmark.scale_grid or [{}]

    for spec in specs:
        spec_base = _cfg_for_data_source(cfg, spec.data_source)
        datamodule, _source_meta = build_training_datamodule(spec_base)
        task = build_task(spec.task_name)
        scale_iter = tqdm(
            base_grid,
            desc=f"{spec.component}:{spec.task_name}",
            disable=not cfg.benchmark.use_tqdm,
        )
        for scale in scale_iter:
            if scale:
                ablation_kwargs = {k: scale[k] for k in _ABLATION_KEYS if k in scale}
                if "pretrained_backbone" in scale:
                    scfg = apply_transformer_scale(
                        spec_base,
                        pretrained_backbone=scale["pretrained_backbone"],
                        **ablation_kwargs,
                    )
                elif "d_model" in scale or "num_layers" in scale or "nhead" in scale:
                    scfg = apply_transformer_scale(
                        spec_base,
                        d_model=scale["d_model"],
                        num_layers=scale["num_layers"],
                        nhead=scale["nhead"],
                        d_latent=scale.get("d_latent"),
                        **ablation_kwargs,
                    )
                else:
                    scfg = apply_transformer_scale(spec_base, **ablation_kwargs)
            else:
                scfg = deepcopy(spec_base)

            if spec.model_name is not None:
                scfg.training = replace(scfg.training, model_name=spec.model_name)
            if scfg.benchmark.train_epochs is not None:
                scfg.training = replace(scfg.training, epochs=scfg.benchmark.train_epochs)

            model = build_model(scfg)
            tag = _scale_tag(scale) if scale else "baseline"
            history: list[dict[str, float]] = []
            if scfg.benchmark.train_before_eval:
                model = fit(
                    scfg,
                    datamodule,
                    model=model,
                    history_out=history,
                    wandb_run_name=f"{scfg.training.model_name}-{spec.component}-{tag}",
                    wandb_group=scfg.training.wandb_group or "benchmark_full",
                    wandb_job_type="benchmark_train",
                    wandb_tags=["benchmark", "phase3", spec.component, spec.task_name, tag],
                    wandb_extra_config={
                        "benchmark_component": spec.component,
                        "benchmark_task": spec.task_name,
                        "benchmark_scale": dict(scale),
                        "benchmark_data_source": spec.data_source,
                    },
                )
            metrics = task.run(model, datamodule, scfg)
            metrics["num_params"] = _count_params(model)
            if history:
                metrics["train_loss_final"] = float(history[-1].get("loss", float("nan")))
            rows.append(_normalize_to_result_row(spec=spec, scale_tag=tag, metrics=metrics))
            if scfg.benchmark.use_tqdm:
                scale_iter.set_postfix(scale=tag)

    snapshot = {
        "benchmark": dict(cfg.benchmark.__dict__),
        "training": {
            "model_name": cfg.training.model_name,
            "epochs": cfg.training.epochs,
            "device": cfg.training.device,
        },
        "specs": [spec.__dict__ for spec in specs],
    }
    return FullBenchmarkResults(rows=rows, config_snapshot=snapshot)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="LLM-generated text auditing benchmark (scale sweep)."
    )
    parser.add_argument(
        "--config-out",
        type=str,
        default="",
        help="Optional path to dump resolved default Config JSON (for debugging).",
    )
    args = parser.parse_args(argv)

    cfg = Config()
    if args.config_out:
        Path(args.config_out).write_text(
            json.dumps(
                {
                    "benchmark": {
                        k: v for k, v in cfg.benchmark.__dict__.items() if not callable(v)
                    },
                    "training_model": cfg.training.model_name,
                    "dataset_K": cfg.dataset.K,
                    "dataset_L": cfg.dataset.L,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    results = run_benchmark(cfg)
    out_dir = Path(cfg.benchmark.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmark_latest.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
