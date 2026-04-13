"""Scale sweep over ``cfg.benchmark.scale_grid`` with HF LM-generated batches and a benchmark task."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import aitchinson_flow.models  # noqa: F401 — populate model REGISTRY

from aitchinson_flow.config import Config
from aitchinson_flow.data.teachers.causal_lm import CausalLMTeacher, CausalLMTeacherDataModule
from aitchinson_flow.llms.registry import build_lm
from aitchinson_flow.models.factory import build_model
from benchmarks.tasks.registry import build_task


def apply_transformer_scale(
    cfg: Config,
    *,
    d_model: int,
    num_layers: int,
    nhead: int,
    d_latent: int | None = None,
) -> Config:
    """Return a deep-copied config with scaled transformer (and matching latent dim)."""
    out = deepcopy(cfg)
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
    return out


def _scale_tag(scale: dict[str, int]) -> str:
    parts = [f"{k}{scale[k]}" for k in sorted(scale)]
    return "_".join(parts) if parts else "baseline"


def run_benchmark(cfg: Config) -> dict[str, dict[str, float]]:
    """Build LM teacher data, then for each scale entry build model and run the benchmark task."""
    import benchmarks.tasks  # noqa: F401 — register tasks (text_audit)

    lm = build_lm(cfg.benchmark.lm_key, cfg.teacher)
    teacher = CausalLMTeacher(lm, cfg)

    prompt_ids = None
    prompt_mask = None
    if cfg.benchmark.use_text8_prompts:
        from benchmarks.text8_prompts import load_text8_prompts  # noqa: PLC0415

        total_prompts = cfg.benchmark.n_batches * cfg.training.B
        prompt_ids, prompt_mask = load_text8_prompts(
            lm,
            n_prompts=total_prompts,
            prompt_length=cfg.benchmark.text8_prompt_length,
        )

    datamodule = CausalLMTeacherDataModule(
        teacher,
        cfg,
        emit_logits=cfg.benchmark.compute_spilled_energy,
        prompt_ids=prompt_ids,
        prompt_attention_mask=prompt_mask,
    )
    task = build_task(cfg.benchmark.task_name)

    grid: list[dict[str, int]] = cfg.benchmark.scale_grid or [{}]
    results: dict[str, dict[str, float]] = {}

    for scale in grid:
        if scale:
            scfg = apply_transformer_scale(
                cfg,
                d_model=scale["d_model"],
                num_layers=scale["num_layers"],
                nhead=scale["nhead"],
                d_latent=scale.get("d_latent"),
            )
        else:
            scfg = deepcopy(cfg)

        model = build_model(scfg)
        tag = _scale_tag(scale) if scale else "baseline"
        results[tag] = task.run(model, datamodule, scfg)

    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="LLM-generated text auditing benchmark (scale sweep).")
    parser.add_argument(
        "--config-out",
        type=str,
        default="",
        help="Optional path to dump resolved default Config JSON (for debugging).",
    )
    args = parser.parse_args(argv)

    cfg = Config()
    if args.config_out:
        # lightweight repr: not full JSON of torch.device; skip or str(device)
        Path(args.config_out).write_text(
            json.dumps(
                {
                    "benchmark": cfg.benchmark.__dict__,
                    "training_model": cfg.training.model_name,
                    "dataset_K": cfg.dataset.K,
                    "dataset_L": cfg.dataset.L,
                },
                indent=2,
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
