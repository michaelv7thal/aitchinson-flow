from __future__ import annotations

from pathlib import Path

from _shared.bootstrap import bootstrap_repo_paths

bootstrap_repo_paths(Path(__file__))

from copy import deepcopy  # noqa: E402
from dataclasses import replace  # noqa: E402

from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.data.teachers.causal_lm import CausalLMTeacher, CausalLMTeacherDataModule  # noqa: E402
from aitchinson_flow.llms.registry import build_lm  # noqa: E402
from aitchinson_flow.training.runner import fit  # noqa: E402


def make_config_for_scale(
    base: Config,
    *,
    d_model: int,
    num_layers: int,
    nhead: int,
    num_inducing: int,
    run_tag: str,
    d_latent: int | None = None,
) -> Config:
    cfg = deepcopy(base)
    d_latent = d_latent if d_latent is not None else d_model

    cfg.transformer = replace(
        cfg.transformer,
        d_model=d_model,
        num_layers=num_layers,
        nhead=nhead,
        d_latent=d_latent,
    )
    # Latent dim lives on transformer; SparseGP reads cfg.transformer.d_latent.
    cfg.gp = replace(cfg.gp, num_inducing=num_inducing)

    assert cfg.transformer.d_model % cfg.transformer.nhead == 0

    cfg.training.checkpoint_dir = f"checkpoints/scale/{run_tag}"
    return cfg


def run_scale_sweep() -> None:
    base = Config()
    base.training.model_name = "bayesian_generator"
    base.training.seed = 123  # same across sizes for comparability
    grid: list[tuple[int, int, int]] = [
        # (d_model, num_layers, nhead)
        (128, 4, 8),
        (256, 6, 8),
        (512, 8, 8),
    ]
    for d_model, num_layers, nhead in grid:
        tag = f"d{d_model}_L{num_layers}_h{nhead}"
        cfg = make_config_for_scale(
            base,
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            num_inducing=base.gp.num_inducing,
            run_tag=tag,
        )
        lm = build_lm(cfg.benchmark.lm_key, cfg.teacher)
        teacher = CausalLMTeacher(lm, cfg)
        dm = CausalLMTeacherDataModule(teacher, cfg)
        fit(cfg, dm)


if __name__ == "__main__":
    run_scale_sweep()
