import argparse

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import tqdm

from copy import deepcopy
from pathlib import Path
from typing import Any, cast, Literal
from dataclasses import replace

from _shared.cli import positive_int, add_training_data_args, apply_training_data_args
from _shared.bootstrap import bootstrap_repo_paths

# Allow direct script execution without editable install.
_REPO_ROOT = bootstrap_repo_paths(Path(__file__))

from aitchinson_flow.training.data_sources import build_training_datamodule
from aitchinson_flow.config import Config
from aitchinson_flow.training.datamodule import DataModule

import aitchinson_flow.models  # noqa: E402,F401  — populate model REGISTRY

from aitchinson_flow.models import build_model
from aitchinson_flow.models.base import TRAINING_LOSS_KEY, GenerativeTrainingModel, LossDict
from aitchinson_flow.training import fit
from aitchinson_flow.training.loops import evaluate, train_epoch
from aitchinson_flow.training.optim import build_optimizer, build_scheduler
from aitchinson_flow.training.seed import seed_all
from aitchinson_flow.training.batch import to_device
from aitchinson_flow.geometry import ilr_inv
from aitchinson_flow.training import save_checkpoint


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=str, default="checkpoints/manual/baseline")
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


def main(argv: list[str] | None = None):
    args = _build_argparser().parse_args(argv)

    cfg = Config()
    apply_training_data_args(cfg, args)

    datamodule, data_source_meta = _build_default_datamodule(cfg=cfg)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s1_cfg = _stage1_config(cfg, epochs=args.stage1_epochs, ckpt_dir=out_dir / "stage1_ckpts")

    s1_history: list[dict[str, float]] = []
    s1_model: nn.Module | None = None
    s1_cfg: Config
    s1_path: Path | None
    stage1_epochs_effective: int = args.stage1_epochs

    model = build_model(s1_cfg)

    training_data = datamodule.train_dataloader()
    num_train_fn = getattr(datamodule, "num_train_samples", None)
    n_train: int | None = None
    if callable(num_train_fn):
        candidate = num_train_fn()
        if isinstance(candidate, int):
            n_train = candidate

    epoch_range = range(0, cfg.training.epochs)
    epoch_pbar = tqdm(
        epoch_range,
        desc="epochs",
        disable=not cfg.training.use_tqdm,
        leave=True,
        unit="epoch",
    )
    prev_train_loss: float | None = None

    seed_all(cfg.training.seed)

    if model is None:
        model = build_model(cfg)
    model = model.to(cfg.training.device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    global_step = 0

    device = torch.device(cfg.training.device)

    for epoch in epoch_pbar:
        m = cast(GenerativeTrainingModel, model)
        model.train()

        agg: dict[str, float] = {}
        counts: dict[str, int] = {}
        step = global_step

        pbar = tqdm(
            training_data,
            desc=f"train epoch {epoch + 1}",
            disable=False,
            leave=False,
        )

        avg_loss = []

        for batch in pbar:
            batch = to_device(batch, cfg.training.device)
            optimizer.zero_grad(set_to_none=True)

            batch = model.prepare_batch(batch)

            log_x1 = batch["log_x"]
            B, L, D = batch["log_x"].shape
            dt = log_x1.dtype
            log_x0 = torch.randn((B, L, D), device=device, dtype=dt)
            gamma = torch.rand(B, device=device, dtype=dt)
            log_x_gamma = (1.0 - gamma[:, None, None]) * log_x0 + gamma[:, None, None] * log_x1
            log_x_gamma.requires_grad_(True)

            with sdpa_kernel(SDPBackend.MATH):
                v_pred, _ = model.forward(log_x_gamma)

                # --- Dot product: g = xγ · f(xγ) ---
                g = (log_x_gamma * v_pred).sum()
                grad_g = torch.autograd.grad(g, log_x_gamma, create_graph=True)[0]

            # --- Squared L2 norm: g = -0.5 * ||f(xγ)||² ---
            # g = -0.5 * (v_pred ** 2).sum()
            # grad_g = torch.autograd.grad(g, log_x_gamma, create_graph=True)[0]

            # Then loss is MSE between grad_g and target
            u_tgt = model._c_gamma(gamma) * (log_x0 - log_x1)

            # v_for_loss = ilr_inv(v_pred, cfg.dataset.K)
            # u_for_loss = ilr_inv(u_tgt, cfg.dataset.K)

            # v_probs = v_for_loss.exp()
            # u_probs = u_for_loss.exp()

            flow_loss = model._velocity_loss_fn(grad_g, u_tgt)

            # flow_loss = model._velocity_loss_fn(v_pred, u_tgt)

            avg_loss.append(flow_loss)

            flow_loss.backward()

            optimizer.step()

            with torch.no_grad():
                per_sample = ((grad_g - u_tgt) ** 2).mean(dim=(-2, -1))  # (B,)
                bins = [(0.0, 0.33, "low_γ"), (0.33, 0.66, "mid_γ"), (0.66, 1.0, "high_γ")]
                bin_stats: dict[str, str] = {}
                for lo, hi, label in bins:
                    mask = (gamma > lo) & (gamma <= hi)
                    if mask.any():
                        bin_stats[label] = f"{per_sample[mask].mean().item():.4f}"
                    else:
                        bin_stats[label] = "-"
                pbar.set_postfix({"loss": f"{flow_loss.item():.4f}", **bin_stats})

        print(f"epoch loss: {sum(avg_loss) / len(avg_loss)}")
        avg_loss = []

        scheduler.step()
        print(f"new learning rate: {scheduler.get_last_lr()}")

    s1_path = out_dir / "stage1.pt"
    save_checkpoint(
        s1_path,
        model=model,
        cfg=s1_cfg,
        optimizer=None,
        epoch=cfg.training.epochs,
        global_step=0,
    )


if __name__ == "__main__":
    main()
