from __future__ import annotations

from pathlib import Path
from typing import Any

import torch.nn as nn
from tqdm.auto import tqdm

from aitchinson_flow.config import Config
from aitchinson_flow.models.base import TRAINING_LOSS_KEY
from aitchinson_flow.models.factory import build_model
from aitchinson_flow.training.checkpoint import save_checkpoint
from aitchinson_flow.training.datamodule import DataModule
from aitchinson_flow.training.loops import evaluate, train_epoch
from aitchinson_flow.training.optim import build_optimizer, build_scheduler
from aitchinson_flow.training.seed import seed_all
from aitchinson_flow.training.wandb_logger import WandbLogger


def fit(
    cfg: Config,
    datamodule: DataModule,
    *,
    model: nn.Module | None = None,
    resume_from: str | Path | None = None,
    history_out: list[dict[str, float]] | None = None,
    wandb_run_name: str | None = None,
    wandb_group: str | None = None,
    wandb_job_type: str | None = None,
    wandb_tags: list[str] | None = None,
    wandb_extra_config: dict[str, Any] | None = None,
) -> nn.Module:
    """Train ``model`` (built from ``cfg`` if not supplied) and return it.

    If ``history_out`` is provided, per-epoch aggregated train metrics are appended to it.
    """
    seed_all(cfg.training.seed)

    if model is None:
        model = build_model(cfg)
    model = model.to(cfg.training.device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    num_train_fn = getattr(datamodule, "num_train_samples", None)
    n_train: int | None = None
    if callable(num_train_fn):
        candidate = num_train_fn()
        if isinstance(candidate, int):
            n_train = candidate
    if n_train is None:
        ds = getattr(train_loader, "dataset", None)
        if ds is not None:
            try:
                n_train = len(ds)
            except TypeError:
                n_train = None
    if n_train is not None and n_train > 0:
        model._kl_normalizer = float(n_train)  # type: ignore[attr-defined]

    start_epoch = 0
    global_step = 0

    if resume_from is not None:
        from aitchinson_flow.training.checkpoint import load_checkpoint

        _, start_epoch, global_step = load_checkpoint(
            resume_from,
            model=model,
            optimizer=optimizer,
            map_location=cfg.training.device,
        )
        start_epoch += 1

    ckpt_dir = Path(cfg.training.checkpoint_dir)
    step_log_every = max(1, cfg.training.log_every)
    logger = WandbLogger(
        cfg,
        run_name=wandb_run_name,
        group=wandb_group,
        job_type=wandb_job_type,
        tags=wandb_tags,
        extra_config=wandb_extra_config,
    )
    logger.watch_model(model)

    epoch_range = range(start_epoch, cfg.training.epochs)
    epoch_pbar = tqdm(
        epoch_range,
        desc="epochs",
        disable=not cfg.training.use_tqdm,
        leave=True,
        unit="epoch",
    )
    prev_train_loss: float | None = None

    try:
        for epoch in epoch_pbar:
            def _log_train_step(step_idx: int, values: dict[str, float]) -> None:
                if not logger.active or (step_idx + 1) % step_log_every != 0:
                    return
                payload = {f"step/train/{k}": float(v) for k, v in values.items()}
                payload["epoch"] = float(epoch + 1)
                logger.log_metrics(payload, step=step_idx + 1)

            step_callback = _log_train_step if cfg.training.wandb_log_steps else None
            metrics, global_step = train_epoch(
                model,
                train_loader,
                optimizer,
                device=cfg.training.device,
                epoch=epoch,
                global_step=global_step,
                grad_clip_norm=cfg.training.grad_clip_norm,
                use_tqdm=cfg.training.use_tqdm,
                step_callback=step_callback,
            )

            epoch_entry: dict[str, float] = {k: float(v) for k, v in metrics.items()}
            wandb_epoch_metrics = {f"train/{k}": float(v) for k, v in metrics.items()}

            train_loss = metrics.get(TRAINING_LOSS_KEY)
            postfix: dict[str, str] = {}
            if train_loss is not None:
                postfix["train"] = f"{train_loss:.4f}"
                if prev_train_loss is not None:
                    postfix["Δtrain"] = f"{train_loss - prev_train_loss:+.4f}"
                prev_train_loss = train_loss

            for k in sorted(metrics):
                if k == TRAINING_LOSS_KEY:
                    continue
                postfix[k] = f"{metrics[k]:.4f}"

            if optimizer.param_groups:
                lr_value = float(optimizer.param_groups[0]["lr"])
                postfix["lr"] = f"{lr_value:.2e}"
                wandb_epoch_metrics["lr"] = lr_value

            if val_loader is not None and (epoch + 1) % cfg.training.eval_every == 0:
                val_metrics = evaluate(
                    model,
                    val_loader,
                    device=cfg.training.device,
                    use_tqdm=cfg.training.use_tqdm,
                )
                vl = val_metrics.get(TRAINING_LOSS_KEY)
                if vl is not None:
                    postfix["val"] = f"{vl:.4f}"
                for k in sorted(val_metrics):
                    if k == TRAINING_LOSS_KEY:
                        continue
                    postfix[f"v_{k}"] = f"{val_metrics[k]:.4f}"
                for k, v in val_metrics.items():
                    epoch_entry[f"val_{k}"] = float(v)
                    wandb_epoch_metrics[f"val/{k}"] = float(v)

            if history_out is not None:
                history_out.append(epoch_entry)

            if cfg.training.use_tqdm and postfix:
                epoch_pbar.set_postfix(postfix, refresh=True)

            wandb_epoch_metrics["epoch"] = float(epoch + 1)
            wandb_epoch_metrics["global_step"] = float(global_step)
            logger.log_metrics(wandb_epoch_metrics, step=global_step)

            if scheduler is not None:
                scheduler.step()

            if (epoch + 1) % cfg.training.checkpoint_every == 0:
                ckpt_path = ckpt_dir / f"epoch_{epoch + 1}.pt"
                save_checkpoint(
                    ckpt_path,
                    model=model,
                    cfg=cfg,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    global_step=global_step,
                )
                if cfg.training.wandb_log_model:
                    logger.log_artifact(
                        ckpt_path,
                        name=f"{cfg.training.model_name}-checkpoint",
                        artifact_type="model",
                        aliases=["latest", f"epoch-{epoch + 1}"],
                        metadata={
                            "epoch": epoch + 1,
                            "global_step": global_step,
                            "model_name": cfg.training.model_name,
                        },
                    )
    finally:
        logger.finish()

    return model
