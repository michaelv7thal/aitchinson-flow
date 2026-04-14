from __future__ import annotations

from pathlib import Path

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


def fit(
    cfg: Config,
    datamodule: DataModule,
    *,
    model: nn.Module | None = None,
    resume_from: str | Path | None = None,
) -> nn.Module:
    """Train ``model`` (built from ``cfg`` if not supplied) and return it."""
    seed_all(cfg.training.seed)

    if model is None:
        model = build_model(cfg)
    model = model.to(cfg.training.device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

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

    epoch_range = range(start_epoch, cfg.training.epochs)
    epoch_pbar = tqdm(
        epoch_range,
        desc="epochs",
        disable=not cfg.training.use_tqdm,
        leave=True,
        unit="epoch",
    )
    prev_train_loss: float | None = None

    for epoch in epoch_pbar:
        metrics, global_step = train_epoch(
            model,
            train_loader,
            optimizer,
            device=cfg.training.device,
            epoch=epoch,
            global_step=global_step,
            grad_clip_norm=cfg.training.grad_clip_norm,
            use_tqdm=cfg.training.use_tqdm,
        )

        train_loss = metrics.get(TRAINING_LOSS_KEY)
        postfix: dict[str, str] = {}
        if train_loss is not None:
            postfix["train"] = f"{train_loss:.4f}"
            if prev_train_loss is not None:
                postfix["Δtrain"] = f"{train_loss - prev_train_loss:+.4f}"
            prev_train_loss = train_loss

        if optimizer.param_groups:
            postfix["lr"] = f"{optimizer.param_groups[0]['lr']:.2e}"

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

        if cfg.training.use_tqdm and postfix:
            epoch_pbar.set_postfix(postfix, refresh=True)

        if scheduler is not None:
            scheduler.step()

        if (epoch + 1) % cfg.training.checkpoint_every == 0:
            save_checkpoint(
                ckpt_dir / f"epoch_{epoch + 1}.pt",
                model=model,
                cfg=cfg,
                optimizer=optimizer,
                epoch=epoch + 1,
                global_step=global_step,
            )

    return model
