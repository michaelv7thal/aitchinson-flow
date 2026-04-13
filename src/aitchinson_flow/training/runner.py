from __future__ import annotations

from pathlib import Path

from aitchinson_flow.config import Config
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
    resume_from: str | Path | None = None,
) -> None:
    seed_all(cfg.training.seed)

    model = build_model(cfg).to(cfg.training.device)
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

    for epoch in range(start_epoch, cfg.training.epochs):
        metrics, global_step = train_epoch(
            model,
            train_loader,
            optimizer,
            device=cfg.training.device,
            epoch=epoch,
            global_step=global_step,
            grad_clip_norm=cfg.training.grad_clip_norm,
        )

        # log metrics (pring / wandb / etc.)

        if val_loader is not None and (epoch + 1) % cfg.training.eval_every == 0:
            evaluate(model, val_loader, device=cfg.training.device)

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
