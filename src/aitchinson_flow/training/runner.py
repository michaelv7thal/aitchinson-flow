from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from aitchinson_flow.config import Config
from aitchinson_flow.training import (
    DataModule,
    seed_all,
    build_optimizer,
    build_scheduler,
    load_checkpoint,
    train_epoch,
    evaluate,
    save_checkpoint,
)
from aitchinson_flow.models import build_model, TRAINING_LOSS_KEY
from aitchinson_flow.losses import anneal_alpha


def _unigram_kl_probe(
    model: nn.Module, datamodule: DataModule, cfg: Config
) -> dict[str, float]:
    """Sample sequences and report unigram KL against the training corpus.

    Catches mode-collapse (e.g. all-space) which the training MSE / γ-bucket
    losses don't expose. Returns 'unigram_kl', 'H_gen', 'H_gt' (nats).
    """
    if not hasattr(datamodule, "splits"):
        return {}
    sample = getattr(model, "sample", None)
    if sample is None:
        return {}
    K = cfg.text8_dataset.K
    L = cfg.text8_dataset.L
    n = cfg.training.sample_eval_n
    steps = cfg.training.sample_eval_steps

    was_training = model.training
    model.eval()
    try:
        x = sample(n, L, max_steps=steps)
        log_probs = model.decode_to_logprobs(x)
        ids = log_probs.argmax(-1).cpu().reshape(-1)
    finally:
        if was_training:
            model.train()

    gen = torch.zeros(K).scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.float))
    gen = (gen + 1e-9) / (gen.sum() + K * 1e-9)

    train_ids = datamodule.splits.train.reshape(-1).long()
    gt = torch.zeros(K).scatter_add_(
        0, train_ids, torch.ones_like(train_ids, dtype=torch.float)
    )
    gt = (gt + 1e-9) / (gt.sum() + K * 1e-9)

    return {
        "unigram_kl": float((gen * (gen.log() - gt.log())).sum()),
        "H_gen": float(-(gen * gen.log()).sum()),
        "H_gt": float(-(gt * gt.log()).sum()),
    }


def fit(
    cfg: Config,
    datamodule: DataModule,
    *,
    model: nn.Module | None = None,
    resume_from: str | Path | None = None,
    history_out: list[dict[str, float]] | None = None,
) -> nn.Module:

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
        leave=True,
        unit="epoch",
    )
    prev_train_loss: float | None = None

    try:
        for epoch in epoch_pbar:
            current_alpha = anneal_alpha(model, cfg, epoch)

            metrics, global_step = train_epoch(
                model,
                train_loader,
                optimizer,
                device=cfg.training.device,
                epoch=epoch,
                global_step=global_step,
                grad_clip_norm=cfg.training.grad_clip_norm,
            )

            epoch_entry: dict[str, float] = {k: float(v) for k, v in metrics.items()}

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

            if current_alpha is not None:
                postfix["α"] = f"{current_alpha:.4f}"

            if optimizer.param_groups:
                lr_value = float(optimizer.param_groups[0]["lr"])
                postfix["lr"] = f"{lr_value:.2e}"

            if val_loader is not None and (epoch + 1) % cfg.training.eval_every == 0:
                val_metrics = evaluate(
                    model,
                    val_loader,
                    device=cfg.training.device,
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

            sev = cfg.training.sample_eval_every
            if sev is not None and sev > 0 and (epoch + 1) % sev == 0:
                probe = _unigram_kl_probe(model, datamodule, cfg)
                for k, v in probe.items():
                    epoch_entry[k] = v
                    postfix[k] = f"{v:.4f}"

            if history_out is not None:
                history_out.append(epoch_entry)

            if postfix:
                epoch_pbar.set_postfix(postfix, refresh=True)

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

    except Exception as e:
        raise e

    save_checkpoint(
        ckpt_dir / "epoch_final.pt",
        model=model,
        cfg=cfg,
        optimizer=None,  # final ckpt is for inference only — strip optimizer state
        epoch=len(epoch_pbar),
        global_step=global_step,
    )

    return model
