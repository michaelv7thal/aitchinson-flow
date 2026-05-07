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


def _ngram_counts_flat(ids2d: torch.Tensor, K: int, n: int) -> torch.Tensor:
    """Return flat (K**n,) counts of contiguous n-grams across rows."""
    L = ids2d.shape[1]
    if L < n:
        return torch.zeros(K**n)
    idx = torch.zeros(ids2d.shape[0], L - n + 1, dtype=torch.long)
    for i in range(n):
        idx = idx + ids2d[:, i : L - n + 1 + i].long() * (K ** (n - 1 - i))
    counts = torch.zeros(K**n)
    counts.scatter_add_(0, idx.reshape(-1), torch.ones(idx.numel()))
    return counts


def _kl_smoothed(gen: torch.Tensor, ref: torch.Tensor) -> float:
    smoothing = 1e-6
    Ksize = gen.numel()
    gp = (gen + smoothing) / (gen.sum() + Ksize * smoothing)
    rp = (ref + smoothing) / (ref.sum() + Ksize * smoothing)
    return float((gp * (gp.log() - rp.log())).sum())


def _unigram_kl_probe(
    model: nn.Module, datamodule: DataModule, cfg: Config
) -> dict[str, float]:
    """Sample sequences and report unigram/bigram/trigram KL against the train corpus.

    Catches mode-collapse (unigram) and tracks per-position joint structure
    (bigram, trigram). Returns nats. Capped at sample_eval_n sequences /
    sample_eval_steps NAG steps to keep the probe cheap.
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
        # Two API conventions: EqM uses max_steps, DFM uses nfe.
        if hasattr(model, "decode_to_logprobs"):
            x = sample(n, L, max_steps=steps)
            log_probs = model.decode_to_logprobs(x)
            ids2d = log_probs.argmax(-1).cpu()  # (n, L)
        else:
            x = sample(n, L, nfe=steps)
            ids2d = x.cpu().long()
    finally:
        if was_training:
            model.train()

    train_ids2d = datamodule.splits.train.long()  # (Ntrain, L)

    gen_uni = _ngram_counts_flat(ids2d, K, 1)
    ref_uni = _ngram_counts_flat(train_ids2d, K, 1)
    gen_bi = _ngram_counts_flat(ids2d, K, 2)
    ref_bi = _ngram_counts_flat(train_ids2d, K, 2)
    gen_tri = _ngram_counts_flat(ids2d, K, 3)
    ref_tri = _ngram_counts_flat(train_ids2d, K, 3)

    # Entropies on unigram for the long-running headline number.
    p_gen = (gen_uni + 1e-9) / (gen_uni.sum() + K * 1e-9)
    p_ref = (ref_uni + 1e-9) / (ref_uni.sum() + K * 1e-9)
    H_gen = float(-(p_gen * p_gen.log()).sum())
    H_ref = float(-(p_ref * p_ref.log()).sum())

    return {
        "unigram_kl": _kl_smoothed(gen_uni, ref_uni),
        "bigram_kl": _kl_smoothed(gen_bi, ref_bi),
        "trigram_kl": _kl_smoothed(gen_tri, ref_tri),
        "H_gen": H_gen,
        "H_gt": H_ref,
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
