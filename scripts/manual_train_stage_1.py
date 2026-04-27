import argparse
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import torch
from tqdm import tqdm

from _shared.bootstrap import bootstrap_repo_paths
from _shared.cli import add_training_data_args, apply_training_data_args, positive_int
from _shared.smoke import make_smoke_config

# Allow direct script execution without editable install.
_REPO_ROOT = bootstrap_repo_paths(Path(__file__))

import aitchinson_flow.models  # noqa: E402,F401  — populate model REGISTRY
from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.models import build_model  # noqa: E402
from aitchinson_flow.models.base import (  # noqa: E402
    TRAINING_LOSS_KEY,
    GenerativeTrainingModel,
)
from aitchinson_flow.training import save_checkpoint  # noqa: E402
from aitchinson_flow.training.batch import to_device  # noqa: E402
from aitchinson_flow.training.data_sources import build_training_datamodule  # noqa: E402
from aitchinson_flow.training.datamodule import DataModule  # noqa: E402
from aitchinson_flow.training.optim import build_optimizer, build_scheduler  # noqa: E402
from aitchinson_flow.training.seed import seed_all  # noqa: E402


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=str, default="checkpoints/manual/stage1")
    p.add_argument("--stage1-epochs", type=positive_int, default=25)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Use a tiny CPU config for end-to-end smoke testing.",
    )
    p.add_argument("--lm-key", type=str, default=None)
    p.add_argument("--top-k", type=positive_int, default=None)
    p.add_argument("--seq-length", type=positive_int, default=None)
    p.add_argument("--char-window-length", type=positive_int, default=None)
    p.add_argument("--corrupt-rate", type=float, default=None)
    p.add_argument("--no-renormalize", action="store_true")

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


def _apply_llm_topk_probs_overrides(cfg: Config, args: argparse.Namespace) -> None:
    """Apply optional Component-2 (llm_topk_probs) CLI overrides to ``cfg``."""
    if args.lm_key is not None:
        cfg.llm_topk_probs.lm_key = args.lm_key
    if args.top_k is not None:
        cfg.dataset.K = args.top_k
    if args.seq_length is not None:
        cfg.dataset.L = args.seq_length
    if args.char_window_length is not None:
        cfg.llm_topk_probs.char_window_length = args.char_window_length
    if args.corrupt_rate is not None:
        cfg.llm_topk_probs.corrupt_rate = args.corrupt_rate
    if args.generation_seed is not None:
        cfg.llm_topk_probs.generation_seed = args.generation_seed
    if args.no_renormalize:
        cfg.llm_topk_probs.renormalize = False


def main(argv: list[str] | None = None):
    args = _build_argparser().parse_args(argv)

    cfg = make_smoke_config() if args.smoke else Config()
    if args.smoke:
        cfg.text8_dataset.max_train_windows = 32
        cfg.text8_dataset.max_eval_windows = 32
        cfg.training.num_workers = 0
    apply_training_data_args(cfg, args)
    _apply_llm_topk_probs_overrides(cfg, args)

    datamodule, _ = _build_default_datamodule(cfg=cfg)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s1_cfg = _stage1_config(cfg, epochs=args.stage1_epochs, ckpt_dir=out_dir / "stage1_ckpts")
    seed_all(s1_cfg.training.seed)
    model = build_model(s1_cfg).to(s1_cfg.training.device)
    model_typed = cast(GenerativeTrainingModel, model)
    optimizer = build_optimizer(model, s1_cfg)
    scheduler = build_scheduler(optimizer, s1_cfg)
    device = torch.device(s1_cfg.training.device)
    training_data = datamodule.train_dataloader()

    epoch_pbar = tqdm(
        range(s1_cfg.training.epochs),
        desc="epochs",
        disable=not s1_cfg.training.use_tqdm,
        leave=True,
        unit="epoch",
    )
    global_step = 0
    for epoch in epoch_pbar:
        model.train()
        pbar = tqdm(
            training_data,
            desc=f"train epoch {epoch + 1}",
            disable=not s1_cfg.training.use_tqdm,
            leave=False,
        )
        loss_meter: list[float] = []
        flow_meter: list[float] = []
        mask_meter: list[float] = []
        for batch in pbar:
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            out = model_typed.training_step(batch, global_step)
            if TRAINING_LOSS_KEY not in out:
                raise KeyError(
                    f"training_step() must return {TRAINING_LOSS_KEY!r}; got keys {sorted(out.keys())}."
                )
            loss = out[TRAINING_LOSS_KEY]
            loss.backward()
            optimizer.step()

            loss_value = float(loss.detach().cpu())
            loss_meter.append(loss_value)
            flow_loss = out.get("flow_loss")
            if torch.is_tensor(flow_loss):
                flow_meter.append(float(flow_loss.detach().cpu()))
            mask_loss = out.get("mask_loss")
            if torch.is_tensor(mask_loss):
                mask_meter.append(float(mask_loss.detach().cpu()))
            postfix: dict[str, str] = {"loss": f"{loss_value:.4f}"}
            if flow_meter:
                postfix["flow"] = f"{flow_meter[-1]:.4f}"
            if mask_meter:
                postfix["mask"] = f"{mask_meter[-1]:.4f}"
            pbar.set_postfix(postfix)
            global_step += 1

        mean_loss = sum(loss_meter) / max(1, len(loss_meter))
        msg = f"epoch {epoch + 1}/{s1_cfg.training.epochs} mean loss: {mean_loss:.6f}"
        if flow_meter:
            msg += f", mean flow: {sum(flow_meter) / len(flow_meter):.6f}"
        if mask_meter:
            msg += f", mean mask: {sum(mask_meter) / len(mask_meter):.6f}"
        print(msg)
        if scheduler is not None:
            scheduler.step()
            print(f"new learning rate: {scheduler.get_last_lr()}")

    s1_path = out_dir / "stage1.pt"
    save_checkpoint(
        s1_path,
        model=model,
        cfg=s1_cfg,
        optimizer=None,
        epoch=s1_cfg.training.epochs,
        global_step=global_step,
    )
    print(f"saved stage1 checkpoint: {s1_path}")


if __name__ == "__main__":
    main()
