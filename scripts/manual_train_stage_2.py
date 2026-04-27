"""Manual Stage-2 training with frozen Stage-1 transformer backbone.

This script mirrors the manual style of ``manual_train_stage_1.py`` but trains
``bayesian_auditor_stage2`` only:

1. Build Stage-2 model.
2. Seed ``backbone.*`` (and optional ``llm_projection.*``) from a Stage-1
   checkpoint.
3. Re-apply Stage-2 freeze invariants.
4. Train only Stage-2 trainable parameters (latent head + GP).
5. Save ``stage2.pt``.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
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
    p.add_argument("--out-dir", type=str, default="checkpoints/manual/stage2")
    p.add_argument("--stage2-epochs", type=positive_int, default=25)
    p.add_argument(
        "--stage1-backbone-ckpt",
        type=str,
        default=None,
        help=(
            "Path to Stage 1 .pt training checkpoint. "
            f"Default: {_REPO_ROOT / 'checkpoints/two_stage/baseline/stage1_ckpts/epoch_25.pt'}"
        ),
    )
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


def _stage2_config(base: Config, *, epochs: int, ckpt_dir: Path) -> Config:
    cfg = deepcopy(base)
    cfg.training = replace(
        cfg.training,
        model_name="bayesian_auditor_stage2",
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


def _filter_state_dict_by_prefix(
    state: dict[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    return {k: v for k, v in state.items() if k.startswith(prefix)}


def _load_stage1_checkpoint_state(
    path: Path,
    *,
    map_location: str | torch.device | None = None,
) -> dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(
            f"Checkpoint at {path} must be a dict with 'model_state_dict' (training format)."
        )
    state = ckpt["model_state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"model_state_dict at {path} is not a mapping.")
    return dict(state)


def _load_backbone_into_stage2(
    stage2_model: nn.Module,
    stage1_state: dict[str, torch.Tensor],
) -> list[str]:
    backbone_state = _filter_state_dict_by_prefix(stage1_state, "backbone.")
    if not backbone_state:
        raise ValueError("Stage 1 checkpoint contains no 'backbone.*' keys; cannot seed Stage 2.")
    remapped = {k.removeprefix("backbone."): v for k, v in backbone_state.items()}
    has_contextual_keys = any(k.startswith("answer_backbone.") for k in remapped)
    if not has_contextual_keys:
        remapped = {f"answer_backbone.{k}": v for k, v in remapped.items()}
    missing, unexpected = stage2_model.backbone.load_state_dict(  # type: ignore[attr-defined]
        remapped,
        strict=False,
    )
    if missing:
        allowed_missing_prefixes = (
            "question_backbone.",
            "cross_attention.",
            "cross_norm.",
        )
        required_missing = [
            k
            for k in missing
            if not any(k.startswith(prefix) for prefix in allowed_missing_prefixes)
            and k != "context_gate"
        ]
        if required_missing:
            raise RuntimeError(
                "Missing required backbone keys when seeding Stage 2 from Stage 1: "
                f"{sorted(required_missing)}"
            )
        print(
            "[warn] Stage 1 checkpoint did not contain contextual backbone keys; "
            f"leaving defaults for: {sorted(missing)}"
        )

    stage1_proj = _filter_state_dict_by_prefix(stage1_state, "llm_projection.")
    if stage1_proj and getattr(stage2_model, "llm_projection", None) is not None:
        proj_missing, proj_unexpected = stage2_model.llm_projection.load_state_dict(  # type: ignore[attr-defined]
            {k.removeprefix("llm_projection."): v for k, v in stage1_proj.items()},
            strict=False,
        )
        if proj_missing:
            raise RuntimeError(
                f"Missing llm_projection keys when seeding Stage 2 from Stage 1: "
                f"{sorted(proj_missing)}"
            )
        unexpected = list(unexpected) + [f"llm_projection.{k}" for k in proj_unexpected]

    return list(unexpected)


def main(argv: list[str] | None = None):
    args = _build_argparser().parse_args(argv)

    cfg = make_smoke_config() if args.smoke else Config()
    if args.smoke:
        # Keep smoke runs short even when the text corpus is large.
        cfg.text8_dataset.max_train_windows = 32
        cfg.text8_dataset.max_eval_windows = 32
        cfg.training.num_workers = 0
    apply_training_data_args(cfg, args)
    _apply_llm_topk_probs_overrides(cfg, args)

    datamodule, _ = _build_default_datamodule(cfg=cfg)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s2_cfg = _stage2_config(cfg, epochs=args.stage2_epochs, ckpt_dir=out_dir / "stage2_ckpts")
    stage1_ckpt = (
        Path(args.stage1_backbone_ckpt)
        if args.stage1_backbone_ckpt
        else _REPO_ROOT / "checkpoints/two_stage/baseline/stage1_ckpts/epoch_25.pt"
    )
    if not stage1_ckpt.is_file():
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {stage1_ckpt}")

    seed_all(s2_cfg.training.seed)
    model = build_model(s2_cfg).to(s2_cfg.training.device)
    stage1_state = _load_stage1_checkpoint_state(stage1_ckpt, map_location=s2_cfg.training.device)
    unexpected = _load_backbone_into_stage2(model, stage1_state)
    if unexpected:
        print(
            f"[warn] Ignored {len(unexpected)} unexpected key(s) during Stage-1->Stage-2 load: "
            f"{sorted(unexpected)}"
        )
    freeze_fn = getattr(model, "_freeze_representations", None)
    if callable(freeze_fn):
        freeze_fn()

    optimizer = build_optimizer(model, s2_cfg)
    scheduler = build_scheduler(optimizer, s2_cfg)
    device = torch.device(s2_cfg.training.device)
    training_data = datamodule.train_dataloader()
    num_train_fn = getattr(datamodule, "num_train_samples", None)
    if callable(num_train_fn):
        candidate = num_train_fn()
        if isinstance(candidate, int) and candidate > 0:
            # Match training.runner.fit behavior so KL is normalized by dataset size.
            setattr(model, "_kl_normalizer", candidate)

    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    frozen = [name for name, p in model.named_parameters() if not p.requires_grad]
    print(f"trainable parameters: {len(trainable)} tensors")
    print(f"frozen parameters: {len(frozen)} tensors")
    if any(name.startswith("backbone.") for name in trainable):
        raise RuntimeError("Backbone parameters unexpectedly trainable in Stage 2.")
    if any(name.startswith("llm_projection.") for name in trainable):
        raise RuntimeError("llm_projection parameters unexpectedly trainable in Stage 2.")

    epoch_range = range(0, s2_cfg.training.epochs)
    epoch_pbar = tqdm(
        epoch_range,
        desc="epochs",
        disable=not s2_cfg.training.use_tqdm,
        leave=True,
        unit="epoch",
    )
    global_step = 0
    model_typed = cast(GenerativeTrainingModel, model)

    for epoch in epoch_pbar:
        model.train()
        pbar = tqdm(
            training_data,
            desc=f"train epoch {epoch + 1}",
            disable=not s2_cfg.training.use_tqdm,
            leave=False,
        )

        loss_meter: list[float] = []
        nll_meter: list[float] = []
        contrastive_meter: list[float] = []
        kl_meter: list[float] = []
        kl_norm_meter: list[float] = []
        anchor_meter: list[float] = []
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

            nll = out.get("nll")
            if torch.is_tensor(nll):
                nll_meter.append(float(nll.detach().cpu()))
            contrastive = out.get("contrastive")
            if torch.is_tensor(contrastive):
                contrastive_meter.append(float(contrastive.detach().cpu()))
            kl = out.get("kl")
            if torch.is_tensor(kl):
                kl_meter.append(float(kl.detach().cpu()))
            kl_norm = out.get("kl_norm")
            if torch.is_tensor(kl_norm):
                kl_norm_meter.append(float(kl_norm.detach().cpu()))
            anchor = out.get("anchor")
            if torch.is_tensor(anchor):
                anchor_meter.append(float(anchor.detach().cpu()))

            postfix: dict[str, str] = {"loss": f"{loss_value:.4f}"}
            if nll_meter:
                postfix["nll"] = f"{nll_meter[-1]:.4f}"
            if contrastive_meter:
                postfix["contrastive"] = f"{contrastive_meter[-1]:.4f}"
            if kl_meter:
                postfix["kl"] = f"{kl_meter[-1]:.4f}"
            if kl_norm_meter:
                postfix["kl_norm"] = f"{kl_norm_meter[-1]:.4f}"
            if anchor_meter:
                postfix["anchor"] = f"{anchor_meter[-1]:.4f}"
            pbar.set_postfix(postfix)

            global_step += 1

        mean_loss = sum(loss_meter) / max(1, len(loss_meter))
        msg = f"epoch {epoch + 1}/{s2_cfg.training.epochs} mean loss: {mean_loss:.6f}"
        if nll_meter:
            msg += f", mean nll: {sum(nll_meter) / len(nll_meter):.6f}"
        if contrastive_meter:
            msg += f", mean contrastive: {sum(contrastive_meter) / len(contrastive_meter):.6f}"
        print(msg)

        if scheduler is not None:
            scheduler.step()
            print(f"new learning rate: {scheduler.get_last_lr()}")

    s2_path = out_dir / "stage2.pt"
    save_checkpoint(
        s2_path,
        model=model,
        cfg=s2_cfg,
        optimizer=None,
        epoch=s2_cfg.training.epochs,
        global_step=global_step,
    )
    print(f"saved stage2 checkpoint: {s2_path}")


if __name__ == "__main__":
    main()
