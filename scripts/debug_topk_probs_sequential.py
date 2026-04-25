"""Bare-bones sequential two-stage training for ``llm_topk_probs``.

This script is intentionally minimal and verbose so debugging is easy:

1. Build ``llm_topk_probs`` datamodule.
2. Train Stage 1 with an explicit epoch/batch loop.
3. Copy ``backbone.*`` (and ``llm_projection.*`` when present) into Stage 2.
4. Train Stage 2 with an explicit epoch/batch loop.
5. Save final Stage 1/Stage 2 checkpoints + a tiny manifest.

No W&B logging, no objective orchestration helpers, no plotting. The console
output is designed for printf-style debugging of data flow and loss values.

Usage:
    python scripts/debug_topk_probs_sequential.py \
        --out-dir checkpoints/debug_topk_probs \
        --stage1-epochs 2 \
        --stage2-epochs 2
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from _shared.bootstrap import bootstrap_repo_paths

# Allow direct script execution without editable install.
bootstrap_repo_paths(Path(__file__))

import aitchinson_flow.models  # noqa: E402,F401  - populate model REGISTRY

from _shared.cli import positive_int  # noqa: E402
from _shared.smoke import make_smoke_config  # noqa: E402
from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.models.base import TRAINING_LOSS_KEY  # noqa: E402
from aitchinson_flow.models.factory import build_model  # noqa: E402
from aitchinson_flow.training.batch import to_device  # noqa: E402
from aitchinson_flow.training.checkpoint import save_checkpoint  # noqa: E402
from aitchinson_flow.training.data_sources import build_training_datamodule  # noqa: E402
from aitchinson_flow.training.optim import build_optimizer, build_scheduler  # noqa: E402


def _numeric(value: Any) -> float | None:
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        return float(value.detach().mean().cpu())
    if isinstance(value, (int, float, bool)):
        return float(value)
    return None


def _summarize_batch(batch: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            summary[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
            }
        else:
            summary[key] = {"type": type(value).__name__}
    return summary


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in rows:
        for key, value in row.items():
            sums[key] = sums.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return {k: sums[k] / max(1, counts[k]) for k in sums}


def _prepare_batch_if_needed(model: torch.nn.Module, batch: dict[str, Any]) -> dict[str, Any]:
    prepare = getattr(model, "prepare_batch", None)
    if callable(prepare):
        prepared = prepare(batch)
        if isinstance(prepared, dict):
            return prepared
        raise TypeError(f"Expected prepare_batch() to return dict, got {type(prepared).__name__}")
    return batch


def _train_stage_verbose(
    *,
    stage_name: str,
    cfg: Config,
    model: torch.nn.Module,
    datamodule: Any,
    max_train_batches: int | None,
    max_val_batches: int | None,
    print_every: int,
) -> tuple[torch.nn.Module, list[dict[str, Any]]]:
    model = model.to(cfg.training.device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    history: list[dict[str, Any]] = []
    global_step = 0
    printed_batch_contract = False

    print(f"\\n=== {stage_name}: start ===")
    print(
        json.dumps(
            {
                "model_name": cfg.training.model_name,
                "epochs": cfg.training.epochs,
                "device": str(cfg.training.device),
                "batch_size": cfg.training.B,
                "lr": cfg.training.lr,
            },
            indent=2,
        )
    )

    for epoch in range(cfg.training.epochs):
        model.train()
        train_rows: list[dict[str, float]] = []
        train_loader = datamodule.train_dataloader()

        for batch_idx, raw_batch in enumerate(train_loader):
            if max_train_batches is not None and batch_idx >= max_train_batches:
                break

            batch = to_device(raw_batch, cfg.training.device)
            if not isinstance(batch, dict):
                raise TypeError(f"Expected dataloader batch to be dict, got {type(batch).__name__}")
            batch = _prepare_batch_if_needed(model, batch)

            if not printed_batch_contract:
                print("\\n[debug] first prepared batch contract")
                print(json.dumps(_summarize_batch(batch), indent=2))
                printed_batch_contract = True

            optimizer.zero_grad(set_to_none=True)
            out = model.training_step(batch, global_step)

            if TRAINING_LOSS_KEY not in out:
                raise KeyError(
                    f"{stage_name} training_step must return key {TRAINING_LOSS_KEY!r}; "
                    f"got keys {sorted(out.keys())}"
                )

            loss = out[TRAINING_LOSS_KEY]
            loss.backward()

            if cfg.training.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip_norm)

            optimizer.step()

            row = {
                key: val
                for key, val in ((k, _numeric(v)) for k, v in out.items())
                if val is not None
            }
            train_rows.append(row)

            if (batch_idx + 1) % max(1, print_every) == 0:
                print(
                    f"[{stage_name}] epoch={epoch + 1}/{cfg.training.epochs} "
                    f"step={batch_idx + 1} loss={row.get(TRAINING_LOSS_KEY, float('nan')):.6f} "
                    f"keys={sorted(row.keys())}"
                )

            global_step += 1

        val_rows: list[dict[str, float]] = []
        val_loader = datamodule.val_dataloader()
        if val_loader is not None and (epoch + 1) % max(1, cfg.training.eval_every) == 0:
            model.eval()
            with torch.no_grad():
                for batch_idx, raw_batch in enumerate(val_loader):
                    if max_val_batches is not None and batch_idx >= max_val_batches:
                        break
                    batch = to_device(raw_batch, cfg.training.device)
                    if not isinstance(batch, dict):
                        raise TypeError(
                            f"Expected val dataloader batch to be dict, got {type(batch).__name__}"
                        )
                    batch = _prepare_batch_if_needed(model, batch)
                    out = model.eval_step(batch)
                    row = {
                        key: val
                        for key, val in ((k, _numeric(v)) for k, v in out.items())
                        if val is not None
                    }
                    val_rows.append(row)

        train_mean = _mean_metrics(train_rows)
        val_mean = _mean_metrics(val_rows)
        epoch_entry: dict[str, Any] = {
            "epoch": epoch + 1,
            "train": train_mean,
            "val": val_mean,
        }
        history.append(epoch_entry)
        print(
            f"[{stage_name}] epoch {epoch + 1} summary: "
            f"train_loss={train_mean.get(TRAINING_LOSS_KEY, float('nan')):.6f} "
            f"val_loss={val_mean.get(TRAINING_LOSS_KEY, float('nan')):.6f}"
        )

        if scheduler is not None:
            scheduler.step()

    print(f"=== {stage_name}: done ===")
    return model, history


def _copy_stage1_into_stage2(stage1: torch.nn.Module, stage2: torch.nn.Module) -> None:
    stage1_state = dict(stage1.state_dict())

    backbone_state = {
        k.removeprefix("backbone."): v for k, v in stage1_state.items() if k.startswith("backbone.")
    }
    if not backbone_state:
        raise ValueError("No backbone.* weights found in Stage 1 model state")

    missing, unexpected = stage2.backbone.load_state_dict(backbone_state, strict=False)  # type: ignore[attr-defined]
    if missing:
        raise RuntimeError(f"Missing Stage 2 backbone keys when loading Stage 1: {sorted(missing)}")
    if unexpected:
        print(f"[debug] unexpected Stage 1 backbone keys ignored: {sorted(unexpected)}")

    proj_state = {
        k.removeprefix("llm_projection."): v
        for k, v in stage1_state.items()
        if k.startswith("llm_projection.")
    }
    if proj_state and getattr(stage2, "llm_projection", None) is not None:
        proj_missing, proj_unexpected = stage2.llm_projection.load_state_dict(  # type: ignore[attr-defined]
            proj_state,
            strict=False,
        )
        if proj_missing:
            raise RuntimeError(
                f"Missing Stage 2 llm_projection keys when loading Stage 1: {sorted(proj_missing)}"
            )
        if proj_unexpected:
            print(
                f"[debug] unexpected Stage 1 llm_projection keys ignored: {sorted(proj_unexpected)}"
            )

    freeze = getattr(stage2, "_freeze_representations", None)
    if callable(freeze):
        freeze()


def _init_inducing_from_data(model: torch.nn.Module, datamodule: Any, cfg: Config) -> None:
    """Initialize Stage 2 GP inducing points from current training batches."""
    model.eval()
    z_samples: list[torch.Tensor] = []
    prepare = getattr(model, "prepare_batch", None)
    with torch.no_grad():
        for raw_batch in datamodule.train_dataloader():
            batch = to_device(raw_batch, cfg.training.device)
            if not isinstance(batch, dict):
                continue
            if callable(prepare):
                prepared = prepare(batch)
                if isinstance(prepared, dict):
                    batch = prepared
            if "log_x" not in batch:
                continue
            z = model._extract_tokens(batch["log_x"])  # type: ignore[attr-defined]
            z_samples.append(z.reshape(-1, z.shape[-1]))
            if sum(t.shape[0] for t in z_samples) >= cfg.gp.num_inducing:
                break
    if not z_samples:
        print("[debug] skipped inducing init: no token latents collected")
        model.train()
        return

    all_z = torch.cat(z_samples, dim=0)
    m = cfg.gp.num_inducing
    if all_z.shape[0] >= m:
        idx = torch.randperm(all_z.shape[0], device=all_z.device)[:m]
    else:
        idx = torch.randint(0, all_z.shape[0], (m,), device=all_z.device)
    model.gp.Z.data.copy_(all_z[idx])  # type: ignore[attr-defined]
    model.train()
    print(f"[debug] initialized {m} inducing points from data")


def _build_base_cfg(args: argparse.Namespace) -> Config:
    cfg = make_smoke_config() if args.smoke else Config()

    cfg.training = replace(
        cfg.training,
        device=torch.device(args.device),
        B=args.batch_size,
        lr=args.lr,
        eval_every=max(1, args.eval_every),
        use_tqdm=False,
        wandb_enabled=False,
        wandb_mode="disabled",
    )

    cfg.training_data.source = "llm_topk_probs"
    cfg.llm_topk_probs.lm_key = args.lm_key
    cfg.llm_topk_probs.char_window_length = args.char_window_length
    cfg.llm_topk_probs.generation_seed = args.generation_seed
    cfg.llm_topk_probs.renormalize = not args.no_renormalize
    if args.corrupt_rate is not None:
        cfg.llm_topk_probs.corrupt_rate = args.corrupt_rate

    cfg.dataset.K = args.top_k
    cfg.dataset.L = args.seq_length

    return cfg


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=str, default="checkpoints/debug_topk_probs")
    p.add_argument("--stage1-epochs", type=positive_int, default=3)
    p.add_argument("--stage2-epochs", type=positive_int, default=3)
    p.add_argument("--batch-size", type=positive_int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--eval-every", type=positive_int, default=1)
    p.add_argument("--print-every", type=positive_int, default=1)
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-val-batches", type=int, default=8)

    p.add_argument("--lm-key", type=str, default="hf_causal")
    p.add_argument("--top-k", type=positive_int, default=64)
    p.add_argument("--seq-length", type=positive_int, default=30)
    p.add_argument("--char-window-length", type=positive_int, default=128)
    p.add_argument("--generation-seed", type=int, default=0)
    p.add_argument("--corrupt-rate", type=float, default=None)
    p.add_argument("--no-renormalize", action="store_true")

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--smoke", action="store_true")
    return p


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _build_argparser().parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = _build_base_cfg(args)
    print("[debug] base config")
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "training_data_source": base_cfg.training_data.source,
                "lm_key": base_cfg.llm_topk_probs.lm_key,
                "top_k": base_cfg.dataset.K,
                "seq_length": base_cfg.dataset.L,
                "char_window_length": base_cfg.llm_topk_probs.char_window_length,
                "device": str(base_cfg.training.device),
            },
            indent=2,
        )
    )

    datamodule, data_meta = build_training_datamodule(base_cfg)
    print("[debug] datamodule meta")
    print(json.dumps(data_meta, indent=2, default=str))

    stage1_cfg = deepcopy(base_cfg)
    stage1_cfg.training = replace(
        stage1_cfg.training,
        model_name="bayesian_auditor_stage1",
        epochs=args.stage1_epochs,
        checkpoint_dir=str(out_dir / "stage1_ckpts"),
    )
    stage1 = build_model(stage1_cfg)
    stage1, stage1_history = _train_stage_verbose(
        stage_name="stage1",
        cfg=stage1_cfg,
        model=stage1,
        datamodule=datamodule,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        print_every=args.print_every,
    )

    stage1_ckpt = out_dir / "stage1.pt"
    save_checkpoint(
        stage1_ckpt,
        model=stage1,
        cfg=stage1_cfg,
        optimizer=None,
        epoch=args.stage1_epochs,
        global_step=0,
    )
    print(f"[debug] wrote {stage1_ckpt}")

    stage2_cfg = deepcopy(base_cfg)
    stage2_cfg.training = replace(
        stage2_cfg.training,
        model_name="bayesian_auditor_stage2",
        epochs=args.stage2_epochs,
        checkpoint_dir=str(out_dir / "stage2_ckpts"),
    )
    stage2 = build_model(stage2_cfg)
    _copy_stage1_into_stage2(stage1, stage2)
    _init_inducing_from_data(stage2, datamodule, stage2_cfg)

    stage2, stage2_history = _train_stage_verbose(
        stage_name="stage2",
        cfg=stage2_cfg,
        model=stage2,
        datamodule=datamodule,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        print_every=args.print_every,
    )

    stage2_ckpt = out_dir / "stage2.pt"
    save_checkpoint(
        stage2_ckpt,
        model=stage2,
        cfg=stage2_cfg,
        optimizer=None,
        epoch=args.stage2_epochs,
        global_step=0,
    )
    print(f"[debug] wrote {stage2_ckpt}")

    manifest: dict[str, Any] = {
        "objective": "debug_llm_topk_probs_sequential",
        "stage1_ckpt": str(stage1_ckpt),
        "stage2_ckpt": str(stage2_ckpt),
        "training_data_source": data_meta,
        "stage1_history": stage1_history,
        "stage2_history": stage2_history,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"[debug] wrote {manifest_path}")
    return manifest


if __name__ == "__main__":
    main()
