"""Inference sanity check: valid sequences → flow matching → simplex → token recovery.

Workflow:
  1. Load trained model checkpoint.
  2. Run forward pass on valid sequences (log_x from dataloader).
  3. Extract velocity prediction v_pred from the forward call.
  4. Map v_pred back to log-simplex via ilr_inv (if ILR mode).
  5. Exponentiate to get simplex probabilities.
  6. Argmax to recover token IDs.
  7. Compare recovered IDs with original token IDs (sanity check).
  8. Extend to AUROC by also scoring invalid sequences.

Usage:
    python scripts/infer_sanity_check.py \\
        --model-ckpt checkpoints/manual/baseline/stage1_ckpts/model_checkpoint_10.pt \\
        --n-batches 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from _shared.bootstrap import bootstrap_repo_paths

bootstrap_repo_paths(Path(__file__))

import aitchinson_flow.models  # noqa: E402,F401 — populate model REGISTRY

from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.geometry import ilr_inv  # noqa: E402
from aitchinson_flow.models.factory import build_model  # noqa: E402
from aitchinson_flow.training.checkpoint import load_checkpoint  # noqa: E402
from aitchinson_flow.training.data_sources import build_training_datamodule  # noqa: E402
from aitchinson_flow.training.batch import to_device  # noqa: E402
from aitchinson_flow.metrics.auroc import safe_auroc  # noqa: E402
from aitchinson_flow.data.corruption import build_invalid_batch  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model-ckpt",
        type=str,
        required=True,
        help="Path to trained model checkpoint (training format with model_state_dict).",
    )
    p.add_argument(
        "--n-batches",
        type=int,
        default=10,
        help="Number of batches to evaluate (None = all).",
    )
    p.add_argument(
        "--compute-auroc",
        action="store_true",
        help="Also compute AUROC by generating and scoring invalid sequences.",
    )
    return p.parse_args()


def _features_to_simplex_probs(features: torch.Tensor, cfg: Config, K: int) -> torch.Tensor:
    """Map ILR/CLR features back to simplex probabilities.

    Args:
        features: (B, L, D) where D = K-1 for ILR or K for CLR.
        cfg: Config with hf_dataset.transform_mode.
        K: Vocabulary size.

    Returns:
        (B, L, K) simplex probabilities that sum to 1 over the last dim.
    """
    mode = cfg.hf_dataset.transform_mode.lower()
    if mode == "ilr":
        log_simplex = ilr_inv(features, K)
    elif mode == "clr":
        log_simplex = features - features.logsumexp(dim=-1, keepdim=True)
    else:
        raise ValueError(f"Unsupported transform mode: {mode}")

    return log_simplex.exp()


@torch.no_grad()
def eval_sanity_check(
    model: nn.Module,
    datamodule,
    cfg: Config,
    n_batches: int | None = None,
) -> dict[str, float]:
    """Run inference sanity check on valid sequences.

    Returns:
        Dictionary with 'token_accuracy' and other diagnostics.
    """
    device = cfg.training.device
    model = model.to(device)
    model.eval()

    loader = datamodule.val_dataloader() or datamodule.train_dataloader()
    K = cfg.dataset.K

    correct_tokens: list[int] = []
    total_tokens: int = 0
    batch_count = 0

    for batch in tqdm(loader, desc="sanity check", disable=False):
        if n_batches is not None and batch_count >= n_batches:
            break
        batch_count += 1

        batch = to_device(batch, device)

        # Path B: prepare batch (translate embeddings → log_x if needed)
        prepare = getattr(model, "prepare_batch", None)
        if prepare is not None:
            batch = prepare(batch)

        if "log_x" not in batch or "token_ids" not in batch:
            continue

        log_x_valid = batch["log_x"]  # (B, L, D) in ILR/CLR coordinates
        token_ids_valid = batch["token_ids"]  # (B, L) original token IDs

        # Forward pass: velocity prediction
        if hasattr(model, "forward"):
            out = model.forward(log_x_valid)
            if isinstance(out, tuple):
                v_pred, _ = out
            else:
                v_pred = out
        else:
            raise ValueError("Model must have a forward method.")

        # Map velocity prediction back to simplex probabilities
        v_probs = _features_to_simplex_probs(v_pred, cfg, K)

        # Argmax to recover token IDs
        recovered_ids = v_probs.argmax(dim=-1)  # (B, L)

        # Sanity check: compare recovered vs original
        matches = (recovered_ids == token_ids_valid).int()
        correct_tokens.append(matches.sum().item())
        total_tokens += token_ids_valid.numel()

        # Early diagnostics
        acc_batch = matches.float().mean().item()
        print(f"  Batch {batch_count}: token accuracy = {acc_batch:.4f}")

    token_accuracy = sum(correct_tokens) / total_tokens if total_tokens > 0 else 0.0
    print(f"\n{'=' * 60}")
    print(f"Sanity Check Results:")
    print(f"  Token accuracy (v_pred → simplex → argmax): {token_accuracy:.4f}")
    print(f"  Total tokens evaluated: {total_tokens}")
    print(f"{'=' * 60}\n")

    return {"token_accuracy": token_accuracy}


@torch.no_grad()
def eval_auroc(
    model: nn.Module,
    datamodule,
    cfg: Config,
    n_batches: int | None = None,
) -> dict[str, float]:
    """Compute AUROC using model score (OOD score if available, else velocity norm).

    Returns:
        Dictionary with 'auroc_score' and related diagnostics.
    """
    device = cfg.training.device
    model = model.to(device)
    model.eval()

    loader = datamodule.val_dataloader() or datamodule.train_dataloader()
    K = cfg.dataset.K

    valid_scores: list[torch.Tensor] = []
    invalid_scores: list[torch.Tensor] = []
    batch_count = 0

    for batch in tqdm(loader, desc="AUROC eval", disable=False):
        if n_batches is not None and batch_count >= n_batches:
            break
        batch_count += 1

        batch = to_device(batch, device)

        # Prepare batch
        prepare = getattr(model, "prepare_batch", None)
        if prepare is not None:
            batch = prepare(batch)

        if "log_x" not in batch:
            continue

        log_x_valid = batch["log_x"]

        # Generate invalid batch (corrupted sequences)
        if "log_x_invalid" not in batch:
            build_invalid_batch(
                batch,
                K=K,
                corrupt_rate=0.3,
                order_mix_rate=0.2,
                order_mix_prob=0.1,
                eps=cfg.hf_dataset.log_simplex_eps,
                label_smoothing=cfg.hf_dataset.label_smoothing,
                transform_mode=cfg.hf_dataset.transform_mode,
                seed=42 + batch_count,
            )
        log_x_invalid = batch.get("log_x_invalid")

        # Score valid sequences
        if hasattr(model, "ood_score"):
            score_valid = model.ood_score(log_x_valid).cpu()
        else:
            # Fallback: velocity norm
            out = model.forward(log_x_valid)
            if isinstance(out, tuple):
                v, _ = out
            else:
                v = out
            score_valid = v.reshape(v.shape[0], -1).norm(dim=1).cpu()

        valid_scores.append(score_valid)

        # Score invalid sequences
        if log_x_invalid is not None:
            if hasattr(model, "ood_score"):
                score_invalid = model.ood_score(log_x_invalid).cpu()
            else:
                out = model.forward(log_x_invalid)
                if isinstance(out, tuple):
                    v, _ = out
                else:
                    v = out
                score_invalid = v.reshape(v.shape[0], -1).norm(dim=1).cpu()

            invalid_scores.append(score_invalid)

    # Compute AUROC
    if valid_scores and invalid_scores:
        v_all = torch.cat(valid_scores).numpy()
        i_all = torch.cat(invalid_scores).numpy()
        auroc = safe_auroc(v_all, i_all)
    else:
        auroc = float("nan")

    print(f"\n{'=' * 60}")
    print(f"AUROC Results:")
    print(f"  AUROC (valid vs corrupted): {auroc:.4f}")
    print(f"  Valid sequences: {len(valid_scores)} batches")
    print(f"  Invalid sequences: {len(invalid_scores)} batches")
    print(f"{'=' * 60}\n")

    return {"auroc_score": auroc}


def main() -> None:
    args = _parse_args()
    ckpt_path = Path(args.model_ckpt)

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    model = build_model(Config())
    cfg_dict, epoch, step = load_checkpoint(ckpt_path, model=model, map_location="cpu")
    print(f"  Epoch: {epoch}, Step: {step}")

    cfg = Config()
    datamodule, _ = build_training_datamodule(cfg)

    print("\n" + "=" * 60)
    print("SANITY CHECK: Token Reconstruction")
    print("=" * 60)
    result_sanity = eval_sanity_check(model, datamodule, cfg, n_batches=args.n_batches)

    if args.compute_auroc:
        print("\n" + "=" * 60)
        print("AUROC: Valid vs Corrupted Sequences")
        print("=" * 60)
        result_auroc = eval_auroc(model, datamodule, cfg, n_batches=args.n_batches)
    else:
        result_auroc = {}

    # Summary
    results = {**result_sanity, **result_auroc}
    print("\nFinal Summary:")
    for key, val in results.items():
        print(f"  {key}: {val:.4f}")


if __name__ == "__main__":
    main()
