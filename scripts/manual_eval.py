"""
Energy-based evaluation for Equilibrium Matching with Explicit Energy (EqM-E).

Usage:
    python eval_energy.py --checkpoint checkpoints/manual/baseline/stage1_ckpts/epoch_25.pt
    python eval_energy.py --checkpoint ... --n-trajectories 5 --corrupt-gamma 0.8 --n-steps 50
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import tqdm

from _shared.bootstrap import bootstrap_repo_paths

_REPO_ROOT = bootstrap_repo_paths(Path(__file__))

from aitchinson_flow.config import Config
from aitchinson_flow.training.data_sources import build_training_datamodule
from aitchinson_flow.training.batch import to_device

import aitchinson_flow.models  # noqa: F401 — populate REGISTRY
from aitchinson_flow.models import build_model
from aitchinson_flow.training.seed import seed_all


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint", type=str, required=True, help="Path to .pt checkpoint (model_state_dict)."
    )
    p.add_argument("--out-dir", type=str, default="eval_outputs", help="Directory to save plots.")
    p.add_argument("--n-trajectories", type=int, default=5, help="Number of trajectories to plot.")
    p.add_argument("--n-steps", type=int, default=100, help="ODE integration steps (noise→data).")
    p.add_argument(
        "--corrupt-gamma",
        type=float,
        default=0.85,
        help="Interpolation gamma for corrupted sequences (closer to 1 = less corrupted).",
    )
    p.add_argument(
        "--n-score-samples", type=int, default=64, help="Number of valid/corrupted pairs to score."
    )
    p.add_argument("--seed", type=int, default=42)
    return p


# ---------------------------------------------------------------------------
# Energy function
# ---------------------------------------------------------------------------


def compute_energy(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Dot-product energy: g(xγ) = xγ · f(xγ), returned as scalar per batch element."""
    x = x.detach().requires_grad_(False)
    with torch.no_grad():
        with sdpa_kernel(SDPBackend.MATH):
            v_pred, _ = model.forward(x)
    # Sum over L and D dims, keep batch dim → shape (B,)
    return (x * v_pred).sum(dim=(-2, -1))


def compute_grad_g(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (energy per sample, grad_g) using dot-product formulation."""
    x = x.detach().requires_grad_(True)
    with sdpa_kernel(SDPBackend.MATH):
        v_pred, _ = model.forward(x)
    g = (x * v_pred).sum()
    grad_g = torch.autograd.grad(g, x)[0]
    # Per-sample energy for logging
    energy_per_sample = (x.detach() * v_pred.detach()).sum(dim=(-2, -1))
    return energy_per_sample, grad_g.detach()


# ---------------------------------------------------------------------------
# Trajectory evaluation
# ---------------------------------------------------------------------------


def eval_trajectories(
    model: nn.Module,
    valid_batch: torch.Tensor,
    device: torch.device,
    n_trajectories: int,
    n_steps: int,
    corrupt_gamma: float,
    out_dir: Path,
):
    """
    For each trajectory:
      - Start from lightly corrupted valid sequence (xγ, gamma=corrupt_gamma)
      - Integrate velocity field from gamma=corrupt_gamma → 1
      - Track energy at each step
    Expect energy to decrease monotonically.
    """
    model.eval()

    # Take first n_trajectories samples
    x1 = valid_batch[:n_trajectories].to(device)  # (T, L, D)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_energy, ax_loss = axes

    gammas = torch.linspace(corrupt_gamma, 1.0, n_steps)
    step_size = (1.0 - corrupt_gamma) / n_steps

    for i in range(n_trajectories):
        x1_i = x1[i].unsqueeze(0)  # (1, L, D)
        eps = torch.randn_like(x1_i)

        # Start from corrupted point
        x_cur = (1 - corrupt_gamma) * eps + corrupt_gamma * x1_i

        energies = []
        grad_norms = []

        for g in tqdm(gammas, desc=f"trajectory {i + 1}/{n_trajectories}", leave=False):
            energy, grad_g = compute_grad_g(model, x_cur)
            energies.append(energy.item())
            grad_norms.append(grad_g.norm().item())

            # Euler step along negative gradient (descend energy)
            x_cur = (x_cur - step_size * grad_g).detach()

        ax_energy.plot(gammas.cpu().numpy(), energies, alpha=0.7, label=f"seq {i + 1}")
        ax_loss.plot(gammas.cpu().numpy(), grad_norms, alpha=0.7, label=f"seq {i + 1}")

    ax_energy.set_xlabel("gamma (corrupt → data)")
    ax_energy.set_ylabel("energy g(xγ)")
    ax_energy.set_title("Energy along trajectory\n(should decrease toward gamma=1)")
    ax_energy.legend()
    ax_energy.grid(True)

    ax_loss.set_xlabel("gamma (corrupt → data)")
    ax_loss.set_ylabel("||∇g|| (gradient norm)")
    ax_loss.set_title("Gradient norm along trajectory\n(should shrink near data)")
    ax_loss.legend()
    ax_loss.grid(True)

    fig.tight_layout()
    path = out_dir / "energy_trajectories.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved trajectory plot → {path}")


# ---------------------------------------------------------------------------
# Valid vs corrupted scoring
# ---------------------------------------------------------------------------


def eval_energy_scores(
    model: nn.Module,
    valid_batch: torch.Tensor,
    device: torch.device,
    n_samples: int,
    corrupt_gamma: float,
    out_dir: Path,
):
    """
    Compare energy of valid sequences vs lightly corrupted versions.
    Valid sequences should have lower (more negative) energy.
    """
    model.eval()

    x1 = valid_batch[:n_samples].to(device)
    eps = torch.randn_like(x1)
    x_corrupt = (1 - corrupt_gamma) * eps + corrupt_gamma * x1

    with torch.no_grad():
        energy_valid = compute_energy(model, x1).cpu()
        energy_corrupt = compute_energy(model, x_corrupt).cpu()

    print("\n--- Energy Scores ---")
    print(f"Valid     | mean: {energy_valid.mean():.4f}  std: {energy_valid.std():.4f}")
    print(f"Corrupted | mean: {energy_corrupt.mean():.4f}  std: {energy_corrupt.std():.4f}")
    delta = energy_valid.mean() - energy_corrupt.mean()
    print(
        f"Δ (valid - corrupt): {delta:.4f}  {'✓ valid < corrupt' if delta < 0 else '✗ unexpected: valid >= corrupt'}"
    )

    # Plot distributions
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = 30
    ax.hist(
        energy_valid.numpy(),
        bins=bins,
        alpha=0.6,
        label=f"valid (μ={energy_valid.mean():.3f})",
        color="steelblue",
    )
    ax.hist(
        energy_corrupt.numpy(),
        bins=bins,
        alpha=0.6,
        label=f"corrupted γ={corrupt_gamma} (μ={energy_corrupt.mean():.3f})",
        color="tomato",
    )
    ax.set_xlabel("energy g(x)")
    ax.set_ylabel("count")
    ax.set_title("Energy distribution: valid vs corrupted sequences")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    path = out_dir / "energy_scores.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved score plot → {path}")

    # Scatter: per-sample valid vs corrupted energy
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(energy_valid.numpy(), energy_corrupt.numpy(), alpha=0.5, s=20)
    lims = [
        min(energy_valid.min(), energy_corrupt.min()).item(),
        max(energy_valid.max(), energy_corrupt.max()).item(),
    ]
    ax.plot(lims, lims, "k--", linewidth=1, label="y=x")
    ax.set_xlabel("energy (valid)")
    ax.set_ylabel("energy (corrupted)")
    ax.set_title(
        "Per-sample: valid vs corrupted energy\n(points above diagonal = corrupt has higher energy ✓)"
    )
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    path = out_dir / "energy_scatter.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved scatter plot → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None):
    args = _build_argparser().parse_args(argv)
    seed_all(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    device = torch.device(cfg.training.device)

    # Build model and load checkpoint
    from aitchinson_flow.training.seed import seed_all
    from copy import deepcopy
    from dataclasses import replace

    cfg_eval = deepcopy(cfg)
    cfg_eval.training = replace(cfg_eval.training, model_name="bayesian_auditor_stage1")

    model = build_model(cfg_eval)
    ckpt = torch.load(args.checkpoint, map_location=device)

    # Support both raw state_dict and wrapped checkpoints
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Load a batch of valid sequences from the datamodule
    datamodule, _ = build_training_datamodule(cfg)
    loader = datamodule.train_dataloader()
    batch = next(iter(loader))
    batch = to_device(batch, cfg.training.device)
    batch = model.prepare_batch(batch)
    valid_sequences = batch["log_x"]  # (B, L, D)

    n_needed = max(args.n_trajectories, args.n_score_samples)
    if valid_sequences.shape[0] < n_needed:
        print(
            f"Warning: only {valid_sequences.shape[0]} samples in first batch, need {n_needed}. "
            f"Consider increasing batch size."
        )

    print(
        f"\nRunning trajectory evaluation ({args.n_trajectories} trajectories, {args.n_steps} steps)..."
    )
    eval_trajectories(
        model=model,
        valid_batch=valid_sequences,
        device=device,
        n_trajectories=args.n_trajectories,
        n_steps=args.n_steps,
        corrupt_gamma=args.corrupt_gamma,
        out_dir=out_dir,
    )

    print(f"\nRunning energy scoring ({args.n_score_samples} valid vs corrupted pairs)...")
    eval_energy_scores(
        model=model,
        valid_batch=valid_sequences,
        device=device,
        n_samples=args.n_score_samples,
        corrupt_gamma=args.corrupt_gamma,
        out_dir=out_dir,
    )

    print("\nDone. All plots saved to:", out_dir)


if __name__ == "__main__":
    main()
