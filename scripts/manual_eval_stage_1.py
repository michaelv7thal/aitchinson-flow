"""
Energy-based evaluation for Equilibrium Matching with Explicit Energy (EqM-E).

Usage:
    python eval_energy.py --checkpoint checkpoints/manual/baseline/stage1_ckpts/epoch_25.pt
    python eval_energy.py --checkpoint ... --n-trajectories 5 --n-steps 50
"""

import argparse
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import tqdm

from _shared.bootstrap import bootstrap_repo_paths
from _shared.cli import add_training_data_args, apply_training_data_args, positive_int

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
        "--corrupt-gammas",
        type=float,
        nargs="+",
        default=[0.99, 0.95, 0.90, 0.85, 0.75, 0.50],
        help="Corruption gammas to sweep over for AUROC evaluation (closer to 1 = less corrupted).",
    )
    p.add_argument(
        "--trajectory-gamma",
        type=float,
        default=0.85,
        help="Gamma to use for trajectory visualization.",
    )
    p.add_argument(
        "--n-score-samples", type=int, default=64, help="Number of valid/corrupted pairs to score."
    )
    p.add_argument(
        "--use-train-loader",
        action="store_true",
        help="Force use of training data instead of validation data.",
    )
    p.add_argument("--lm-key", type=str, default=None)
    p.add_argument("--top-k", type=positive_int, default=None)
    p.add_argument("--seq-length", type=positive_int, default=None)
    p.add_argument("--char-window-length", type=positive_int, default=None)
    p.add_argument("--corrupt-rate", type=float, default=None)
    p.add_argument("--no-renormalize", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    add_training_data_args(p)
    p.add_argument(
        "--grid-corrupt-rates",
        type=float,
        nargs="+",
        default=None,
        help="Optional corruption-rate grid (y-axis) for rate×gamma heatmaps.",
    )
    p.add_argument(
        "--grid-gammas",
        type=float,
        nargs="+",
        default=None,
        help="Optional gamma grid (x-axis) for rate×gamma heatmaps.",
    )
    p.add_argument(
        "--grid-samples",
        type=positive_int,
        default=64,
        help="Number of valid/invalid pairs per corruption rate for grid evaluation.",
    )
    return p


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


# ---------------------------------------------------------------------------
# Energy function
# ---------------------------------------------------------------------------


def compute_energy(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Dot-product energy: g(xγ) = xγ · f(xγ), returned per batch element (B,)."""
    x = x.detach().requires_grad_(False)
    with torch.no_grad():
        with sdpa_kernel(SDPBackend.MATH):
            v_pred, _ = model.forward(x)
    return (x * v_pred).sum(dim=(-2, -1))


def compute_grad_g(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (energy per sample (B,), grad_g) using dot-product formulation."""
    x = x.detach().requires_grad_(True)
    with sdpa_kernel(SDPBackend.MATH):
        v_pred, _ = model.forward(x)
    g = (x * v_pred).sum()
    grad_g = torch.autograd.grad(g, x)[0]
    energy_per_sample = (x.detach() * v_pred.detach()).sum(dim=(-2, -1))
    return energy_per_sample, grad_g.detach()


# ---------------------------------------------------------------------------
# AUROC helpers
# ---------------------------------------------------------------------------


def compute_auroc(positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> float:
    """Binary AUROC. positive_scores should be larger for the positive class."""
    pos = positive_scores.reshape(-1).float()
    neg = negative_scores.reshape(-1).float()
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).float().mean()
    eq = (pos[:, None] == neg[None, :]).float().mean()
    return (gt + 0.5 * eq).item()


def compute_roc_curve(
    positive_scores: torch.Tensor, negative_scores: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """ROC curve (FPR, TPR) where larger scores → positive class."""
    pos = positive_scores.reshape(-1).float()
    neg = negative_scores.reshape(-1).float()
    scores = torch.cat([pos, neg])
    labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])

    thresholds = torch.unique(scores).sort(descending=True).values
    thresholds = torch.cat(
        [
            torch.tensor([float("inf")]),
            thresholds,
            torch.tensor([float("-inf")]),
        ]
    )

    pos_count = labels.sum()
    neg_count = (1.0 - labels).sum()
    tprs, fprs = [], []
    for thr in thresholds:
        pred_pos = (scores >= thr).float()
        tp = (pred_pos * labels).sum()
        fp = (pred_pos * (1.0 - labels)).sum()
        tprs.append(tp / pos_count if pos_count > 0 else torch.tensor(0.0))
        fprs.append(fp / neg_count if neg_count > 0 else torch.tensor(0.0))

    return torch.stack(fprs).cpu(), torch.stack(tprs).cpu()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_valid_sequences(
    model: nn.Module,
    cfg: Config,
    n_needed: int,
    use_train_loader: bool,
    device: torch.device,
) -> torch.Tensor:
    """
    Load valid sequences from the validation loader if available, else training loader.
    Accumulates batches until n_needed samples are collected.
    """
    datamodule, _ = build_training_datamodule(cfg)

    loader = None
    if not use_train_loader:
        val_fn = getattr(datamodule, "val_dataloader", None)
        if callable(val_fn):
            try:
                loader = val_fn()
                print("Using validation dataloader.")
            except Exception as e:
                print(f"val_dataloader() failed ({e}), falling back to train loader.")
                loader = None

    if loader is None:
        loader = datamodule.train_dataloader()
        if not use_train_loader:
            print(
                "Warning: no validation loader found, using training data. Pass --use-train-loader to silence."
            )

    all_seqs = []
    collected = 0
    for batch in loader:
        batch = to_device(batch, str(device))
        batch = model.prepare_batch(batch)
        seqs = batch["log_x"]
        all_seqs.append(seqs)
        collected += seqs.shape[0]
        if collected >= n_needed:
            break

    sequences = torch.cat(all_seqs, dim=0)[:n_needed]
    print(f"Loaded {sequences.shape[0]} valid sequences (shape: {tuple(sequences.shape)}).")
    return sequences


def _load_valid_invalid_pairs(
    model: nn.Module,
    cfg: Config,
    n_needed: int,
    use_train_loader: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load matching valid/invalid feature pairs (requires ``log_x_invalid``)."""
    datamodule, _ = build_training_datamodule(cfg)

    loader = None
    if not use_train_loader:
        val_fn = getattr(datamodule, "val_dataloader", None)
        if callable(val_fn):
            try:
                loader = val_fn()
            except Exception:
                loader = None
    if loader is None:
        loader = datamodule.train_dataloader()

    all_valid: list[torch.Tensor] = []
    all_invalid: list[torch.Tensor] = []
    collected = 0
    for batch in loader:
        batch = to_device(batch, device)
        batch = model.prepare_batch(batch)
        if "log_x_invalid" not in batch:
            raise ValueError(
                "Datamodule batch is missing 'log_x_invalid'; "
                "rate×gamma grid requires a source that emits invalid examples."
            )
        valid = batch["log_x"]
        invalid = batch["log_x_invalid"]
        all_valid.append(valid)
        all_invalid.append(invalid)
        collected += valid.shape[0]
        if collected >= n_needed:
            break

    valid_out = torch.cat(all_valid, dim=0)[:n_needed]
    invalid_out = torch.cat(all_invalid, dim=0)[:n_needed]
    return valid_out, invalid_out


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
    For each trajectory, start from a lightly corrupted valid sequence and
    integrate toward gamma=1, tracking energy and gradient norm.
    Energy should decrease monotonically toward gamma=1.
    """
    model.eval()
    x1 = valid_batch[:n_trajectories].to(device)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_energy, ax_grad = axes

    gammas = torch.linspace(corrupt_gamma, 1.0, n_steps)
    step_size = (1.0 - corrupt_gamma) / n_steps

    for i in range(n_trajectories):
        x1_i = x1[i].unsqueeze(0)
        eps = torch.randn_like(x1_i)
        x_cur = (1 - corrupt_gamma) * eps + corrupt_gamma * x1_i

        energies, grad_norms = [], []
        for _ in tqdm(gammas, desc=f"trajectory {i + 1}/{n_trajectories}", leave=False):
            energy, grad_g = compute_grad_g(model, x_cur)
            energies.append(energy.item())
            grad_norms.append(grad_g.norm().item())
            x_cur = (x_cur - step_size * grad_g).detach()

        ax_energy.plot(gammas.cpu().numpy(), energies, alpha=0.7, label=f"seq {i + 1}")
        ax_grad.plot(gammas.cpu().numpy(), grad_norms, alpha=0.7, label=f"seq {i + 1}")

    ax_energy.set_xlabel("gamma (corrupt → data)")
    ax_energy.set_ylabel("energy g(xγ)")
    ax_energy.set_title(
        f"Energy along trajectory (start γ={corrupt_gamma})\n(should decrease toward γ=1)"
    )
    ax_energy.legend()
    ax_energy.grid(True)

    ax_grad.set_xlabel("gamma (corrupt → data)")
    ax_grad.set_ylabel("||∇g|| (gradient norm)")
    ax_grad.set_title("Gradient norm along trajectory\n(should shrink near data)")
    ax_grad.legend()
    ax_grad.grid(True)

    fig.tight_layout()
    path = out_dir / "energy_trajectories.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved trajectory plot → {path}")


# ---------------------------------------------------------------------------
# Multi-gamma AUROC sweep
# ---------------------------------------------------------------------------


def eval_gamma_sweep(
    model: nn.Module,
    valid_batch: torch.Tensor,
    device: torch.device,
    corrupt_gammas: list[float],
    n_samples: int,
    out_dir: Path,
):
    """
    Sweep over corruption gammas and compute AUROC at each level.
    Reveals at what corruption level the model starts failing to separate
    valid from corrupted sequences.
    """
    model.eval()
    x1 = valid_batch[:n_samples].to(device)
    energy_valid = compute_energy(model, x1).cpu()
    mean_valid = energy_valid.mean().item()

    aurocs, mean_corrupts, gaps = [], [], []
    gammas_sorted = sorted(corrupt_gammas, reverse=True)

    print("\n--- Multi-gamma AUROC Sweep ---")
    print(f"{'gamma':>8}  {'AUROC':>8}  {'μ_valid':>10}  {'μ_corrupt':>10}  {'Δ':>10}")
    print("-" * 55)

    # Collect all corrupt energies first (for ROC plot reuse)
    all_corrupt_energies = {}
    for gamma in gammas_sorted:
        eps = torch.randn_like(x1)
        x_corrupt = (1 - gamma) * eps + gamma * x1
        energy_corrupt = compute_energy(model, x_corrupt).cpu()
        all_corrupt_energies[gamma] = energy_corrupt

        auroc = compute_auroc(positive_scores=energy_corrupt, negative_scores=energy_valid)
        mean_corrupt = energy_corrupt.mean().item()
        gap = mean_valid - mean_corrupt

        aurocs.append(auroc)
        mean_corrupts.append(mean_corrupt)
        gaps.append(gap)
        print(
            f"{gamma:>8.2f}  {auroc:>8.4f}  {mean_valid:>10.2f}  {mean_corrupt:>10.2f}  {gap:>10.2f}"
        )

    # AUROC vs gamma + energy gap
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax_auroc = axes[0]
    ax_auroc.plot(gammas_sorted, aurocs, "o-", color="steelblue")
    ax_auroc.axhline(0.5, color="k", linestyle="--", linewidth=1, alpha=0.5, label="random")
    ax_auroc.axhline(1.0, color="green", linestyle="--", linewidth=1, alpha=0.5, label="perfect")
    ax_auroc.set_xlabel("corruption gamma (1.0 = no corruption)")
    ax_auroc.set_ylabel("AUROC")
    ax_auroc.set_title("AUROC vs corruption level\n(how subtle a corruption can the model detect?)")
    ax_auroc.set_ylim(0.4, 1.05)
    x_pad = (max(gammas_sorted) - min(gammas_sorted)) * 0.08 + 0.01
    ax_auroc.set_xlim(max(gammas_sorted) + x_pad, min(gammas_sorted) - x_pad)
    ax_auroc.legend()
    ax_auroc.grid(True)

    ax_gap = axes[1]
    ax_gap.plot(gammas_sorted, gaps, "o-", color="tomato")
    ax_gap.axhline(0, color="k", linestyle="--", linewidth=1, alpha=0.5)
    ax_gap.set_xlabel("corruption gamma (1.0 = no corruption)")
    ax_gap.set_ylabel("energy gap (μ_valid - μ_corrupt)")
    ax_gap.set_title("Energy gap vs corruption level\n(negative = valid has lower energy ✓)")
    ax_gap.set_xlim(max(gammas_sorted) + x_pad, min(gammas_sorted) - x_pad)
    ax_gap.grid(True)

    fig.tight_layout()
    path = out_dir / "gamma_sweep.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved gamma sweep → {path}")

    # ROC curves overlaid for all gammas
    fig, ax = plt.subplots(figsize=(7, 7))
    cmap = plt.cm.plasma
    colors = [cmap(i / max(len(gammas_sorted) - 1, 1)) for i in range(len(gammas_sorted))]

    for gamma, color, auroc in zip(gammas_sorted, colors, aurocs):
        energy_corrupt = all_corrupt_energies[gamma]
        fpr, tpr = compute_roc_curve(positive_scores=energy_corrupt, negative_scores=energy_valid)
        ax.plot(
            fpr.numpy(),
            tpr.numpy(),
            color=color,
            alpha=0.8,
            label=f"γ={gamma:.2f} (AUC={auroc:.3f})",
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="random")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC curves across corruption levels")
    ax.legend(fontsize=8)
    ax.grid(True)
    fig.tight_layout()
    path = out_dir / "roc_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved ROC curves → {path}")


# ---------------------------------------------------------------------------
# Single-gamma distribution plots
# ---------------------------------------------------------------------------


def eval_energy_distributions(
    model: nn.Module,
    valid_batch: torch.Tensor,
    device: torch.device,
    corrupt_gamma: float,
    n_samples: int,
    out_dir: Path,
):
    """Histogram and per-sample scatter of energy for valid vs corrupted at a single gamma."""
    model.eval()
    x1 = valid_batch[:n_samples].to(device)
    eps = torch.randn_like(x1)
    x_corrupt = (1 - corrupt_gamma) * eps + corrupt_gamma * x1

    energy_valid = compute_energy(model, x1).cpu()
    energy_corrupt = compute_energy(model, x_corrupt).cpu()

    print(f"\n--- Energy Distributions (γ={corrupt_gamma}) ---")
    print(f"Valid     | mean: {energy_valid.mean():.4f}  std: {energy_valid.std():.4f}")
    print(f"Corrupted | mean: {energy_corrupt.mean():.4f}  std: {energy_corrupt.std():.4f}")
    delta = energy_valid.mean() - energy_corrupt.mean()
    print(
        f"Δ (valid - corrupt): {delta:.4f}  {'✓ valid < corrupt' if delta < 0 else '✗ unexpected: valid >= corrupt'}"
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax_hist = axes[0]
    ax_hist.hist(
        energy_valid.numpy(),
        bins=30,
        alpha=0.6,
        label=f"valid (μ={energy_valid.mean():.2f})",
        color="steelblue",
    )
    ax_hist.hist(
        energy_corrupt.numpy(),
        bins=30,
        alpha=0.6,
        label=f"corrupted γ={corrupt_gamma} (μ={energy_corrupt.mean():.2f})",
        color="tomato",
    )
    ax_hist.set_xlabel("energy g(x)")
    ax_hist.set_ylabel("count")
    ax_hist.set_title(f"Energy distribution: valid vs corrupted (γ={corrupt_gamma})")
    ax_hist.legend()
    ax_hist.grid(True)

    ax_scatter = axes[1]
    idx = torch.arange(energy_valid.shape[0]).numpy()
    ax_scatter.scatter(idx, energy_valid.numpy(), alpha=0.7, s=20, color="steelblue", label="valid")
    ax_scatter.scatter(
        idx,
        energy_corrupt.numpy(),
        alpha=0.7,
        s=20,
        color="tomato",
        label=f"corrupted γ={corrupt_gamma}",
    )
    ax_scatter.set_xlabel("sample index")
    ax_scatter.set_ylabel("energy g(x)")
    ax_scatter.set_title("Per-sample energy: valid vs corrupted")
    ax_scatter.legend()
    ax_scatter.grid(True)

    fig.tight_layout()
    path = out_dir / "energy_distributions.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved distribution plots → {path}")


def eval_corruption_gamma_grid(
    model: nn.Module,
    base_cfg: Config,
    *,
    corrupt_rates: list[float],
    gammas: list[float],
    n_samples: int,
    use_train_loader: bool,
    device: torch.device,
    out_dir: Path,
) -> None:
    """Grid AUROC evaluation over corruption rate (rows) × gamma (cols)."""
    model.eval()
    rates = sorted(corrupt_rates)
    gammas_sorted = sorted(gammas, reverse=True)

    auroc_grid = torch.zeros((len(rates), len(gammas_sorted)), dtype=torch.float32)
    gap_grid = torch.zeros_like(auroc_grid)

    for i, rate in enumerate(rates):
        cfg_rate = deepcopy(base_cfg)
        # Route the corruption-rate override to the active datasource family.
        source = cfg_rate.training_data.source
        if source == "llm_topk_probs":
            cfg_rate.llm_topk_probs.corrupt_rate = rate
        elif source == "llm_topk":
            cfg_rate.llm_embedding_dataset.corrupt_rate = rate
        elif source in ("dna",):
            cfg_rate.dna_dataset.train_corrupt_rate = rate
            cfg_rate.dna_dataset.eval_corrupt_rate = rate
        elif source in ("medical",):
            cfg_rate.medical_dataset.train_corrupt_rate = rate
            cfg_rate.medical_dataset.eval_corrupt_rate = rate
        else:
            cfg_rate.text8_dataset.train_corrupt_rate = rate
            cfg_rate.text8_dataset.eval_corrupt_rate = rate

        valid, invalid = _load_valid_invalid_pairs(
            model=model,
            cfg=cfg_rate,
            n_needed=n_samples,
            use_train_loader=use_train_loader,
            device=device,
        )
        x_valid = valid.to(device)
        x_invalid = invalid.to(device)
        e_valid = compute_energy(model, x_valid).cpu()
        mu_valid = e_valid.mean().item()

        for j, gamma in enumerate(gammas_sorted):
            x_mix = gamma * x_valid + (1.0 - gamma) * x_invalid
            e_mix = compute_energy(model, x_mix).cpu()
            auroc_grid[i, j] = compute_auroc(positive_scores=e_mix, negative_scores=e_valid)
            gap_grid[i, j] = mu_valid - e_mix.mean().item()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax_auc, ax_gap = axes

    im_auc = ax_auc.imshow(auroc_grid.numpy(), aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax_auc.set_title("AUROC over corruption_rate × gamma")
    ax_auc.set_xlabel("gamma (1.0 = clean)")
    ax_auc.set_ylabel("corruption rate")
    ax_auc.set_xticks(range(len(gammas_sorted)))
    ax_auc.set_xticklabels([f"{g:.2f}" for g in gammas_sorted], rotation=45, ha="right")
    ax_auc.set_yticks(range(len(rates)))
    ax_auc.set_yticklabels([f"{r:.2f}" for r in rates])
    fig.colorbar(im_auc, ax=ax_auc, fraction=0.046, pad=0.04, label="AUROC")

    # Annotate each heatmap cell with its AUROC value for quick readability.
    auc_np = auroc_grid.numpy()
    for yi in range(len(rates)):
        for xi in range(len(gammas_sorted)):
            v = float(auc_np[yi, xi])
            txt_color = "black" if v >= 0.70 else "white"
            ax_auc.text(
                xi,
                yi,
                f"{v:.2f}",
                ha="center",
                va="center",
                color=txt_color,
                fontsize=8,
                fontweight="bold",
            )

    # Overlay the AUROC=0.8 contour as a practical "strong separation" frontier.
    if len(rates) >= 2 and len(gammas_sorted) >= 2:
        x = torch.arange(len(gammas_sorted), dtype=torch.float32).numpy()
        y = torch.arange(len(rates), dtype=torch.float32).numpy()
        cs = ax_auc.contour(
            x,
            y,
            auc_np,
            levels=[0.8],
            colors=["white"],
            linewidths=1.8,
        )
        if cs.allsegs and any(len(seg) > 0 for seg in cs.allsegs):
            ax_auc.clabel(cs, fmt={0.8: "AUC=0.80"}, inline=True, fontsize=8)

    vmax = float(torch.max(torch.abs(gap_grid)).item()) if gap_grid.numel() else 1.0
    vmax = max(vmax, 1e-6)
    im_gap = ax_gap.imshow(
        gap_grid.numpy(),
        aspect="auto",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )
    ax_gap.set_title("Energy gap μ_valid - μ_corrupt")
    ax_gap.set_xlabel("gamma (1.0 = clean)")
    ax_gap.set_ylabel("corruption rate")
    ax_gap.set_xticks(range(len(gammas_sorted)))
    ax_gap.set_xticklabels([f"{g:.2f}" for g in gammas_sorted], rotation=45, ha="right")
    ax_gap.set_yticks(range(len(rates)))
    ax_gap.set_yticklabels([f"{r:.2f}" for r in rates])
    fig.colorbar(im_gap, ax=ax_gap, fraction=0.046, pad=0.04, label="Δ energy")

    fig.tight_layout()
    out_path = out_dir / "corruption_gamma_grid.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved corruption×gamma grid → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None):
    args = _build_argparser().parse_args(argv)
    seed_all(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    apply_training_data_args(cfg, args)
    _apply_llm_topk_probs_overrides(cfg, args)
    device = torch.device(cfg.training.device)

    cfg_eval = deepcopy(cfg)
    cfg_eval.training = replace(cfg_eval.training, model_name="bayesian_auditor_stage1")

    model = build_model(cfg_eval)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    n_needed = max(args.n_trajectories, args.n_score_samples)
    valid_sequences = _load_valid_sequences(
        model=model,
        cfg=cfg,
        n_needed=n_needed,
        use_train_loader=args.use_train_loader,
        device=device,
    )

    print(
        f"\nRunning trajectory evaluation ({args.n_trajectories} trajectories, "
        f"{args.n_steps} steps, γ={args.trajectory_gamma})..."
    )
    eval_trajectories(
        model=model,
        valid_batch=valid_sequences,
        device=device,
        n_trajectories=args.n_trajectories,
        n_steps=args.n_steps,
        corrupt_gamma=args.trajectory_gamma,
        out_dir=out_dir,
    )

    print(f"\nRunning gamma sweep over {args.corrupt_gammas}...")
    eval_gamma_sweep(
        model=model,
        valid_batch=valid_sequences,
        device=device,
        corrupt_gammas=args.corrupt_gammas,
        n_samples=args.n_score_samples,
        out_dir=out_dir,
    )

    eval_energy_distributions(
        model=model,
        valid_batch=valid_sequences,
        device=device,
        corrupt_gamma=args.trajectory_gamma,
        n_samples=args.n_score_samples,
        out_dir=out_dir,
    )

    if args.grid_corrupt_rates and args.grid_gammas:
        print(
            f"\nRunning corruption×gamma grid (rates={args.grid_corrupt_rates}, "
            f"gammas={args.grid_gammas}, n={args.grid_samples})..."
        )
        eval_corruption_gamma_grid(
            model=model,
            base_cfg=cfg,
            corrupt_rates=args.grid_corrupt_rates,
            gammas=args.grid_gammas,
            n_samples=args.grid_samples,
            use_train_loader=args.use_train_loader,
            device=device,
            out_dir=out_dir,
        )
    elif args.grid_corrupt_rates or args.grid_gammas:
        print("Skipping corruption×gamma grid: pass both --grid-corrupt-rates and --grid-gammas.")

    print("\nDone. All plots saved to:", out_dir)


if __name__ == "__main__":
    main()
