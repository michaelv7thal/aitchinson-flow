"""EqM healing demo: scrambled text recovery + energy as uncertainty.

Two figures are produced:

Figure 1 — Healing heatmap
  For a few example sequences, shows three stages side by side:
    - Original   : real text8 characters
    - Scrambled  : random characters injected at some positions
    - Recovered  : after running NAG-GD
  Each cell is coloured by the per-position gradient norm (∥∇E∥_i).
  Green  = low gradient  = model is confident = near data manifold
  Red    = high gradient = model is uncertain = far from data manifold

Figure 2 — Energy trajectory
  Max gradient norm across all positions as a function of NAG-GD step,
  for each scramble rate. Shows convergence and that the model "heals"
  more corrupted sequences more slowly.

Usage
-----
    python scripts/eval_healing_demo.py \\
        --checkpoint checkpoints/eqm/best.pt \\
        --scramble-rates 0.0 0.2 0.5 \\
        --n-steps 200 \\
        --n-examples 4 \\
        --output-dir plots/healing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def _bootstrap(anchor: Path) -> None:
    repo_root = anchor.resolve().parent.parent
    src = repo_root / "src"
    for p in (str(src), str(repo_root)):
        if p not in sys.path:
            sys.path.insert(0, p)

_bootstrap(Path(__file__))

import aitchinson_flow.models  # noqa: F401
from aitchinson_flow.config import Config
from aitchinson_flow.training.checkpoint import load_checkpoint
from aitchinson_flow.models import build_model, EquilibriumFlowMatching
from aitchinson_flow.data.transforms import token_ids_to_features
from aitchinson_flow.data.char_window_dataset import CHAR2ID

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ID2CHAR: dict[int, str] = {v: k for k, v in CHAR2ID.items()}


def ids_to_str(token_ids: torch.Tensor) -> str:
    return "".join(ID2CHAR.get(i.item(), "?") for i in token_ids)


def scramble(token_ids: torch.Tensor, rate: float, K: int = 27) -> torch.Tensor:
    """Replace `rate` fraction of positions with a uniform random character."""
    mask = torch.rand_like(token_ids.float()) < rate
    noise = torch.randint(0, K, token_ids.shape, device=token_ids.device)
    return torch.where(mask, noise, token_ids)


# ---------------------------------------------------------------------------
# Core: run NAG-GD and record trajectory
# ---------------------------------------------------------------------------

def heal_and_record(
    model: EquilibriumFlowMatching,
    token_ids: torch.Tensor,        # (B, L) clean token IDs
    scramble_rate: float,
    n_steps: int,
    record_every: int = 5,
) -> dict:
    """Scramble → CLR encode → NAG-GD → decode, recording energy along the way."""
    cfg = model.cfg
    K = cfg.text8_dataset.K
    L = cfg.text8_dataset.L
    ls = cfg.transformation.label_smoothing
    device = next(model.parameters()).device

    token_ids = token_ids.to(device)
    scrambled_ids = scramble(token_ids, scramble_rate, K)

    # Convert scrambled tokens to CLR features as the starting point
    x = token_ids_to_features(scrambled_ids, K, label_smoothing=ls)
    x = x + 0.05 * torch.randn_like(x)
    x = x - x.mean(dim=-1, keepdim=True)

    # NAG-GD with trajectory recording
    eta = cfg.eqm.sample_eta
    mu  = cfg.eqm.sample_mu

    grad_norms: list[float] = []       # max grad norm per step (convergence)
    uncertainty_start: torch.Tensor   # per-position grad norm at step 0
    uncertainty_end:   torch.Tensor   # per-position grad norm at final step

    x_last = x.clone()
    grad = model(x)

    uncertainty_start = model.position_uncertainty(x).cpu()

    for step in range(n_steps):
        g_max = grad.reshape(x.shape[0], -1).norm(dim=-1).max().item()
        if step % record_every == 0:
            grad_norms.append(g_max)
        if g_max < cfg.eqm.sample_g_min:
            break
        x_last = x
        x = x - eta * grad
        grad = model(x + mu * (x - x_last))

    uncertainty_end = model.position_uncertainty(x).cpu()

    # Decode
    log_probs = model.decode_to_logprobs(x)
    recovered_ids = log_probs.argmax(dim=-1).cpu()

    return {
        "original_ids":    token_ids.cpu(),
        "scrambled_ids":   scrambled_ids.cpu(),
        "recovered_ids":   recovered_ids,
        "uncertainty_start": uncertainty_start,   # (B, L)
        "uncertainty_end":   uncertainty_end,     # (B, L)
        "grad_norms":      grad_norms,
        "record_every":    record_every,
    }


# ---------------------------------------------------------------------------
# Figure 1: healing heatmap
# ---------------------------------------------------------------------------

def _uncertainty_color(u: torch.Tensor) -> np.ndarray:
    """Map per-position uncertainty (B, L) → RGBA image (B, L, 4)."""
    u_np = u.numpy().astype(float)
    u_norm = np.clip(u_np / (u_np.max() + 1e-8), 0, 1)
    cmap = plt.get_cmap("RdYlGn_r")
    return cmap(u_norm)


def plot_healing_heatmap(
    results: list[dict],          # one per scramble rate
    scramble_rates: list[float],
    n_examples: int,
    out_path: Path,
) -> None:
    """
    Grid: rows = (scramble rates × examples), columns = (original | scrambled | recovered)
    Cell background = uncertainty colour.
    """
    n_rates = len(results)
    # One row per example per scramble rate; 3 columns (orig / scrambled / recovered)
    n_rows = n_rates * n_examples
    fig, axes = plt.subplots(
        n_rows, 3,
        figsize=(18, 1.4 * n_rows + 0.6),
        gridspec_kw={"hspace": 0.05, "wspace": 0.05},
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Original", "Scrambled", "Recovered"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, pad=4)

    cmap = plt.get_cmap("RdYlGn_r")
    norm = mcolors.Normalize(vmin=0, vmax=1)

    for ri, (rate, res) in enumerate(zip(scramble_rates, results)):
        orig      = res["original_ids"]     # (B, L)
        scram     = res["scrambled_ids"]
        rec       = res["recovered_ids"]
        u_start   = res["uncertainty_start"]  # (B, L)
        u_end     = res["uncertainty_end"]

        # Normalise uncertainty across this result for consistent colours
        u_all = torch.cat([u_start, u_end], dim=0)
        u_max = u_all.max().item() + 1e-8

        for ei in range(min(n_examples, orig.shape[0])):
            row = ri * n_examples + ei
            ax_orig, ax_scr, ax_rec = axes[row, 0], axes[row, 1], axes[row, 2]

            orig_str = ids_to_str(orig[ei])
            scram_str = ids_to_str(scram[ei])
            rec_str  = ids_to_str(rec[ei])

            u_s_norm = (u_start[ei] / u_max).clamp(0, 1).numpy()  # (L,)
            u_e_norm = (u_end[ei]   / u_max).clamp(0, 1).numpy()

            for ax, text, u_norm, label_prefix in (
                (ax_orig, orig_str,  np.zeros_like(u_s_norm), ""),
                (ax_scr,  scram_str, u_s_norm,                ""),
                (ax_rec,  rec_str,   u_e_norm,                ""),
            ):
                ax.set_xlim(0, len(text))
                ax.set_ylim(0, 1)
                ax.axis("off")

                for ci, (ch, u_val) in enumerate(zip(text, u_norm)):
                    colour = cmap(u_val)
                    ax.add_patch(
                        plt.Rectangle((ci, 0.05), 0.92, 0.9,
                                      facecolor=colour, edgecolor="none")
                    )
                    brightness = 0.299*colour[0] + 0.587*colour[1] + 0.114*colour[2]
                    fg = "black" if brightness > 0.55 else "white"
                    ax.text(ci + 0.46, 0.5, ch,
                            ha="center", va="center",
                            fontsize=8, fontfamily="monospace",
                            color=fg)

            # Row label
            if ei == 0:
                ax_orig.set_ylabel(
                    f"scramble={int(rate*100)}%", fontsize=9,
                    rotation=0, labelpad=60, va="center"
                )
                ax_orig.yaxis.set_visible(True)

    # Colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical",
                        fraction=0.01, pad=0.01, shrink=0.6)
    cbar.set_label("∥∇E∥ (normalised)\ngreen=confident  red=uncertain", fontsize=9)

    fig.suptitle("EqM Healing Demo — character uncertainty (∥∇E∥ per position)", fontsize=12, y=1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Heatmap saved to {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: convergence / energy trajectory
# ---------------------------------------------------------------------------

def plot_energy_trajectory(
    results: list[dict],
    scramble_rates: list[float],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    cmap = plt.get_cmap("viridis")
    colours = [cmap(i / max(len(scramble_rates) - 1, 1)) for i in range(len(scramble_rates))]

    for rate, res, colour in zip(scramble_rates, results, colours):
        every = res["record_every"]
        steps = [i * every for i in range(len(res["grad_norms"]))]
        ax.semilogy(steps, res["grad_norms"], label=f"scramble={int(rate*100)}%",
                    color=colour, linewidth=2)

    ax.axhline(y=results[0]["grad_norms"][-1] if results else 1e-3,
               color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("NAG-GD step")
    ax.set_ylabel("max ∥∇E∥  (log scale)")
    ax.set_title("Energy convergence during healing\n(lower = closer to data manifold)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Trajectory saved to {out_path}")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def _load_examples(cfg: Config, n: int, device: torch.device) -> torch.Tensor:
    from aitchinson_flow.training import build_training_datamodule
    dm, _ = build_training_datamodule(cfg)
    loader = dm.val_dataloader()
    if loader is None:
        raise RuntimeError("No validation dataloader available.")
    for batch in loader:
        return batch["token_ids"][:n].to(device)
    raise RuntimeError("Empty dataloader.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--scramble-rates", type=float, nargs="+",
                   default=[0.0, 0.2, 0.5], metavar="R")
    p.add_argument("--n-steps",    type=int, default=200)
    p.add_argument("--n-examples", type=int, default=4)
    p.add_argument("--output-dir", type=str, default="plots/healing")
    p.add_argument("--device",     type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    cfg = Config()
    cfg.training.model_name = "EqM"

    print(f"Loading EqM from {args.checkpoint}")
    model = build_model(cfg)
    load_checkpoint(args.checkpoint, model=model, map_location=device)
    model.to(device).eval()
    assert isinstance(model, EquilibriumFlowMatching)

    token_ids = _load_examples(cfg, args.n_examples, device)
    print(f"Loaded {token_ids.shape[0]} examples, L={token_ids.shape[1]}")

    results = []
    for rate in args.scramble_rates:
        print(f"Running healing at scramble_rate={rate:.0%} for {args.n_steps} steps...")
        res = heal_and_record(
            model, token_ids,
            scramble_rate=rate,
            n_steps=args.n_steps,
        )
        results.append(res)

        # Print a few examples to console
        for i in range(min(2, token_ids.shape[0])):
            print(f"  orig:      {ids_to_str(res['original_ids'][i])}")
            print(f"  scrambled: {ids_to_str(res['scrambled_ids'][i])}")
            print(f"  recovered: {ids_to_str(res['recovered_ids'][i])}")
            print()

    out_dir = Path(args.output_dir)
    plot_healing_heatmap(
        results, args.scramble_rates, args.n_examples,
        out_dir / "healing_heatmap.png",
    )
    plot_energy_trajectory(
        results, args.scramble_rates,
        out_dir / "energy_trajectory.png",
    )


if __name__ == "__main__":
    main()
