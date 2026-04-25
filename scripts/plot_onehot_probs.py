"""Plot one-hot simplex probabilities per token for one valid and one invalid sequence.

The raw-text (Path A) pipeline encodes each character as a near-one-hot
distribution over the 27-char text8 vocabulary (a=0 … z=25, space=26),
then applies label smoothing (or a small additive eps) before the ILR transform.

This script shows the same three plots as ``plot_topk_probs.py`` but for the
one-hot case so you can compare how peaked/flat each distribution is before and
after corruption.

    Valid token   → one-hot spike on the true char  → entropy ≈ 0
    Corrupted tok → one-hot spike on a random char  → same entropy, different position

The signal in the one-hot case is therefore about *which* character is active,
not the shape of the distribution.  Both plots are produced for comparison.

Usage:
    python scripts/plot_onehot_probs.py --out-dir plots/onehot_inspect

Optional flags:
    --seq-length L       Number of character positions to show (default: 30)
    --corrupt-rate float Fraction of positions corrupted (default: 0.15)
    --window-idx N       Which text8 window to use (default: 0)
    --token-pos N        Position to zoom in on in the bar chart (default: 0)
    --label-smoothing α  Label-smoothing mix with uniform (default: 0.0 → eps path)
    --eps float          Additive constant before log when label_smoothing=0
    --seed int           RNG seed for corruption (default: 42)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

from _shared.bootstrap import bootstrap_repo_paths

bootstrap_repo_paths(Path(__file__))

from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.data.corruption import corrupt_token_ids  # noqa: E402
from aitchinson_flow.data.text8_datamodule import (  # noqa: E402
    VOCAB_SIZE as TEXT8_VOCAB_SIZE,
    _ALPHABET,
    _load_text8_splits_cfg,
)
from aitchinson_flow.data.transforms.discrete import _smoothed_log_row  # noqa: E402


# ---------------------------------------------------------------------------
# Core: build simplex probability rows from char-id sequence
# ---------------------------------------------------------------------------

def _onehot_probs(
    token_ids: torch.Tensor,
    K: int,
    *,
    eps: float,
    label_smoothing: float,
) -> np.ndarray:
    """Return (L, K) ndarray of simplex probabilities for a 1-D token_ids tensor."""
    log_rows = _smoothed_log_row(
        token_ids,
        K,
        eps=eps,
        label_smoothing=label_smoothing,
    )
    probs = log_rows.exp()
    # Normalise so rows sum exactly to 1 for plotting.
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs.numpy()


# ---------------------------------------------------------------------------
# Shared plot helpers (same structure as plot_topk_probs.py)
# ---------------------------------------------------------------------------

def _heatmap(
    ax: plt.Axes,
    probs: np.ndarray,
    title: str,
    token_strs: list[str],
) -> plt.cm.ScalarMappable:
    im = ax.imshow(probs, aspect="auto", origin="upper", cmap="plasma", vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Character (vocabulary index)", fontsize=8)
    ax.set_ylabel("Token position", fontsize=8)
    ax.set_yticks(range(len(token_strs)))
    ax.set_yticklabels(
        [f"{i}: '{t}'" for i, t in enumerate(token_strs)],
        fontsize=6,
    )
    ax.set_xticks(range(probs.shape[1]))
    ax.set_xticklabels(list(_ALPHABET), fontsize=5)
    return im


def _bar_chart(
    ax: plt.Axes,
    valid_probs: np.ndarray,
    invalid_probs: np.ndarray,
    pos: int,
    char: str,
) -> None:
    K = valid_probs.shape[0]
    x = np.arange(K)
    width = 0.4
    ax.bar(x - width / 2, valid_probs, width, label="valid", color="#1f77b4", alpha=0.85)
    ax.bar(x + width / 2, invalid_probs, width, label="invalid", color="#d62728", alpha=0.85)
    ax.set_title(
        f"Simplex distribution at position {pos} (char: {char!r})",
        fontsize=9,
        fontweight="bold",
    )
    ax.set_xlabel("Character (vocabulary index)", fontsize=8)
    ax.set_ylabel("Probability", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(list(_ALPHABET), fontsize=5)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=7)


def _entropy_plot(
    ax: plt.Axes,
    valid_probs: np.ndarray,
    invalid_probs: np.ndarray,
    token_strs: list[str],
) -> None:
    def _ent(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-30, 1.0)
        return -(p * np.log(p)).sum(axis=-1)

    L = valid_probs.shape[0]
    positions = np.arange(L)
    ax.plot(positions, _ent(valid_probs), "o-", color="#1f77b4", label="valid", lw=1.2, ms=4)
    ax.plot(positions, _ent(invalid_probs), "s--", color="#d62728", label="invalid", lw=1.2, ms=4)
    ax.set_title("Per-token entropy of simplex distribution", fontsize=9, fontweight="bold")
    ax.set_xlabel("Token position", fontsize=8)
    ax.set_ylabel("Entropy (nats)", fontsize=8)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{i}:'{t}'" for i, t in enumerate(token_strs)],
        rotation=60,
        ha="right",
        fontsize=6,
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=str, default="plots/onehot_inspect")
    p.add_argument("--seq-length", type=int, default=30)
    p.add_argument("--corrupt-rate", type=float, default=0.15)
    p.add_argument("--window-idx", type=int, default=0)
    p.add_argument("--token-pos", type=int, default=0)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    K = TEXT8_VOCAB_SIZE  # 27
    L = args.seq_length

    cfg = Config()
    cfg.dataset.K = K
    cfg.dataset.L = L

    print(f"[info] text8 vocab size K={K}, alphabet: {_ALPHABET!r}")
    print(f"[info] label_smoothing={args.label_smoothing}, eps={args.eps}")

    # --- load one text8 window (each window has L characters = L tokens here) ---
    print("[info] loading text8 train split ...")
    train_windows, _, _ = _load_text8_splits_cfg(cfg, cfg.text8_dataset.cache_dir, L)
    total_windows = train_windows.shape[0]
    idx = args.window_idx % total_windows
    token_ids = train_windows[idx]  # (L,) long, values in 0..26
    token_strs = [_ALPHABET[int(t)] for t in token_ids]
    raw_text = "".join(token_strs)
    print(f"[info] window #{idx}/{total_windows}: {raw_text!r}")

    # --- corrupt ---
    corrupt_ids = corrupt_token_ids(
        token_ids.unsqueeze(0),
        vocab_size=K,
        corrupt_rate=args.corrupt_rate,
        seed=args.seed,
    ).squeeze(0)
    changed = (corrupt_ids != token_ids)
    changed_positions = changed.nonzero(as_tuple=False).squeeze(-1).tolist()
    corrupt_strs = [_ALPHABET[int(t)] for t in corrupt_ids]
    print(f"[info] corrupted positions ({len(changed_positions)}): {changed_positions}")
    print(f"[info] corrupted text: {''.join(corrupt_strs)!r}")

    # --- build simplex probs ---
    vp = _onehot_probs(token_ids, K, eps=args.eps, label_smoothing=args.label_smoothing)
    ip = _onehot_probs(corrupt_ids, K, eps=args.eps, label_smoothing=args.label_smoothing)

    token_pos = max(0, min(args.token_pos, L - 1))

    # --- Figure 1: heatmaps + bar chart ---
    fig1, axes1 = plt.subplots(1, 3, figsize=(20, max(6, L * 0.35)))
    fig1.suptitle(
        f"One-hot simplex probs (K={K}) | window #{idx} | corrupt_rate={args.corrupt_rate} | "
        f"label_smoothing={args.label_smoothing}",
        fontsize=11,
        fontweight="bold",
    )

    im = _heatmap(axes1[0], vp, "Valid sequence", token_strs)
    _heatmap(axes1[1], ip, "Invalid sequence (corrupted)", corrupt_strs)

    for ax in axes1[:2]:
        for pos in changed_positions:
            ax.axhline(pos - 0.5, color="cyan", lw=0.8, alpha=0.7)
            ax.axhline(pos + 0.5, color="cyan", lw=0.8, alpha=0.7)

    _bar_chart(
        axes1[2],
        vp[token_pos],
        ip[token_pos],
        token_pos,
        token_strs[token_pos],
    )

    cbar = fig1.colorbar(im, ax=axes1[:2], orientation="vertical", fraction=0.02, pad=0.01)
    cbar.set_label("Probability", fontsize=7)

    fig1.tight_layout(rect=[0, 0, 1, 0.95])
    out1 = out_dir / "heatmap_and_bar.png"
    fig1.savefig(out1, dpi=140, bbox_inches="tight")
    plt.close(fig1)
    print(f"[info] wrote {out1}")

    # --- Figure 2: per-token entropy ---
    fig2, ax2 = plt.subplots(figsize=(max(10, L * 0.5), 4))
    _entropy_plot(ax2, vp, ip, token_strs)
    fig2.tight_layout()
    out2 = out_dir / "entropy_per_token.png"
    fig2.savefig(out2, dpi=140, bbox_inches="tight")
    plt.close(fig2)
    print(f"[info] wrote {out2}")

    # --- Figure 3: |valid - invalid| difference heatmap ---
    diff = np.abs(vp - ip)
    fig3, ax3 = plt.subplots(figsize=(10, max(4, L * 0.3)))
    im3 = ax3.imshow(diff, aspect="auto", origin="upper", cmap="viridis")
    ax3.set_title("|valid probs − invalid probs| per position/char", fontsize=9, fontweight="bold")
    ax3.set_xlabel("Character (vocabulary index)", fontsize=8)
    ax3.set_ylabel("Token position", fontsize=8)
    ax3.set_yticks(range(L))
    ax3.set_yticklabels([f"{i}: '{t}'" for i, t in enumerate(token_strs)], fontsize=6)
    ax3.set_xticks(range(K))
    ax3.set_xticklabels(list(_ALPHABET), fontsize=5)
    for pos in changed_positions:
        ax3.axhline(pos - 0.5, color="red", lw=0.8, alpha=0.7)
        ax3.axhline(pos + 0.5, color="red", lw=0.8, alpha=0.7)
    fig3.colorbar(im3, ax=ax3, fraction=0.03, pad=0.01)
    fig3.tight_layout()
    out3 = out_dir / "diff_heatmap.png"
    fig3.savefig(out3, dpi=140, bbox_inches="tight")
    plt.close(fig3)
    print(f"[info] wrote {out3}")

    print("\nDone. Outputs:")
    for f in [out1, out2, out3]:
        print(f"  {f}")


if __name__ == "__main__":
    main()
