"""Multi-signal token heatmaps: overlay all three UQ components on the same axis.

Each figure shows the same set of sequences across three stacked heatmap rows:
  Row 1 — Structural energy   (Component 1 GP mean)
  Row 2 — Contextual energy   (Component 2 GP mean)
  Row 3 — Spilled energy      (Component 3, shape (N, L-1))

Token positions are aligned along the x-axis.  Spilled energy is padded to
length L by prepending NaN at position 0 (no preceding logit is available for
the first token), so all three signals share the same column grid.

Corruption-mask overlays (cyan ×) are applied uniformly to all three rows when
a ``corrupt_mask`` of shape ``(N, L)`` is supplied.

Public API
----------
plot_three_signal_heatmap
    Main entry point — save a stacked PNG.
plot_three_signal_lines
    Overlay line plots of all three signals for a single sequence.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _as_2d(arr: np.ndarray | None, name: str) -> np.ndarray | None:
    if arr is None:
        return None
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        a = a[np.newaxis, :]
    if a.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D, got shape {arr.shape}")
    return a


def _pad_spilled(spilled: np.ndarray, target_len: int) -> np.ndarray:
    """Prepend NaN column so spilled energy aligns with token positions."""
    n, sl = spilled.shape
    if sl == target_len:
        return spilled
    if sl == target_len - 1:
        pad = np.full((n, 1), np.nan)
        return np.concatenate([pad, spilled], axis=1)
    return spilled  # mismatched — pass through; caller sees truncated display


def _overlay_mask(ax: plt.Axes, mask: np.ndarray, n: int) -> None:
    """Draw cyan × at corruption positions on the given axes."""
    mask_slice = mask[:n].astype(bool)
    rows, cols = np.where(mask_slice)
    if rows.size:
        ax.scatter(
            cols, rows, marker="x", s=18, color="cyan",
            linewidths=0.8, zorder=3, label="corrupted",
        )


# ---------------------------------------------------------------------------
# Main plot functions
# ---------------------------------------------------------------------------


def plot_three_signal_heatmap(
    structural_energy: np.ndarray | None,
    contextual_energy: np.ndarray | None,
    spilled_energy: np.ndarray | None,
    out_path: Path | str,
    *,
    n_samples: int = 32,
    corrupt_mask: np.ndarray | None = None,
    title: str = "three-component UQ signals",
    cmaps: tuple[str, str, str] = ("magma", "magma", "plasma"),
) -> None:
    """Stacked heatmap of structural energy, contextual energy, and spilled energy.

    Args:
        structural_energy: Component 1 GP mean, shape ``(N, L)`` or ``(L,)``.
        contextual_energy: Component 2 GP mean, shape ``(N, L)`` or ``(L,)``.
        spilled_energy: Component 3 scores, shape ``(N, L-1)`` or ``(N, L)``.
        out_path: Output PNG path.
        n_samples: Maximum number of sequences to render.
        corrupt_mask: Boolean mask shape ``(N, L)`` with True at corrupted
            positions — rendered as cyan × overlays on every row.
        title: Figure suptitle.
        cmaps: Three matplotlib colormaps for (structural, contextual, spilled).
    """
    se = _as_2d(structural_energy, "structural_energy")
    ce = _as_2d(contextual_energy, "contextual_energy")
    spe = _as_2d(spilled_energy, "spilled_energy")

    available = [(name, arr, cmap) for name, arr, cmap in [
        ("structural energy (C1)", se, cmaps[0]),
        ("contextual energy (C2)", ce, cmaps[1]),
        ("spilled energy (C3)", spe, cmaps[2]),
    ] if arr is not None]

    if not available:
        return

    # Determine common sequence count and token length
    n = min(n_samples, min(arr.shape[0] for _, arr, _ in available))
    target_len = max(arr.shape[1] for _, arr, _ in available)

    # Pad spilled energy to match token length
    padded: list[tuple[str, np.ndarray, str]] = []
    for name, arr, cmap in available:
        if "spilled" in name:
            arr = _pad_spilled(arr, target_len)
        padded.append((name, arr[:n], cmap))

    cmask = _as_2d(corrupt_mask, "corrupt_mask")[:n] if corrupt_mask is not None else None

    n_rows = len(padded)
    height_per_row = max(2.0, n * 0.18)
    fig = plt.figure(figsize=(max(6, target_len * 0.2), height_per_row * n_rows + 1.2))
    gs = gridspec.GridSpec(n_rows, 1, hspace=0.4)

    for row_idx, (name, arr, cmap) in enumerate(padded):
        ax = fig.add_subplot(gs[row_idx])
        im = ax.imshow(arr, aspect="auto", cmap=cmap, interpolation="nearest")
        ax.set_ylabel("sequence")
        ax.set_title(name, fontsize=9)
        if row_idx == n_rows - 1:
            ax.set_xlabel("token position")
        else:
            ax.set_xticklabels([])
        cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
        cbar.ax.tick_params(labelsize=7)
        if cmask is not None:
            _overlay_mask(ax, cmask, n)
            if row_idx == 0:
                ax.legend(loc="upper right", fontsize=7)

    fig.suptitle(title, fontsize=10, y=0.98)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_three_signal_lines(
    structural_energy: np.ndarray | None,
    contextual_energy: np.ndarray | None,
    spilled_energy: np.ndarray | None,
    out_path: Path | str,
    *,
    sequence_index: int = 0,
    corrupt_mask: np.ndarray | None = None,
    tokens: list[str] | None = None,
    title: str | None = None,
) -> None:
    """Line plot of all three UQ signals for a single sequence.

    Overlays structural energy (C1), contextual energy (C2), and spilled
    energy (C3) on the same x-axis so signal alignment is visible.  Spilled
    energy is plotted at position i+1 (where it is defined) with a dashed line.

    Args:
        structural_energy: Shape ``(N, L)`` or ``(L,)``; uses row
            ``sequence_index``.
        contextual_energy: Shape ``(N, L)`` or ``(L,)``; same indexing.
        spilled_energy: Shape ``(N, L-1)`` or ``(L-1,)``; plotted at
            positions ``1..L-1``.
        out_path: Output PNG path.
        sequence_index: Which sequence (row) to render.
        corrupt_mask: Boolean mask shape ``(N, L)``; corrupted positions are
            highlighted as vertical red spans.
        tokens: Optional list of ``L`` token strings for x-tick labels.
        title: Figure title (auto-generated if None).
    """
    def _row(arr: np.ndarray | None, idx: int) -> np.ndarray | None:
        if arr is None:
            return None
        a = np.asarray(arr, dtype=np.float64)
        if a.ndim == 1:
            return a
        if a.ndim == 2 and idx < a.shape[0]:
            return a[idx]
        return None

    se = _row(structural_energy, sequence_index)
    ce = _row(contextual_energy, sequence_index)
    spe = _row(spilled_energy, sequence_index)

    if se is None and ce is None and spe is None:
        return

    L = max(
        len(x) for x in [se, ce] if x is not None
    ) if any(x is not None for x in [se, ce]) else (len(spe) + 1 if spe is not None else 1)
    xs = np.arange(L)

    fig, ax = plt.subplots(figsize=(max(6, L * 0.25), 4))

    if se is not None:
        ax.plot(xs[: len(se)], se, "-o", markersize=3, color="C0", label="structural energy (C1)")
    if ce is not None:
        ax.plot(xs[: len(ce)], ce, "-s", markersize=3, color="C2", label="contextual energy (C2)")
    if spe is not None:
        # spilled is defined at positions 1..L (between tokens), plot at i+1
        spe_xs = np.arange(1, len(spe) + 1)
        ax.plot(spe_xs, spe, "--^", markersize=3, color="C3", label="spilled energy (C3)", alpha=0.8)

    # Shade corrupted positions
    if corrupt_mask is not None:
        mask_arr = np.asarray(corrupt_mask)
        if mask_arr.ndim == 2 and sequence_index < mask_arr.shape[0]:
            row = mask_arr[sequence_index].astype(bool)
        elif mask_arr.ndim == 1:
            row = mask_arr.astype(bool)
        else:
            row = None
        if row is not None:
            for pos in np.where(row)[0]:
                ax.axvspan(pos - 0.5, pos + 0.5, color="red", alpha=0.15)

    ax.set_xlabel("token position")
    ax.set_ylabel("signal value")
    if tokens is not None and len(tokens) == L:
        step = max(1, L // 20)
        ax.set_xticks(xs[::step])
        ax.set_xticklabels(tokens[::step], rotation=45, ha="right", fontsize=7)
    ax.legend(fontsize=8, loc="best")
    seq_title = title if title is not None else f"UQ signals — sequence {sequence_index}"
    ax.set_title(seq_title)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def save_three_signal_plots(
    out_dir: Path | str,
    *,
    structural_energy_valid: np.ndarray | None = None,
    structural_energy_invalid: np.ndarray | None = None,
    contextual_energy_valid: np.ndarray | None = None,
    contextual_energy_invalid: np.ndarray | None = None,
    spilled_energy_valid: np.ndarray | None = None,
    spilled_energy_invalid: np.ndarray | None = None,
    corrupt_mask_invalid: np.ndarray | None = None,
    n_samples: int = 32,
    label: str = "phase5",
) -> dict[str, str]:
    """Write all three-signal heatmaps and a per-sequence line example.

    Convenience wrapper that calls :func:`plot_three_signal_heatmap` for valid
    and invalid sequences separately, then :func:`plot_three_signal_lines` for
    the first sequence of each set.

    Returns:
        Dict mapping plot-key to saved file path.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    for split, se, ce, spe, cmask in [
        ("valid", structural_energy_valid, contextual_energy_valid, spilled_energy_valid, None),
        ("invalid", structural_energy_invalid, contextual_energy_invalid, spilled_energy_invalid, corrupt_mask_invalid),
    ]:
        if all(x is None for x in [se, ce, spe]):
            continue
        hmap_path = out / f"{label}_three_signal_heatmap_{split}.png"
        plot_three_signal_heatmap(
            se, ce, spe, hmap_path,
            n_samples=n_samples,
            corrupt_mask=cmask,
            title=f"three-component UQ — {split} sequences",
        )
        written[f"three_signal_heatmap_{split}"] = str(hmap_path)

        lines_path = out / f"{label}_three_signal_lines_{split}_seq0.png"
        plot_three_signal_lines(
            se, ce, spe, lines_path,
            sequence_index=0,
            corrupt_mask=cmask,
            title=f"UQ signals — {split} sequence 0",
        )
        written[f"three_signal_lines_{split}"] = str(lines_path)

    return written


__all__ = [
    "plot_three_signal_heatmap",
    "plot_three_signal_lines",
    "save_three_signal_plots",
]
