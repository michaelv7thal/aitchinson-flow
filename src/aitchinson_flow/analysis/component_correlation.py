"""Inter-component correlation analysis for the three UQ signals.

Computes Pearson and Spearman rank correlations between:
  - Structural energy (Component 1 GP mean)
  - Contextual energy (Component 2 GP mean)
  - Spilled energy    (Component 3)

Supports two aggregation levels:
  sequence — per-sequence mean energies, shape (N,) each
  token    — all per-token values flattened, shape (N*L,) each

Produces a correlation-matrix heatmap and pair-wise scatter plots.

Public API
----------
compute_correlation_matrix
    Return Pearson and Spearman matrices plus p-values.
plot_correlation_heatmap
    Render the Pearson / Spearman correlation matrices side-by-side.
plot_correlation_scatter
    Pair-wise scatter plots for every signal combination.
compute_and_plot_correlations
    Convenience orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import matplotlib
import numpy as np
from scipy import stats as scipy_stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SIGNAL_NAMES = ("structural", "contextual", "spilled")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class PairCorrelation:
    """Pearson and Spearman statistics for a single pair of signals.

    Attributes:
        pearson_r: Pearson correlation coefficient.
        pearson_p: Two-tailed p-value for the Pearson test.
        spearman_r: Spearman rank correlation.
        spearman_p: Two-tailed p-value for the Spearman test.
        n: Number of samples used.
    """

    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float
    n: int


@dataclass
class CorrelationMatrix:
    """Full pairwise correlation results for all three UQ signals.

    Attributes:
        signal_names: Ordered list of signal labels.
        pairs: Dict mapping ``(name_a, name_b)`` to :class:`PairCorrelation`.
        pearson_matrix: 3×3 Pearson-r matrix (signal_names ordering).
        spearman_matrix: 3×3 Spearman-r matrix.
        level: Aggregation level used (``"sequence"`` or ``"token"``).
    """

    signal_names: list[str]
    pairs: dict[tuple[str, str], PairCorrelation]
    pearson_matrix: np.ndarray
    spearman_matrix: np.ndarray
    level: str

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict."""
        pairs_out: dict[str, dict] = {}
        for (a, b), pc in self.pairs.items():
            pairs_out[f"{a}_vs_{b}"] = {
                "pearson_r": pc.pearson_r,
                "pearson_p": pc.pearson_p,
                "spearman_r": pc.spearman_r,
                "spearman_p": pc.spearman_p,
                "n": pc.n,
            }
        return {
            "level": self.level,
            "signal_names": self.signal_names,
            "pairs": pairs_out,
            "pearson_matrix": self.pearson_matrix.tolist(),
            "spearman_matrix": self.spearman_matrix.tolist(),
        }


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def _prepare_signal(
    arr: np.ndarray | None,
    level: Literal["sequence", "token"],
    name: str,
) -> np.ndarray | None:
    """Flatten and aggregate a signal array to a 1-D vector."""
    if arr is None:
        return None
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        return a
    if a.ndim == 2:
        if level == "sequence":
            return np.nanmean(a, axis=1)  # (N,)
        return a.ravel()  # (N*L,)
    raise ValueError(f"{name} must be 1-D or 2-D, got shape {arr.shape}")


def _pair_correlation(x: np.ndarray, y: np.ndarray) -> PairCorrelation:
    """Compute Pearson and Spearman for two aligned vectors; drop non-finite."""
    mask = np.isfinite(x) & np.isfinite(y)
    xf, yf = x[mask], y[mask]
    n = int(mask.sum())
    if n < 3:
        return PairCorrelation(
            pearson_r=float("nan"),
            pearson_p=float("nan"),
            spearman_r=float("nan"),
            spearman_p=float("nan"),
            n=n,
        )
    pr, pp = scipy_stats.pearsonr(xf, yf)
    sr, sp = scipy_stats.spearmanr(xf, yf)
    return PairCorrelation(
        pearson_r=float(pr),
        pearson_p=float(pp),
        spearman_r=float(sr),
        spearman_p=float(sp),
        n=n,
    )


def compute_correlation_matrix(
    structural_energy: np.ndarray | None,
    contextual_energy: np.ndarray | None,
    spilled_energy: np.ndarray | None,
    *,
    level: Literal["sequence", "token"] = "sequence",
) -> CorrelationMatrix:
    """Compute pairwise Pearson and Spearman correlations for the three signals.

    Args:
        structural_energy: Component 1 GP mean, shape ``(N,)`` or ``(N, L)``.
        contextual_energy: Component 2 GP mean, matching shape.
        spilled_energy: Component 3 values, shape ``(N,)``, ``(N, L-1)``, or
            ``(N, L)``.  NaN-padded positions are excluded automatically.
        level: ``"sequence"`` aggregates each array to ``(N,)`` by row-mean;
            ``"token"`` flattens to ``(N*L,)`` before correlating.

    Returns:
        :class:`CorrelationMatrix` with all pairwise statistics.
    """
    signals_raw = [
        (SIGNAL_NAMES[0], structural_energy),
        (SIGNAL_NAMES[1], contextual_energy),
        (SIGNAL_NAMES[2], spilled_energy),
    ]

    prepared: list[tuple[str, np.ndarray | None]] = [
        (name, _prepare_signal(arr, level, name)) for name, arr in signals_raw
    ]

    # Find minimum length among non-None signals for alignment
    lengths = [v.shape[0] for _, v in prepared if v is not None]
    min_len = min(lengths) if lengths else 0

    aligned: dict[str, np.ndarray | None] = {}
    for name, v in prepared:
        aligned[name] = v[:min_len] if v is not None else None

    names = [n for n in SIGNAL_NAMES if aligned.get(n) is not None]
    n_sig = len(names)

    pearson_mat = np.full((n_sig, n_sig), np.nan)
    spearman_mat = np.full((n_sig, n_sig), np.nan)
    pairs: dict[tuple[str, str], PairCorrelation] = {}

    for i, na in enumerate(names):
        pearson_mat[i, i] = 1.0
        spearman_mat[i, i] = 1.0
        for j, nb in enumerate(names):
            if j <= i:
                continue
            xa = aligned[na]
            xb = aligned[nb]
            if xa is None or xb is None:
                continue
            pc = _pair_correlation(xa, xb)
            pairs[(na, nb)] = pc
            pairs[(nb, na)] = pc
            pearson_mat[i, j] = pearson_mat[j, i] = pc.pearson_r
            spearman_mat[i, j] = spearman_mat[j, i] = pc.spearman_r

    return CorrelationMatrix(
        signal_names=names,
        pairs=pairs,
        pearson_matrix=pearson_mat,
        spearman_matrix=spearman_mat,
        level=level,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _annotated_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> None:
    """Draw an annotated heatmap on ``ax``."""
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title(title, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = matrix[i, j]
            text = f"{val:.2f}" if np.isfinite(val) else "–"
            ax.text(j, i, text, ha="center", va="center", fontsize=9,
                    color="white" if abs(val) > 0.5 else "black")
    return im  # type: ignore[return-value]


def plot_correlation_heatmap(
    result: CorrelationMatrix,
    out_path: Path | str,
    *,
    title: str | None = None,
) -> None:
    """Render Pearson and Spearman correlation matrices side-by-side.

    Args:
        result: Output of :func:`compute_correlation_matrix`.
        out_path: Output PNG path.
        title: Figure suptitle; auto-generated from ``result.level`` if None.
    """
    n = len(result.signal_names)
    if n == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(max(7, n * 2.5), max(3.5, n * 1.5)))
    im_p = _annotated_heatmap(
        axes[0], result.pearson_matrix, result.signal_names, "Pearson r"
    )
    im_s = _annotated_heatmap(
        axes[1], result.spearman_matrix, result.signal_names, "Spearman ρ"
    )

    plt.colorbar(im_p, ax=axes[0], fraction=0.046, pad=0.04)
    plt.colorbar(im_s, ax=axes[1], fraction=0.046, pad=0.04)

    sup = title or f"UQ component correlations ({result.level}-level)"
    fig.suptitle(sup, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_correlation_scatter(
    structural_energy: np.ndarray | None,
    contextual_energy: np.ndarray | None,
    spilled_energy: np.ndarray | None,
    out_path: Path | str,
    *,
    level: Literal["sequence", "token"] = "sequence",
    max_points: int = 5000,
    title: str | None = None,
) -> CorrelationMatrix:
    """Pair-wise scatter plots with Pearson r and Spearman ρ annotations.

    Args:
        structural_energy: Component 1 energy.
        contextual_energy: Component 2 energy.
        spilled_energy: Component 3 energy.
        out_path: Output PNG path.
        level: Aggregation level passed to :func:`compute_correlation_matrix`.
        max_points: Subsample to this many points for plotting speed.
        title: Figure suptitle.

    Returns:
        :class:`CorrelationMatrix` (also used for annotations).
    """
    result = compute_correlation_matrix(
        structural_energy, contextual_energy, spilled_energy, level=level
    )
    names = result.signal_names
    n_sig = len(names)
    if n_sig < 2:
        return result

    prepared: dict[str, np.ndarray] = {}
    for name, arr in [
        (SIGNAL_NAMES[0], structural_energy),
        (SIGNAL_NAMES[1], contextual_energy),
        (SIGNAL_NAMES[2], spilled_energy),
    ]:
        v = _prepare_signal(arr, level, name)
        if v is not None and name in names:
            prepared[name] = v

    # Align lengths
    min_len = min(v.shape[0] for v in prepared.values()) if prepared else 0
    for k in prepared:
        prepared[k] = prepared[k][:min_len]

    # Subsample
    rng = np.random.default_rng(42)
    if min_len > max_points:
        idx = rng.choice(min_len, size=max_points, replace=False)
        for k in prepared:
            prepared[k] = prepared[k][idx]

    fig, axes = plt.subplots(n_sig, n_sig, figsize=(n_sig * 3.5, n_sig * 3.2))
    axes = np.atleast_2d(axes)

    for i, na in enumerate(names):
        for j, nb in enumerate(names):
            ax = axes[i, j]
            if i == j:
                xa = prepared.get(na)
                if xa is not None:
                    finite = xa[np.isfinite(xa)]
                    ax.hist(finite, bins=30, color="C0", alpha=0.7, density=True)
                    ax.set_title(na, fontsize=8)
                ax.set_xlabel("")
                ax.set_ylabel("")
                continue
            xa = prepared.get(na)
            xb = prepared.get(nb)
            if xa is None or xb is None:
                ax.axis("off")
                continue
            mask = np.isfinite(xa) & np.isfinite(xb)
            ax.scatter(xb[mask], xa[mask], s=4, alpha=0.4, color="C0", rasterized=True)
            pc = result.pairs.get((na, nb))
            if pc is not None:
                ann = f"r={pc.pearson_r:.2f}\nρ={pc.spearman_r:.2f}"
                ax.text(
                    0.05, 0.95, ann, transform=ax.transAxes,
                    va="top", fontsize=7,
                    bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
                )
            if i == n_sig - 1:
                ax.set_xlabel(nb, fontsize=8)
            if j == 0:
                ax.set_ylabel(na, fontsize=8)

    sup = title or f"UQ signal pair-wise scatter ({level}-level, n≤{max_points})"
    fig.suptitle(sup, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return result


def compute_and_plot_correlations(
    structural_energy: np.ndarray | None,
    contextual_energy: np.ndarray | None,
    spilled_energy: np.ndarray | None,
    out_dir: Path | str,
    *,
    label: str = "phase5",
    levels: tuple[str, ...] = ("sequence", "token"),
) -> dict[str, CorrelationMatrix]:
    """Run correlation analysis at each aggregation level and save plots.

    Args:
        structural_energy: Component 1 energy, shape ``(N,)`` or ``(N, L)``.
        contextual_energy: Component 2 energy, matching shape.
        spilled_energy: Component 3 energy.
        out_dir: Output directory for PNG files.
        label: Filename prefix.
        levels: Which aggregation levels to run.

    Returns:
        Dict mapping level name to its :class:`CorrelationMatrix`.
    """
    import json

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, CorrelationMatrix] = {}

    for level in levels:
        if level not in ("sequence", "token"):
            continue
        hmap_path = out / f"{label}_corr_heatmap_{level}.png"
        scatter_path = out / f"{label}_corr_scatter_{level}.png"

        corr = plot_correlation_scatter(
            structural_energy, contextual_energy, spilled_energy,
            scatter_path, level=level,  # type: ignore[arg-type]
        )
        plot_correlation_heatmap(corr, hmap_path)
        results[level] = corr

        json_path = out / f"{label}_corr_{level}.json"
        json_path.write_text(json.dumps(corr.to_dict(), indent=2), encoding="utf-8")

    return results


__all__ = [
    "PairCorrelation",
    "CorrelationMatrix",
    "compute_correlation_matrix",
    "plot_correlation_heatmap",
    "plot_correlation_scatter",
    "compute_and_plot_correlations",
]
