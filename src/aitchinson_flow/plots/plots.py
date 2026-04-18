"""Stage-aware diagnostic plots for the Bayesian Auditor pipeline.

Two stages emit different plot payloads:

* Stage 1 (`BayesianAuditorStage1`) — geometric Hilbert energy, no GP.
* Stage 2 (`BayesianAuditorStage2`) — GP predictive mean + epistemic variance,
  plus inducing points in the latent space.

`save_stage_plots` is the single orchestrator that consumes a `StagePlotData`
bundle and writes every requested artifact for one stage into `out_dir`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _finite(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr).ravel()
    return a[np.isfinite(a)]


def _as_np(x: np.ndarray | None) -> np.ndarray | None:
    if x is None:
        return None
    arr = np.asarray(x)
    return arr if arr.size else None


@dataclass
class StagePlotData:
    """Plot-ready payload emitted by `text_audit` for one stage.

    All fields are optional; plots with missing data are skipped silently.

    Attributes:
        stage: Stage tag used in titles and filenames (``"stage1"`` or ``"stage2"``).
        energy_token_valid / energy_token_invalid: Per-token energies, shape
            ``(N, L)``. Stage 1: ``-d_H`` Hilbert energy. Stage 2: GP predictive
            mean.
        energy_seq_valid / energy_seq_invalid: Per-sequence energies, shape
            ``(N,)``. Typically the mean of ``energy_token_*`` over positions.
        variance_token_valid / variance_token_invalid: Per-token variances,
            shape ``(N, L)``. Stage 2: GP epistemic variance. Stage 1:
            velocity-norm surrogate (optional; typically unused for Stage 1
            plots).
        variance_seq_valid / variance_seq_invalid: Per-sequence variances,
            shape ``(N,)``.
        latent_tokens_valid / latent_tokens_invalid: Token latents used by the
            GP head, shape ``(N, L, d_latent)``. Projected to 2D for the
            density plot.
        inducing_points: GP inducing locations, shape ``(M, d_latent)``
            (Stage 2 only; ``None`` for Stage 1).
        history: Optional per-epoch training history (list of dicts). May
            contain ``val_*`` keys for validation metrics.
    """

    stage: str
    energy_token_valid: np.ndarray | None = None
    energy_token_invalid: np.ndarray | None = None
    energy_seq_valid: np.ndarray | None = None
    energy_seq_invalid: np.ndarray | None = None
    variance_token_valid: np.ndarray | None = None
    variance_token_invalid: np.ndarray | None = None
    variance_seq_valid: np.ndarray | None = None
    variance_seq_invalid: np.ndarray | None = None
    latent_tokens_valid: np.ndarray | None = None
    latent_tokens_invalid: np.ndarray | None = None
    inducing_points: np.ndarray | None = None
    history: list[dict[str, float]] | None = None


def plot_histogram(
    valid: np.ndarray | None,
    invalid: np.ndarray | None,
    out_path: Path | str,
    *,
    title: str,
    xlabel: str,
    bins: int = 40,
) -> None:
    """Overlay histogram of valid vs invalid scalar scores."""
    v = _finite(valid) if valid is not None else np.empty(0)
    i = _finite(invalid) if invalid is not None else np.empty(0)
    if v.size == 0 and i.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    if v.size:
        ax.hist(v, bins=bins, alpha=0.6, label=f"valid (n={v.size})", color="C0")
    if i.size:
        ax.hist(i, bins=bins, alpha=0.6, label=f"invalid (n={i.size})", color="C3")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_loss_curves(
    history: list[dict[str, float]] | None,
    out_path: Path | str,
    *,
    title: str = "training losses per epoch",
    keys: Iterable[str] | None = None,
) -> None:
    """Plot every numeric metric in ``history``, pairing ``val_*`` with base.

    Training keys render as solid lines; matching ``val_<key>`` entries render
    as dashed lines in the same colour so train/val pairs are visually
    comparable on a shared axis.
    """
    if not history:
        return
    all_keys: set[str] = set()
    for entry in history:
        all_keys.update(entry.keys())
    if not all_keys:
        return

    base_keys: list[str]
    if keys is not None:
        base_keys = [k for k in keys if k in all_keys or f"val_{k}" in all_keys]
    else:
        base_keys = sorted(k for k in all_keys if not k.startswith("val_"))

    if not base_keys:
        return

    xs = np.arange(1, len(history) + 1)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for idx, k in enumerate(base_keys):
        color = f"C{idx % 10}"
        train_ys = [float(h[k]) for h in history if k in h]
        train_xs = [x for x, h in zip(xs, history) if k in h]
        if train_ys:
            ax.plot(train_xs, train_ys, color=color, marker="o", markersize=3, label=f"train/{k}")

        val_key = f"val_{k}"
        val_ys = [float(h[val_key]) for h in history if val_key in h]
        val_xs = [x for x, h in zip(xs, history) if val_key in h]
        if val_ys:
            ax.plot(
                val_xs,
                val_ys,
                color=color,
                marker="s",
                markersize=3,
                linestyle="--",
                label=f"val/{k}",
            )
    ax.set_xlabel("epoch")
    ax.set_ylabel("metric value")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_token_heatmap(
    values: np.ndarray | None,
    out_path: Path | str,
    *,
    title: str,
    cbar_label: str,
    n_samples: int = 32,
    cmap: str = "magma",
) -> None:
    """Heatmap of up to ``n_samples`` sequences × L token-level values."""
    if values is None:
        return
    arr = np.asarray(values)
    if arr.ndim != 2 or arr.size == 0:
        return
    n = min(n_samples, arr.shape[0])
    mat = arr[:n]
    fig, ax = plt.subplots(figsize=(max(6, mat.shape[1] * 0.2), max(3, n * 0.18)))
    im = ax.imshow(mat, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xlabel("token position")
    ax.set_ylabel("sequence index")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _project_2d(
    *arrays: np.ndarray,
) -> tuple[list[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Fit a PCA (via SVD) on the concatenation and project each input to 2D.

    Returns the per-input projections plus the 2D basis ``(mean, components)``
    so additional points (e.g. inducing locations) can be projected later.
    """
    stacked = np.concatenate(arrays, axis=0)
    mean = stacked.mean(axis=0, keepdims=True)
    centered = stacked - mean
    if centered.shape[1] == 1:
        proj = np.concatenate([centered, np.zeros_like(centered)], axis=1)
        components = np.array([[1.0], [0.0]])
    elif centered.shape[1] == 2:
        proj = centered
        components = np.eye(2)
    else:
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
        components = vt[:2]
        proj = centered @ components.T
    outputs: list[np.ndarray] = []
    cursor = 0
    for arr in arrays:
        outputs.append(proj[cursor : cursor + arr.shape[0]])
        cursor += arr.shape[0]
    return outputs, (mean.squeeze(0), components)


def plot_latent_density(
    latents_valid: np.ndarray | None,
    latents_invalid: np.ndarray | None,
    out_path: Path | str,
    *,
    title: str,
    inducing_points: np.ndarray | None = None,
    max_points: int = 8000,
) -> None:
    """2D PCA scatter of token latents, optionally overlaying inducing points.

    Input token tensors of shape ``(N, L, d)`` are flattened to ``(N*L, d)``.
    Valid and invalid point clouds are colour-coded; inducing points (Stage 2)
    are drawn as black stars.
    """

    def _flatten(x: np.ndarray | None) -> np.ndarray | None:
        arr = _as_np(x)
        if arr is None:
            return None
        if arr.ndim == 3:
            arr = arr.reshape(-1, arr.shape[-1])
        elif arr.ndim != 2:
            return None
        return arr

    v = _flatten(latents_valid)
    i = _flatten(latents_invalid)
    z = _as_np(inducing_points)

    if v is None and i is None and z is None:
        return

    pieces: list[np.ndarray] = []
    if v is not None:
        pieces.append(v)
    if i is not None:
        pieces.append(i)
    if z is not None and z.ndim == 2:
        pieces.append(z)

    projected, (mean, components) = _project_2d(*pieces)
    cursor = 0
    v_2d = projected[cursor] if v is not None else None
    if v is not None:
        cursor += 1
    i_2d = projected[cursor] if i is not None else None
    if i is not None:
        cursor += 1
    z_2d = projected[cursor] if z is not None and z.ndim == 2 else None

    rng = np.random.default_rng(0)

    def _subsample(arr: np.ndarray | None) -> np.ndarray | None:
        if arr is None or arr.shape[0] <= max_points:
            return arr
        idx = rng.choice(arr.shape[0], size=max_points, replace=False)
        return arr[idx]

    v_2d = _subsample(v_2d)
    i_2d = _subsample(i_2d)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    if v_2d is not None and v_2d.size:
        ax.scatter(
            v_2d[:, 0],
            v_2d[:, 1],
            s=6,
            alpha=0.35,
            color="C0",
            label=f"valid tokens (n={v_2d.shape[0]})",
            rasterized=True,
        )
    if i_2d is not None and i_2d.size:
        ax.scatter(
            i_2d[:, 0],
            i_2d[:, 1],
            s=6,
            alpha=0.35,
            color="C3",
            label=f"invalid tokens (n={i_2d.shape[0]})",
            rasterized=True,
        )
    if z_2d is not None and z_2d.size:
        ax.scatter(
            z_2d[:, 0],
            z_2d[:, 1],
            s=90,
            marker="*",
            color="black",
            edgecolors="white",
            linewidths=0.6,
            label=f"inducing (n={z_2d.shape[0]})",
            zorder=5,
        )
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    del mean, components  # unused beyond projection


def save_stage_plots(out_dir: Path | str, data: StagePlotData) -> dict[str, str]:
    """Render every available plot for a single stage into ``out_dir``.

    Returns a dict ``{plot_key: relative_filename}`` for artifacts actually
    written (useful for surfacing in training manifests).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stage = data.stage
    written: dict[str, str] = {}

    # ---- Loss curves (train + val pairs) -----------------------------------
    if data.history:
        fname = f"{stage}_loss_curves.png"
        plot_loss_curves(
            data.history,
            out / fname,
            title=f"{stage}: training/validation losses per epoch",
        )
        written["loss_curves"] = fname

    # ---- Token-level energy histogram --------------------------------------
    etv = _as_np(data.energy_token_valid)
    eti = _as_np(data.energy_token_invalid)
    if etv is not None or eti is not None:
        fname = f"{stage}_hist_token_energy.png"
        plot_histogram(
            etv.ravel() if etv is not None else None,
            eti.ravel() if eti is not None else None,
            out / fname,
            title=f"{stage}: per-token energy (valid vs invalid)",
            xlabel="token energy",
        )
        written["hist_token_energy"] = fname

    # ---- Sequence-level energy histogram -----------------------------------
    esv = _as_np(data.energy_seq_valid)
    esi = _as_np(data.energy_seq_invalid)
    if esv is None and etv is not None:
        esv = etv.mean(axis=1)
    if esi is None and eti is not None:
        esi = eti.mean(axis=1)
    if esv is not None or esi is not None:
        fname = f"{stage}_hist_sequence_energy.png"
        plot_histogram(
            esv,
            esi,
            out / fname,
            title=f"{stage}: per-sequence energy (valid vs invalid)",
            xlabel="sequence energy (mean over tokens)",
        )
        written["hist_sequence_energy"] = fname

    # ---- Token variance histogram (Stage 2) --------------------------------
    vtv = _as_np(data.variance_token_valid)
    vti = _as_np(data.variance_token_invalid)
    if vtv is not None or vti is not None:
        fname = f"{stage}_hist_token_variance.png"
        plot_histogram(
            vtv.ravel() if vtv is not None else None,
            vti.ravel() if vti is not None else None,
            out / fname,
            title=f"{stage}: per-token variance (valid vs invalid)",
            xlabel="token variance",
        )
        written["hist_token_variance"] = fname

    # ---- Sequence-level variance histogram (Stage 2) -----------------------
    vsv = _as_np(data.variance_seq_valid)
    vsi = _as_np(data.variance_seq_invalid)
    if vsv is None and vtv is not None:
        vsv = vtv.mean(axis=1)
    if vsi is None and vti is not None:
        vsi = vti.mean(axis=1)
    if vsv is not None or vsi is not None:
        fname = f"{stage}_hist_sequence_variance.png"
        plot_histogram(
            vsv,
            vsi,
            out / fname,
            title=f"{stage}: per-sequence variance (valid vs invalid)",
            xlabel="sequence variance (mean over tokens)",
        )
        written["hist_sequence_variance"] = fname

    # ---- Heatmaps ----------------------------------------------------------
    if etv is not None:
        fname = f"{stage}_heatmap_token_energy_valid.png"
        plot_token_heatmap(
            etv,
            out / fname,
            title=f"{stage}: valid sequences — energy per token",
            cbar_label="token energy",
        )
        written["heatmap_token_energy_valid"] = fname
    if eti is not None:
        fname = f"{stage}_heatmap_token_energy_invalid.png"
        plot_token_heatmap(
            eti,
            out / fname,
            title=f"{stage}: invalid sequences — energy per token",
            cbar_label="token energy",
        )
        written["heatmap_token_energy_invalid"] = fname

    if vtv is not None:
        fname = f"{stage}_heatmap_token_variance_valid.png"
        plot_token_heatmap(
            vtv,
            out / fname,
            title=f"{stage}: valid sequences — variance per token",
            cbar_label="token variance",
            cmap="viridis",
        )
        written["heatmap_token_variance_valid"] = fname
    if vti is not None:
        fname = f"{stage}_heatmap_token_variance_invalid.png"
        plot_token_heatmap(
            vti,
            out / fname,
            title=f"{stage}: invalid sequences — variance per token",
            cbar_label="token variance",
            cmap="viridis",
        )
        written["heatmap_token_variance_invalid"] = fname

    # ---- Latent density with inducing points -------------------------------
    ltv = _as_np(data.latent_tokens_valid)
    lti = _as_np(data.latent_tokens_invalid)
    ip = _as_np(data.inducing_points)
    if ltv is not None or lti is not None or ip is not None:
        fname = f"{stage}_latent_density.png"
        ip_label = " with inducing points" if ip is not None else ""
        plot_latent_density(
            ltv,
            lti,
            out / fname,
            title=f"{stage}: token-latent PCA{ip_label}",
            inducing_points=ip,
        )
        written["latent_density"] = fname

    return written
