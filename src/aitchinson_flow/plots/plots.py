"""Stage-aware diagnostic plots for the Bayesian Auditor pipeline.

Two stages emit different plot payloads:

* Stage 1 (`BayesianAuditorStage1`) — geometric Hilbert energy, no GP.
* Stage 2 (`BayesianAuditorStage2`) — GP predictive mean + epistemic variance,
  plus inducing points in the latent space.

`save_stage_plots` is the single orchestrator that consumes a `StagePlotData`
bundle and writes every requested artifact for one stage into `out_dir`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aitchinson_flow.metrics.auroc import safe_auroc


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
        energy_seq_valid / energy_seq_invalid: Per-sequence scalar energies,
            shape ``(N,)``. Populated explicitly by the audit task;
            ``save_stage_plots`` does not derive from token-mean anymore.
        variance_token_valid / variance_token_invalid: Per-token variances,
            shape ``(N, L)``. Stage 2: GP epistemic variance. Stage 1:
            velocity-norm surrogate (optional; typically unused for Stage 1
            plots).
        variance_seq_valid / variance_seq_invalid: Per-sequence scalar
            variances, shape ``(N,)``. Explicit, not derived.
        latent_tokens_valid / latent_tokens_invalid: Token latents used by the
            GP head, shape ``(N, L, d_latent)``. Projected to 2D for the
            density plot.
        inducing_points: GP inducing locations, shape ``(M, d_latent)``
            (Stage 2 only; ``None`` for Stage 1).
        corrupt_mask_invalid: Ground-truth corruption mask for invalid
            sequences, shape ``(N, L)``, 0/1. True where the invalid token_id
            differs from the clean token_id.
        auditor_score_valid / auditor_score_invalid: Sequence-level auditor
            anomaly scores, shape ``(N,)``. Used for the combined ROC.
        spilled_seq_scalar_valid / spilled_seq_scalar_invalid: Sequence-level
            spilled-energy anomaly scores, shape ``(N,)``.
        spilled_token_valid / spilled_token_invalid: Per-position spilled
            energy, shape ``(N, L-1)``.
        geometric_energy_valid / geometric_energy_invalid: Sequence-level
            Stage 1 geometric anomaly scores, shape ``(N,)`` (Stage 1 only).
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
    corrupt_mask_invalid: np.ndarray | None = None
    auditor_score_valid: np.ndarray | None = None
    auditor_score_invalid: np.ndarray | None = None
    spilled_seq_scalar_valid: np.ndarray | None = None
    spilled_seq_scalar_invalid: np.ndarray | None = None
    spilled_token_valid: np.ndarray | None = None
    spilled_token_invalid: np.ndarray | None = None
    geometric_energy_valid: np.ndarray | None = None
    geometric_energy_invalid: np.ndarray | None = None
    history: list[dict[str, float]] | None = None


def plot_histogram(
    valid: np.ndarray | None,
    invalid: np.ndarray | None,
    out_path: Path | str,
    *,
    title: str,
    xlabel: str,
    bins: int = 40,
    annotation: str | None = None,
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
    if annotation:
        ax.text(
            0.02,
            0.97,
            annotation,
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )
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
    """Plot each numeric metric in ``history`` on its own subplot.

    Loss components can differ in scale by orders of magnitude (e.g. ``kl``
    vs ``nll``); shared-axis plots collapse the smaller ones to zero. Each
    base key gets its own row sharing only the x-axis so every component
    stays legible. Matching ``val_<key>`` entries render as dashed lines.
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
    n = len(base_keys)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(7.0, 2.2 * n), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, base_keys):
        train_ys = [float(h[k]) for h in history if k in h]
        train_xs = [x for x, h in zip(xs, history) if k in h]
        if train_ys:
            ax.plot(train_xs, train_ys, color="C0", marker="o", markersize=3, label=f"train/{k}")

        val_key = f"val_{k}"
        val_ys = [float(h[val_key]) for h in history if val_key in h]
        val_xs = [x for x, h in zip(xs, history) if val_key in h]
        if val_ys:
            ax.plot(
                val_xs,
                val_ys,
                color="C0",
                marker="s",
                markersize=3,
                linestyle="--",
                label=f"val/{k}",
            )
        ax.set_title(k, fontsize=9)
        ax.set_ylabel(k, fontsize=8)
        ax.legend(fontsize=7, loc="best")
    axes[-1].set_xlabel("epoch")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_total_loss(
    history: list[dict[str, float]] | None,
    out_path: Path | str,
    *,
    title: str = "total loss per epoch",
) -> None:
    """Single-axes overview: ``loss`` and ``val_loss`` only.

    Produced alongside ``plot_loss_curves`` so the quick-look is always one
    figure while component diagnostics live in the multi-row file.
    """
    if not history:
        return
    train_ys = [float(h["loss"]) for h in history if "loss" in h]
    train_xs = [i + 1 for i, h in enumerate(history) if "loss" in h]
    val_ys = [float(h["val_loss"]) for h in history if "val_loss" in h]
    val_xs = [i + 1 for i, h in enumerate(history) if "val_loss" in h]
    if not train_ys and not val_ys:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    if train_ys:
        ax.plot(train_xs, train_ys, color="C0", marker="o", markersize=3, label="train/loss")
    if val_ys:
        ax.plot(
            val_xs, val_ys, color="C0", marker="s", markersize=3, linestyle="--", label="val/loss"
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.legend()
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
    corrupt_mask: np.ndarray | None = None,
) -> None:
    """Heatmap of up to ``n_samples`` sequences × L token-level values.

    When ``corrupt_mask`` is supplied with matching shape, cyan ``x`` markers
    are overlaid on cells where the invalid token differs from the clean
    token — giving a direct visual check on whether high energy/variance
    co-locates with the actual corruption.
    """
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
    full_title = title
    if corrupt_mask is not None:
        mask = np.asarray(corrupt_mask)
        if mask.shape == arr.shape:
            mask_slice = mask[:n].astype(bool)
            rows, cols = np.where(mask_slice)
            if rows.size:
                ax.scatter(
                    cols,
                    rows,
                    marker="x",
                    s=18,
                    color="cyan",
                    linewidths=0.8,
                    zorder=3,
                    label="scrambled",
                )
                full_title = f"{title}  (x = scrambled token)"
                ax.legend(loc="upper right", fontsize=7)
    ax.set_title(full_title)
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


def plot_corruption_comparison(
    values_invalid: np.ndarray | None,
    corrupt_mask: np.ndarray | None,
    out_path: Path | str,
    *,
    title: str,
    xlabel: str,
    bins: int = 40,
    plot_kind: str = "violin",
) -> None:
    """Compare per-token scores at corrupted vs clean positions in invalid seqs.

    This is the primary experimental check: within invalid sequences, tokens
    that were actually scrambled should score higher than tokens that were
    left alone. A violin plot surfaces the distributional shift; histogram
    mode is kept as an alternative for distribution shape inspection.
    """
    if values_invalid is None or corrupt_mask is None:
        return
    vals = np.asarray(values_invalid)
    mask = np.asarray(corrupt_mask)
    if vals.shape != mask.shape or vals.size == 0:
        return
    mask_bool = mask.astype(bool)
    corrupted = _finite(vals[mask_bool])
    clean = _finite(vals[~mask_bool])
    if corrupted.size == 0 and clean.size == 0:
        return

    auc = safe_auroc(clean, corrupted)
    mu_clean = float(clean.mean()) if clean.size else float("nan")
    mu_corr = float(corrupted.mean()) if corrupted.size else float("nan")
    annotation = (
        f"μ_clean={mu_clean:+.3f}\nμ_scrambled={mu_corr:+.3f}\nAUROC={auc:.3f}"
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    if plot_kind == "hist":
        if clean.size:
            ax.hist(clean, bins=bins, alpha=0.6, color="C2", label=f"clean (n={clean.size})")
        if corrupted.size:
            ax.hist(
                corrupted,
                bins=bins,
                alpha=0.6,
                color="C3",
                label=f"scrambled (n={corrupted.size})",
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        ax.legend()
    else:
        parts = [d for d in (clean, corrupted) if d.size]
        positions = [p for p, d in zip([0, 1], (clean, corrupted)) if d.size]
        ax.violinplot(parts, positions=positions, showmedians=True)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(
            [f"clean (n={clean.size})", f"scrambled (n={corrupted.size})"]
        )
        ax.set_ylabel(xlabel)
    ax.set_title(title)
    ax.text(
        0.02,
        0.97,
        annotation,
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_roc_curves(
    methods: dict[str, tuple[np.ndarray | None, np.ndarray | None]],
    out_path: Path | str,
    *,
    title: str = "ROC: valid vs invalid",
) -> dict[str, float]:
    """Render one ROC per method on a shared axes; return ``{name: AUROC}``."""
    from sklearn.metrics import roc_curve  # noqa: PLC0415

    aurocs: dict[str, float] = {}
    fig, ax = plt.subplots(figsize=(5.5, 5))
    any_plotted = False
    for name, pair in methods.items():
        v_raw, i_raw = pair
        v = _finite(v_raw) if v_raw is not None else np.empty(0)
        i = _finite(i_raw) if i_raw is not None else np.empty(0)
        if v.size == 0 or i.size == 0:
            print(f"plot_roc_curves: skipping {name!r} (empty class)")
            continue
        auc = safe_auroc(v, i)
        labels = np.concatenate([np.zeros(v.size), np.ones(i.size)])
        scores = np.concatenate([v, i])
        if not np.isfinite(scores).all():
            scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        fpr, tpr, _ = roc_curve(labels, scores)
        ax.plot(fpr, tpr, label=f"{name} (AUROC={auc:.3f})")
        aurocs[name] = auc
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        return aurocs

    ax.plot([0, 1], [0, 1], "--", color="grey", alpha=0.6, label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return aurocs


def stage_plot_data_from_scores(
    scores: dict[str, np.ndarray | None],
    *,
    stage: str,
    history: list[dict[str, float]] | None = None,
) -> StagePlotData:
    """Map a benchmark ``_scores`` dict onto a :class:`StagePlotData` payload.

    Benchmark tasks (``text_audit``) and the two-stage orchestration script
    both produce the same ``_scores`` dict. Centralizing the mapping here keeps
    plotting consumers aligned and avoids duplicating the per-key wiring in
    every entrypoint.
    """
    return StagePlotData(
        stage=stage,
        energy_token_valid=scores.get("auditor_energy_seq_valid"),
        energy_token_invalid=scores.get("auditor_energy_seq_invalid"),
        variance_token_valid=scores.get("auditor_var_seq_valid"),
        variance_token_invalid=scores.get("auditor_var_seq_invalid"),
        energy_seq_valid=scores.get("auditor_energy_scalar_valid"),
        energy_seq_invalid=scores.get("auditor_energy_scalar_invalid"),
        variance_seq_valid=scores.get("auditor_variance_scalar_valid"),
        variance_seq_invalid=scores.get("auditor_variance_scalar_invalid"),
        latent_tokens_valid=scores.get("latent_tokens_valid"),
        latent_tokens_invalid=scores.get("latent_tokens_invalid"),
        inducing_points=scores.get("inducing_points"),
        corrupt_mask_invalid=scores.get("corrupt_mask_invalid"),
        auditor_score_valid=scores.get("auditor_valid"),
        auditor_score_invalid=scores.get("auditor_invalid"),
        spilled_seq_scalar_valid=scores.get("spilled_valid"),
        spilled_seq_scalar_invalid=scores.get("spilled_invalid"),
        spilled_token_valid=scores.get("spilled_token_valid"),
        spilled_token_invalid=scores.get("spilled_token_invalid"),
        geometric_energy_valid=scores.get("energy_valid"),
        geometric_energy_invalid=scores.get("energy_invalid"),
        history=history,
    )


def save_benchmark_plots(
    out_dir: Path | str,
    *,
    scores: dict[str, np.ndarray | None],
    loss_history: list[dict[str, float]] | None = None,
    stage: str = "benchmark",
) -> dict[str, str]:
    """Render :class:`StagePlotData` artifacts from a benchmark ``_scores`` dict.

    Thin wrapper around :func:`save_stage_plots` that keeps benchmark runners
    from having to construct :class:`StagePlotData` themselves.
    """
    data = stage_plot_data_from_scores(scores, stage=stage, history=loss_history)
    return save_stage_plots(out_dir, data)


def save_stage_plots(out_dir: Path | str, data: StagePlotData) -> dict[str, str]:
    """Render every available plot for a single stage into ``out_dir``.

    Returns a dict ``{plot_key: relative_filename}`` for artifacts actually
    written (useful for surfacing in training manifests).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stage = data.stage
    written: dict[str, str] = {}

    # ---- Loss curves: per-component subplots + total-loss overview ---------
    if data.history:
        components_name = f"{stage}_loss_components.png"
        plot_loss_curves(
            data.history,
            out / components_name,
            title=f"{stage}: training/validation loss components per epoch",
        )
        written["loss_components"] = components_name

        total_name = f"{stage}_loss_total.png"
        plot_total_loss(
            data.history,
            out / total_name,
            title=f"{stage}: total loss per epoch",
        )
        written["loss_total"] = total_name

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

    # ---- Sequence-level energy histogram (explicit scalars from task) ------
    esv = _as_np(data.energy_seq_valid)
    esi = _as_np(data.energy_seq_invalid)
    if esv is not None or esi is not None:
        fname = f"{stage}_hist_sequence_energy.png"
        auc = safe_auroc(esv, esi)
        sep = (
            float(np.asarray(esi).mean() - np.asarray(esv).mean())
            if esv is not None and esi is not None
            else float("nan")
        )
        plot_histogram(
            esv,
            esi,
            out / fname,
            title=f"{stage}: per-sequence energy (valid vs invalid)",
            xlabel="sequence energy",
            annotation=f"AUROC={auc:.3f}\nΔμ={sep:+.3f}",
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

    # ---- Sequence-level variance histogram (explicit scalars from task) ----
    vsv = _as_np(data.variance_seq_valid)
    vsi = _as_np(data.variance_seq_invalid)
    if vsv is not None or vsi is not None:
        fname = f"{stage}_hist_sequence_variance.png"
        auc = safe_auroc(vsv, vsi)
        sep = (
            float(np.asarray(vsi).mean() - np.asarray(vsv).mean())
            if vsv is not None and vsi is not None
            else float("nan")
        )
        plot_histogram(
            vsv,
            vsi,
            out / fname,
            title=f"{stage}: per-sequence variance (valid vs invalid)",
            xlabel="sequence variance",
            annotation=f"AUROC={auc:.3f}\nΔμ={sep:+.3f}",
        )
        written["hist_sequence_variance"] = fname

    # ---- Heatmaps (invalid heatmaps overlay the corruption mask) -----------
    cmask = _as_np(data.corrupt_mask_invalid)
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
            corrupt_mask=cmask,
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
            corrupt_mask=cmask,
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

    # ---- Corruption diagnostics: scrambled-vs-clean tokens within invalid --
    if cmask is not None and eti is not None:
        fname = f"{stage}_corrupt_vs_clean_energy.png"
        plot_corruption_comparison(
            eti,
            cmask,
            out / fname,
            title=f"{stage}: invalid sequences — energy at scrambled vs clean tokens",
            xlabel="token energy",
        )
        written["corrupt_vs_clean_energy"] = fname
    if cmask is not None and vti is not None:
        fname = f"{stage}_corrupt_vs_clean_variance.png"
        plot_corruption_comparison(
            vti,
            cmask,
            out / fname,
            title=f"{stage}: invalid sequences — variance at scrambled vs clean tokens",
            xlabel="token variance",
        )
        written["corrupt_vs_clean_variance"] = fname

    # ---- Combined ROC: sequence + token level, auditor + baselines ---------
    methods: dict[str, tuple[np.ndarray | None, np.ndarray | None]] = {}
    asv = _as_np(data.auditor_score_valid)
    asi = _as_np(data.auditor_score_invalid)
    if asv is not None and asi is not None:
        methods["auditor (seq)"] = (asv, asi)
    ssv = _as_np(data.spilled_seq_scalar_valid)
    ssi = _as_np(data.spilled_seq_scalar_invalid)
    if ssv is not None and ssi is not None:
        methods["spilled (seq)"] = (ssv, ssi)
    gev = _as_np(data.geometric_energy_valid)
    gei = _as_np(data.geometric_energy_invalid)
    if gev is not None and gei is not None:
        methods["geometric energy (seq)"] = (gev, gei)
    if etv is not None and eti is not None:
        methods["auditor energy (token, all-invalid)"] = (etv.ravel(), eti.ravel())
        if cmask is not None and cmask.shape == eti.shape:
            corrupted_inv = eti[cmask.astype(bool)]
            if corrupted_inv.size:
                methods["auditor energy (token, scrambled-only)"] = (etv.ravel(), corrupted_inv)
    stv = _as_np(data.spilled_token_valid)
    sti = _as_np(data.spilled_token_invalid)
    if stv is not None and sti is not None:
        methods["spilled (token)"] = (stv.ravel(), sti.ravel())

    if methods:
        fname = f"{stage}_roc_combined.png"
        aurocs = plot_roc_curves(
            methods,
            out / fname,
            title=f"{stage}: ROC — valid vs invalid",
        )
        written["roc_combined"] = fname
        summary_name = f"{stage}_auroc_summary.json"
        (out / summary_name).write_text(json.dumps(aurocs, indent=2), encoding="utf-8")
        written["auroc_summary"] = summary_name

    return written


def _row_to_mapping(row: Any) -> Mapping[str, Any]:
    if isinstance(row, Mapping):
        return row
    to_dict = getattr(row, "to_dict", None)
    if callable(to_dict):
        out = to_dict()
        if isinstance(out, Mapping):
            return out
    raise TypeError(f"Unsupported benchmark row payload type: {type(row)!r}")


def plot_benchmark_table(
    rows: Iterable[Any],
    *,
    out_path: Path | str,
    metric: str = "auroc_combined",
    aggregate: str = "best",
) -> dict[str, Any]:
    """Plot component×task AUROC heatmap from unified benchmark rows.

    Args:
        rows: Iterable of row dicts or dataclasses exposing ``to_dict()`` with
            at least ``component``, ``task``, and the requested ``metric``.
        out_path: Output heatmap image path.
        metric: Metric field to visualize.
        aggregate: How to combine multiple scales per (component, task):
            ``"best"`` (max) or ``"mean"``.
    """
    if aggregate not in {"best", "mean"}:
        raise ValueError(f"aggregate must be 'best' or 'mean', got {aggregate!r}")

    parsed: list[Mapping[str, Any]] = [_row_to_mapping(row) for row in rows]
    components: list[str] = []
    tasks: list[str] = []
    grouped: dict[tuple[str, str], list[float]] = {}

    for row in parsed:
        component = str(row.get("component", "unknown"))
        task = str(row.get("task", "unknown"))
        if component not in components:
            components.append(component)
        if task not in tasks:
            tasks.append(task)

        raw = row.get(metric)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(value):
            continue
        grouped.setdefault((component, task), []).append(value)

    if not components or not tasks:
        raise ValueError("No benchmark rows were provided for heatmap plotting.")

    matrix = np.full((len(components), len(tasks)), np.nan, dtype=np.float64)
    values_summary: dict[str, dict[str, float | None]] = {}
    for i, component in enumerate(components):
        row_summary: dict[str, float | None] = {}
        for j, task in enumerate(tasks):
            vals = grouped.get((component, task), [])
            if not vals:
                row_summary[task] = None
                continue
            if aggregate == "best":
                agg = float(np.max(vals))
            else:
                agg = float(np.mean(vals))
            matrix[i, j] = agg
            row_summary[task] = agg
        values_summary[component] = row_summary

    finite = matrix[np.isfinite(matrix)]
    vmin = float(np.min(finite)) if finite.size else 0.0
    vmax = float(np.max(finite)) if finite.size else 1.0
    if abs(vmax - vmin) < 1e-9:
        vmax = vmin + 1e-6

    fig, ax = plt.subplots(figsize=(max(6, len(tasks) * 1.2), max(3, len(components) * 0.9)))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(tasks)))
    ax.set_xticklabels(tasks, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(components)))
    ax.set_yticklabels(components)
    ax.set_title(f"benchmark table: {metric} ({aggregate} across scales)")
    ax.set_xlabel("task")
    ax.set_ylabel("component")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric)

    for i in range(len(components)):
        for j in range(len(tasks)):
            val = matrix[i, j]
            text = "--" if not np.isfinite(val) else f"{val:.3f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color="white")

    fig.tight_layout()
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)

    summary: dict[str, Any] = {
        "metric": metric,
        "aggregate": aggregate,
        "components": components,
        "tasks": tasks,
        "values": values_summary,
        "heatmap_path": str(output),
    }
    summary_path = output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary
