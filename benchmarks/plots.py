"""Matplotlib plots for the text-audit benchmark."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _finite(arr: np.ndarray) -> np.ndarray:
    return arr[np.isfinite(arr)]


def plot_score_histograms(
    valid: np.ndarray,
    invalid: np.ndarray,
    out_path: Path | str,
    *,
    title: str,
    xlabel: str = "anomaly score",
    bins: int = 40,
) -> None:
    valid = _finite(np.asarray(valid).ravel())
    invalid = _finite(np.asarray(invalid).ravel())
    fig, ax = plt.subplots(figsize=(6, 4))
    if valid.size:
        ax.hist(valid, bins=bins, alpha=0.6, label=f"valid (n={valid.size})", color="C0")
    if invalid.size:
        ax.hist(
            invalid, bins=bins, alpha=0.6, label=f"invalid (n={invalid.size})", color="C3"
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_roc(
    methods: dict[str, tuple[np.ndarray, np.ndarray]],
    out_path: Path | str,
    *,
    title: str = "ROC: auditor vs spilled energy",
) -> None:
    """``methods[name] = (valid_scores, invalid_scores)`` — label valid=0, invalid=1."""
    from sklearn.metrics import roc_auc_score, roc_curve  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, (v, i) in methods.items():
        v = _finite(np.asarray(v).ravel())
        i = _finite(np.asarray(i).ravel())
        if v.size == 0 or i.size == 0:
            continue
        labels = np.concatenate([np.zeros(v.size), np.ones(i.size)])
        scores = np.concatenate([v, i])
        auc = roc_auc_score(labels, scores)
        fpr, tpr, _ = roc_curve(labels, scores)
        ax.plot(fpr, tpr, label=f"{name} (AUROC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", alpha=0.6, label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_loss_curve(
    history: list[dict[str, float]],
    out_path: Path | str,
    *,
    keys: tuple[str, ...] = ("loss", "flow_loss", "mean_loss", "var_loss"),
    title: str = "training losses per epoch",
) -> None:
    if not history:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = np.arange(1, len(history) + 1)
    for k in keys:
        ys = [float(h[k]) for h in history if k in h]
        if len(ys) == len(history):
            ax.plot(xs, ys, marker="o", markersize=3, label=k)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_sequence_spilled(
    sp_valid: np.ndarray,
    sp_invalid: np.ndarray,
    out_path: Path | str,
    *,
    title: str = "spilled energy per position",
) -> None:
    """``sp_*`` shape ``(B, L-1)``."""
    if sp_valid is None or sp_invalid is None:
        return
    sp_valid = np.asarray(sp_valid)
    sp_invalid = np.asarray(sp_invalid)
    if sp_valid.size == 0 or sp_invalid.size == 0:
        return
    positions = np.arange(sp_valid.shape[1])
    v_mu, v_sd = sp_valid.mean(axis=0), sp_valid.std(axis=0)
    i_mu, i_sd = sp_invalid.mean(axis=0), sp_invalid.std(axis=0)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(positions, v_mu, color="C0", label="valid mean")
    ax.fill_between(positions, v_mu - v_sd, v_mu + v_sd, color="C0", alpha=0.2)
    ax.plot(positions, i_mu, color="C3", label="invalid mean")
    ax.fill_between(positions, i_mu - i_sd, i_mu + i_sd, color="C3", alpha=0.2)
    ax.set_xlabel("position")
    ax.set_ylabel("spilled energy  (lower = more anomalous)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_all_plots(
    out_dir: Path | str,
    *,
    scores: dict[str, np.ndarray | None],
    loss_history: list[dict[str, float]] | None = None,
) -> None:
    """Write ROC, histograms, loss, per-sequence plots under ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    v_a = scores.get("auditor_valid")
    i_a = scores.get("auditor_invalid")
    v_s = scores.get("spilled_valid")
    i_s = scores.get("spilled_invalid")

    methods: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if v_a is not None and i_a is not None and v_a.size and i_a.size:
        methods["auditor"] = (v_a, i_a)
    if v_s is not None and i_s is not None and v_s.size and i_s.size:
        methods["spilled energy"] = (v_s, i_s)
    if methods:
        plot_roc(methods, out / "roc.png")

    if v_a is not None and i_a is not None and v_a.size and i_a.size:
        plot_score_histograms(
            v_a, i_a, out / "hist_auditor.png",
            title="auditor: valid vs invalid", xlabel="GP variance",
        )
    if v_s is not None and i_s is not None and v_s.size and i_s.size:
        plot_score_histograms(
            v_s, i_s, out / "hist_spilled.png",
            title="spilled energy: valid vs invalid", xlabel="-mean ΔE",
        )

    plot_sequence_spilled(
        scores.get("spilled_seq_valid"),
        scores.get("spilled_seq_invalid"),
        out / "sequence_spilled.png",
    )
    if loss_history:
        plot_loss_curve(loss_history, out / "loss_curve.png")
