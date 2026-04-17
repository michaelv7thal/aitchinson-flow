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
    sp_valid: np.ndarray | None,
    sp_invalid: np.ndarray | None,
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


def plot_sequence_series(
    valid: np.ndarray | None,
    invalid: np.ndarray | None,
    out_path: Path | str,
    *,
    title: str,
    ylabel: str,
) -> None:
    """Plot per-position mean ± std for valid/invalid arrays shaped ``(B, L)``."""
    if valid is None or invalid is None:
        return
    valid = np.asarray(valid)
    invalid = np.asarray(invalid)
    if valid.size == 0 or invalid.size == 0:
        return
    if valid.ndim != 2 or invalid.ndim != 2:
        return

    positions = np.arange(valid.shape[1])
    v_mu, v_sd = valid.mean(axis=0), valid.std(axis=0)
    i_mu, i_sd = invalid.mean(axis=0), invalid.std(axis=0)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(positions, v_mu, color="C0", label="valid mean")
    ax.fill_between(positions, v_mu - v_sd, v_mu + v_sd, color="C0", alpha=0.2)
    ax.plot(positions, i_mu, color="C3", label="invalid mean")
    ax.fill_between(positions, i_mu - i_sd, i_mu + i_sd, color="C3", alpha=0.2)
    ax.set_xlabel("position")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_scaling_auroc(
    results: dict[str, dict],
    out_path: Path | str,
    *,
    auroc_key: str = "auroc_auditor",
    title: str = "GP UQ quality vs backbone parameter count",
) -> None:
    """Scatter/line of AUROC vs trainable parameter count across scale entries."""
    param_counts: list[int] = []
    aurocs: list[float] = []
    labels: list[str] = []

    for tag, metrics in results.items():
        n = metrics.get("num_params")
        a = metrics.get(auroc_key)
        if n is None or a is None or not np.isfinite(a):
            continue
        param_counts.append(int(n))
        aurocs.append(float(a))
        labels.append(tag)

    if not param_counts:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(param_counts, aurocs, zorder=3)
    for x, y, lbl in zip(param_counts, aurocs, labels):
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.plot(param_counts, aurocs, "--", alpha=0.5, color="grey")
    ax.set_xlabel("trainable parameters")
    ax.set_ylabel("AUROC")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_gp_vs_spilled_auroc(
    sweep_results: dict[str, dict],
    out_path: Path | str,
    *,
    title: str = "AUROC vs corruption rate: GP auditor vs spilled energy",
) -> None:
    """Dual line chart of auditor/spilled AUROC across corruption rates."""
    rates = sorted(float(k) for k in sweep_results)
    auditor_aurocs = [sweep_results[str(r)].get("auroc_auditor", float("nan")) for r in rates]
    spilled_aurocs = [sweep_results[str(r)].get("auroc_spilled", float("nan")) for r in rates]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rates, auditor_aurocs, marker="o", label="GP auditor")
    ax.plot(rates, spilled_aurocs, marker="s", label="spilled energy")
    ax.set_xlabel("corruption rate")
    ax.set_ylabel("AUROC")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_gp_vs_spilled_scatter(
    gp_scores: np.ndarray,
    spilled_scores: np.ndarray,
    out_path: Path | str,
    *,
    title: str | None = None,
) -> None:
    """Scatter of GP variance vs spilled energy per sequence, with Pearson r."""
    gp_scores = _finite(np.asarray(gp_scores).ravel())
    spilled_scores = _finite(np.asarray(spilled_scores).ravel())
    n = min(len(gp_scores), len(spilled_scores))
    if n == 0:
        return
    gp_scores = gp_scores[:n]
    spilled_scores = spilled_scores[:n]

    r = float(np.corrcoef(gp_scores, spilled_scores)[0, 1])
    plot_title = title or f"GP var vs spilled energy  (r={r:.3f})"

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(gp_scores, spilled_scores, alpha=0.4, s=12, rasterized=True)
    ax.set_xlabel("GP variance")
    ax.set_ylabel("spilled energy score")
    ax.set_title(plot_title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_healing_trajectory(
    pre_var: np.ndarray,
    post_var: np.ndarray,
    out_path: Path | str,
    *,
    title: str = "GP variance before vs after healing",
) -> None:
    """Violin/box comparison of pre and post healing variance."""
    pre_var = _finite(np.asarray(pre_var).ravel())
    post_var = _finite(np.asarray(post_var).ravel())
    if pre_var.size == 0:
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.violinplot([pre_var, post_var], positions=[0, 1], showmedians=True)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["before healing", "after healing"])
    ax.set_ylabel("GP variance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_healing_success_by_corruption(
    results_by_rate: dict[str, dict],
    out_path: Path | str,
    *,
    title: str = "Healing success rate vs corruption level",
) -> None:
    """Line plot of healing success rate across corruption rates."""
    rates = sorted(float(k) for k in results_by_rate)
    success = [results_by_rate[str(r)].get("healing_success_rate", float("nan")) for r in rates]
    post_auroc = [results_by_rate[str(r)].get("post_auroc", float("nan")) for r in rates]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rates, success, marker="o", label="success rate")
    ax.plot(rates, post_auroc, marker="s", linestyle="--", label="post-healing AUROC")
    ax.set_xlabel("corruption rate")
    ax.set_ylabel("metric value")
    ax.set_title(title)
    ax.set_ylim(0, 1)
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
    plot_sequence_series(
        scores.get("auditor_energy_seq_valid"),
        scores.get("auditor_energy_seq_invalid"),
        out / "sequence_auditor_energy.png",
        title="auditor energy per position",
        ylabel="auditor energy",
    )
    plot_sequence_series(
        scores.get("auditor_var_seq_valid"),
        scores.get("auditor_var_seq_invalid"),
        out / "sequence_auditor_variance.png",
        title="auditor variance per position",
        ylabel="auditor variance",
    )

    # GP vs spilled energy scatter (all valid sequences)
    gp_v = scores.get("auditor_valid")
    sp_v = scores.get("spilled_valid")
    if gp_v is not None and sp_v is not None and gp_v.size and sp_v.size:
        plot_gp_vs_spilled_scatter(gp_v, sp_v, out / "gp_vs_spilled_valid.png")
    gp_i = scores.get("auditor_invalid")
    sp_i = scores.get("spilled_invalid")
    if gp_i is not None and sp_i is not None and gp_i.size and sp_i.size:
        plot_gp_vs_spilled_scatter(gp_i, sp_i, out / "gp_vs_spilled_invalid.png")

    # Healing trajectory (if scores include pre/post variance)
    pre_inv = scores.get("pre_invalid")
    post_inv = scores.get("post_invalid")
    if pre_inv is not None and post_inv is not None:
        plot_healing_trajectory(pre_inv, post_inv, out / "healing_trajectory.png")

    if loss_history:
        plot_loss_curve(loss_history, out / "loss_curve.png")
