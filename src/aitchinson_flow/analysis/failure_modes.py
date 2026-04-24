"""Failure-mode analysis: identify sequences where UQ components disagree.

A "disagreement" occurs when one component flags a sequence as high-energy
(above its percentile threshold) while another does not.  This surfaces:
  - Sequences with structural anomalies undetected by the contextual signal
  - Sequences the LLM treats as plausible but that are structurally corrupt
  - Spilled-energy outliers not captured by either GP signal

Three disagreement modes
------------------------
structural_not_contextual
    Structural energy high, contextual energy low (geometry anomaly,
    semantically plausible to the LLM).
contextual_not_structural
    Contextual energy high, structural energy low (semantically surprising
    but structurally valid; e.g. factual hallucination).
gp_not_spilled / spilled_not_gp
    Either GP signal fires but spilled energy does not, or vice versa.

Public API
----------
find_disagreements
    Compute disagreement masks and summary statistics.
DisagreementReport
    Container returned by :func:`find_disagreements`.
format_inspection_table
    Pretty-print a subset of disagreeing sequences for manual review.
save_disagreement_report
    Write a JSON report + optional CSV to disk.
plot_disagreement_scatter
    2-D scatter of structural vs contextual energy with disagreements marked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class DisagreementReport:
    """Output of :func:`find_disagreements`.

    Attributes:
        n_total: Total number of sequences analysed.
        thresholds: Dict of ``{signal_name: threshold_value}`` used.
        structural_not_contextual: Indices where structural energy is high but
            contextual energy is not.
        contextual_not_structural: Indices where contextual energy is high but
            structural energy is not.
        gp_not_spilled: Indices where either GP energy is high but spilled
            energy is not.
        spilled_not_gp: Indices where spilled energy is high but neither GP
            energy is.
        both_agree_high: Indices where all available signals are high
            (consensus anomalies).
        all_agree_low: Indices where all available signals are low (consensus
            in-distribution).
        labels: Optional label array (0=valid, 1=invalid) of length
            ``n_total``, used to compute per-disagreement-type precision.
        precision: Dict ``{disagreement_type: precision}`` (fraction of
            disagreeing sequences that are truly invalid).
    """

    n_total: int
    thresholds: dict[str, float]
    structural_not_contextual: np.ndarray
    contextual_not_structural: np.ndarray
    gp_not_spilled: np.ndarray
    spilled_not_gp: np.ndarray
    both_agree_high: np.ndarray
    all_agree_low: np.ndarray
    labels: np.ndarray | None = None
    precision: dict[str, float] = field(default_factory=dict)

    def summary(self) -> dict[str, int | float]:
        """Return a scalar summary dict suitable for JSON serialisation."""
        out: dict[str, int | float] = {
            "n_total": self.n_total,
            "n_structural_not_contextual": int(len(self.structural_not_contextual)),
            "n_contextual_not_structural": int(len(self.contextual_not_structural)),
            "n_gp_not_spilled": int(len(self.gp_not_spilled)),
            "n_spilled_not_gp": int(len(self.spilled_not_gp)),
            "n_both_agree_high": int(len(self.both_agree_high)),
            "n_all_agree_low": int(len(self.all_agree_low)),
        }
        out.update({f"precision_{k}": v for k, v in self.precision.items()})
        return out


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def _percentile_threshold(arr: np.ndarray, percentile: float) -> float:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, percentile))


def find_disagreements(
    structural_energy: np.ndarray | None,
    contextual_energy: np.ndarray | None,
    spilled_energy: np.ndarray | None,
    *,
    labels: np.ndarray | None = None,
    threshold_percentile: float = 75.0,
) -> DisagreementReport:
    """Find sequences where at least one UQ component disagrees with another.

    Thresholds are computed per-signal from the provided samples at
    ``threshold_percentile``.  A signal is "high" when its value exceeds
    the threshold.

    Args:
        structural_energy: Component 1 sequence-level energy, shape ``(N,)``.
        contextual_energy: Component 2 sequence-level energy, shape ``(N,)``.
        spilled_energy: Component 3 sequence-level anomaly scores, shape
            ``(N,)``.  Pass per-sequence mean values.
        labels: Optional ground-truth labels, shape ``(N,)``; 0=valid,
            1=invalid.
        threshold_percentile: Percentile used to binarise each signal.

    Returns:
        :class:`DisagreementReport`.
    """
    signals: dict[str, np.ndarray | None] = {
        "structural": None if structural_energy is None else np.asarray(structural_energy, dtype=np.float64).ravel(),
        "contextual": None if contextual_energy is None else np.asarray(contextual_energy, dtype=np.float64).ravel(),
        "spilled":    None if spilled_energy    is None else np.asarray(spilled_energy,    dtype=np.float64).ravel(),
    }

    present = {k: v for k, v in signals.items() if v is not None}
    if not present:
        raise ValueError("At least one signal must be provided")

    n = min(v.shape[0] for v in present.values())
    for k in present:
        present[k] = present[k][:n]

    thresholds: dict[str, float] = {}
    high_masks: dict[str, np.ndarray] = {}
    for name, arr in present.items():
        thr = _percentile_threshold(arr, threshold_percentile)
        thresholds[name] = thr
        high_masks[name] = arr > thr

    def _get(name: str) -> np.ndarray | None:
        return high_masks.get(name)

    h_struct = _get("structural")
    h_ctx = _get("contextual")
    h_spill = _get("spilled")
    h_gp = (
        (h_struct | h_ctx) if (h_struct is not None and h_ctx is not None)
        else h_struct if h_struct is not None
        else h_ctx
    )

    # structural high, contextual not high (or contextual absent)
    if h_struct is not None and h_ctx is not None:
        struct_not_ctx = np.where(h_struct & ~h_ctx)[0]
        ctx_not_struct = np.where(h_ctx & ~h_struct)[0]
    elif h_struct is not None:
        struct_not_ctx = np.where(h_struct)[0]
        ctx_not_struct = np.empty(0, dtype=np.intp)
    elif h_ctx is not None:
        struct_not_ctx = np.empty(0, dtype=np.intp)
        ctx_not_struct = np.where(h_ctx)[0]
    else:
        struct_not_ctx = np.empty(0, dtype=np.intp)
        ctx_not_struct = np.empty(0, dtype=np.intp)

    # GP high, spilled not high
    if h_gp is not None and h_spill is not None:
        gp_not_spill = np.where(h_gp & ~h_spill)[0]
        spill_not_gp = np.where(h_spill & ~h_gp)[0]
    elif h_gp is not None:
        gp_not_spill = np.where(h_gp)[0]
        spill_not_gp = np.empty(0, dtype=np.intp)
    elif h_spill is not None:
        gp_not_spill = np.empty(0, dtype=np.intp)
        spill_not_gp = np.where(h_spill)[0]
    else:
        gp_not_spill = np.empty(0, dtype=np.intp)
        spill_not_gp = np.empty(0, dtype=np.intp)

    all_high_masks = [m for m in [h_struct, h_ctx, h_spill] if m is not None]
    if all_high_masks:
        consensus_high = np.ones(n, dtype=bool)
        consensus_low = np.ones(n, dtype=bool)
        for m in all_high_masks:
            consensus_high &= m
            consensus_low &= ~m
        both_agree_high = np.where(consensus_high)[0]
        all_agree_low = np.where(consensus_low)[0]
    else:
        both_agree_high = np.empty(0, dtype=np.intp)
        all_agree_low = np.arange(n, dtype=np.intp)

    report = DisagreementReport(
        n_total=n,
        thresholds=thresholds,
        structural_not_contextual=struct_not_ctx,
        contextual_not_structural=ctx_not_struct,
        gp_not_spilled=gp_not_spill,
        spilled_not_gp=spill_not_gp,
        both_agree_high=both_agree_high,
        all_agree_low=all_agree_low,
        labels=np.asarray(labels)[:n] if labels is not None else None,
    )

    # Compute precision for each disagreement type
    if report.labels is not None:
        lbl = report.labels
        for dtype, idx in [
            ("structural_not_contextual", struct_not_ctx),
            ("contextual_not_structural", ctx_not_struct),
            ("gp_not_spilled", gp_not_spill),
            ("spilled_not_gp", spill_not_gp),
            ("both_agree_high", both_agree_high),
        ]:
            if len(idx) > 0:
                report.precision[dtype] = float(lbl[idx].mean())

    return report


# ---------------------------------------------------------------------------
# Inspection / formatting
# ---------------------------------------------------------------------------


def format_inspection_table(
    report: DisagreementReport,
    *,
    disagreement_type: str = "structural_not_contextual",
    structural_energy: np.ndarray | None = None,
    contextual_energy: np.ndarray | None = None,
    spilled_energy: np.ndarray | None = None,
    texts: Sequence[str] | None = None,
    max_rows: int = 20,
) -> str:
    """Return a plain-text table of the top disagreeing sequences.

    Args:
        report: Output of :func:`find_disagreements`.
        disagreement_type: Which disagreement set to inspect; one of
            ``"structural_not_contextual"``, ``"contextual_not_structural"``,
            ``"gp_not_spilled"``, ``"spilled_not_gp"``, ``"both_agree_high"``.
        structural_energy: Full energy arrays for display.
        contextual_energy: Full energy arrays for display.
        spilled_energy: Full energy arrays for display.
        texts: Optional sequence of raw text strings for each sequence.
        max_rows: Maximum rows to display.

    Returns:
        Multi-line string table.
    """
    idx_arr = getattr(report, disagreement_type, None)
    if idx_arr is None or len(idx_arr) == 0:
        return f"No sequences in disagreement type '{disagreement_type}'."

    rows = idx_arr[:max_rows]
    lines = [f"Disagreement type: {disagreement_type}  (showing {len(rows)} of {len(idx_arr)})"]
    lines.append("")

    col_widths = {"idx": 6, "struct_e": 10, "ctx_e": 10, "spill_e": 10, "label": 7}
    header = (
        f"{'idx':>6}  {'struct_e':>10}  {'ctx_e':>10}  {'spill_e':>10}  {'label':>7}"
    )
    if texts is not None:
        header += "  text"
    lines.append(header)
    lines.append("-" * len(header))

    se = np.asarray(structural_energy) if structural_energy is not None else None
    ce = np.asarray(contextual_energy) if contextual_energy is not None else None
    spe = np.asarray(spilled_energy) if spilled_energy is not None else None

    for i in rows:
        se_val = f"{float(se.ravel()[i]):.4f}" if se is not None and i < se.ravel().shape[0] else "–"
        ce_val = f"{float(ce.ravel()[i]):.4f}" if ce is not None and i < ce.ravel().shape[0] else "–"
        spe_val = f"{float(spe.ravel()[i]):.4f}" if spe is not None and i < spe.ravel().shape[0] else "–"
        lbl = (
            "invalid" if report.labels is not None and report.labels[i] == 1
            else "valid" if report.labels is not None
            else "–"
        )
        row = f"{i:>6}  {se_val:>10}  {ce_val:>10}  {spe_val:>10}  {lbl:>7}"
        if texts is not None and i < len(texts):
            snippet = texts[i][:80].replace("\n", " ")
            row += f"  {snippet}"
        lines.append(row)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_disagreement_report(
    report: DisagreementReport,
    out_dir: Path | str,
    *,
    label: str = "phase5",
) -> dict[str, str]:
    """Write the disagreement report to JSON and per-type index files.

    Args:
        report: Output of :func:`find_disagreements`.
        out_dir: Output directory.
        label: Filename prefix.

    Returns:
        Dict mapping artifact key to file path.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    summary_path = out / f"{label}_disagreement_summary.json"
    summary_data = {
        "summary": report.summary(),
        "thresholds": report.thresholds,
    }
    summary_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
    written["summary"] = str(summary_path)

    for dtype in (
        "structural_not_contextual",
        "contextual_not_structural",
        "gp_not_spilled",
        "spilled_not_gp",
        "both_agree_high",
        "all_agree_low",
    ):
        idx = getattr(report, dtype)
        if len(idx) == 0:
            continue
        idx_path = out / f"{label}_disagreement_{dtype}_indices.json"
        idx_path.write_text(json.dumps(idx.tolist()), encoding="utf-8")
        written[dtype] = str(idx_path)

    return written


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_disagreement_scatter(
    structural_energy: np.ndarray | None,
    contextual_energy: np.ndarray | None,
    report: DisagreementReport,
    out_path: Path | str,
    *,
    max_points: int = 3000,
    title: str = "Component 1 vs Component 2 energy: disagreements",
) -> None:
    """Scatter of structural vs contextual energy with disagreements marked.

    Points are colour-coded:
      - Grey: both agree (or only one signal present)
      - Blue: structural high, contextual low
      - Orange: contextual high, structural low
      - Red stars: both signals agree high (consensus anomalies)

    Args:
        structural_energy: Shape ``(N,)`` — Component 1 sequence energies.
        contextual_energy: Shape ``(N,)`` — Component 2 sequence energies.
        report: Output of :func:`find_disagreements`.
        out_path: Output PNG path.
        max_points: Subsample background to this many points for clarity.
        title: Figure title.
    """
    if structural_energy is None or contextual_energy is None:
        return

    se = np.asarray(structural_energy, dtype=np.float64).ravel()
    ce = np.asarray(contextual_energy, dtype=np.float64).ravel()
    n = min(se.shape[0], ce.shape[0])
    se, ce = se[:n], ce[:n]

    rng = np.random.default_rng(0)
    all_idx = np.arange(n)
    bg_mask = np.ones(n, dtype=bool)
    for spec_idx in [
        report.structural_not_contextual,
        report.contextual_not_structural,
        report.both_agree_high,
    ]:
        if len(spec_idx):
            bg_mask[spec_idx] = False
    bg_idx = all_idx[bg_mask]
    if bg_idx.shape[0] > max_points:
        bg_idx = rng.choice(bg_idx, size=max_points, replace=False)

    fig, ax = plt.subplots(figsize=(6, 5.5))

    if bg_idx.size:
        ax.scatter(se[bg_idx], ce[bg_idx], s=5, alpha=0.3, color="grey",
                   label=f"other (n={bg_idx.size})", rasterized=True)

    snc = report.structural_not_contextual
    if len(snc):
        ax.scatter(se[snc], ce[snc], s=20, alpha=0.7, color="C0",
                   label=f"struct↑ ctx↓ (n={len(snc)})", zorder=3)

    cns = report.contextual_not_structural
    if len(cns):
        ax.scatter(se[cns], ce[cns], s=20, alpha=0.7, color="C1",
                   label=f"ctx↑ struct↓ (n={len(cns)})", zorder=3)

    bah = report.both_agree_high
    if len(bah):
        ax.scatter(se[bah], ce[bah], s=60, marker="*", color="C3",
                   edgecolors="darkred", linewidths=0.5,
                   label=f"both high (n={len(bah)})", zorder=4)

    # Threshold lines
    thr_s = report.thresholds.get("structural")
    thr_c = report.thresholds.get("contextual")
    if thr_s is not None and np.isfinite(thr_s):
        ax.axvline(thr_s, color="C0", linestyle="--", linewidth=0.9, alpha=0.6)
    if thr_c is not None and np.isfinite(thr_c):
        ax.axhline(thr_c, color="C1", linestyle="--", linewidth=0.9, alpha=0.6)

    ax.set_xlabel("structural energy (C1)")
    ax.set_ylabel("contextual energy (C2)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def run_failure_mode_analysis(
    structural_energy: np.ndarray | None,
    contextual_energy: np.ndarray | None,
    spilled_energy: np.ndarray | None,
    out_dir: Path | str,
    *,
    labels: np.ndarray | None = None,
    threshold_percentile: float = 75.0,
    label: str = "phase5",
    print_tables: bool = False,
) -> DisagreementReport:
    """Full failure-mode analysis pipeline: find, save, and plot.

    Args:
        structural_energy: Component 1 sequence-level energies.
        contextual_energy: Component 2 sequence-level energies.
        spilled_energy: Component 3 sequence-level anomaly scores.
        out_dir: Directory to write output files.
        labels: Ground-truth labels (0=valid, 1=invalid).
        threshold_percentile: Percentile threshold for binarising each signal.
        label: Output filename prefix.
        print_tables: Print inspection tables to stdout.

    Returns:
        :class:`DisagreementReport`.
    """
    report = find_disagreements(
        structural_energy, contextual_energy, spilled_energy,
        labels=labels,
        threshold_percentile=threshold_percentile,
    )

    save_disagreement_report(report, out_dir, label=label)

    scatter_path = Path(out_dir) / f"{label}_disagreement_scatter.png"
    plot_disagreement_scatter(
        structural_energy, contextual_energy, report, scatter_path
    )

    if print_tables:
        for dtype in ("structural_not_contextual", "contextual_not_structural",
                      "gp_not_spilled", "spilled_not_gp"):
            print(format_inspection_table(
                report,
                disagreement_type=dtype,
                structural_energy=structural_energy,
                contextual_energy=contextual_energy,
                spilled_energy=spilled_energy,
            ))
            print()

    return report


__all__ = [
    "DisagreementReport",
    "find_disagreements",
    "format_inspection_table",
    "save_disagreement_report",
    "plot_disagreement_scatter",
    "run_failure_mode_analysis",
]
