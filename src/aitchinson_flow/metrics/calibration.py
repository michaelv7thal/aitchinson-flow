"""GP variance calibration: Expected Calibration Error and reliability diagrams.

Two calibration modes
---------------------
coverage
    For N(mean, variance) predictions on hold-out valid sequences (y=0),
    the empirical coverage at confidence level α should equal α:

        coverage(α) = fraction of samples with |y − mean| ≤ z_α × σ_total

    where σ_total = sqrt(epistemic_variance + noise_var).  A well-calibrated
    GP has coverage(α) ≈ α for all α.

anomaly
    Treat the sequence-level GP mean-energy as a soft anomaly score,
    normalise to [0, 1] via min-max, bin by predicted probability, and
    compare to the fraction actually anomalous in each bin:

        ECE = Σ_b (|b| / N) × |frac_anomalous_b − mean_score_b|
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np
from scipy import stats as scipy_stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class CoverageCalibrationResult:
    """Output of :func:`coverage_calibration`.

    Attributes:
        alphas: Confidence levels tested, shape ``(n_levels,)``.
        empirical_coverage: Fraction of valid samples inside the α-credible
            interval at each level, shape ``(n_levels,)``.
        ece: Expected calibration error — mean absolute deviation from the
            diagonal, scalar.
    """

    alphas: np.ndarray
    empirical_coverage: np.ndarray
    ece: float


@dataclass
class AnomalyCalibrationResult:
    """Output of :func:`anomaly_calibration`.

    Attributes:
        bin_centers: Midpoint of each probability bin, shape ``(num_bins,)``.
        bin_fractions: Fraction of samples actually anomalous per bin,
            shape ``(num_bins,)``.
        bin_counts: Number of samples per bin, shape ``(num_bins,)``.
        ece: Expected calibration error, scalar.
    """

    bin_centers: np.ndarray
    bin_fractions: np.ndarray
    bin_counts: np.ndarray
    ece: float


@dataclass
class CalibrationReport:
    """Combined report returned by :func:`compute_calibration`.

    Attributes:
        coverage: GP interval-coverage calibration (None when insufficient data).
        anomaly: Anomaly-score ECE calibration (None when insufficient data).
        plots: Dict of ``{plot_key: absolute_path}`` for saved PNG files.
    """

    coverage: CoverageCalibrationResult | None = None
    anomaly: AnomalyCalibrationResult | None = None
    plots: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core calibration functions
# ---------------------------------------------------------------------------


def coverage_calibration(
    mean_valid: np.ndarray,
    variance_valid: np.ndarray,
    *,
    noise_var: float = 0.0,
    y_target: float = 0.0,
    alphas: Sequence[float] | None = None,
) -> CoverageCalibrationResult:
    """Compute interval-coverage calibration for GP predictions on valid data.

    Args:
        mean_valid: GP predictive means for valid samples, any shape; flattened
            internally.
        variance_valid: GP epistemic variances (matching shape to
            ``mean_valid``).
        noise_var: Homoscedastic aleatoric noise variance added to epistemic
            variance to form the total predictive variance.
        y_target: Target value the GP should predict for valid tokens (0.0).
        alphas: Confidence levels to evaluate.  Defaults to
            ``[0.05, 0.10, …, 0.95]``.

    Returns:
        :class:`CoverageCalibrationResult` with empirical coverage per α and
        the scalar ECE.
    """
    if alphas is None:
        alphas = np.linspace(0.05, 0.95, 19).tolist()
    alpha_arr = np.asarray(alphas, dtype=np.float64)

    mu = np.asarray(mean_valid, dtype=np.float64).ravel()
    var = np.asarray(variance_valid, dtype=np.float64).ravel()
    if mu.size == 0 or var.size == 0:
        raise ValueError("mean_valid and variance_valid must be non-empty")

    sigma = np.sqrt(np.clip(var + noise_var, 1e-12, None))
    residuals = np.abs(y_target - mu)

    empirical = np.empty_like(alpha_arr)
    for idx, alpha in enumerate(alpha_arr):
        z = scipy_stats.norm.ppf((1.0 + float(alpha)) / 2.0)
        empirical[idx] = float((residuals <= z * sigma).mean())

    ece = float(np.mean(np.abs(empirical - alpha_arr)))
    return CoverageCalibrationResult(
        alphas=alpha_arr,
        empirical_coverage=empirical,
        ece=ece,
    )


def anomaly_calibration(
    energy_valid: np.ndarray,
    energy_invalid: np.ndarray,
    *,
    num_bins: int = 10,
) -> AnomalyCalibrationResult:
    """Compute anomaly-score ECE treating GP energy as anomaly probability.

    The energy scores from valid and invalid sequences are combined, min-max
    normalised to [0, 1], and then binned.  In each bin the fraction of
    truly-anomalous samples is compared with the average predicted score.

    Args:
        energy_valid: Sequence-level GP mean-energy for valid samples, shape
            ``(N_valid,)``.
        energy_invalid: Sequence-level GP mean-energy for invalid samples,
            shape ``(N_invalid,)``.
        num_bins: Number of equal-width probability bins.

    Returns:
        :class:`AnomalyCalibrationResult`.
    """
    ev = np.asarray(energy_valid, dtype=np.float64).ravel()
    ei = np.asarray(energy_invalid, dtype=np.float64).ravel()
    if ev.size == 0 or ei.size == 0:
        raise ValueError("energy_valid and energy_invalid must be non-empty")

    labels = np.concatenate([np.zeros(ev.size), np.ones(ei.size)])
    scores = np.concatenate([ev, ei])

    finite_mask = np.isfinite(scores)
    labels = labels[finite_mask]
    scores = scores[finite_mask]

    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min < 1e-12:
        probs = np.full_like(scores, 0.5)
    else:
        probs = (scores - s_min) / (s_max - s_min)

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_fractions = np.full(num_bins, np.nan)
    bin_counts = np.zeros(num_bins, dtype=np.int64)

    for b in range(num_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        in_bin = (probs >= lo) & (probs < hi if b < num_bins - 1 else probs <= hi)
        bin_counts[b] = int(in_bin.sum())
        if bin_counts[b] > 0:
            bin_fractions[b] = float(labels[in_bin].mean())

    n_total = labels.size
    ece = 0.0
    for b in range(num_bins):
        if np.isfinite(bin_fractions[b]) and bin_counts[b] > 0:
            ece += (bin_counts[b] / n_total) * abs(bin_fractions[b] - bin_centers[b])

    return AnomalyCalibrationResult(
        bin_centers=bin_centers,
        bin_fractions=bin_fractions,
        bin_counts=bin_counts,
        ece=ece,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_reliability_diagram(
    result: CoverageCalibrationResult,
    out_path: Path | str,
    *,
    title: str = "GP coverage reliability diagram",
) -> None:
    """Render a reliability (calibration) diagram for interval coverage.

    The diagonal represents perfect calibration.  Points above the diagonal
    indicate over-coverage (conservative); below indicates under-coverage.

    Args:
        result: Output of :func:`coverage_calibration`.
        out_path: Path to save the PNG.
        title: Figure title.
    """
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", alpha=0.7, label="perfect calibration")
    ax.plot(
        result.alphas,
        result.empirical_coverage,
        "o-",
        color="C0",
        markersize=5,
        label=f"GP (ECE={result.ece:.4f})",
    )
    ax.fill_between(result.alphas, result.alphas, result.empirical_coverage, alpha=0.15, color="C0")
    ax.set_xlabel("expected coverage (α)")
    ax.set_ylabel("empirical coverage")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_anomaly_calibration(
    result: AnomalyCalibrationResult,
    out_path: Path | str,
    *,
    title: str = "Anomaly-score calibration diagram",
) -> None:
    """Render a calibration diagram for GP energy as an anomaly score.

    Bar height = fraction actually anomalous in that bin; bar width spans
    the bin.  A perfectly calibrated score would have bar height equal to
    the bin center (shown as a diagonal line).

    Args:
        result: Output of :func:`anomaly_calibration`.
        out_path: Path to save the PNG.
        title: Figure title.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    width = result.bin_centers[1] - result.bin_centers[0] if len(result.bin_centers) > 1 else 0.1
    fracs = np.where(np.isfinite(result.bin_fractions), result.bin_fractions, 0.0)
    ax.bar(
        result.bin_centers,
        fracs,
        width=width * 0.85,
        color="C0",
        alpha=0.7,
        label="empirical fraction anomalous",
    )
    ax.plot([0, 1], [0, 1], "--", color="grey", alpha=0.7, label="perfect calibration")
    ax.set_xlabel("predicted anomaly probability (normalised energy)")
    ax.set_ylabel("fraction anomalous")
    ax.set_title(f"{title}\nECE={result.ece:.4f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.bar(result.bin_centers, result.bin_counts, width=width * 0.85, color="C2", alpha=0.7)
    ax2.set_xlabel("predicted anomaly probability (normalised energy)")
    ax2.set_ylabel("sample count")
    ax2.set_title("sample count per bin")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def compute_calibration(
    *,
    mean_valid: np.ndarray | None = None,
    variance_valid: np.ndarray | None = None,
    noise_var: float = 0.0,
    energy_valid: np.ndarray | None = None,
    energy_invalid: np.ndarray | None = None,
    num_bins: int = 10,
    alphas: Sequence[float] | None = None,
    out_dir: Path | str | None = None,
    label: str = "gp",
) -> CalibrationReport:
    """Compute both calibration modes and optionally save diagnostic plots.

    All array arguments are optional; omitting them skips the corresponding
    calibration mode.

    Args:
        mean_valid: GP means for valid tokens/sequences (coverage calibration).
        variance_valid: GP epistemic variances for valid tokens/sequences.
        noise_var: Aleatoric noise variance (added to epistemic for coverage).
        energy_valid: Sequence-level GP energy for valid samples (anomaly ECE).
        energy_invalid: Sequence-level GP energy for invalid samples.
        num_bins: Bins for anomaly calibration histogram.
        alphas: Confidence levels for coverage calibration.
        out_dir: Directory to write PNG diagnostics.  No plots if ``None``.
        label: Prefix for output filenames.

    Returns:
        :class:`CalibrationReport` with populated fields for each mode run.
    """
    report = CalibrationReport()

    if mean_valid is not None and variance_valid is not None:
        try:
            report.coverage = coverage_calibration(
                mean_valid,
                variance_valid,
                noise_var=noise_var,
                alphas=alphas,
            )
        except Exception as exc:  # noqa: BLE001
            import warnings
            warnings.warn(f"coverage_calibration failed: {exc}", stacklevel=2)

    if energy_valid is not None and energy_invalid is not None:
        try:
            report.anomaly = anomaly_calibration(
                energy_valid,
                energy_invalid,
                num_bins=num_bins,
            )
        except Exception as exc:  # noqa: BLE001
            import warnings
            warnings.warn(f"anomaly_calibration failed: {exc}", stacklevel=2)

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if report.coverage is not None:
            path = out / f"{label}_reliability_diagram.png"
            plot_reliability_diagram(
                report.coverage, path, title=f"{label}: GP coverage reliability diagram"
            )
            report.plots["reliability_diagram"] = str(path)
        if report.anomaly is not None:
            path = out / f"{label}_anomaly_calibration.png"
            plot_anomaly_calibration(
                report.anomaly, path, title=f"{label}: anomaly-score calibration"
            )
            report.plots["anomaly_calibration"] = str(path)

    return report


__all__ = [
    "CoverageCalibrationResult",
    "AnomalyCalibrationResult",
    "CalibrationReport",
    "coverage_calibration",
    "anomaly_calibration",
    "plot_reliability_diagram",
    "plot_anomaly_calibration",
    "compute_calibration",
]
