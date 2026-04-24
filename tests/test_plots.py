"""Smoke tests for stage-aware diagnostic plots."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aitchinson_flow.plots.plots import (
    StagePlotData,
    plot_benchmark_table,
    plot_corruption_comparison,
    plot_loss_curves,
    plot_total_loss,
    save_stage_plots,
)


def _assert_file_nonempty(path: Path) -> None:
    assert path.exists(), f"expected plot {path} to be written"
    assert path.stat().st_size > 0, f"plot {path} is empty"


def test_save_stage_plots_minimal_stage1(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    data = StagePlotData(
        stage="stage1",
        energy_seq_valid=rng.normal(0.0, 1.0, size=8),
        energy_seq_invalid=rng.normal(1.0, 1.0, size=8),
    )
    written = save_stage_plots(tmp_path, data)
    assert "hist_sequence_energy" in written
    _assert_file_nonempty(tmp_path / written["hist_sequence_energy"])
    # No token matrices, no auxiliary scores → no ROC, no corruption diagnostics.
    assert "roc_combined" not in written
    assert "auroc_summary" not in written
    assert "corrupt_vs_clean_energy" not in written
    assert "heatmap_token_energy_valid" not in written


def test_save_stage_plots_full_stage2(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    n, L = 8, 10
    energy_valid = rng.normal(0.0, 1.0, size=(n, L))
    energy_invalid = rng.normal(1.0, 1.0, size=(n, L))
    variance_valid = rng.gamma(1.0, 0.3, size=(n, L))
    variance_invalid = rng.gamma(2.0, 0.3, size=(n, L))
    mask = (rng.random((n, L)) > 0.8).astype(np.int8)

    data = StagePlotData(
        stage="stage2",
        energy_token_valid=energy_valid,
        energy_token_invalid=energy_invalid,
        variance_token_valid=variance_valid,
        variance_token_invalid=variance_invalid,
        energy_seq_valid=energy_valid.mean(axis=1),
        energy_seq_invalid=energy_invalid.mean(axis=1),
        variance_seq_valid=variance_valid.mean(axis=1),
        variance_seq_invalid=variance_invalid.mean(axis=1),
        corrupt_mask_invalid=mask,
        auditor_score_valid=rng.normal(0.0, 1.0, size=n),
        auditor_score_invalid=rng.normal(1.0, 1.0, size=n),
        spilled_seq_scalar_valid=rng.normal(0.0, 1.0, size=n),
        spilled_seq_scalar_invalid=rng.normal(0.5, 1.0, size=n),
        spilled_token_valid=rng.normal(0.0, 1.0, size=(n, L - 1)),
        spilled_token_invalid=rng.normal(0.5, 1.0, size=(n, L - 1)),
        history=[
            {"loss": 1.0, "velocity_loss": 0.7, "kl": 1e-4, "val_loss": 1.1},
            {"loss": 0.8, "velocity_loss": 0.5, "kl": 0.9e-4, "val_loss": 0.9},
        ],
    )

    written = save_stage_plots(tmp_path, data)
    for key in (
        "loss_components",
        "loss_total",
        "hist_sequence_energy",
        "hist_sequence_variance",
        "heatmap_token_energy_invalid",
        "heatmap_token_variance_invalid",
        "corrupt_vs_clean_energy",
        "corrupt_vs_clean_variance",
        "roc_combined",
        "auroc_summary",
    ):
        assert key in written, f"missing {key}"
        _assert_file_nonempty(tmp_path / written[key])

    summary = json.loads((tmp_path / written["auroc_summary"]).read_text())
    assert isinstance(summary, dict)
    assert len(summary) >= 2
    for name, auc in summary.items():
        assert 0.0 <= auc <= 1.0, f"{name} AUROC out of range: {auc}"


def test_plot_loss_curves_subplots(tmp_path: Path) -> None:
    history = [
        {"loss": 1.0, "velocity_loss": 0.7, "mask_loss": 0.2, "val_loss": 1.1},
        {"loss": 0.8, "velocity_loss": 0.5, "mask_loss": 0.15, "val_loss": 0.9},
        {"loss": 0.6, "velocity_loss": 0.4, "mask_loss": 0.1, "val_loss": 0.7},
    ]
    out = tmp_path / "losses.png"
    plot_loss_curves(history, out, title="test losses")
    _assert_file_nonempty(out)

    total_out = tmp_path / "total.png"
    plot_total_loss(history, total_out)
    _assert_file_nonempty(total_out)


def test_plot_corruption_comparison_shape_mismatch(tmp_path: Path) -> None:
    values = np.random.default_rng(0).normal(0, 1, size=(8, 10))
    bad_mask = np.zeros((8, 5), dtype=np.int8)
    out = tmp_path / "compare.png"
    plot_corruption_comparison(
        values, bad_mask, out, title="shape mismatch", xlabel="x"
    )
    assert not out.exists(), "plot must not be written on shape mismatch"


def test_plot_benchmark_table_handles_missing_metric(tmp_path: Path) -> None:
    rows = [
        {
            "component": "structural",
            "task": "text_audit",
            "scale": "baseline",
            "auroc_combined": 0.91,
        },
        {
            "component": "contextual",
            "task": "trivia_audit",
            "scale": "baseline",
            "auroc_combined": None,
        },
    ]
    out = tmp_path / "benchmark_table.png"
    summary = plot_benchmark_table(rows, out_path=out, metric="auroc_combined", aggregate="best")
    _assert_file_nonempty(out)
    sidecar = out.with_suffix(".json")
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["metric"] == "auroc_combined"
    assert summary["values"]["contextual"]["trivia_audit"] is None
