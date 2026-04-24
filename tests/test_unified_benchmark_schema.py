from __future__ import annotations

from pathlib import Path

from benchmarks.results_schema import BenchmarkResultRow, FullBenchmarkResults
from benchmarks.runner import ComponentTaskSpec, _normalize_to_result_row


def test_full_benchmark_results_roundtrip(tmp_path: Path) -> None:
    row = BenchmarkResultRow(
        component="structural",
        task="text_audit",
        scale="d_model128_nhead8_num_layers4",
        auroc_energy=0.91,
        auroc_variance=0.88,
        auroc_combined=0.895,
        auroc_spilled=0.8,
        energy_gap=1.2,
        variance_ratio=2.4,
        per_token_auroc_at_corruption=0.84,
        metadata={"num_params": 1234},
    )
    payload = FullBenchmarkResults(rows=[row], config_snapshot={"seed": 7})
    path = tmp_path / "full_benchmark.json"
    payload.to_json_file(path)

    loaded = FullBenchmarkResults.from_json_file(path)
    assert len(loaded.rows) == 1
    assert loaded.rows[0].component == "structural"
    assert loaded.rows[0].task == "text_audit"
    assert loaded.rows[0].auroc_combined == 0.895
    assert loaded.rows[0].metadata["num_params"] == 1234
    assert loaded.config_snapshot["seed"] == 7


def test_normalize_to_result_row_prefers_task_metrics() -> None:
    spec = ComponentTaskSpec(
        component="contextual",
        task_name="trivia_audit",
        data_source="trivia",
        model_name="bayesian_auditor_stage2",
    )
    row = _normalize_to_result_row(
        spec=spec,
        scale_tag="baseline",
        metrics={
            "auroc_trivia": 0.81,
            "auroc_energy": 0.77,
            "var_correct_mean": 0.2,
            "var_incorrect_mean": 0.6,
            "num_params": 555,
        },
    )
    assert row.component == "contextual"
    assert row.task == "trivia_audit"
    assert row.scale == "baseline"
    assert row.auroc_variance == 0.81
    assert row.auroc_energy == 0.77
    assert row.auroc_combined == (0.81 + 0.77) / 2.0
    assert row.variance_ratio == 3.0
    assert row.metadata["num_params"] == 555

