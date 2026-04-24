"""Run the Phase 3 unified benchmark matrix and write one consolidated artifact."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from _shared.bootstrap import bootstrap_repo_paths

bootstrap_repo_paths(Path(__file__))

from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.plots import plot_benchmark_table  # noqa: E402
from benchmarks.runner import ComponentTaskSpec, run_unified_benchmark  # noqa: E402


def _default_specs() -> list[ComponentTaskSpec]:
    """Default component/task matrix for Phase 3."""
    return [
        ComponentTaskSpec(
            component="structural",
            task_name="text_audit",
            data_source="text8",
            model_name="bayesian_auditor_stage1",
        ),
        ComponentTaskSpec(
            component="structural",
            task_name="dna_audit",
            data_source="dna",
            model_name="bayesian_auditor_stage1",
        ),
        ComponentTaskSpec(
            component="structural",
            task_name="medical_audit",
            data_source="medical",
            model_name="bayesian_auditor_stage1",
        ),
        ComponentTaskSpec(
            component="contextual",
            task_name="text_audit",
            data_source="text8",
            model_name="bayesian_auditor_stage2",
        ),
        ComponentTaskSpec(
            component="contextual",
            task_name="trivia_audit",
            data_source="trivia",
            model_name="bayesian_auditor_stage2",
        ),
        ComponentTaskSpec(
            component="contextual",
            task_name="medical_audit",
            data_source="medical",
            model_name="bayesian_auditor_stage2",
        ),
        ComponentTaskSpec(
            component="spilled",
            task_name="text_audit",
            data_source="text8",
            model_name="bayesian_auditor_stage1",
        ),
        ComponentTaskSpec(
            component="spilled",
            task_name="trivia_audit",
            data_source="trivia",
            model_name="bayesian_auditor_stage1",
        ),
        ComponentTaskSpec(
            component="spilled",
            task_name="dna_audit",
            data_source="dna",
            model_name="bayesian_auditor_stage1",
        ),
        ComponentTaskSpec(
            component="spilled",
            task_name="medical_audit",
            data_source="medical",
            model_name="bayesian_auditor_stage1",
        ),
    ]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full Phase 3 benchmark matrix.")
    parser.add_argument(
        "--output",
        type=str,
        default="results/full_benchmark.json",
        help="Path to consolidated JSON results file.",
    )
    parser.add_argument(
        "--heatmap-output",
        type=str,
        default="",
        help="Optional path for benchmark heatmap PNG (defaults near --output).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="auroc_combined",
        help="Metric key for heatmap cells.",
    )
    parser.add_argument(
        "--aggregate",
        type=str,
        default="best",
        choices=["best", "mean"],
        help="How to aggregate multiple scales into one component-task heatmap cell.",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Skip train-before-eval in benchmark configs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    cfg = Config()
    if args.no_train:
        cfg.benchmark = replace(cfg.benchmark, train_before_eval=False)

    specs = _default_specs()
    results = run_unified_benchmark(cfg, specs=specs)

    output_path = Path(args.output)
    results.to_json_file(output_path)

    heatmap_path = Path(args.heatmap_output) if args.heatmap_output else output_path.with_suffix(
        ""
    ).with_name(f"{output_path.stem}_heatmap.png")
    table = plot_benchmark_table(
        results.rows,
        out_path=heatmap_path,
        metric=args.metric,
        aggregate=args.aggregate,
    )
    print(f"wrote {output_path}")
    print(f"wrote {heatmap_path}")
    print(f"rows={len(results.rows)} metric={table['metric']} aggregate={table['aggregate']}")


if __name__ == "__main__":
    main()

