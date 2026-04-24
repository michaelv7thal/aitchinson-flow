"""Phase 4 healing evaluation script.

Evaluates all three Phase 4 healing strategies (targeted_resample,
simplex_project, beam_rerank) plus the legacy EqM baseline on a trained
auditor checkpoint, then writes a consolidated JSON results file and prints
a summary table.

Usage
-----
::

    python scripts/run_healing_eval.py \\
        --auditor-ckpt checkpoints/stage2_epoch25.pt \\
        --healer-ckpt  checkpoints/stage1_epoch25.pt \\
        --output       results/healing_eval.json

Optional flags:
    ``--strategies``    Comma-separated subset, e.g. ``targeted_resample,eqm``.
    ``--threshold``     Per-token / sequence energy threshold (default 1.0).
    ``--max-iter``      Max resample iterations (default 5).
    ``--lambda-energy`` Beam reranking energy penalty λ (default 1.0).
    ``--heal-steps``    EqM integration steps (default: healer model default).
    ``--data-source``   Benchmark data source: ``text8`` | ``dna`` | ``medical``
                        | ``trivia`` (default ``text8``).
    ``--n-batches``     Evaluation batches per strategy (default 20).
    ``--no-train``      Skip training; assume checkpoint is already trained.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from _shared.bootstrap import bootstrap_repo_paths

bootstrap_repo_paths(Path(__file__))

from aitchinson_flow.config import Config, HealingConfig  # noqa: E402
from aitchinson_flow.models.factory import build_model  # noqa: E402
from aitchinson_flow.training.datamodule import DataModule  # noqa: E402
from benchmarks.tasks.healing_audit import HealingAuditTask  # noqa: E402


_ALL_STRATEGIES = ("targeted_resample", "simplex_project", "beam_rerank", "eqm")

_DATA_SOURCE_MAP: dict[str, str] = {
    "text8": "text_audit",
    "dna": "dna_audit",
    "medical": "medical_audit",
    "trivia": "trivia_audit",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Phase 4 healing evaluation.")
    p.add_argument("--auditor-ckpt", type=str, required=True,
                   help="Path to trained auditor checkpoint (Stage 2 or BayesianAuditor).")
    p.add_argument("--healer-ckpt", type=str, required=True,
                   help="Path to trained healer checkpoint (Stage 1 / EquilibriumAuditor).")
    p.add_argument("--output", type=str, default="results/healing_eval.json",
                   help="Path for consolidated JSON results.")
    p.add_argument("--strategies", type=str, default=",".join(_ALL_STRATEGIES),
                   help="Comma-separated healing strategies to evaluate.")
    p.add_argument("--threshold", type=float, default=1.0,
                   help="Energy threshold for flagging tokens/sequences.")
    p.add_argument("--max-iter", type=int, default=5,
                   help="Max resampling iterations (targeted_resample only).")
    p.add_argument("--lambda-energy", type=float, default=1.0,
                   help="Energy penalty weight λ for beam reranking.")
    p.add_argument("--heal-steps", type=int, default=None,
                   help="EqM integration steps (None → healer default).")
    p.add_argument("--data-source", type=str, default="text8",
                   choices=list(_DATA_SOURCE_MAP),
                   help="Benchmark data source.")
    p.add_argument("--n-batches", type=int, default=20,
                   help="Number of evaluation batches per strategy.")
    p.add_argument("--no-train", action="store_true",
                   help="Skip training; evaluate checkpoint directly.")
    return p.parse_args(argv)


def _load_auditor(ckpt_path: str, cfg: Config):
    """Load and return auditor from checkpoint."""
    import torch  # noqa: PLC0415

    model = build_model(cfg)
    state = torch.load(ckpt_path, map_location="cpu")
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model


def _build_datamodule(cfg: Config, data_source: str) -> DataModule:
    """Build the evaluation datamodule for the chosen data source."""
    from aitchinson_flow.training.data_sources import build_training_datamodule  # noqa: PLC0415

    source_map = {
        "text8": "raw_text",
        "dna": "dna",
        "medical": "medical",
        "trivia": "qa_pairs",
    }
    cfg_patched = replace(
        cfg,
        training_data=replace(cfg.training_data, source=source_map[data_source]),
    )
    return build_training_datamodule(cfg_patched)


def _run_strategy(
    strategy: str,
    auditor,
    cfg: Config,
    datamodule: DataModule,
) -> dict[str, Any]:
    """Run a single healing strategy and return its result dict."""
    cfg_with_strategy = replace(
        cfg,
        healing=replace(cfg.healing, strategy=strategy),
    )
    task = HealingAuditTask()
    return task.run(auditor, datamodule, cfg_with_strategy)


def _print_table(rows: list[dict[str, Any]]) -> None:
    """Print a compact comparison table to stdout."""
    header = (
        f"{'Strategy':<22} {'pre_auroc':>9} {'post_auroc':>10} "
        f"{'Δauroc':>8} {'energy_red':>11} {'success':>8} {'tok_acc':>9}"
    )
    print("\n" + "-" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        tok_acc = r.get("token_accuracy", float("nan"))
        tok_str = f"{tok_acc:.4f}" if tok_acc == tok_acc else "  n/a  "
        print(
            f"{r['strategy']:<22} "
            f"{r['pre_auroc']:>9.4f} "
            f"{r['post_auroc']:>10.4f} "
            f"{r['auroc_improvement']:>8.4f} "
            f"{r.get('mean_energy_reduction', float('nan')):>11.4f} "
            f"{r['healing_success_rate']:>8.4f} "
            f"{tok_str:>9}"
        )
    print("-" * len(header) + "\n")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    unknown = [s for s in strategies if s not in _ALL_STRATEGIES]
    if unknown:
        raise ValueError(f"Unknown strategies: {unknown}; valid: {_ALL_STRATEGIES}")

    cfg = Config()
    # Wire healer checkpoint into benchmark config
    cfg = replace(
        cfg,
        benchmark=replace(
            cfg.benchmark,
            healer_ckpt=args.healer_ckpt,
            train_before_eval=not args.no_train,
            n_batches=args.n_batches,
        ),
        healing=HealingConfig(
            strategy=strategies[0],       # overridden per-run inside _run_strategy
            threshold=args.threshold,
            max_iter=args.max_iter,
            lambda_energy=args.lambda_energy,
            heal_steps=args.heal_steps,
        ),
    )

    print(f"Loading auditor from {args.auditor_ckpt} …")
    auditor = _load_auditor(args.auditor_ckpt, cfg)

    print(f"Building datamodule for data_source={args.data_source} …")
    datamodule = _build_datamodule(cfg, args.data_source)

    results: list[dict[str, Any]] = []
    for strategy in strategies:
        print(f"\nRunning strategy: {strategy} …")
        result = _run_strategy(strategy, auditor, cfg, datamodule)
        # Drop raw score arrays before serialising (not JSON-serialisable)
        serialisable = {k: v for k, v in result.items() if k != "_scores"}
        results.append(serialisable)
        print(
            f"  pre_auroc={result['pre_auroc']:.4f}  "
            f"post_auroc={result['post_auroc']:.4f}  "
            f"Δ={result['auroc_improvement']:+.4f}  "
            f"success={result['healing_success_rate']:.4f}"
        )

    _print_table(results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "data_source": args.data_source,
        "auditor_ckpt": args.auditor_ckpt,
        "healer_ckpt": args.healer_ckpt,
        "threshold": args.threshold,
        "max_iter": args.max_iter,
        "lambda_energy": args.lambda_energy,
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
