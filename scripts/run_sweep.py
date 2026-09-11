"""Sweep orchestrator: run a list of {name, overrides} entries end-to-end.

Each entry produces:
  runs/<name>/config.json           full Config dump
  runs/<name>/history.jsonl         one entry per epoch
  runs/<name>/eval.json             eval_full.py output for the final ckpt
  runs/<name>/epoch_final.pt        final-only model checkpoint
and appends one row to runs/sweep_results.jsonl.

Idempotent: if runs/<name>/eval.json already exists the cell is skipped, so
the script can be safely re-run after a context reset.

Sweep spec format (YAML or JSON, list of objects). `overrides` is a flat dict
of dotted keys, e.g. `training.epochs`, `transformer.d_model`, `eqm.lambda_ce`.
A `null` value resets the field to its dataclass default.

Example:
    - name: ep25_default
      overrides: {training.epochs: 25}
    - name: bb_d512_l6
      overrides: {transformer.d_model: 512, transformer.nhead: 8, transformer.num_layers: 6}

Usage:
    python scripts/run_sweep.py --sweep sweeps/phase1.yaml [--only ep25_default] [--n 256] [--steps 200]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402

import aitchinson_flow.models  # noqa: E402,F401 — populate registry
from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.training import (  # noqa: E402
    build_training_datamodule,
    build_wandb_logger,
    fit,
    seed_all,
)

from scripts.eval_full import evaluate_checkpoint  # noqa: E402


def _set_dotted(cfg: Config, key: str, value: Any) -> None:
    """Apply a flat dotted override to a nested dataclass."""
    parts = key.split(".")
    *parents, leaf = parts
    obj: Any = cfg
    parent_chain: list[tuple[Any, str]] = []
    for p in parents:
        parent_chain.append((obj, p))
        obj = getattr(obj, p)
    if not is_dataclass(obj):
        raise TypeError(f"override path '{key}' does not end at a dataclass field")
    if leaf not in {f.name for f in fields(obj)}:
        raise KeyError(
            f"unknown override field '{key}' (no '{leaf}' in {type(obj).__name__})"
        )
    new = replace(obj, **{leaf: value})
    # Walk back up replacing each parent so we don't mutate frozen instances.
    for parent, name in reversed(parent_chain):
        new = (
            replace(parent, **{name: new})
            if is_dataclass(parent) and parent is not cfg
            else new
        )
        if parent is cfg:
            setattr(cfg, name, new)
            return
    # If we got here, the leaf field lives directly on cfg, which doesn't happen
    # because Config has only nested dataclass fields, but cover it for safety.
    setattr(cfg, parents[0], new)


def _config_to_dict(cfg: Config) -> dict[str, Any]:
    d = asdict(cfg)
    d["training"]["device"] = str(cfg.training.device)
    if "loader_settings" in d:
        d["loader_settings"]["device_type"] = str(cfg.loader_settings.device_type)
    return d


def _build_cfg(overrides: dict[str, Any]) -> Config:
    cfg = Config()
    for k, v in overrides.items():
        _set_dotted(cfg, k, v)
    return cfg


def _run_one(
    name: str,
    overrides: dict[str, Any],
    runs_root: Path,
    *,
    eval_n: int,
    eval_steps: int,
    redo_eval: bool,
    eval_only: bool = False,
    eval_only_ckpt: str | Path | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    eval_steps_override: int | None = None,
    wandb_enabled: bool = False,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_group: str | None = None,
) -> dict[str, Any]:
    run_dir = runs_root / name
    eval_path = run_dir / "eval.json"
    if eval_path.exists() and not redo_eval:
        print(f"[skip] {name}: eval.json exists")
        return json.loads(eval_path.read_text())

    # Eval-only branch (Phase R): skip training, evaluate the supplied ckpt
    # directly with the given sample_kwargs. eval_steps may be overridden
    # per-cell (the SDE sweep mixes 128 and 64 NFE).
    if eval_only:
        if eval_only_ckpt is None:
            raise ValueError(f"eval_only row {name!r} requires a 'ckpt' field")
        run_dir.mkdir(parents=True, exist_ok=True)
        nstep = eval_steps_override if eval_steps_override is not None else eval_steps
        print(f"[eval-only] {name}: ckpt={eval_only_ckpt} kwargs={sample_kwargs}")
        result = evaluate_checkpoint(
            eval_only_ckpt,
            n_samples=eval_n,
            n_steps=nstep,
            sample_kwargs=sample_kwargs or None,
            overrides=overrides or None,
        )
        result["run_name"] = name
        result["eval_only"] = True
        result["overrides"] = overrides
        eval_path.write_text(json.dumps(result, indent=2))
        sweep_results = runs_root / "sweep_results.jsonl"
        with sweep_results.open("a") as fh:
            fh.write(json.dumps(result) + "\n")
        summary = (
            f"{name}: KL_uni={result['unigram_kl']:.4f} "
            f"KL_bi={result['bigram_kl']:.4f} "
            f"KL_tri={result['trigram_kl']:.4f} "
            f"H_ratio={result['H_ratio']:.3f}"
        )
        print(summary)
        return result

    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = _build_cfg(overrides)
    cfg.training = replace(cfg.training, checkpoint_dir=str(run_dir))
    # Disable per-epoch checkpointing — only keep epoch_final.pt.
    cfg.training = replace(cfg.training, checkpoint_every=10**9)

    if wandb_enabled:
        wandb_overrides: dict[str, Any] = {
            "enabled": True,
            "run_name": name,
            "group": wandb_group,
            "tags": tuple(t for t in ("sweep", wandb_group) if t),
        }
        if wandb_project is not None:
            wandb_overrides["project"] = wandb_project
        if wandb_entity is not None:
            wandb_overrides["entity"] = wandb_entity
        cfg.wandb = replace(cfg.wandb, **wandb_overrides)

    (run_dir / "config.json").write_text(json.dumps(_config_to_dict(cfg), indent=2))

    print(f"[run] {name}: {overrides}")
    seed_all(cfg.training.seed)
    dm, _ = build_training_datamodule(cfg)

    final_ckpt = run_dir / "epoch_final.pt"
    if not final_ckpt.exists():
        history: list[dict[str, float]] = []
        wandb_logger = build_wandb_logger(cfg, run_dir=run_dir)
        fit(cfg=cfg, datamodule=dm, history_out=history, wandb_logger=wandb_logger)
        (run_dir / "history.jsonl").write_text(
            "\n".join(json.dumps(h) for h in history)
        )
    else:
        print(f"  reusing existing {final_ckpt}")

    print(f"[eval] {name}")
    result = evaluate_checkpoint(final_ckpt, n_samples=eval_n, n_steps=eval_steps)
    result["run_name"] = name
    result["overrides"] = overrides
    eval_path.write_text(json.dumps(result, indent=2))

    summary = (
        f"{name}: KL_uni={result['unigram_kl']:.4f} "
        f"KL_bi={result['bigram_kl']:.4f} "
        f"KL_tri={result['trigram_kl']:.4f} "
        f"H_ratio={result['H_ratio']:.3f}"
    )
    print(summary)

    sweep_results = runs_root / "sweep_results.jsonl"
    with sweep_results.open("a") as fh:
        fh.write(json.dumps(result) + "\n")

    return result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sweep", required=True, help="YAML or JSON list of {name, overrides}"
    )
    p.add_argument("--only", default=None, help="comma-separated subset of run names")
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--n", type=int, default=256, dest="eval_n")
    p.add_argument("--steps", type=int, default=200, dest="eval_steps")
    p.add_argument(
        "--redo-eval", action="store_true", help="rerun eval.json even if it exists"
    )
    p.add_argument(
        "--wandb",
        action="store_true",
        help="enable W&B logging (one run per sweep entry)",
    )
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-entity", type=str, default=None)
    args = p.parse_args(argv)

    text = Path(args.sweep).read_text()
    spec = yaml.safe_load(text)
    if not isinstance(spec, list):
        raise ValueError("sweep file must be a YAML/JSON list")

    only = set(args.only.split(",")) if args.only else None
    runs_root = Path(args.runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)

    wandb_group = f"sweep:{Path(args.sweep).stem}" if args.wandb else None

    for entry in spec:
        name = entry["name"]
        if only is not None and name not in only:
            continue
        overrides = entry.get("overrides") or {}
        eval_only = bool(entry.get("eval_only", False))
        eval_only_ckpt = entry.get("ckpt")
        sample_kwargs = entry.get("sample_kwargs")
        eval_steps_override = entry.get("eval_steps")
        try:
            _run_one(
                name,
                overrides,
                runs_root,
                eval_n=args.eval_n,
                eval_steps=args.eval_steps,
                redo_eval=args.redo_eval,
                eval_only=eval_only,
                eval_only_ckpt=eval_only_ckpt,
                sample_kwargs=sample_kwargs,
                eval_steps_override=eval_steps_override,
                wandb_enabled=args.wandb,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                wandb_group=wandb_group,
            )
        except Exception as e:
            err_path = runs_root / name / "error.txt"
            err_path.parent.mkdir(parents=True, exist_ok=True)
            err_path.write_text(f"{type(e).__name__}: {e}\n")
            print(f"[error] {name}: {type(e).__name__}: {e}", file=sys.stderr)
            # don't crash the whole sweep — continue to next cell
            continue


if __name__ == "__main__":
    main()
