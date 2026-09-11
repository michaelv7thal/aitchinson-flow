"""Re-train a single run from its saved config.json (no YAML needed).

Used to regenerate weights for runs/<cell>/ directories that have a
config.json + eval.json from a prior local training session but no
loadable .pt checkpoint on this filesystem.

Idempotent: skips if epoch_final.pt already exists (unless --force).

Usage:
    .venv/bin/python scripts/retrain_latent_from_config.py runs/latent_d32_untied_ce0
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path


def _bootstrap() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))


_bootstrap()

import aitchinson_flow.models  # noqa: E402,F401 — populate registry
from aitchinson_flow.training import (  # noqa: E402
    build_training_datamodule,
    fit,
    seed_all,
)
from scripts.eval_full import _config_from_payload  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="run directory containing config.json")
    ap.add_argument("--force", action="store_true",
                    help="re-train even if epoch_final.pt already exists")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    cfg_path = run_dir / "config.json"
    ckpt_out = run_dir / "epoch_final.pt"

    if not cfg_path.is_file():
        print(f"[error] no config.json in {run_dir}", file=sys.stderr)
        return 1
    if ckpt_out.is_file() and not args.force:
        print(f"[skip] {ckpt_out} already exists (use --force to retrain)")
        return 0

    cfg_dict = json.loads(cfg_path.read_text())
    cfg = _config_from_payload({"cfg": cfg_dict})
    cfg.training = replace(
        cfg.training,
        checkpoint_dir=str(run_dir),
        checkpoint_every=10**9,  # only save epoch_final.pt
    )

    print(f"[retrain] {run_dir.name}: model={cfg.training.model_name} "
          f"epochs={cfg.training.epochs} L={cfg.training.L}")
    seed_all(cfg.training.seed)
    dm, _ = build_training_datamodule(cfg)
    fit(cfg=cfg, datamodule=dm, history_out=run_dir / "history.jsonl")

    if ckpt_out.is_file():
        print(f"[done] wrote {ckpt_out}")
        return 0
    print(f"[warn] training completed but {ckpt_out} not found", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
