#!/usr/bin/env python3
"""The matched benchmark budget of the nine tab:gen arms, dumped to JSON.

The paper calls the model-selection benchmark "budget-matched" (introduction
contribution bullet, ``tab:setup`` "Model selection" column, ``tab:gen``
caption).  Nothing on disk carried that claim in a checkable form: each arm's
``train_meta.json`` records only ``L``, ``arm``, ``batch``, ``max_train_windows``,
``scale``, ``seed``, ``stage`` and ``status``, with no epoch or capacity field,
and ``bench.log`` carries one stale header (``TRAIN SFLMEBM ... d_model=768/8L``)
that contradicts the checkpoint it is supposed to describe.

The budget actually lives in the ``cfg`` block every checkpoint carries next to
``model_state_dict``.  This script reads it from all nine arms, plus the shared
evaluation settings from each arm's ``eval_all.json``, and writes one JSON so
``check_claims.py`` can read the budget the same way it reads every other number.

Resolve an arm's budget by its checkpoint ``cfg``, never by a ``bench.log``
header.

Output (committed):
  runs/sflm_bench_a100_20g_L256/bench_budget.json

Usage:
  uv run python scripts/bench_budget.py           # write the file
  uv run python scripts/bench_budget.py --check   # verify it matches the disk
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/sflm_bench_a100_20g_L256"
OUT = RUN / "bench_budget.json"

# arm directory -> the row label tab:gen prints for it (docs/paper-map.md)
ARMS = [
    ("FMonCLR", "FM on clr"),
    ("SFM", "Statistical FM (Cheng)"),
    ("FisherFM", "Fisher FM (Davis, smoothed target)"),
    ("SFLM", "Hyperspherical flow"),
    ("DFM", "Discrete FM"),
    ("DirichletFM", "Dirichlet FM"),
    ("EqM_OneHot", "EqM, deterministic clr"),
    ("EqM", "EqM, Dirichlet-thickened"),
    ("EqMAE", "EqM, VAE latent"),
]

# the fields tab:setup's "Model selection" column prints, and where they live
SHARED_KEYS = [
    ("d_model", ("transformer", "d_model")),
    ("num_layers", ("transformer", "num_layers")),
    ("nhead", ("transformer", "nhead")),
    ("epochs", ("training", "epochs")),
    ("batch_size", ("training", "B")),
    ("seed", ("training", "seed")),
    ("L", ("training", "L")),
    ("K", ("training", "K")),
    ("max_train_windows", ("text8_dataset", "max_train_windows")),
]


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_of(arm_dir: Path) -> Path:
    """The weights tab:gen reads: epoch_final.pt, or the one epoch_N.pt kept."""
    final = arm_dir / "epoch_final.pt"
    if final.exists():
        return final
    kept = sorted(arm_dir.glob("epoch_[0-9]*.pt"))
    if len(kept) == 1:
        return kept[0]
    # SFLM keeps epoch_10.pt (the paper's) and epoch_20.pt (a provenance check)
    ten = arm_dir / "epoch_10.pt"
    if ten.exists():
        return ten
    raise SystemExit(f"{arm_dir}: cannot tell which checkpoint tab:gen reads")


def collect() -> dict:
    arms = []
    for name, row in ARMS:
        d = RUN / name
        ckpt = checkpoint_of(d)
        blob = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
        cfg = blob["cfg"]
        rec = {
            "arm": name,
            "paper_row": row,
            "checkpoint": str(ckpt.relative_to(ROOT)),
            "checkpoint_md5": md5(ckpt),
            "checkpoint_epoch": blob.get("epoch"),
            "global_step": blob.get("global_step"),
        }
        for key, (section, field) in SHARED_KEYS:
            rec[key] = cfg[section][field]
        ev = json.loads((d / "eval_all.json").read_text())
        rec["eval_epoch"] = ev.get("epoch")
        rec["eval_n_samples"] = ev.get("n_samples")
        rec["eval_sample_steps"] = ev.get("sample_steps")
        arms.append(rec)

    shared = {}
    for key, _ in SHARED_KEYS:
        values = {a[key] for a in arms}
        if len(values) != 1:
            raise SystemExit(f"arms disagree on {key}: {sorted(values)}")
        shared[key] = values.pop()
    for key in ("eval_epoch", "eval_n_samples", "eval_sample_steps"):
        values = {a[key] for a in arms}
        if len(values) != 1:
            raise SystemExit(f"arms disagree on {key}: {sorted(values)}")
        shared[key] = values.pop()
    shared["n_arms"] = len(arms)
    shared["training_characters"] = shared["max_train_windows"] * shared["L"]

    return {
        "what": "the matched benchmark budget of the nine tab:gen arms",
        "written_by": "scripts/bench_budget.py",
        "run_root": str(RUN.relative_to(ROOT)),
        "paper_loc": "chapters/results.tex:tab:setup 'Model selection' column",
        "shared": shared,
        "arms": arms,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="compare against the written file")
    args = ap.parse_args()
    data = collect()
    if args.check:
        have = json.loads(OUT.read_text())
        print("MATCH" if have == data else "DIFFERS")
        raise SystemExit(0 if have == data else 1)
    OUT.write_text(json.dumps(data, indent=2) + "\n")
    s = data["shared"]
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  {s['n_arms']} arms, all at d_model {s['d_model']}, {s['num_layers']} layers / "
          f"{s['nhead']} heads, {s['epochs']} epochs, {s['max_train_windows']:,} windows "
          f"({s['training_characters']:,} characters), batch {s['batch_size']}, seed {s['seed']}, L={s['L']}")
    print(f"  evaluated at epoch {s['eval_epoch']}, n={s['eval_n_samples']} samples, "
          f"{s['eval_sample_steps']} sampler steps")


if __name__ == "__main__":
    main()
