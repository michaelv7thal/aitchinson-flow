"""Run the four UQ signals on the trained Hilbert FM model and report AUROC.

The labelled task is **reconstruction-from-`t_start`**: given a real text8
window, build the path target, walk to ``t_start``, then re-sample to
``t_max``. The argmax of the final prediction is "correct" iff it matches the
original token. AUROC is computed against the binary error label per position.

Usage::

    python -m hilbert_fm.run_uq <ckpt.pt> [t_start] [M]
"""
from __future__ import annotations

import json
import os
import sys

import torch

from .data import encode, get_batch, load_corpus, split_train_val
from .sample import load_model
from .uq import auroc, ensemble_disagreement, reconstruction_trajectory


def run(
    ckpt_path: str,
    t_start: float = 0.5,
    n_steps: int = 25,
    M: int = 4,
    n_batches: int = 8,
    batch_size: int = 64,
    seed: int = 0,
):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(ckpt_path, device)

    text = load_corpus()
    ids = encode(text)
    _, val_ids = split_train_val(ids)

    cfg = dict(cfg)
    cfg["batch_size"] = batch_size

    pieces = {"U_spread": [], "U_traj": [], "L_excess": [], "U_ensemble": [], "err": []}

    for b in range(n_batches):
        batch = get_batch(val_ids, batch_size, cfg["L"], device)
        out = reconstruction_trajectory(model, batch, cfg, device, t_start, n_steps)
        U_ens = ensemble_disagreement(model, batch, cfg, device, t_start, n_steps, M=M)
        err = 1.0 - out["correct"]
        pieces["U_spread"].append(out["U_spread"])
        pieces["U_traj"].append(out["U_traj"])
        pieces["L_excess"].append(out["L_excess"])
        pieces["U_ensemble"].append(U_ens)
        pieces["err"].append(err)
        print(
            f"[batch {b + 1}/{n_batches}] err_rate={float(err.mean()):.3f} "
            f"U_traj_mean={float(out['U_traj'].mean()):.3f}"
        )

    cat = {k: torch.cat([t.flatten() for t in v]) for k, v in pieces.items()}
    err = cat["err"].long()
    n_pos = int((err == 1).sum())
    n_total = int(err.numel())
    print(f"\nN={n_total} positions, errors={n_pos} ({n_pos / n_total:.1%})")

    # AUROC: higher score should predict error=1.
    # U_spread is anti-correlated with error (high = peaked = confident),
    # so we report AUROC for both the raw signal and its negation.
    rows = [
        ("U_spread (raw — should be < 0.5)", cat["U_spread"]),
        ("-U_spread  (flipped: low spread = uncertain)", -cat["U_spread"]),
        ("U_traj", cat["U_traj"]),
        ("L_excess", cat["L_excess"]),
        ("U_ensemble", cat["U_ensemble"]),
    ]
    print("\nSignal                                          AUROC")
    print("-" * 60)
    summary = {}
    for name, score in rows:
        a = auroc(score, err)
        summary[name] = a
        print(f"{name:48s} {a:6.4f}")

    out_path = os.path.join(os.path.dirname(ckpt_path), "uq_summary.json")
    with open(out_path, "w") as f:
        json.dump(
            dict(
                t_start=t_start, n_steps=n_steps, M=M, n_batches=n_batches,
                batch_size=batch_size, n_total=n_total, n_errors=n_pos,
                auroc=summary,
                signal_means={k: float(cat[k].mean()) for k in
                              ("U_spread", "U_traj", "L_excess", "U_ensemble")},
            ),
            f,
            indent=2,
        )
    print(f"\nSaved {out_path}")
    return summary


if __name__ == "__main__":
    args = sys.argv[1:]
    ckpt = args[0] if args else "hilbert_fm/runs/default/final.pt"
    t_start = float(args[1]) if len(args) > 1 else 0.5
    M = int(args[2]) if len(args) > 2 else 4
    run(ckpt, t_start=t_start, M=M)
