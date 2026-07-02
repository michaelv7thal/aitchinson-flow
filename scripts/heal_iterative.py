"""Iterative NLL healing — repeat localize+inpaint N times, store every iteration.

Tests whether repeatedly applying the heal drives corrupted text toward FLUENT /
self-consistent strings (low denoiser-NLL) even as it DRIFTS away from the
original clean ground truth (token accuracy vs clean falls). Each iteration
RE-LOCALIZES on the current (partially-healed) text with the training-free
denoiser-NLL, thresholds at a fixed calibrated operating point, pins the
unflagged positions and regenerates the flagged ones with the Dirichlet field,
then feeds the output back in.

Per iteration k = 0..N (k=0 is the corrupted input), aggregated over corruption
seeds, it stores:
  acc_vs_clean   — fidelity: P(token == original clean)         [want ↑ then watch drift]
  mean_nll       — plausibility: mean denoiser-NLL of the text  [want ↓ = more fluent]
  fix_rate       — P(originally-corrupt token -> correct)
  damage_rate    — P(originally-clean token   -> broken)
  net_per_corrupt
  drift_from_prev— P(token changed vs previous iteration)       [convergence: -> 0]
  frac_flagged   — fraction of positions the localizer flagged this iteration
plus decoded example text for a few sequences at every iteration so the drift is
readable.  Reuses the building blocks of scripts/heal_dirichlet.py.

Usage:
    uv run python scripts/heal_iterative.py \
        --ckpt runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM/epoch_final.pt \
        --split test --iters 5 --n-demo 32 --n-seeds 3 \
        --out heal_out/heal_iterative_nll.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch


def _bootstrap() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))


_bootstrap()

from scripts.ood_variance_perpos import _load_dirichletfm  # noqa: E402
from scripts.heal_dirichlet import (  # noqa: E402
    _decode,
    make_nll_localizer,
    train_localizer,
    calibrate_threshold,
    inpaint,
)
from aitchinson_flow.training import build_training_datamodule  # noqa: E402
from aitchinson_flow.data.corruption import corrupt_token_ids  # noqa: E402


def _mean_std(xs):
    xs = [float(x) for x in xs if x == x and abs(float(x)) != float("inf")]  # drop NaN/inf
    if not xs:
        return float("nan"), 0.0
    return (sum(xs) / len(xs), statistics.pstdev(xs) if len(xs) > 1 else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="heal_out/heal_iterative_nll.json")
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--localizer", choices=["nll", "linear"], default="nll")
    ap.add_argument("--t-nll", type=float, default=3.0, help="[nll] denoiser-NLL path-time")
    ap.add_argument("--t-eval", type=float, default=None, help="[linear] feature path-time")
    ap.add_argument("--iters", type=int, default=5, help="# heal repetitions")
    ap.add_argument("--fit-seqs", type=int, default=256, help="localizer fit + cal slice")
    ap.add_argument("--n-demo", type=int, default=32)
    ap.add_argument("--n-seeds", type=int, default=3, help="# corruption realizations")
    ap.add_argument("--corrupt-rate", type=float, default=0.15)
    ap.add_argument("--train-rate", type=float, default=0.3, help="[linear] negatives rate")
    ap.add_argument("--target-fpr", type=float, default=0.02,
                    help="clean-FPR operating point for the heal mask (fixed across iters)")
    ap.add_argument("--margin", type=float, default=4.0)
    ap.add_argument("--head-steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--nfe", type=int, default=100)
    ap.add_argument("--n-example", type=int, default=3, help="# sequences to dump text for")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = _load_dirichletfm(args.ckpt, device)
    K = cfg.text8_dataset.K
    L = cfg.text8_dataset.L
    t_eval = float(args.t_eval if args.t_eval is not None else cfg.dfm_svgp.t_eval)
    print(f"[iter-heal] ckpt={args.ckpt}")
    print(f"[iter-heal] localizer={args.localizer} iters={args.iters} split={args.split} "
          f"corrupt_rate={args.corrupt_rate} fpr={args.target_fpr} nfe={args.nfe}")

    dm, _ = build_training_datamodule(cfg)
    loaders = {"train": dm.train_dataloader, "val": dm.val_dataloader,
               "test": dm.test_dataloader}
    vl = loaders[args.split]() or dm.train_dataloader()
    need = 2 * args.fit_seqs + args.n_demo
    seqs = []
    for b in vl:
        seqs.append(b["token_ids"].long())
        if sum(s.shape[0] for s in seqs) >= need:
            break
    seqs = torch.cat(seqs)
    if seqs.shape[0] < need:
        raise SystemExit(f"split yielded {seqs.shape[0]} < required {need}")
    fit_tok = seqs[: args.fit_seqs]
    cal_tok = seqs[args.fit_seqs : 2 * args.fit_seqs]
    demo_clean = seqs[2 * args.fit_seqs : 2 * args.fit_seqs + args.n_demo]
    print(f"[iter-heal] splits: fit={fit_tok.shape[0]} cal={cal_tok.shape[0]} "
          f"demo={demo_clean.shape[0]}")

    # ---- localizer (mask) + a denoiser-NLL scorer for the plausibility metric --
    if args.localizer == "nll":
        score, _ = make_nll_localizer(model, args.t_nll, device)
    else:
        score, _ = train_localizer(model, fit_tok, t_eval, device,
                                   train_rate=args.train_rate, margin=args.margin,
                                   steps=args.head_steps, lr=args.lr, seed=args.seed)
    nll_score, _ = make_nll_localizer(model, args.t_nll, device)  # plausibility (always NLL)

    # fixed threshold at the target FPR (consistent NLL/energy scale across iters)
    cal_corr = corrupt_token_ids(cal_tok.clone(), vocab_size=K,
                                 corrupt_rate=args.corrupt_rate, seed=args.seed + 1)
    thr, cal_stats = calibrate_threshold(score, cal_tok, cal_corr, target_fpr=args.target_fpr)
    print(f"[iter-heal] threshold={thr:.3f}  cal P/R/F1="
          f"{cal_stats['precision']:.3f}/{cal_stats['recall']:.3f}/{cal_stats['f1']:.3f}")

    clean = demo_clean
    n_corrupt_tot = n_clean_tot = 0  # for reference

    # per (seed): list over iterations of metric dicts; iter 0 = corrupted input
    per_seed_iters: list[list[dict]] = []
    example_texts: list[dict] = []  # decoded trajectory for the first n_example seqs

    for s in range(args.n_seeds):
        corr = corrupt_token_ids(clean.clone(), vocab_size=K,
                                 corrupt_rate=args.corrupt_rate, seed=args.seed + 100 + s)
        true_corrupt = (corr != clean)
        n_corrupt = int(true_corrupt.sum()); n_clean = int((~true_corrupt).sum())
        n_corrupt_tot += n_corrupt; n_clean_tot += n_clean

        def metrics(cur, prev, flagged):
            acc = float((cur == clean).float().mean())
            mnll = float(nll_score(cur).mean())
            fixed = int(((cur == clean) & true_corrupt).sum())
            damaged = int(((cur != clean) & ~true_corrupt).sum())
            drift = float((cur != prev).float().mean())
            return {"acc_vs_clean": acc, "mean_nll": mnll,
                    "fix_rate": fixed / max(n_corrupt, 1),
                    "damage_rate": damaged / max(n_clean, 1),
                    "net_per_corrupt": (fixed - damaged) / max(n_corrupt, 1),
                    "drift_from_prev": drift,
                    "frac_flagged": (float(flagged.float().mean()) if flagged is not None
                                     else float("nan"))}

        iters_m = [metrics(corr, corr, None)]            # k=0: the corrupted input
        traj = [{"iter": 0, "text": [_decode(corr[b]) for b in range(min(args.n_example,
                                                                         clean.shape[0]))]}]
        cur = corr
        for k in range(1, args.iters + 1):
            E = score(cur)
            mask = E > thr
            healed = inpaint(model, cur, mask, nfe=args.nfe, device=device)
            iters_m.append(metrics(healed, cur, mask))
            traj.append({"iter": k,
                         "text": [_decode(healed[b]) for b in
                                  range(min(args.n_example, clean.shape[0]))],
                         "n_flagged": [int(mask[b].sum()) for b in
                                       range(min(args.n_example, clean.shape[0]))]})
            cur = healed
        per_seed_iters.append(iters_m)
        if s == 0:
            for b in range(min(args.n_example, clean.shape[0])):
                example_texts.append({
                    "idx": b, "clean": _decode(clean[b]),
                    "trajectory": [{"iter": t["iter"], "text": t["text"][b],
                                    "n_flagged": (t.get("n_flagged", [None] * 99)[b]
                                                  if "n_flagged" in t else None)}
                                   for t in traj],
                })
        print(f"  seed {s}: acc {iters_m[0]['acc_vs_clean']:.3f} -> "
              f"{iters_m[-1]['acc_vs_clean']:.3f} | nll "
              f"{iters_m[0]['mean_nll']:.3f} -> {iters_m[-1]['mean_nll']:.3f}")

    # ---- aggregate over seeds, per iteration ----------------------------------
    keys = ["acc_vs_clean", "mean_nll", "fix_rate", "damage_rate",
            "net_per_corrupt", "drift_from_prev", "frac_flagged"]
    per_iteration = []
    for k in range(args.iters + 1):
        row = {"iter": k}
        for key in keys:
            m, sd = _mean_std([per_seed_iters[s][k][key] for s in range(args.n_seeds)])
            row[key] = m
            row[key + "_std"] = sd
        per_iteration.append(row)

    print(f"\n{'iter':>4} {'acc_clean':>10} {'mean_nll':>9} {'fix':>6} {'dmg':>6} "
          f"{'net/cor':>8} {'drift':>7} {'flagged':>8}")
    for r in per_iteration:
        print(f"{r['iter']:>4} {r['acc_vs_clean']:>10.4f} {r['mean_nll']:>9.4f} "
              f"{r['fix_rate']:>6.3f} {r['damage_rate']:>6.3f} "
              f"{r['net_per_corrupt']:>+8.3f} {r['drift_from_prev']:>7.4f} "
              f"{r['frac_flagged']:>8.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "ckpt": args.ckpt, "K": K, "L": L, "split": args.split,
        "localizer": args.localizer, "t_nll": args.t_nll, "t_eval": t_eval,
        "iters": args.iters, "n_demo": int(demo_clean.shape[0]),
        "n_seeds": args.n_seeds, "corrupt_rate": args.corrupt_rate,
        "target_fpr": args.target_fpr, "threshold": thr, "nfe": args.nfe,
        "calibration": cal_stats,
        "note": "iter 0 = corrupted input; each iter re-localizes on the current "
                "text, pins unflagged, regenerates flagged, then feeds back",
        "per_iteration": per_iteration,
        "examples": example_texts,
    }, indent=2))
    print(f"\nWrote {out_path}")

    # readable drift for the first example
    if example_texts:
        ex = example_texts[0]
        print(f"\n--- example[0] drift (clean vs each iteration) ---")
        print(f"  clean : {ex['clean']!r}")
        for t in ex["trajectory"]:
            tag = "corrupt" if t["iter"] == 0 else f"iter {t['iter']}"
            fl = "" if t["n_flagged"] is None else f"  [{t['n_flagged']} flagged]"
            print(f"  {tag:>7}: {t['text']!r}{fl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
