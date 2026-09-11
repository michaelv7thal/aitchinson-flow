"""Scratch: does the one-class SVGP variance OOD signal need more inducing points?

Sweeps M (inducing points) on cached raw 1280-d features, fixed small lengthscale.
As M -> N the SVGP -> exact GP, so this separates an APPROXIMATION ceiling
(variance improves with M) from a SIGNAL ceiling (flat in M ~ the ~0.70 wall we
keep hitting). BLR (full-rank exact linear-kernel analog) is also ~0.70, which
predicts a signal ceiling.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch

repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo / "src")); sys.path.append(str(repo))
from scripts.scratch_gp_oneclass_sweep import auroc, fit_oneclass, eval_ev  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(args.cache)
    zc, zh, yh, d = blob["zc"].to(device), blob["zh"].to(device), blob["yh"], blob["d"]
    bh, Lh = blob["bh"], blob["Lh"]
    print(f"[data] zc={tuple(zc.shape)} zh={tuple(zh.shape)} d={d}")
    print(f"{'M':>6}  {'V_clean':>8}  AUROC_var")
    for M in (256, 512, 1024, 2048, 4096):
        gp = fit_oneclass(zc, d, M, 0.1, False, 0.05, args.steps, 1e-2, 4096, 0.01, device, args.seed)
        Eh, Vh = eval_ev(gp, zh, bh, Lh)
        with torch.no_grad():
            vcl = gp(zc[:8192]).variance.mean().item()
        print(f"{M:>6}  {vcl:>8.3f}  {auroc(Vh[yh], Vh[~yh]):.4f}")


if __name__ == "__main__":
    main()
