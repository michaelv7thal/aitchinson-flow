"""Aggregate the band-γ sweep (sweeps/band_gamma.yaml) against the published cell.

Three questions, one table:

  1. Does it still collapse?      KL_uni, KL_bi, H_gen from runs/<cell>/eval.json.
     Reference points: unigram collapse is KL_uni ~ 0.031; the iid-unigram
     bigram floor is 1.692; uniform output is H = ln 27 = 3.296.

  2. Where does the descent stop?  Mean per-position radius of the generated
     state. The published cell lands at 12.33 (the vertex). A band cell with
     gamma_star=0.05 should stop near 0.78, the radius of x_{0.05}. Landing at
     12.3 instead means the field never learned to stop; landing at 0.51 means
     it never moved.

  3. Does it use the neighbours?   Left to scripts/recovery_native_path.py,
     whose rows this script reads if present: initialise at x_gamma with the
     true x_1, descend, and compare argmax accuracy before and after. Any
     positive gap beyond the uniform->unigram shift is context being used.

Run (after the sweep finishes, GPU free):
    uv run python scripts/band_report.py --cells compu_mse_det,ctrl_noce,band_g05,band_strict
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import aitchinson_flow.models  # noqa: E402,F401
from aitchinson_flow.models import build_model  # noqa: E402
from scripts.eval_full import _config_from_payload  # noqa: E402

UNIGRAM_BIGRAM_FLOOR = 1.692  # iid draws from the corpus unigram, same measure
LN27 = 2.302585 + 0.9932518   # ln 27 = 3.2958


def gen_radius(cell: str, n: int, steps: int, seed: int):
    """Sample from noise; report where the descent stopped."""
    ck = ROOT / "runs" / cell / "epoch_final.pt"
    if not ck.exists():
        return None
    payload = torch.load(ck, map_location="cpu", weights_only=False)
    cfg = _config_from_payload(payload)
    model = build_model(cfg).to(cfg.training.device)
    model.load_state_dict(payload.get("model_state_dict", payload))
    model.eval()
    torch.manual_seed(seed)
    L = cfg.text8_dataset.L
    with torch.no_grad():
        x = model.sample(n, L, max_steps=steps)
        lp = model.decode_to_logprobs(x)
        pmax = lp.exp().max(-1).values.mean().item()
    out = {
        "radius_out": float(x.norm(dim=-1).mean()),
        "p_max": pmax,
        "gamma_star": float(getattr(cfg.eqm, "gamma_star", float("nan"))),
        "gamma_lo": float(getattr(cfg.eqm, "gamma_lo", 0.0)),
        "gamma_hi": float(getattr(cfg.eqm, "gamma_hi", 1.0)),
        "decay": cfg.eqm.decay_strategy,
        "lambda_ce": float(cfg.eqm.lambda_ce),
        "grad_clip": cfg.eqm.sample_grad_clip,
    }
    del model
    torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="compu_mse_det,ctrl_noce,band_g05,band_strict")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-radius", action="store_true")
    args = ap.parse_args()

    cells = [c for c in args.cells.split(",") if c.strip()]
    rows = {}
    for cell in cells:
        r = {}
        ev = ROOT / "runs" / cell / "eval.json"
        if ev.exists():
            e = json.load(open(ev))
            r.update(KL_uni=e.get("unigram_kl"), KL_bi=e.get("bigram_kl"),
                     H_gen=e.get("H_gen"), sample0=(e.get("samples") or [""])[0])
        if not args.skip_radius:
            g = gen_radius(cell, args.n, args.steps, args.seed)
            if g:
                r.update(g)
        rec = ROOT / "runs" / cell / "recovery_native.json"
        if rec.exists():
            r["native"] = sorted(json.load(open(rec))["rows"], key=lambda x: x["gamma"])
        rows[cell] = r

    print(f"\n{'cell':16}{'decay':7}{'γ range':>13}{'γ*':>6}{'λ_ce':>6}"
          f"{'KL_uni':>9}{'KL_bi':>8}{'H_gen':>7}{'radius':>8}{'p_max':>7}")
    print("-" * 87)
    for cell, r in rows.items():
        def f(k, w, p=3, d="-"):
            v = r.get(k)
            return f"{v:>{w}.{p}f}" if isinstance(v, (int, float)) else f"{d:>{w}}"
        rng = (f"[{r['gamma_lo']:.3f},{r['gamma_hi']:.3f}]"
               if "gamma_lo" in r else "-")
        print(f"{cell:16}{str(r.get('decay','-')):7}{rng:>13}"
              f"{f('gamma_star',6,3)}{f('lambda_ce',6,2)}"
              f"{f('KL_uni',9,4)}{f('KL_bi',8,3)}{f('H_gen',7,3)}"
              f"{f('radius_out',8,2)}{f('p_max',7,3)}")
    print(f"\nreference   unigram collapse KL_uni≈0.031 | iid-unigram KL_bi={UNIGRAM_BIGRAM_FLOOR} "
          f"| uniform H={LN27:.3f} | data radius 12.27")

    print("\nsamples")
    for cell, r in rows.items():
        if r.get("sample0"):
            print(f"  {cell:16}'{r['sample0'][:56]}'")

    any_native = any("native" in r for r in rows.values())
    if any_native:
        print("\nnative-path recovery: argmax accuracy before -> after the descent")
        for cell, r in rows.items():
            if "native" not in r:
                continue
            print(f"  {cell}")
            print(f"    {'γ':>7}{'r_in':>8}{'r_out':>8}{'acc_in':>9}{'acc_out':>9}{'Δ':>9}")
            for x in r["native"]:
                print(f"    {x['gamma']:>7.3f}{x['radius']:>8.2f}"
                      f"{x.get('radius_out', float('nan')):>8.2f}"
                      f"{x['token_acc_perturbed']:>9.4f}{x['token_acc']:>9.4f}"
                      f"{x['delta']:>+9.4f}")

    out = ROOT / "runs" / "band_report.json"
    json.dump(rows, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
