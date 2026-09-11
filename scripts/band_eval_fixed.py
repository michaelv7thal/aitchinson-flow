"""Re-evaluate the band cells with a sampler step matched to their equilibrium.

sweeps/band_gamma.yaml scaled eqm.sample_grad_clip by gamma_star to reproduce
the published descent dynamics at 1/20 scale. That was the wrong knob: the
clipped gradient norm then sits at exactly the clip at every step, so the
`sample_return_best` rule (keep the lowest mean-||grad|| iterate) never sees an
improvement over step 0 and hands back the initial noise. The generation
numbers in runs/band_*/eval.json are therefore the source distribution, not the
model's output.

Scaling the step size instead leaves the clip free to bind and release: with
eta=0.005 and clip=1.0 the descent lands at radius 0.807 against the 0.784 of
x_{0.05}, and the gradient dips to 0.84 there, so the best-iterate rule picks
the equilibrium.

Run:
    uv run python scripts/band_eval_fixed.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.eval_full import evaluate_checkpoint  # noqa: E402

CELLS = ("band_g05", "band_strict")
SAMPLE_KWARGS = {"eta": 0.005, "grad_clip": 1.0}
N, STEPS = 256, 400


def main() -> None:
    for cell in CELLS:
        ck = ROOT / "runs" / cell / "epoch_final.pt"
        if not ck.exists():
            print(f"{cell}: no checkpoint yet", flush=True)
            continue
        r = evaluate_checkpoint(str(ck), n_samples=N, n_steps=STEPS,
                                sample_kwargs=SAMPLE_KWARGS)
        out = ROOT / "runs" / cell / "eval_fixed.json"
        json.dump(r, open(out, "w"), indent=2)
        print(f"{cell}: KL_uni={r['unigram_kl']:.4f} KL_bi={r['bigram_kl']:.3f} "
              f"H_gen={r['H_gen']:.3f} (H_gt={r['H_gt']:.3f})", flush=True)
        for s in r["samples"][:3]:
            print("   ", repr(s), flush=True)


if __name__ == "__main__":
    main()
