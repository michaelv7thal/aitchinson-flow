"""Phase T (CAPSTONE_EXPERIMENTS.md §6) — 6-cell W1 comparison table.

Loads two checkpoints (eqm and eqm_consgrad) and evaluates each at three
sampler settings, producing a 2×3 = 6-cell table.

Sampler settings:
* (a) Euler on the model's output     — "Euler-on-f"
* (b) Euler on ∇⟨x, model_output⟩    — "Euler-on-grad"
* (c) NAG-GD on the conservative gradient (default for EqM)

For the eqm_consgrad model, (a) is the natural sampler (its output IS
the gradient), so (a) ≡ (b) trivially; we still emit both rows for
table symmetry.

Output: ``runs/capstone/T/phaseT_w1_compare.{json,md}``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.eval_full import evaluate_checkpoint  # noqa: E402


def _evaluate(ckpt: str, *, sample_kwargs: dict, n: int, steps: int, label: str) -> dict:
    res = evaluate_checkpoint(
        ckpt,
        n_samples=n,
        n_steps=steps,
        sample_kwargs=sample_kwargs or None,
    )
    res["cell_label"] = label
    return res


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eqm-ckpt", required=True)
    p.add_argument("--consgrad-ckpt", required=True)
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--steps-euler", type=int, default=128)
    p.add_argument("--steps-nag", type=int, default=200)
    p.add_argument("--out", default="runs/capstone/T/phaseT_w1_compare.json")
    args = p.parse_args(argv)

    settings = [
        ("eqm",       args.eqm_ckpt,      "euler",  False, args.steps_euler, "Euler on f"),
        ("eqm",       args.eqm_ckpt,      "euler",  True,  args.steps_euler, "Euler on grad"),
        ("eqm",       args.eqm_ckpt,      "nag",    None,  args.steps_nag,   "NAG"),
        ("consgrad",  args.consgrad_ckpt, "euler",  False, args.steps_euler, "Euler on output"),
        ("consgrad",  args.consgrad_ckpt, "euler",  True,  args.steps_euler, "Euler on output (use_grad)"),
        ("consgrad",  args.consgrad_ckpt, "nag",    None,  args.steps_nag,   "NAG"),
    ]

    rows = []
    for fam, ckpt, method, use_grad, steps, label in settings:
        kwargs = {"method": method}
        if use_grad is not None:
            kwargs["use_grad"] = bool(use_grad)
        print(f"[eval] {fam} / {label} ({ckpt})")
        try:
            res = _evaluate(ckpt, sample_kwargs=kwargs, n=args.n, steps=steps, label=label)
        except Exception as e:  # noqa: BLE001
            print(f"  failed: {type(e).__name__}: {e}")
            res = {"error": f"{type(e).__name__}: {e}"}
        rows.append({"family": fam, "label": label, "result": res})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"[write] {out}")

    # Markdown table
    md = ["# Phase T — W1 6-cell comparison", ""]
    md.append("| family | sampler | KL_uni | KL_bi | KL_tri | H_ratio |")
    md.append("|---|---|---:|---:|---:|---:|")
    for r in rows:
        res = r.get("result", {})
        if "error" in res:
            md.append(f"| {r['family']} | {r['label']} | error | — | — | — |")
            continue
        md.append(
            f"| {r['family']} | {r['label']} | "
            f"{res.get('unigram_kl', float('nan')):.4f} | "
            f"{res.get('bigram_kl', float('nan')):.4f} | "
            f"{res.get('trigram_kl', float('nan')):.4f} | "
            f"{res.get('H_ratio', float('nan')):.3f} |"
        )
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print(f"[write] {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
