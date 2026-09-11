"""EqM_OneHot on K=4 DNA — vocab-size (K) isolation experiment.

Trains the *exact* text8 ``EqM_OneHot`` recipe (imported from the bench harness,
local scale: d512/6L/8H, batch 16, L=40, 10k train windows, 10 epochs,
gradient_lambda=3.0, gamma_power=0.5, dirichlet_sampling=False) on biological
promoter DNA with K=4 instead of text8's K=27. The ONLY deliberate change is the
vocabulary: data flows through the same CharWindowDataset / CorruptingCollate /
CLR-feature path via DNADataModule (routed by cfg.text8_dataset.windows_path).

Then it scores generation with the canonical scorecard (scripts/eval_all.py,
same metric definitions as the baseline's eval_all.json) and prints the
head-to-head vs the text8 K=27 baseline.

CAVEAT (logged in the output): DNA (K=4) also has very different statistics from
English (adjacent bases are far closer to independent), so this isolates K only
up to that data-structure confound. See AskUserQuestion design note.

Usage:
    uv run python scripts/exp_dna_eqm_onehot.py \
        --windows-path data_cache/dna/promoters_L40.pt \
        --out-dir runs/dna_K_isolation/EqM_OneHot_dna --epochs 10
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.append(str(_ROOT))


import aitchinson_flow.models  # noqa: E402,F401  populate REGISTRY
from aitchinson_flow.training import build_training_datamodule, fit, seed_all  # noqa: E402
from scripts.train_for_sflm_bench import _model_cfg  # noqa: E402
from scripts.eval_all import evaluate  # noqa: E402

# text8 K=27 EqM_OneHot baseline (runs/sflm_bench_local/EqM_OneHot/eval_all.json).
TEXT8_BASELINE = {
    "K": 27,
    "KL_uni": 0.0693,
    "KL_bi": 1.5694,
    "KL_tri": 6.9696,
    "H_gen": 2.7772,
    "H_gt": 2.8484,
    "H_ratio": 0.9750,
    "per_pos_entropy": 2.3112,
    "collapsed": False,
}


def _build_cfg(args) -> "object":
    """Matched EqM_OneHot recipe (bench, local scale) + DNA K=4 override."""
    cfg = _model_cfg("EqM_OneHot", args.epochs, args.scale, out_dir=args.out_dir, seed=args.seed)
    cfg.text8_dataset = replace(cfg.text8_dataset, windows_path=args.windows_path, K=4)
    cfg.training = replace(cfg.training, K=4)
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows-path", default="data_cache/dna/promoters_L40.pt")
    ap.add_argument("--out-dir", default="runs/dna_K_isolation/EqM_OneHot_dna")
    ap.add_argument("--scale", default="local")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=256, help="eval generation samples")
    ap.add_argument("--steps", type=int, default=200, help="sampler steps")
    ap.add_argument("--skip-train", action="store_true",
                    help="evaluate an existing epoch_final.pt without retraining")
    args = ap.parse_args()

    cfg = _build_cfg(args)
    ckpt = Path(args.out_dir) / "epoch_final.pt"

    if not args.skip_train and not ckpt.exists():
        print(f"[exp_dna] training EqM_OneHot on DNA K=4 (L={cfg.text8_dataset.L}, "
              f"epochs={args.epochs}, scale={args.scale}) -> {args.out_dir}", flush=True)
        seed_all(args.seed)
        dm, meta = build_training_datamodule(cfg)
        print(f"[exp_dna] data source: {meta.get('source')} "
              f"train={dm.splits.train.shape} K={cfg.text8_dataset.K}", flush=True)
        fit(cfg=cfg, datamodule=dm, model=None, wandb_logger=None)
    else:
        print(f"[exp_dna] using existing checkpoint {ckpt}", flush=True)

    print(f"[exp_dna] scoring generation (n={args.n}, steps={args.steps}) ...", flush=True)
    res = evaluate(ckpt, model_kind="EqM_OneHot", split="test",
                   n_samples=args.n, n_steps=args.steps)
    res["windows_path"] = args.windows_path
    res["note_confound"] = (
        "Isolates K up to a data-structure confound: DNA bigram/trigram "
        "structure differs from English (adjacent bases ~ closer to independent)."
    )
    out_json = Path(args.out_dir) / "eval_all.json"
    out_json.write_text(json.dumps(res, indent=2, default=str))
    print(f"[exp_dna] wrote {out_json}")

    # ---- Head-to-head table -------------------------------------------------
    b = TEXT8_BASELINE
    rows = [
        ("KL_uni", "KL_uni"), ("KL_bi", "KL_bi"), ("KL_tri", "KL_tri"),
        ("H_gen", "H_gen"), ("H_gt", "H_gt"), ("H_ratio", "H_ratio"),
        ("per_pos_entropy", "per_pos_entropy"),
    ]
    print("\n" + "=" * 64)
    print("  EqM_OneHot  —  DNA (K=4)  vs  text8 (K=27)")
    print("=" * 64)
    print(f"  {'metric':<18}{'text8 K=27':>14}{'DNA K=4':>14}")
    print("  " + "-" * 46)
    for label, key in rows:
        tv, dv = b.get(key), res.get(key)
        ts = f"{tv:.4f}" if isinstance(tv, (int, float)) else str(tv)
        ds = f"{dv:.4f}" if isinstance(dv, (int, float)) else str(dv)
        print(f"  {label:<18}{ts:>14}{ds:>14}")
    print("  " + "-" * 46)
    print(f"  {'collapsed':<18}{str(b['collapsed']):>14}{str(res.get('collapsed')):>14}")
    print(f"  {'sample[0]':<18}  {(res.get('samples') or ['?'])[0]!r}")
    print("=" * 64)
    print("  NB: KL/H are vs each modality's OWN n-gram reference, so KL_bi/tri")
    print("      compare 'how much sequential structure was captured' relative")
    print("      to each corpus. Lower KL_bi/tri at K=4 ⇒ small-K helps — but is")
    print("      confounded by DNA's near-independent base statistics.")


if __name__ == "__main__":
    main()
