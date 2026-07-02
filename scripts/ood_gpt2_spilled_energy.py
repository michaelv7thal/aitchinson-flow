"""GPT-2 Spilled-Energy OOD baseline on text8 — MATCHED to the DirichletFM-NLL sweep.

The project's "forced baseline": per-char spilled energy under a frozen pretrained
GPT-2, SE_t = logsumexp(logits_t) - logits_t[char_t] = -log p_GPT2(char_t | left
context). Zero-train, single forward pass. Implementation reuses the canonical
``scripts.bench_sflm_ebm._gpt2_per_char_SE`` (BPE-tokenise the decoded char string
with offset mapping; per-BPE-token NLL spread uniformly over the chars it spans).

Run on the SAME test sequences (same --fit-seqs offset, --n, --split) and the SAME
corruption ladder + seeds as ``scripts/ood_denoiser_nll.py`` so the sequence- and
per-token AUROC columns are a controlled apples-to-apples baseline for the
DirichletFM-NLL detector. (SE needs no fit set; --fit-seqs only aligns the eval
slice with the NLL sweep.)

Usage:
    uv run python scripts/ood_gpt2_spilled_energy.py \
        --ckpt runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM/epoch_final.pt \
        --split test --rates 0.05,0.1,0.15,0.2,0.25,0.3,0.5,0.7,1.0 \
        --out ood_out/gpt2_se/gpt2_spilled_energy_sweep.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace as _replace
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

from scripts.bench_sflm_ebm import _load_gpt2, _gpt2_per_char_SE  # noqa: E402
from scripts.ood_variance_perpos import _auroc  # noqa: E402
from scripts.eval_full import _config_from_payload  # noqa: E402
from aitchinson_flow.training import build_training_datamodule  # noqa: E402
from aitchinson_flow.data.corruption import (  # noqa: E402
    corrupt_token_ids,
    partially_shuffle_token_ids,
)


def _corrupt(tok, scheme, rate, K, seed):
    if scheme == "replace":
        return corrupt_token_ids(tok, vocab_size=K, corrupt_rate=rate, seed=seed)
    if scheme == "shuffle":
        return partially_shuffle_token_ids(tok, shuffle_rate=rate, seed=seed)
    out = corrupt_token_ids(tok, vocab_size=K, corrupt_rate=rate, seed=seed)
    return partially_shuffle_token_ids(out, shuffle_rate=rate, seed=seed + 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True,
                    help="DirichletFM ckpt — used ONLY for its cfg/datamodule so the "
                         "test sequences match the NLL sweep (GPT-2 does the scoring)")
    ap.add_argument("--ref-lm", default="gpt2", help="HF causal LM (gpt2, distilgpt2, ...)")
    ap.add_argument("--out", default="ood_out/gpt2_se/gpt2_spilled_energy_sweep.json")
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--fit-seqs", type=int, default=512,
                    help="eval-slice offset; must match the NLL sweep (SE uses no fit)")
    ap.add_argument("--n", type=int, default=256, help="# eval sequences per split")
    ap.add_argument("--schemes", type=str, default="replace,shuffle,both")
    ap.add_argument("--rates", type=str, default="0.1,0.3,0.5,0.7,1.0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = _config_from_payload(payload)
    cfg.training = _replace(cfg.training, device=device)
    K = cfg.text8_dataset.K
    dm, _ = build_training_datamodule(cfg)
    loaders = {"train": dm.train_dataloader, "val": dm.val_dataloader,
               "test": dm.test_dataloader}
    vl = loaders[args.split]() or dm.train_dataloader()
    seqs = []
    for b in vl:
        seqs.append(b["token_ids"].long())
        if sum(s.shape[0] for s in seqs) >= args.fit_seqs + args.n:
            break
    seqs = torch.cat(seqs)
    pos_tok = seqs[args.fit_seqs : args.fit_seqs + args.n]
    print(f"[gpt2-se] ref_lm={args.ref_lm} split={args.split} K={K} "
          f"eval={pos_tok.shape[0]} (slice [{args.fit_seqs}:{args.fit_seqs+args.n}])")

    model, tok = _load_gpt2(args.ref_lm, device)

    def score(t):
        return _gpt2_per_char_SE(model, tok, t, chunk=args.chunk)  # (B,L) per-char SE

    Sp = score(pos_tok)
    Sp_seq = Sp.mean(1).cpu().numpy()
    clean_mean = float(Sp.mean())
    print(f"[gpt2-se] clean per-char SE mean = {clean_mean:.4f}")
    rows = [{"scheme": None, "rate": 0.0, "n": int(pos_tok.shape[0]),
             "se_seq_mean": clean_mean}]
    print(f"\n{'scheme':>9} {'rate':>5} {'AUROC_seq_SE':>13} {'AUROC_tok_SE':>13}")
    schemes = [s for s in args.schemes.split(",") if s.strip()]
    rates = [float(r) for r in args.rates.split(",") if r.strip()]
    for scheme in schemes:
        for r in rates:
            ot = _corrupt(pos_tok.clone(), scheme, r, K, args.seed + int(1000 * r))
            So = score(ot)
            lab = np.r_[np.zeros(len(Sp_seq)), np.ones(So.shape[0])]
            au_seq = _auroc(np.r_[Sp_seq, So.mean(1).cpu().numpy()], lab)
            changed = (ot != pos_tok)
            au_tok = float("nan")
            if changed.any() and (~changed).any():
                cm = changed.cpu().numpy().reshape(-1).astype(int)
                au_tok = _auroc(So.cpu().numpy().reshape(-1), cm)
            print(f"{scheme:>9} {r:>5.2f} {au_seq:>13.4f} {au_tok:>13.4f}")
            rows.append({"scheme": scheme, "rate": r, "n": int(ot.shape[0]),
                         "auroc_seq_se": au_seq, "auroc_token_se": au_tok,
                         "se_seq_mean": float(So.mean())})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "ref_lm": args.ref_lm, "split": args.split, "K": K,
        "n": int(pos_tok.shape[0]), "fit_seqs_offset": args.fit_seqs,
        "detector": "GPT2_SpilledEnergy",
        "detector_long": "per-char spilled energy under a frozen pretrained GPT-2 "
                         "(zero-train); matched to the DirichletFM-NLL sweep",
        "rows": rows,
    }, indent=2))
    print(f"\nWrote {out_path}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
            rws = [r for r in rows if r.get("scheme")]
            schs = []
            for r in rws:
                if r["scheme"] not in schs:
                    schs.append(r["scheme"])
            for ax, key, title in [(axes[0], "auroc_seq_se", "Sequence SE AUROC"),
                                   (axes[1], "auroc_token_se", "Per-token SE AUROC")]:
                for sc in schs:
                    pts = [(r["rate"], r.get(key)) for r in rws if r["scheme"] == sc]
                    pts = [(x, y) for x, y in pts if y is not None and y == y]
                    if pts:
                        xs, ys = zip(*sorted(pts))
                        ax.plot(xs, ys, marker="o", label=sc)
                ax.axhline(0.5, ls="--", lw=0.8, color="grey")
                ax.set_ylim(0.0, 1.02); ax.grid(alpha=0.3)
                ax.set_title(title); ax.set_xlabel("corruption rate"); ax.set_ylabel("AUROC")
            axes[0].legend(title="scheme", fontsize=9)
            fig.suptitle(f"GPT-2 Spilled Energy — OOD AUROC vs corruption ladder "
                         f"({args.ref_lm}, split={args.split}, n={pos_tok.shape[0]})")
            fig.tight_layout()
            p = out_path.with_name(out_path.stem + "_auroc.png")
            fig.savefig(p, dpi=150); plt.close(fig)
            print(f"Wrote {p}")
        except Exception as e:
            print(f"[plot] skipped: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
