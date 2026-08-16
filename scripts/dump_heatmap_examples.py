"""Dump per-character denoiser-NLL heatmap examples at multiple corruption rates.

Regenerates the ``examples`` block of ``ood_denoiser_nll.py`` for the paper's
heatmap figure (fig:ood-heatmap), at the bench's rate 0.30 AND additional rates
(default adds 0.15), so the paper can show the same window at a realistic and a
heavy corruption rate. Every bench convention is reused exactly: test split,
fit/eval slicing, t_nll, the 5%-FPR clean-quantile flag threshold, and the
example corruption seed 7 of ``_bench_common.heal_style_examples``. Because the
corruption RNG is seeded identically per scheme, the positions corrupted at a
lower rate are a subset of those at a higher rate (verified at dump time), so
panels at different rates are directly comparable.

Also stores, as sidecar metadata, the train-corpus occurrence counts of the
words the clean window's false positives land on (the paper's "honestly rare"
caption claim), so that claim has a stored source.

Usage:
    uv run python scripts/dump_heatmap_examples.py \
        --ckpt runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM_converge/epoch_final.pt \
        --out bench_ood_final/nll/heatmap_examples.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _bootstrap() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))


_bootstrap()

from scripts.ood_variance_perpos import _load_dirichletfm  # noqa: E402
from scripts._bench_common import _decode  # noqa: E402
from aitchinson_flow.training import build_training_datamodule  # noqa: E402
from aitchinson_flow.data.corruption import (  # noqa: E402
    corrupt_token_ids,
    corrupt_false_info,
    build_vocab_by_len,
)

_ALPH = "abcdefghijklmnopqrstuvwxyz "


def _det_mean_xt(tok: torch.Tensor, t: float, K: int, device: str) -> torch.Tensor:
    """Deterministic Dirichlet mean beta/sum(beta), as in ood_denoiser_nll.py."""
    B, L = tok.shape
    beta = torch.ones(B, L, K, device=device)
    beta.scatter_(-1, tok.to(device).long().unsqueeze(-1), float(t))
    return beta / beta.sum(-1, keepdim=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="bench_ood_final/nll/heatmap_examples.json")
    ap.add_argument("--rates", type=str, default="0.15,0.3")
    ap.add_argument("--t-nll", type=float, default=3.0)
    ap.add_argument("--fit-seqs", type=int, default=512)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42, help="dataset seed (bench value)")
    ap.add_argument("--example-seed", type=int, default=7,
                    help="corruption seed of heal_style_examples (bench value)")
    ap.add_argument("--example-indices", type=str, default="0,1",
                    help="eval-window indices to dump as examples")
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--fpr", type=float, default=0.05)
    ap.add_argument("--max-pos", type=int, default=120)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--check-against", default="bench_ood_final/nll/denoiser_nll_sweep.json",
                    help="bench sweep JSON to cross-check the clean/0.30 examples against")
    ap.add_argument("--rare-words", type=str, default="lewinsky,samille,dc,clinton,monica",
                    help="words to count in the train corpus (caption rarity claim)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = _load_dirichletfm(args.ckpt, device)
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
    fit_tok = seqs[: args.fit_seqs]
    pos_tok = seqs[args.fit_seqs : args.fit_seqs + args.n]
    ex_idx = [int(s) for s in args.example_indices.split(",") if s.strip()]
    ex_tok = pos_tok[ex_idx]
    print(f"[dump] split={args.split} fit={fit_tok.shape[0]} eval={pos_tok.shape[0]} "
          f"t_nll={args.t_nll} device={device}")

    @torch.no_grad()
    def score(tok: torch.Tensor) -> torch.Tensor:
        Ns = []
        for i in range(0, tok.shape[0], args.chunk):
            tb = tok[i : i + args.chunk]
            x_n = _det_mean_xt(tb, args.t_nll, K, device)
            tt = torch.full((tb.shape[0],), float(args.t_nll), device=device)
            logits = model.forward(x_n, tt)
            nll = -F.log_softmax(logits, -1).gather(
                -1, tb.to(device).long().unsqueeze(-1)).squeeze(-1)
            Ns.append(nll.cpu())
        return torch.cat(Ns)

    # flag threshold: (1 - fpr) quantile of the clean per-token scores (bench rule)
    clean_all = score(pos_tok).numpy().reshape(-1)
    flag_thr = float(np.quantile(clean_all, 1.0 - args.fpr))
    print(f"[dump] flag thr={flag_thr:.4f} @ fpr={args.fpr}")

    fit_txt = " ".join(
        "".join(_ALPH[int(i)] if int(i) < len(_ALPH) else "?" for i in row)
        for row in fit_tok)
    by_len = build_vocab_by_len(fit_txt)

    rates = [float(r) for r in args.rates.split(",") if r.strip()]
    variants = [("clean", 0.0, ex_tok.clone())]
    for r in rates:
        tag = str(int(round(r * 100)))
        variants.append((f"replace{tag}", r, corrupt_token_ids(
            ex_tok.clone(), vocab_size=K, corrupt_rate=r, seed=args.example_seed)))
        variants.append((f"falseinfo{tag}", r, corrupt_false_info(
            ex_tok.clone(), r, by_len, seed=args.example_seed)))

    examples = []
    for tag, rate, tk in variants:
        sc = score(tk)
        changed = tk != ex_tok
        flagged = sc > flag_thr
        for b in range(tk.shape[0]):
            examples.append({
                "which": tag, "rate": rate, "idx": ex_idx[b],
                "flag_thr": flag_thr, "unit": "char",
                "clean_text": _decode(ex_tok[b].tolist(), args.max_pos),
                "text": _decode(tk[b].tolist(), args.max_pos),
                "NLL_t": [round(float(x), 4) for x in sc[b][: args.max_pos].tolist()],
                "corrupted": [bool(x) for x in changed[b][: args.max_pos].tolist()],
                "flagged": [bool(x) for x in flagged[b][: args.max_pos].tolist()],
            })

    # same-seed corruptions at a lower rate must be a subset of the higher rate
    nesting = {}
    for scheme in ("replace", "falseinfo"):
        for lo, hi in zip(sorted(rates)[:-1], sorted(rates)[1:]):
            lo_ex = [e for e in examples if e["which"] == f"{scheme}{int(round(lo*100))}"]
            hi_ex = [e for e in examples if e["which"] == f"{scheme}{int(round(hi*100))}"]
            nested = all(
                all(not c_lo or c_hi for c_lo, c_hi in zip(a["corrupted"], b["corrupted"]))
                for a, b in zip(lo_ex, hi_ex))
            nesting[f"{scheme}:{lo}<={hi}"] = nested
            print(f"[dump] nesting {scheme} {lo} within {hi}: {nested}")

    # cross-check clean / rate-0.30 examples against the bench sweep, if present
    check = {}
    ref_path = Path(args.check_against)
    if ref_path.exists():
        ref = {(e["which"], e["idx"]): e
               for e in json.loads(ref_path.read_text())["examples"]}
        for e in examples:
            r = ref.get((e["which"], e["idx"]))
            if r is None:
                continue
            dnll = max(abs(a - b) for a, b in zip(e["NLL_t"], r["NLL_t"]))
            check[f'{e["which"]}#{e["idx"]}'] = {
                "max_abs_nll_diff": dnll,
                "masks_equal": (e["corrupted"] == r["corrupted"]
                                and e["flagged"] == r["flagged"]),
            }
        print(f"[dump] cross-check vs {ref_path}:")
        for k, v in check.items():
            print(f"    {k}: max|dNLL|={v['max_abs_nll_diff']:.4f} "
                  f"masks_equal={v['masks_equal']}")

    # rarity of the clean window's false-positive words, counted in the train corpus
    rarity = {}
    try:
        from aitchinson_flow.data.hf_text_loader import load_text_column
        train_txt = load_text_column(cfg.text8_dataset, split="train")
        if not isinstance(train_txt, str):
            train_txt = " ".join(train_txt)
        padded = f" {train_txt} "
        for w in [w for w in args.rare_words.split(",") if w.strip()]:
            rarity[w] = padded.count(f" {w} ")
        print(f"[dump] train-corpus word counts ({len(train_txt)} chars): {rarity}")
    except Exception as exc:  # rarity is sidecar metadata, never fail the dump
        print(f"[dump] rarity count skipped: {exc}")
        rarity = {"error": str(exc)}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "ckpt": args.ckpt, "t_nll": args.t_nll, "split": args.split,
        "n": args.n, "fit_seqs": args.fit_seqs, "seed": args.seed,
        "example_seed": args.example_seed, "fpr": args.fpr,
        "flag_thr": flag_thr, "rates": rates, "max_pos": args.max_pos,
        "nesting": nesting, "cross_check": check,
        "train_word_counts": rarity,
        "examples": examples,
    }, indent=2))
    print(f"[dump] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
