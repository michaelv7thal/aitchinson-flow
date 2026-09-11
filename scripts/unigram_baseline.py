"""Phase 6 — trivial unigram baseline for the capstone.

Samples sequences i.i.d. from the empirical unigram distribution of the
training corpus and reports the same KL/entropy metrics that
``scripts/eval_full.py`` reports for trained models. If the EqM model
doesn't beat this, the training is broken.

Output: writes a JSON record matching the ``eval.json`` schema, plus a
``samples`` field with 8 decoded sequences.

Usage:
    python scripts/unigram_baseline.py \\
        --out runs/unigram_baseline/eval.json --n 256
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def _bootstrap() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))


_bootstrap()

from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.data.char_window_dataset import CHAR2ID  # noqa: E402
from aitchinson_flow.training import build_training_datamodule  # noqa: E402

ALPHABET = "".join(sorted(CHAR2ID, key=CHAR2ID.__getitem__))


def _ngram_counts_flat(ids2d: torch.Tensor, K: int, n: int) -> torch.Tensor:
    L = ids2d.shape[1]
    if L < n:
        return torch.zeros(K**n)
    idx = torch.zeros(ids2d.shape[0], L - n + 1, dtype=torch.long)
    for i in range(n):
        idx = idx + ids2d[:, i : L - n + 1 + i].long() * (K ** (n - 1 - i))
    counts = torch.zeros(K**n)
    counts.scatter_add_(0, idx.reshape(-1), torch.ones(idx.numel()))
    return counts


def _kl_smoothed(gen: torch.Tensor, ref: torch.Tensor) -> float:
    smoothing = 1e-6
    Ksize = gen.numel()
    gp = (gen + smoothing) / (gen.sum() + Ksize * smoothing)
    rp = (ref + smoothing) / (ref.sum() + Ksize * smoothing)
    return float((gp * (gp.log() - rp.log())).sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--n", type=int, default=256, help="number of sampled sequences")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg = Config()
    K = cfg.text8_dataset.K
    L = cfg.text8_dataset.L

    dm, _ = build_training_datamodule(cfg)
    train_ids = dm.splits.train.long()  # (Ntrain, L)
    val_ids = dm.splits.val.long()       # noqa: F841 — referenced only via dm

    # Empirical unigram on the training corpus.
    uni_counts = _ngram_counts_flat(train_ids, K, 1)
    uni_p = uni_counts / uni_counts.sum()

    # Sample n sequences of length L i.i.d. from uni_p.
    sampled = torch.multinomial(uni_p, args.n * L, replacement=True).view(args.n, L)

    gen_uni = _ngram_counts_flat(sampled, K, 1)
    gen_bi = _ngram_counts_flat(sampled, K, 2)
    gen_tri = _ngram_counts_flat(sampled, K, 3)
    ref_uni = _ngram_counts_flat(train_ids, K, 1)
    ref_bi = _ngram_counts_flat(train_ids, K, 2)
    ref_tri = _ngram_counts_flat(train_ids, K, 3)

    p_gen = (gen_uni + 1e-9) / (gen_uni.sum() + K * 1e-9)
    p_ref = (ref_uni + 1e-9) / (ref_uni.sum() + K * 1e-9)
    H_gen = float(-(p_gen * p_gen.log()).sum())
    H_ref = float(-(p_ref * p_ref.log()).sum())

    samples_text = ["".join(ALPHABET[int(i)] for i in row) for row in sampled[:8]]

    record = {
        "ckpt": None,
        "model": "unigram_baseline",
        "epoch": None,
        "global_step": None,
        "n_samples": args.n,
        "sample_steps": None,
        "unigram_kl": _kl_smoothed(gen_uni, ref_uni),
        "bigram_kl": _kl_smoothed(gen_bi, ref_bi),
        "trigram_kl": _kl_smoothed(gen_tri, ref_tri),
        "H_gen": H_gen,
        "H_gt": H_ref,
        "H_ratio": H_gen / H_ref if H_ref > 0 else float("nan"),
        "samples": samples_text,
        "run_name": "unigram_baseline",
        "overrides": {},
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2))

    print(
        f"unigram_baseline: KL_uni={record['unigram_kl']:.4f} "
        f"KL_bi={record['bigram_kl']:.4f} "
        f"KL_tri={record['trigram_kl']:.4f} "
        f"H_ratio={record['H_ratio']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
