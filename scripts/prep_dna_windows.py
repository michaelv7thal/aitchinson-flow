"""Prepare a K=4 DNA windows tensor for the EqM_OneHot K-isolation experiment.

Downloads a biological promoter/enhancer DNA dataset from the HuggingFace
genomic-benchmarks collection, tokenizes the {A,C,G,T} alphabet to ids 0..3,
and slices each sequence into length-L windows — the exact same windows-tensor
contract that ``data/char_window_dataset.py::text_to_windows`` produces for
text8, only with K=4 instead of K=27.

Windowing is done *within* each sequence (no cross-sequence junctions) and any
window containing a non-ACGT base (e.g. ``N``) is dropped, so every emitted
token id is in {0,1,2,3}. Train/val/test split sizes mirror the text8 local
baseline (10k train / 5k eval) so the EqM_OneHot comparison changes only K.

Output: a ``.pt`` file ``{"train","val","test","meta"}`` consumable by
``data/dna_datamodule.py::DNADataModule``.

Usage:
    uv run python scripts/prep_dna_windows.py \
        --dataset katarinagresova/Genomic_Benchmarks_human_nontata_promoters \
        --L 40 --out data_cache/dna/promoters_L40.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import datasets
import torch

# DNA alphabet — fixed order so id<->base is stable across runs/scripts.
DNA_ALPHABET = "ACGT"
DNA_CHAR2ID = {c: i for i, c in enumerate(DNA_ALPHABET)}
DNA_K = len(DNA_ALPHABET)  # 4


def _seqs_to_windows(seqs: list[str], L: int) -> torch.Tensor:
    """Slice each sequence into non-overlapping length-L windows over ACGT.

    Windows are taken within a single sequence (offsets 0, L, 2L, ...); the
    trailing < L bases are dropped. Windows containing any non-ACGT character
    are skipped so every id is in {0,1,2,3}. Returns an (N, L) long tensor.
    """
    rows: list[list[int]] = []
    dropped = 0
    for s in seqs:
        s = s.strip().upper()
        n = (len(s) // L) * L
        for i in range(0, n, L):
            w = s[i : i + L]
            ids = [DNA_CHAR2ID.get(c, -1) for c in w]
            if min(ids) < 0:  # non-ACGT base present → drop this window
                dropped += 1
                continue
            rows.append(ids)
    if not rows:
        return torch.empty(0, L, dtype=torch.long), dropped
    return torch.tensor(rows, dtype=torch.long), dropped


def _load_seqs(dataset: str, split: str, text_column: str) -> list[str]:
    ds = datasets.load_dataset(dataset, split=split)
    if text_column not in ds.column_names:
        # genomic-benchmarks uses "seq"; fall back to the first string column.
        cand = [c for c in ds.column_names if c != "label"]
        text_column = cand[0]
    return [str(x) for x in ds[text_column]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset",
        default="katarinagresova/Genomic_Benchmarks_human_nontata_promoters",
        help="HuggingFace dataset id (genomic-benchmarks promoter/enhancer).",
    )
    ap.add_argument("--text-column", default="seq")
    ap.add_argument("--L", type=int, default=40)
    ap.add_argument("--max-train-windows", type=int, default=10_000)
    ap.add_argument("--max-eval-windows", type=int, default=5_000)
    ap.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Fraction of the train split carved off for validation "
        "(genomic-benchmarks ships only train/test).",
    )
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="data_cache/dna/promoters_L40.pt")
    args = ap.parse_args()

    print(f"[prep_dna] dataset={args.dataset} L={args.L}")
    train_seqs = _load_seqs(args.dataset, "train", args.text_column)
    test_seqs = _load_seqs(args.dataset, "test", args.text_column)
    print(f"[prep_dna] raw sequences: train={len(train_seqs)} test={len(test_seqs)}")

    # Deterministic shuffle of sequences before windowing so the capped train
    # split is not dominated by one label block (genomic-benchmarks orders by
    # label) and so val is a representative carve-out.
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(train_seqs), generator=g).tolist()
    train_seqs = [train_seqs[i] for i in perm]
    n_val_seqs = int(len(train_seqs) * args.val_frac)
    val_seqs = train_seqs[:n_val_seqs]
    tr_seqs = train_seqs[n_val_seqs:]

    train_w, d_tr = _seqs_to_windows(tr_seqs, args.L)
    val_w, d_va = _seqs_to_windows(val_seqs, args.L)
    test_w, d_te = _seqs_to_windows(test_seqs, args.L)
    print(
        f"[prep_dna] windows (pre-cap): train={train_w.shape[0]} "
        f"val={val_w.shape[0]} test={test_w.shape[0]} "
        f"(dropped non-ACGT windows: {d_tr + d_va + d_te})"
    )

    # Shuffle windows deterministically, then cap to mirror the text8 baseline.
    def _cap(w: torch.Tensor, n: int | None, salt: int) -> torch.Tensor:
        if w.shape[0] == 0:
            return w
        gg = torch.Generator().manual_seed(args.seed + salt)
        w = w[torch.randperm(w.shape[0], generator=gg)]
        return w if n is None else w[:n]

    train_w = _cap(train_w, args.max_train_windows, 1)
    val_w = _cap(val_w, args.max_eval_windows, 2)
    test_w = _cap(test_w, args.max_eval_windows, 3)

    # Sanity: every id must be a valid base.
    for name, w in [("train", train_w), ("val", val_w), ("test", test_w)]:
        if w.numel() and (int(w.min()) < 0 or int(w.max()) >= DNA_K):
            raise ValueError(f"{name} has out-of-range ids: min={w.min()} max={w.max()}")

    # Report unigram base composition (GC content sanity check).
    flat = train_w.reshape(-1)
    comp = {DNA_ALPHABET[i]: round((flat == i).float().mean().item(), 4) for i in range(DNA_K)}
    print(f"[prep_dna] train base composition: {comp}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "train": train_w,
            "val": val_w,
            "test": test_w,
            "meta": {
                "dataset": args.dataset,
                "alphabet": DNA_ALPHABET,
                "K": DNA_K,
                "L": args.L,
                "base_composition": comp,
            },
        },
        out,
    )
    print(
        f"[prep_dna] saved -> {out}  "
        f"(train={train_w.shape[0]} val={val_w.shape[0]} test={test_w.shape[0]}, K={DNA_K})"
    )


if __name__ == "__main__":
    main()
