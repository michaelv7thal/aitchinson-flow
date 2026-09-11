"""Learn PPMI-SVD character embeddings from text8.

For K=27 characters this is computationally trivial (27×27 matrices). The
output is a fixed embedding matrix that ``EqMLatent`` can load when
``cfg.embedding.fixed_path`` is set, eliminating the moving-target problem
of jointly training embeddings with the flow model.

Algorithm (Levy & Goldberg, 2014; Bullinaria & Levy, 2007):
  1. Count co-occurrences of (i, j) within a symmetric window of width w
     across the training corpus.
  2. Compute PMI(i, j) = log(p(i, j) / (p(i) · p(j))).
  3. Take PPMI = max(PMI, 0).
  4. SVD: PPMI = U Σ V^T.
  5. Embedding = U[:, :d] · diag(Σ[:d])^β   (β=0.5 is the standard
     "shifted" weighting).

Output: ``--out`` is a ``torch.save``-able dict with keys
    embeddings    (K, d) float tensor
    K, d, window_size, beta, source     metadata for reproducibility

Usage:
    python scripts/learn_ppmi_svd_embeddings.py --out data/ppmi_svd_d32.pt \\
        --d 32 --window 5
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
from aitchinson_flow.training import build_training_datamodule  # noqa: E402


def co_occurrence_matrix(
    sequences: torch.Tensor,
    K: int,
    window: int = 5,
) -> torch.Tensor:
    """Symmetric co-occurrence count matrix.

    ``sequences`` shape (N, L) of long token ids. Returns a (K, K) tensor C
    where ``C[i, j]`` is the number of times token j appeared within a
    window of width ≤ ``window`` of token i, across all sequences.

    Self-pairs (i, i within distance 0) are not counted.
    """
    N, L = sequences.shape
    C = torch.zeros(K, K, dtype=torch.float64)
    for offset in range(1, window + 1):
        if offset >= L:
            break
        left = sequences[:, :-offset].reshape(-1)   # (N*(L-offset),)
        right = sequences[:, offset:].reshape(-1)
        idx = left * K + right
        flat = torch.bincount(idx.long(), minlength=K * K)
        C_off = flat.view(K, K).to(torch.float64)
        # Symmetric: count (i, j) and (j, i) for each offset.
        C += C_off + C_off.t()
    return C


def ppmi(C: torch.Tensor) -> torch.Tensor:
    """C → PPMI matrix.

    PMI(i, j) = log(C[i,j] · total / (row_i · col_j)).
    PPMI = max(PMI, 0). Returns (K, K) float64 tensor.
    """
    C = C.to(torch.float64)
    total = C.sum().clamp(min=1.0)
    row = C.sum(dim=1, keepdim=True).clamp(min=1.0)  # (K, 1)
    col = C.sum(dim=0, keepdim=True).clamp(min=1.0)  # (1, K)
    pmi = torch.log((C * total).clamp(min=1e-12) / (row * col))
    pmi = torch.where(C > 0, pmi, torch.zeros_like(pmi))  # 0 where no co-occ
    return pmi.clamp(min=0.0)


def svd_embedding(M: torch.Tensor, d: int, beta: float = 0.5) -> torch.Tensor:
    """Truncated SVD: M = U Σ V^T → ``embed = U[:, :d] · diag(Σ[:d])^β``.

    β=0.5 (default) gives the "shifted" weighting that's the standard
    word2vec-equivalent. β=1.0 corresponds to plain Eckart-Young
    factorisation; β=0 returns just U[:, :d] with no scaling.
    """
    U, S, _ = torch.linalg.svd(M, full_matrices=False)
    d = min(d, S.shape[0])
    return (U[:, :d] * S[:d].pow(beta).unsqueeze(0)).to(torch.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--K", type=int, default=27)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    cfg = Config()
    dm, _ = build_training_datamodule(cfg)
    train_ids = dm.splits.train.long()  # (N, L)
    print(f"corpus: {tuple(train_ids.shape)}, K={args.K}")

    C = co_occurrence_matrix(train_ids, K=args.K, window=args.window)
    print(f"co-occurrence: total={float(C.sum()):.0f}, "
          f"min={float(C.min()):.0f}, max={float(C.max()):.0f}")

    M = ppmi(C)
    nz = (M > 0).float().mean().item()
    print(f"PPMI: nz={nz:.3f}, mean={float(M.mean()):.4f}, max={float(M.max()):.4f}")

    embed = svd_embedding(M, d=args.d, beta=args.beta)
    print(f"embed: shape={tuple(embed.shape)}, "
          f"norm_mean={float(embed.norm(dim=-1).mean()):.4f}, "
          f"pairwise_min={float(torch.cdist(embed, embed).fill_diagonal_(float('inf')).min()):.4f}")

    out = {
        "embeddings": embed,
        "K": args.K,
        "d": args.d,
        "window_size": args.window,
        "beta": args.beta,
        "source": "text8/ppmi_svd",
        "co_occurrence_total": float(C.sum()),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
