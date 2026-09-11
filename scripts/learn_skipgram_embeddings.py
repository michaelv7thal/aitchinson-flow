"""Learn skip-gram character embeddings on text8 (Mikolov et al., 2013).

Trains two ``nn.Embedding(K, d)`` matrices — one for the center token, one
for the context — by maximising ``log p(context | center)`` over all
(center, context) pairs within a symmetric window. The center matrix is
saved as the fixed-embedding artefact for ``EqMLatent``; the context
matrix is discarded after training.

For K=27 chars this is tiny and converges quickly on CPU. Output schema
matches ``learn_ppmi_svd_embeddings.py``:

    embeddings    (K, d) float tensor   — center embedding
    K, d, window_size, source           metadata

Usage:
    python scripts/learn_skipgram_embeddings.py --out data/skipgram_d27.pt \\
        --d 27 --window 5 --epochs 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def build_pairs(sequences: torch.Tensor, window: int) -> tuple[torch.Tensor, torch.Tensor]:
    """All (center, context) pairs within a symmetric window across sequences.

    Returns two flat long tensors of equal length.
    """
    centers: list[torch.Tensor] = []
    contexts: list[torch.Tensor] = []
    L = sequences.shape[1]
    for offset in range(1, window + 1):
        if offset >= L:
            break
        # pair (i, i+offset)
        c1 = sequences[:, :-offset].reshape(-1)
        x1 = sequences[:, offset:].reshape(-1)
        # pair (i+offset, i)  — symmetric counterpart
        c2 = sequences[:, offset:].reshape(-1)
        x2 = sequences[:, :-offset].reshape(-1)
        centers.append(c1)
        contexts.append(x1)
        centers.append(c2)
        contexts.append(x2)
    return torch.cat(centers).long(), torch.cat(contexts).long()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--d", type=int, default=27)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--K", type=int, default=27)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    cfg = Config()
    dm, _ = build_training_datamodule(cfg)
    train_ids = dm.splits.train.long()
    print(f"corpus: {tuple(train_ids.shape)}, K={args.K}")

    centers, contexts = build_pairs(train_ids, window=args.window)
    n_pairs = centers.shape[0]
    print(f"pairs: {n_pairs:,d}")

    center_embed = nn.Embedding(args.K, args.d).to(device)
    context_embed = nn.Embedding(args.K, args.d).to(device)
    nn.init.normal_(center_embed.weight, std=0.5 / args.d ** 0.5)
    nn.init.normal_(context_embed.weight, std=0.5 / args.d ** 0.5)

    optim = torch.optim.AdamW(
        list(center_embed.parameters()) + list(context_embed.parameters()),
        lr=args.lr,
    )

    centers = centers.to(device)
    contexts = contexts.to(device)

    for epoch in range(args.epochs):
        perm = torch.randperm(n_pairs, device=device)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, n_pairs, args.batch):
            idx = perm[i : i + args.batch]
            c = centers[idx]
            x = contexts[idx]
            logits = center_embed(c) @ context_embed.weight.t()  # (B, K)
            loss = F.cross_entropy(logits, x)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total_loss += float(loss)
            n_batches += 1
        avg = total_loss / max(n_batches, 1)
        print(f"epoch {epoch + 1}/{args.epochs}: loss={avg:.4f}")

    embed = center_embed.weight.detach().cpu()
    norm = embed.norm(dim=-1)
    pwise = torch.cdist(embed, embed).fill_diagonal_(float("inf"))
    print(f"embed: shape={tuple(embed.shape)}, "
          f"norm_mean={float(norm.mean()):.4f}, "
          f"pairwise_min={float(pwise.min()):.4f}")

    out = {
        "embeddings": embed,
        "K": args.K,
        "d": args.d,
        "window_size": args.window,
        "epochs": args.epochs,
        "source": "text8/skipgram",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
