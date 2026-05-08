"""Phase V — train the SAPLMA-style 3-layer MLP probe on the wiki cache.

Output goes to ``runs/capstone/checkpoints/saplma_wiki/probe.pt`` —
loadable by ``aitchinson_flow.baselines.saplma.score_saplma``.

Usage:
    python scripts/train_saplma.py --cache data/wiki_cache_gpt2.pt \\
        --out runs/capstone/checkpoints/saplma_wiki/probe.pt \\
        --epochs 25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from aitchinson_flow.baselines.saplma import SaplmaProbe, train_saplma  # noqa: E402
from aitchinson_flow.data.wiki import load_wiki_cache  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="data/wiki_cache_gpt2.pt")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    cache = load_wiki_cache(args.cache)
    H = int(cache["clean_h"].shape[-1])
    probe = train_saplma(
        cache,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        train_frac=args.train_frac,
        device=args.device,
        seed=args.seed,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": probe.state_dict(), "in_dim": H}, out)
    print(f"[write] {out}")


if __name__ == "__main__":
    main()
