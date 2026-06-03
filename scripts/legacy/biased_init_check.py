"""Diagnostic: re-evaluate a pure-FM EqMLatent checkpoint with biased noise init.

Hypothesis (user-posed): pure FM with fixed embeddings fails on K=27 text not
because of time-conditioning vs CE, but because the 27 embeddings are
symmetrically arranged around their centroid. The FM regression fit produces
an energy field whose global minimum sits at the centroid (all 27 attractors
balance). NAG-GD from a centred Gaussian falls into that minimum.

Test: re-run sampling from a *biased* init that breaks the symmetry:
    x_init = σ·randn(B, L, d) + α·embed[random_token(B, L)]

For α > 0, each (B, L) position is started inside one of the 27 basins
(its random_token's basin). NAG follows the local gradient toward the
nearest data mode, which should be that random token's embedding row.

Compares argmax(x_final @ embed.T) against the random token labels
to gauge basin-fidelity, plus reports KL_uni / KL_bi against the corpus.

Usage:
    python scripts/biased_init_check.py \\
        --ckpt runs/latent_fixed_skipgram_d27_tied_pureFM_ep20/epoch_final.pt \\
        --alphas 0,0.25,0.5,1.0,1.5,2.0
"""

from __future__ import annotations

import argparse
import json
import math
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

import aitchinson_flow.models  # noqa: E402,F401 — populate registry
from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.data.char_window_dataset import CHAR2ID  # noqa: E402
from aitchinson_flow.models import build_model  # noqa: E402
from aitchinson_flow.training import build_training_datamodule  # noqa: E402
from scripts.eval_full import _config_from_payload  # noqa: E402

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
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--alphas", type=str, default="0,0.25,0.5,1.0,1.5,2.0")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = _config_from_payload(payload)
    device = cfg.training.device
    model = build_model(cfg).to(device)
    state = payload["model_state_dict"] if "model_state_dict" in payload else payload
    model.load_state_dict(state)
    model.eval()

    dm, _ = build_training_datamodule(cfg)
    train_ids = dm.splits.train.long()
    K = cfg.text8_dataset.K
    L = cfg.text8_dataset.L
    sigma = cfg.eqm.source_sigma

    embed = model.embed.weight.detach().to(device)
    centroid = embed.mean(dim=0)
    print(f"K={K}, L={L}, d={embed.shape[1]}, σ={sigma}")
    print(f"embed: norm_mean={embed.norm(dim=-1).mean().item():.4f}  "
          f"centroid_norm={centroid.norm().item():.4f}  "
          f"pairwise_min={torch.cdist(embed, embed).fill_diagonal_(float('inf')).min().item():.4f}")
    print()

    ref_uni = _ngram_counts_flat(train_ids, K, 1)
    ref_bi = _ngram_counts_flat(train_ids, K, 2)
    ref_tri = _ngram_counts_flat(train_ids, K, 3)

    rows: list[dict] = []
    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    print(f"{'α':>6} {'KL_uni':>8} {'KL_bi':>8} {'KL_tri':>8} "
          f"{'H_gen':>7} {'init_basin_acc':>15} {'sample[0]':>}")
    print("-" * 100)
    for alpha in alphas:
        torch.manual_seed(args.seed)
        # Biased init: each (b, l) starts in the basin of a random token,
        # plus σ·noise for variety.
        rand_ids = torch.randint(0, K, (args.n, L), device=device)
        x_init = sigma * torch.randn(args.n, L, embed.shape[1], device=device) \
                 + alpha * embed[rand_ids]

        with torch.no_grad():
            x = model.sample(args.n, L, max_steps=args.steps, x_init=x_init)
            log_probs = model.decode_to_logprobs(x)
            ids = log_probs.argmax(-1).cpu()

        gen_uni = _ngram_counts_flat(ids, K, 1)
        gen_bi = _ngram_counts_flat(ids, K, 2)
        gen_tri = _ngram_counts_flat(ids, K, 3)

        kl_u = _kl_smoothed(gen_uni, ref_uni)
        kl_b = _kl_smoothed(gen_bi, ref_bi)
        kl_t = _kl_smoothed(gen_tri, ref_tri)
        p = (gen_uni + 1e-9) / (gen_uni.sum() + K * 1e-9)
        H_gen = float(-(p * p.log()).sum())

        # How often does the final argmax recover the *seed* token (i.e., does
        # NAG stay in the basin it was initialised in)?
        init_basin_acc = float((ids == rand_ids.cpu()).float().mean())

        sample0 = "".join(ALPHABET[int(i)] for i in ids[0])
        print(f"{alpha:>6.2f} {kl_u:>8.4f} {kl_b:>8.4f} {kl_t:>8.4f} "
              f"{H_gen:>7.4f} {init_basin_acc:>15.3f}  {sample0!r}")
        rows.append({
            "alpha": alpha, "KL_uni": kl_u, "KL_bi": kl_b, "KL_tri": kl_t,
            "H_gen": H_gen, "init_basin_acc": init_basin_acc, "sample0": sample0,
        })

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"rows": rows}, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
