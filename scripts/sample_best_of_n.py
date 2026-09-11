"""Best-of-N unconditional sampling — generate B·N samples, score each by
``model.energy(x)``, return the lowest-energy K per row.

For each of B "slots", run N independent sampling chains from independent
noise inits, score the final iterates by the model's energy, and keep
the best. Trivially parallel. Catches one of the failure modes single-
chain NAG-GD has: getting stuck in a spurious basin near the noise init
when there's a better basin not far away that another chain found.

Writes ``runs/<cell>/eval_bestof<N>.json`` with the same KL/H scorecard
as ``scripts/eval_full.py`` for direct comparison.

Usage:
    python scripts/sample_best_of_n.py \\
        --ckpt runs/ae_d256_l2_z64/eqm/epoch_final.pt \\
        --n-samples 256 --best-of 8 --steps 200 \\
        --sampler nag \\
        --out runs/ae_d256_l2_z64/eqm/eval_bestof8.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


def _bootstrap() -> None:
    repo = Path(__file__).resolve().parent.parent
    src = repo / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    if str(repo) not in sys.path:
        sys.path.append(str(repo))


_bootstrap()

import aitchinson_flow.models  # noqa: E402,F401
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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-samples", type=int, default=256, help="output sample count B")
    ap.add_argument("--best-of", type=int, default=8, help="N candidate chains per slot")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--sampler", type=str, default=None,
                    choices=("nag", "euler", "sde", None),
                    help="override cfg.eqm.sampler")
    ap.add_argument("--sde-alpha", type=float, default=None,
                    help="if --sampler=sde, override cfg.eqm.sde_alpha")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = _config_from_payload(payload)
    device = cfg.training.device
    if args.sampler is not None:
        from dataclasses import replace
        eqm_overrides: dict[str, Any] = {"sampler": args.sampler}
        if args.sde_alpha is not None:
            eqm_overrides["sde_alpha"] = float(args.sde_alpha)
        cfg.eqm = replace(cfg.eqm, **eqm_overrides)
    K = cfg.text8_dataset.K
    L = cfg.text8_dataset.L

    model = build_model(cfg).to(device)
    state = payload["model_state_dict"] if "model_state_dict" in payload else payload
    model.load_state_dict(state)
    model.eval()

    # Reference n-gram stats from the train split for the headline KLs.
    dm, _ = build_training_datamodule(cfg)
    train_ids = dm.splits.train.long()
    ref_uni = _ngram_counts_flat(train_ids, K, 1)
    ref_bi = _ngram_counts_flat(train_ids, K, 2)
    ref_tri = _ngram_counts_flat(train_ids, K, 3)
    p_ref = (ref_uni + 1e-9) / (ref_uni.sum() + K * 1e-9)
    H_ref = float(-(p_ref * p_ref.log()).sum())

    B = int(args.n_samples)
    N = int(args.best_of)
    print(f"=== best-of-{N} on {args.ckpt} (B={B}, sampler={cfg.eqm.sampler}) ===")
    torch.manual_seed(int(args.seed))

    # Run N chains sequentially — each is a size-B sample. We keep a
    # running best (lowest energy) per slot, so total GPU memory stays
    # bounded at one chain's worth (no B*N tensor ever materialises).
    best_x: torch.Tensor | None = None
    best_E: torch.Tensor | None = None
    E_sum = torch.zeros(B, device=device)
    for n_idx in range(N):
        x = model.sample(B, L, max_steps=args.steps)
        with torch.no_grad():
            E = model.energy(x)
        E_sum = E_sum + E
        if best_x is None:
            best_x = x.detach()
            best_E = E.detach()
        else:
            improved = E < best_E
            if improved.any():
                idx = improved.nonzero(as_tuple=True)[0]
                best_x[idx] = x[idx].detach()
                best_E[idx] = E[idx].detach()
        # Free intermediate tensor before the next chain.
        del x, E
        torch.cuda.empty_cache()
    E_mean_all = E_sum / float(N)

    log_probs = model.decode_to_logprobs(best_x)
    ids2d = log_probs.argmax(-1).cpu()

    gen_uni = _ngram_counts_flat(ids2d, K, 1)
    gen_bi = _ngram_counts_flat(ids2d, K, 2)
    gen_tri = _ngram_counts_flat(ids2d, K, 3)
    p = (gen_uni + 1e-9) / (gen_uni.sum() + K * 1e-9)
    H_gen = float(-(p * p.log()).sum())

    out = {
        "ckpt": str(args.ckpt),
        "best_of": N,
        "n_samples": B,
        "sampler": cfg.eqm.sampler,
        "sde_alpha": float(getattr(cfg.eqm, "sde_alpha", 0.0)),
        "steps": int(args.steps),
        "unigram_kl": _kl_smoothed(gen_uni, ref_uni),
        "bigram_kl": _kl_smoothed(gen_bi, ref_bi),
        "trigram_kl": _kl_smoothed(gen_tri, ref_tri),
        "H_gen": H_gen,
        "H_gt": H_ref,
        "H_ratio": H_gen / max(H_ref, 1e-9),
        "energy_mean_best": float(best_E.mean().item()),
        "energy_mean_all": float(E_mean_all.mean().item()),
        "energy_gap_mean": float((E_mean_all - best_E).mean().item()),
        "samples": [
            "".join(ALPHABET[int(i)] for i in ids2d[s])
            for s in range(min(8, B))
        ],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"unigram_kl={out['unigram_kl']:.4f} bigram_kl={out['bigram_kl']:.4f} "
          f"H_ratio={out['H_ratio']:.3f} E_best={out['energy_mean_best']:.3f}")
    print(f"sample[0]: {out['samples'][0]!r}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
