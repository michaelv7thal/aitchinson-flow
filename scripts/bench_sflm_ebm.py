"""Spilled-energy benchmark: SFLMEBM vs EqM / EqMLatent / DFM on text8.

Two questions, both about whether the model's energy localises corruption:

  (1) SEQUENCE level — clean vs corrupted AUROC. Each model's natural
      sequence score (``model.energy`` for the EqM family; mean
      per-position spilled energy for DFM).

  (2) PER-POSITION level — *spilled-energy localisation*. For substitution
      corruption we know exactly which positions changed. Score every
      position by its spilled energy ``SE(i) = logsumexp_v logits_i −
      logits_i[token_i]`` (uniform across models — all expose per-position
      logits) and report AUROC of corrupted-vs-clean positions *within the
      corrupted sequences*. This is the headline metric the user asked
      for: does the energy spill at the right token?

EqM-family models additionally expose ``position_uncertainty`` (Riemannian
‖∇E‖ per position); reported alongside SE where available.

Reads runs/sflm_bench/<model>/epoch_final.pt (see
scripts/train_for_sflm_bench.py). Writes runs/sflm_bench/bench.json + a
printed table.

Usage:  python scripts/bench_sflm_ebm.py [--n 256]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.append(str(_ROOT))

import aitchinson_flow.models  # noqa: E402,F401
from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.data.corruption import (  # noqa: E402
    corrupt_token_ids,
    partially_shuffle_token_ids,
)
from aitchinson_flow.data.transforms import token_ids_to_features  # noqa: E402
from aitchinson_flow.models import build_model  # noqa: E402
from aitchinson_flow.training import build_training_datamodule  # noqa: E402
from scripts.eval_full import _config_from_payload  # noqa: E402

MODELS = ["SFLMEBM", "EqM", "EqMLatent", "DFM"]


def _auroc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """AUROC with `pos` the class that should score higher."""
    pos, neg = pos.flatten().float().cpu(), neg.flatten().float().cpu()
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    comb = torch.cat([pos, neg])
    order = comb.argsort()
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, comb.numel() + 1, dtype=torch.float)
    return float(
        (ranks[: pos.numel()].sum() - pos.numel() * (pos.numel() + 1) / 2)
        / (pos.numel() * neg.numel())
    )


@torch.no_grad()
def _per_pos_logits(model, name: str, ids: torch.Tensor, cfg: Config):
    """(B, L, K) per-position logits via each model's natural path, and the
    sequence-level energy score (None for DFM → fall back to mean SE)."""
    K = cfg.text8_dataset.K
    if name == "DFM":
        t = torch.full((ids.shape[0],), 0.99, device=ids.device)
        return model.forward(ids, t), None
    if name == "EqM":  # simplex: CLR features, no encode()
        feats = token_ids_to_features(
            ids, K, label_smoothing=cfg.transformation.label_smoothing
        )
        return model.decode_to_logprobs(feats), model.energy(feats)
    # SFLMEBM / EqMLatent: learned embedding lookup
    z = model.encode(ids)
    return model.decode_to_logprobs(z), model.energy(z)


@torch.no_grad()
def _spilled_energy(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """SE(i) = logsumexp_v logits_i − logits_i[token_i].  (B, L)."""
    lse = torch.logsumexp(logits, dim=-1)
    picked = logits.gather(-1, ids.long().unsqueeze(-1)).squeeze(-1)
    return lse - picked


def _position_uncertainty(model, name: str, ids: torch.Tensor, cfg: Config):
    if name == "DFM" or not hasattr(model, "position_uncertainty"):
        return None
    if name == "EqM":
        feats = token_ids_to_features(
            ids, cfg.text8_dataset.K,
            label_smoothing=cfg.transformation.label_smoothing,
        )
        return model.position_uncertainty(feats)
    return model.position_uncertainty(model.encode(ids))


# Reusable, well-trained baseline checkpoints (d1024). No fresh retrain
# needed — these are *stronger* than a local d512 retrain, so a local
# SFLMEBM win against them is a tougher, more honest result. Scale is not
# matched (caveat); the matched apples-to-apples is `--scale cluster`,
# where train_for_sflm_bench.py trains all four at d1024. plain "DFM" has
# no reusable text8 checkpoint in runs/ → only available from the cluster
# run (runs/sflm_bench_cluster/DFM).
EXISTING_BASELINES = {
    "EqM": "runs/dphase4_lambda_recalib/epoch_final.pt",
    "EqMLatent": "runs/latent_d128_trainable_tied_ce0_ep20/epoch_final.pt",
}


def _load(name: str, device, root: str):
    # SFLMEBM: from the freshly trained sflm_bench_<scale> run. Baselines:
    # prefer the sflm_bench run if present (cluster), else fall back to the
    # reusable existing checkpoint (local).
    ckpt = Path(f"{root}/{name}/epoch_final.pt")
    if not ckpt.exists() and name in EXISTING_BASELINES:
        ckpt = Path(EXISTING_BASELINES[name])
        print(f"[{name}] reusing existing checkpoint {ckpt}")
    if not ckpt.exists():
        return None, None
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = _config_from_payload(payload)
    model = build_model(cfg).to(device)
    state = payload.get("model_state_dict", payload)
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--scale", choices=["local", "cluster"], default="local",
                    help="which sflm_bench_<scale> checkpoints to benchmark")
    args = ap.parse_args()
    root = f"runs/sflm_bench_{args.scale}"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Clean held-out text8 windows (datamodule is model-independent here).
    base_cfg = Config()
    dm, _ = build_training_datamodule(base_cfg)
    val = dm.splits.val.long()
    g = torch.Generator().manual_seed(args.seed)
    pick = torch.randperm(val.shape[0], generator=g)[: args.n]
    clean = val[pick].to(device)
    K = base_cfg.text8_dataset.K

    # Corruption ladder. `mask` marks changed positions (per-position GT).
    corruptions = {}
    for r in (0.1, 0.3):
        c = corrupt_token_ids(clean, vocab_size=K, corrupt_rate=r, seed=args.seed)
        corruptions[f"subst_{r}"] = (c, c != clean)
    sh = partially_shuffle_token_ids(clean, shuffle_rate=1.0, seed=args.seed + 7)
    corruptions["shuffle"] = (sh, sh != clean)
    rnd = torch.randint(0, K, clean.shape, device=device,
                         generator=torch.Generator(device=device).manual_seed(args.seed))
    corruptions["rand"] = (rnd, torch.ones_like(rnd, dtype=torch.bool))

    results: dict = {}
    for name in MODELS:
        model, cfg = _load(name, device, root)
        if model is None:
            print(f"[skip] {name}: no checkpoint")
            continue

        cl_logits, cl_E = _per_pos_logits(model, name, clean, cfg)
        cl_SE = _spilled_energy(cl_logits, clean)               # (B, L)
        cl_seq = cl_E if cl_E is not None else cl_SE.mean(-1)    # (B,)
        cl_pu = _position_uncertainty(model, name, clean, cfg)

        res = {"seq_auroc": {}, "pospair_se_auroc": {}, "pospair_pu_auroc": {}}
        for cname, (cids, cmask) in corruptions.items():
            co_logits, co_E = _per_pos_logits(model, name, cids, cfg)
            co_SE = _spilled_energy(co_logits, cids)
            co_seq = co_E if co_E is not None else co_SE.mean(-1)
            # (1) sequence-level clean vs corrupted (corrupted scores higher)
            res["seq_auroc"][cname] = _auroc(co_seq, cl_seq)
            # (2) per-position localisation: changed vs unchanged positions
            #     *inside the corrupted sequences*.
            if cmask.any() and (~cmask).any():
                res["pospair_se_auroc"][cname] = _auroc(
                    co_SE[cmask], co_SE[~cmask]
                )
                if cl_pu is not None:
                    co_pu = _position_uncertainty(model, name, cids, cfg)
                    res["pospair_pu_auroc"][cname] = _auroc(
                        co_pu[cmask], co_pu[~cmask]
                    )
        results[name] = res

    Path(root).mkdir(parents=True, exist_ok=True)
    Path(f"{root}/bench.json").write_text(json.dumps(results, indent=2))

    # ---- report ----
    cs = list(corruptions)
    print("\n" + "=" * 78)
    print("(1) SEQUENCE-level AUROC  (clean vs corrupted; 0.5=chance, 1=perfect)")
    print("=" * 78)
    print(f"{'model':10s} " + " ".join(f"{c:>10s}" for c in cs))
    for n, r in results.items():
        print(f"{n:10s} " + " ".join(
            f"{r['seq_auroc'].get(c, float('nan')):10.3f}" for c in cs))

    print("\n" + "=" * 78)
    print("(2) PER-POSITION spilled-energy localisation AUROC")
    print("    (changed vs unchanged positions within corrupted seqs)")
    print("=" * 78)
    loc = [c for c in cs if c != "rand"]  # rand changes all positions
    print(f"{'model':10s} " + " ".join(f"{c:>10s}" for c in loc))
    for n, r in results.items():
        print(f"{n:10s} " + " ".join(
            f"{r['pospair_se_auroc'].get(c, float('nan')):10.3f}" for c in loc))

    print("\n--- per-position via Riemannian ‖∇E‖ (EqM family only) ---")
    print(f"{'model':10s} " + " ".join(f"{c:>10s}" for c in loc))
    for n, r in results.items():
        if r["pospair_pu_auroc"]:
            print(f"{n:10s} " + " ".join(
                f"{r['pospair_pu_auroc'].get(c, float('nan')):10.3f}"
                for c in loc))
    print(f"\nWrote {root}/bench.json")


if __name__ == "__main__":
    main()
