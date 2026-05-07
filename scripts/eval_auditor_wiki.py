"""Auditor scorecard on the WikiText-2 cache (Phase F, F1 decision).

For each (clean, invalid) chunk we compute:

* ``E_seq``       — sequence-level grad-norm² of ⟨x, f(x; γ_aud)⟩
* ``E_seq_dot``   — sequence-level signed energy ⟨x, f(x; γ_aud)⟩
* ``U_pos_mean``  — per-position grad-norm averaged across L
* ``U_pos_max``   — per-position grad-norm max
* ``SE_seq``      — Spilled Energy (LM NLL summed over positions)
                    pulled directly from the cache; included as the
                    zero-train baseline that the prototype reports at
                    Seq AUROC ≈ 0.998 on this task.

Headline outputs:

* ``Seq AUROC``   on each statistic, clean vs invalid (sequence-level).
* ``Tok AUROC``   on per-position grad-norm restricted to *corrupted*
                  positions vs clean positions in the same sequences.

Decision (TRAINING_PROTOCOL.md §6 Phase F): F1 *passes* if ``Seq AUROC ≥
0.99`` AND ``Tok AUROC ≥ 0.95`` (within 1 point of the prototype's
0.999/0.996). Always include the SE row; if SE alone matches/beats the
trained auditor, frame the auditor's contribution as *complementary*
(structural / per-position / generative) rather than discriminative.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import aitchinson_flow.models  # noqa: E402,F401 — populate registry
from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.data.wiki import WikiAuditorDataset, load_wiki_cache  # noqa: E402
from aitchinson_flow.models import build_model  # noqa: E402

from scripts.eval_full import _config_from_payload  # noqa: E402


def _grad_norm_per_pos(model, x: torch.Tensor, gamma_value: float) -> torch.Tensor:
    """Return (B, L) per-position grad-norm of ⟨x, f(x; γ)⟩ at γ=gamma_value."""
    time_cond = getattr(model.cfg.eqm, "time_conditioning", "off")
    B = x.shape[0]
    x_req = x.detach().requires_grad_(True)
    if time_cond != "off":
        gamma = torch.full((B,), float(gamma_value), device=x.device, dtype=x.dtype)
    else:
        gamma = None
    v = model.forward(x_req, gamma)
    energy = (x_req * v).sum()
    grad = torch.autograd.grad(energy, x_req, create_graph=False)[0].detach()
    return grad.norm(dim=-1)  # (B, L)


def _signed_energy(model, x: torch.Tensor, gamma_value: float) -> torch.Tensor:
    """Return (B,) signed energy E(x) = Σ ⟨x, f(x; γ)⟩."""
    with torch.no_grad():
        time_cond = getattr(model.cfg.eqm, "time_conditioning", "off")
        B = x.shape[0]
        if time_cond != "off":
            gamma = torch.full((B,), float(gamma_value), device=x.device, dtype=x.dtype)
        else:
            gamma = None
        v = model.forward(x, gamma)
        return (x * v).sum(dim=(-1, -2))


def _roc_auc(neg: torch.Tensor, pos: torch.Tensor) -> float:
    n_pos, n_neg = pos.numel(), neg.numel()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    combined = torch.cat([pos.flatten(), neg.flatten()])
    order = combined.argsort()
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, combined.numel() + 1, dtype=torch.float)
    pos_ranks = ranks[: n_pos]
    return float((pos_ranks.sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def evaluate_auditor(
    ckpt_path: str | Path,
    cache_path: str | Path,
    *,
    n_eval: int | None = None,
    batch_size: int = 16,
) -> dict[str, Any]:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = _config_from_payload(payload)
    device = cfg.training.device
    gamma = float(cfg.eqm.auditor_gamma)

    model = build_model(cfg).to(device)
    state = payload["model_state_dict"] if "model_state_dict" in payload else payload
    model.load_state_dict(state)
    model.eval()

    cache = load_wiki_cache(cache_path)
    ds = WikiAuditorDataset(cache)
    n_total = len(ds)
    n = n_total if n_eval is None else min(n_eval, n_total)

    # Validation slice — same convention as training datamodule's split:
    # train uses idx[:n_train], val uses idx[n_train:]. We score on the
    # *full* cache here for tightest AUROC; if you want held-out only,
    # pass --n_eval = n_total - n_train.
    grad_norm_clean: list[torch.Tensor] = []
    grad_norm_invalid: list[torch.Tensor] = []
    signed_clean: list[torch.Tensor] = []
    signed_invalid: list[torch.Tensor] = []
    SE_clean: list[torch.Tensor] = []
    SE_invalid: list[torch.Tensor] = []
    SE_pos_clean: list[torch.Tensor] = []
    SE_pos_invalid: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []

    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        items = [ds[i] for i in range(s, e)]
        x_clean = torch.stack([it["x"] for it in items]).to(device)
        x_invalid = torch.stack([it["x_invalid"] for it in items]).to(device)
        mask = torch.stack([it["mask_corrupt"] for it in items])

        gn_c = _grad_norm_per_pos(model, x_clean, gamma).cpu()
        gn_i = _grad_norm_per_pos(model, x_invalid, gamma).cpu()
        sg_c = _signed_energy(model, x_clean, gamma).cpu()
        sg_i = _signed_energy(model, x_invalid, gamma).cpu()

        se_c = torch.stack([it["SE_pos_clean"] for it in items])
        se_i = torch.stack([it["SE_pos_invalid"] for it in items])

        grad_norm_clean.append(gn_c)
        grad_norm_invalid.append(gn_i)
        signed_clean.append(sg_c)
        signed_invalid.append(sg_i)
        SE_clean.append(se_c.sum(dim=-1))
        SE_invalid.append(se_i.sum(dim=-1))
        SE_pos_clean.append(se_c)
        SE_pos_invalid.append(se_i)
        masks.append(mask)

    GN_clean = torch.cat(grad_norm_clean, dim=0)  # (n, L)
    GN_invalid = torch.cat(grad_norm_invalid, dim=0)  # (n, L)
    SGN_clean = torch.cat(signed_clean, dim=0)  # (n,)
    SGN_invalid = torch.cat(signed_invalid, dim=0)  # (n,)
    SE_seq_clean = torch.cat(SE_clean, dim=0)  # (n,)
    SE_seq_invalid = torch.cat(SE_invalid, dim=0)  # (n,)
    SE_pos_clean_t = torch.cat(SE_pos_clean, dim=0)  # (n, L)
    SE_pos_invalid_t = torch.cat(SE_pos_invalid, dim=0)  # (n, L)
    Mask = torch.cat(masks, dim=0)  # (n, L)

    # Sequence-level statistics.
    E_seq_clean = GN_clean.pow(2).sum(dim=-1)
    E_seq_invalid = GN_invalid.pow(2).sum(dim=-1)
    U_mean_clean = GN_clean.mean(dim=-1)
    U_mean_invalid = GN_invalid.mean(dim=-1)
    U_max_clean = GN_clean.max(dim=-1).values
    U_max_invalid = GN_invalid.max(dim=-1).values

    seq_auc = {
        "E_seq_grad_sq": _roc_auc(E_seq_clean, E_seq_invalid),
        "E_seq_signed":  _roc_auc(SGN_clean, SGN_invalid),
        "U_pos_mean":    _roc_auc(U_mean_clean, U_mean_invalid),
        "U_pos_max":     _roc_auc(U_max_clean, U_max_invalid),
        "SE_seq":        _roc_auc(SE_seq_clean, SE_seq_invalid),
    }

    # Token-level: per-position grad-norm at *corrupted* invalid positions
    # (positive class) vs the same positions in the clean sequence
    # (negative class). For SE, same convention.
    pos_clean = GN_clean[Mask]
    pos_inv = GN_invalid[Mask]
    se_pos_clean = SE_pos_clean_t[Mask]
    se_pos_inv = SE_pos_invalid_t[Mask]
    tok_auc = {
        "U_pos":  _roc_auc(pos_clean, pos_inv),
        "SE_pos": _roc_auc(se_pos_clean, se_pos_inv),
    }

    # Aggregate stats.
    def _agg(t: torch.Tensor) -> dict[str, float]:
        return {
            "mean": float(t.mean()), "std": float(t.std()) if t.numel() > 1 else 0.0,
            "median": float(t.median()),
        }

    summary = {
        "ckpt": str(ckpt_path),
        "cache": str(cache_path),
        "model_name": cfg.training.model_name,
        "auditor_gamma": gamma,
        "n_eval": n,
        "n_corrupt_positions": int(Mask.sum().item()),
        "seq_auc": seq_auc,
        "tok_auc": tok_auc,
        "stats": {
            "GN_seq_clean":  _agg(E_seq_clean),
            "GN_seq_invalid": _agg(E_seq_invalid),
            "U_mean_clean":  _agg(U_mean_clean),
            "U_mean_invalid": _agg(U_mean_invalid),
            "SE_seq_clean":  _agg(SE_seq_clean),
            "SE_seq_invalid": _agg(SE_seq_invalid),
            "Signed_clean":  _agg(SGN_clean),
            "Signed_invalid": _agg(SGN_invalid),
        },
    }
    return summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument(
        "--cache", default="data/wiki_cache_gpt2.pt", help="wiki cache path"
    )
    p.add_argument("--n", type=int, default=None, dest="n_eval")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args(argv)

    summary = evaluate_auditor(
        args.ckpt,
        args.cache,
        n_eval=args.n_eval,
        batch_size=args.batch_size,
    )

    print(
        f"ckpt={Path(args.ckpt).parent.name}/{Path(args.ckpt).name}  "
        f"model={summary['model_name']}  γ_aud={summary['auditor_gamma']}  "
        f"n={summary['n_eval']}  corrupt_pos={summary['n_corrupt_positions']}"
    )
    print("Seq AUROC:")
    for k, v in summary["seq_auc"].items():
        print(f"  {k:<16s}  AUC={v:.4f}")
    print("Tok AUROC:")
    for k, v in summary["tok_auc"].items():
        print(f"  {k:<16s}  AUC={v:.4f}")

    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
