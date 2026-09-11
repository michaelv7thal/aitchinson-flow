"""Field-probe diagnostic for an EqMLatent checkpoint.

Tests the "FM converged to a degenerate field" hypothesis by measuring the
model's velocity field f(x) and conservative gradient ∇⟨x, f(x)⟩ at three
qualitatively different input types:

  1. Pure noise   x_0 ~ N(0, σ²I)
  2. Data points  x_1 = embed(token_ids)
  3. Mid-points   x_γ = (1−γ)·x_0 + γ·x_1  for γ ∈ {0.25, 0.5, 0.75}

For each, we report:
  * ‖f(x)‖  — average L2 magnitude of the raw velocity output (per position)
  * ‖∇E‖    — average L2 magnitude of the conservative gradient (per position)
  * cos(f, x_0 − x_1)  — direction alignment with the FM target (this should be
    ≈ +1 if the model has learned the right field)
  * cos(f, centroid − x) — does the field point toward the centroid (the trap)?
  * Direction-collapse score — fraction of variance of f(x) explained by its
    first principal direction (a degenerate field outputs ~the same direction
    everywhere, so this score → 1 means collapse)

Also reports the embedding centroid and statistics for context.

Usage:
    python scripts/field_probe.py --ckpt runs/.../epoch_final.pt
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

import aitchinson_flow.models  # noqa: E402,F401
from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.models import build_model  # noqa: E402
from aitchinson_flow.training import build_training_datamodule  # noqa: E402
from scripts.eval_full import _config_from_payload  # noqa: E402


def _conservative_grad(model, x, gamma):
    with torch.enable_grad():
        x_req = x.detach().requires_grad_(True)
        v = model.forward(x_req, gamma)
        energy = (x_req * v).sum()
        g = torch.autograd.grad(energy, x_req, create_graph=False)[0].detach()
    return v.detach(), g


def _direction_collapse(v: torch.Tensor) -> float:
    """Fraction of variance of v explained by its top principal component.

    v is shape (..., d). We flatten leading dims, mean-center, do SVD, and
    return s[0]² / Σ s[i]². 1.0 = perfectly directionally collapsed.
    """
    flat = v.reshape(-1, v.shape[-1])
    flat = flat - flat.mean(dim=0, keepdim=True)
    if flat.shape[0] < 2 or flat.shape[1] < 1:
        return float("nan")
    s = torch.linalg.svdvals(flat)
    s2 = s.pow(2)
    return float(s2[0] / s2.sum())


def _safe_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a_n = a.norm(dim=-1).clamp(min=1e-9)
    b_n = b.norm(dim=-1).clamp(min=1e-9)
    return float(((a * b).sum(dim=-1) / (a_n * b_n)).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=64, help="batch size for probe")
    ap.add_argument("--gammas", type=str, default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = _config_from_payload(payload)
    device = cfg.training.device
    model = build_model(cfg).to(device)
    state = payload["model_state_dict"] if "model_state_dict" in payload else payload
    model.load_state_dict(state)
    model.eval()

    dm, _ = build_training_datamodule(cfg)
    val_ids = dm.splits.val.long().to(device)[: args.n]
    K = cfg.text8_dataset.K
    L = cfg.text8_dataset.L
    sigma = cfg.eqm.source_sigma
    embed = model.embed.weight.detach().to(device)
    d = embed.shape[1]
    centroid = embed.mean(dim=0)
    print(f"K={K}  L={L}  d={d}  σ={sigma}  "
          f"embed.norm={embed.norm(dim=-1).mean().item():.3f}  "
          f"centroid.norm={centroid.norm().item():.3f}  "
          f"pairwise_min={torch.cdist(embed, embed).fill_diagonal_(float('inf')).min().item():.3f}")

    time_cond = getattr(cfg.eqm, "time_conditioning", "off")
    print(f"time_conditioning={time_cond!r}")

    x0 = sigma * torch.randn(args.n, L, d, device=device)
    x1 = embed[val_ids]

    rows: list[dict] = []
    print()
    print(f"{'γ':>5} {'||f(x)||':>10} {'||∇E||':>10} "
          f"{'cos(∇E, x0−x1)':>15} {'cos(f, c−x)':>13} {'col_score':>10}")
    print("-" * 80)
    for g_str in args.gammas.split(","):
        g = float(g_str.strip())
        if not g_str.strip():
            continue
        x_g = (1.0 - g) * x0 + g * x1
        gamma_b = (
            torch.full((args.n,), g, device=device, dtype=x_g.dtype)
            if time_cond != "off" else None
        )
        f_val, grad_E = _conservative_grad(model, x_g, gamma_b)
        f_norm = f_val.norm(dim=-1).mean().item()
        grad_norm = grad_E.norm(dim=-1).mean().item()
        # The training target's direction is (x0 − x1); cos(∇E, x0−x1) should
        # be ≈ +1 for a well-trained field at γ < 1. (Sign because ∇E ≈
        # c(γ)·(x0−x1) with c≥0.)
        cos_target = _safe_cos(grad_E, x0 - x1)
        # Centroid pull: cos(f, centroid − x). +1 = field pushes toward
        # centroid (trap); −1 = away.
        cos_centroid = _safe_cos(f_val, centroid[None, None, :] - x_g)
        col_score = _direction_collapse(f_val)
        print(f"{g:>5.2f} {f_norm:>10.4f} {grad_norm:>10.4f} "
              f"{cos_target:>15.4f} {cos_centroid:>13.4f} {col_score:>10.4f}")
        rows.append({
            "gamma": g, "f_norm_mean": f_norm, "grad_E_norm_mean": grad_norm,
            "cos_target": cos_target, "cos_centroid": cos_centroid,
            "direction_collapse_score": col_score,
        })

    # Where does sampling actually converge?
    print("\n=== Where does the model actually sample to? ===")
    with torch.no_grad():
        x_gen = model.sample(args.n, L, max_steps=200)
    x_gen_flat = x_gen.reshape(-1, d)
    print(f"  generated point norm_mean: {x_gen_flat.norm(dim=-1).mean().item():.4f}")
    print(f"  centroid norm: {centroid.norm().item():.4f}")
    print(f"  distance(generated, centroid): "
          f"{(x_gen_flat - centroid).norm(dim=-1).mean().item():.4f}")
    # Distance to nearest embedding row.
    d_to_embed = torch.cdist(x_gen_flat, embed).min(dim=-1).values
    print(f"  distance(generated, nearest embed row): "
          f"mean={d_to_embed.mean().item():.4f}, "
          f"min={d_to_embed.min().item():.4f}")
    # Decode and report which tokens dominate.
    log_probs = model.decode_to_logprobs(x_gen)
    ids = log_probs.argmax(-1).cpu()
    top_tokens, counts = torch.unique(ids, return_counts=True)
    sorted_idx = counts.argsort(descending=True)
    top_tokens = top_tokens[sorted_idx][:5]
    counts = counts[sorted_idx][:5]
    total = ids.numel()
    print(f"  top-5 sampled tokens (out of {K}):")
    from aitchinson_flow.data.char_window_dataset import CHAR2ID
    alphabet = "".join(sorted(CHAR2ID, key=CHAR2ID.__getitem__))
    for tok, c in zip(top_tokens.tolist(), counts.tolist()):
        print(f"    {alphabet[tok]!r}: {c} ({100*c/total:.1f}%)")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"rows": rows, "K": K, "L": L, "d": d}, indent=2))
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
