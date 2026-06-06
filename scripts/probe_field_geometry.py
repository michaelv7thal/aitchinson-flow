"""E2d — field-geometry probe (curl fraction + cos(g, g*) vs γ) in CLR space.

For a collapsing arm (EqM / FMonCLR) the learned vector field is supposed to be
the conservative gradient of a scalar energy and to match the analytic FM target
``g* = c(γ)·(x0 − x1)``. This probe quantifies two geometric defects across the
interpolation coordinate γ ∈ {0.1, 0.25, 0.5, 0.75, 1.0}:

  (a) **curl fraction** — ‖antisym(J)‖²_F / ‖J‖²_F of the field Jacobian
      J = ∂g/∂x, estimated by central finite differences on a random subset of
      flattened coordinates (no second-order autograd needed). A conservative
      (gradient) field has curl_frac ≈ 0; a large fraction means the learned
      field is *not* the gradient of any scalar.
  (b) **cos(g_θ, g*)** — cosine alignment of the field with the analytic FM
      target direction (x0 − x1). Low cos near the data basin (γ→1) is the
      flat-field / wrong-geometry signature.

Operates purely in CLR (R^K per token); it does NOT touch ``model.embed`` —
that is the EqMLatent-only path handled by the separate ``scripts/field_probe.py``
(left untouched). Targets the simplex arms EqM / EqM_OneHot / FMonCLR, which
expose a conservative-gradient field (``_compute_grad``) or a velocity
(``forward``).

Usage:
  python scripts/probe_field_geometry.py --ckpt runs/<eqm>/epoch_final.pt --n 64
  python scripts/probe_field_geometry.py --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.append(str(_ROOT))

import torch  # noqa: E402

import aitchinson_flow.models  # noqa: E402,F401  populate REGISTRY
from aitchinson_flow.models.factory import build_model  # noqa: E402
from aitchinson_flow.data.transforms import token_ids_to_features  # noqa: E402

GAMMAS = (0.1, 0.25, 0.5, 0.75, 1.0)


def _field(model, x: torch.Tensor) -> torch.Tensor:
    """The model's vector field at x (B,L,K). EqM-family → conservative
    gradient ∇⟨x,f⟩; FMonCLR → the velocity head output."""
    if hasattr(model, "_compute_grad"):
        return model._compute_grad(x).detach()
    # FMonCLR / plain velocity field (time-conditioning off by default → γ=None)
    with torch.no_grad():
        return model.forward(x).detach()


@torch.no_grad()
def _curl_fraction(model, x: torch.Tensor, *, m_coords: int, eps: float,
                   seed: int) -> float:
    """Monte-Carlo curl fraction of J=∂g/∂x on a random coordinate subset S.

    Perturbs each coord c∈S by ±eps (central diff) to read column J[:,c] for the
    whole batch in one field eval; then curl_frac is the antisymmetric fraction
    of the |S|×|S| principal submatrix, averaged over the batch. 2|S| field
    evals total. Returns a scalar in [0, 1]."""
    B, L, K = x.shape
    D = L * K
    g = torch.Generator().manual_seed(seed)
    m = min(m_coords, D)
    coords = torch.randperm(D, generator=g)[:m]
    flat = x.reshape(B, D)
    # columns[k] = J[:, :, coords[k]]  shape (B, D)
    cols = torch.empty(B, m, D)
    for k, c in enumerate(coords.tolist()):
        xp = flat.clone()
        xp[:, c] += eps
        xm = flat.clone()
        xm[:, c] -= eps
        gp = _field(model, xp.reshape(B, L, K)).reshape(B, D)
        gm = _field(model, xm.reshape(B, L, K)).reshape(B, D)
        cols[:, k, :] = (gp - gm) / (2.0 * eps)
    # submatrix J_sub[b, i, j] = ∂g_{coords[i]} / ∂x_{coords[j]} = cols[b, j, coords[i]]
    idx = coords
    Jsub = cols[:, :, idx].transpose(1, 2)  # (B, m_i, m_j)
    A = 0.5 * (Jsub - Jsub.transpose(1, 2))  # antisymmetric part
    num = (A ** 2).sum(dim=(1, 2))
    den = (Jsub ** 2).sum(dim=(1, 2)).clamp(min=1e-12)
    return float((num / den).mean())


def probe(model, *, K: int, L: int, n: int, source_sigma: float,
          m_coords: int, eps: float, seed: int) -> dict:
    torch.manual_seed(seed)
    # x1: CLR features of random tokens (data-manifold endpoint); x0: zero-mean
    # Gaussian noise source (the EqM source). Both centred (V_d).
    ids = torch.randint(0, K, (n, L))
    x1 = token_ids_to_features(ids, K)
    x1 = x1 - x1.mean(dim=-1, keepdim=True)
    x0 = source_sigma * torch.randn(n, L, K)
    x0 = x0 - x0.mean(dim=-1, keepdim=True)
    out: dict[str, dict] = {}
    for gamma in GAMMAS:
        x_g = (1.0 - gamma) * x0 + gamma * x1
        g = _field(model, x_g)
        tgt = x0 - x1  # FM target direction (scale-free); u_tgt = c(γ)·(x0−x1)
        cos = torch.nn.functional.cosine_similarity(
            g.reshape(n, -1), tgt.reshape(n, -1), dim=-1
        ).mean()
        curl = _curl_fraction(model, x_g, m_coords=m_coords, eps=eps, seed=seed)
        out[f"{gamma:.2f}"] = {"curl_frac": curl, "cos_g_gstar": float(cos)}
    return out


def _load(ckpt: str):
    from scripts.eval_full import _config_from_payload
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = _config_from_payload(payload)
    model = build_model(cfg)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, cfg


def _smoke() -> None:
    from dataclasses import replace
    from aitchinson_flow.config import Config
    cfg = Config()
    cfg.training = replace(cfg.training, model_name="EqM", L=12, K=27)
    cfg.transformer = replace(cfg.transformer, d_model=32, nhead=2, num_layers=2)
    cfg.text8_dataset = replace(cfg.text8_dataset, L=12, K=27)
    model = build_model(cfg).eval()
    res = probe(model, K=27, L=12, n=4, source_sigma=cfg.eqm.source_sigma,
                m_coords=16, eps=1e-3, seed=0)
    assert set(res) == {f"{g:.2f}" for g in GAMMAS}, res.keys()
    for gv in res.values():
        assert 0.0 <= gv["curl_frac"] <= 1.0001, gv
        assert -1.0001 <= gv["cos_g_gstar"] <= 1.0001, gv
    print("OK probe_field_geometry smoke:",
          {k: (round(v["curl_frac"], 3), round(v["cos_g_gstar"], 3))
           for k, v in res.items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--m-coords", type=int, default=24,
                    help="random coords for the curl-Jacobian estimate")
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        _smoke()
        return
    if not args.ckpt:
        ap.error("--ckpt is required unless --smoke")
    model, cfg = _load(args.ckpt)
    res = probe(model, K=cfg.text8_dataset.K, L=cfg.text8_dataset.L, n=args.n,
                source_sigma=cfg.eqm.source_sigma, m_coords=args.m_coords,
                eps=args.eps, seed=args.seed)
    rec = {"ckpt": args.ckpt, "model_name": cfg.training.model_name, "gammas": res}
    out = args.out or str(Path(args.ckpt).parent / "field_geometry.json")
    Path(out).write_text(json.dumps(rec, indent=2))
    print(f"wrote {out}")
    for g, v in res.items():
        print(f"  γ={g}  curl_frac={v['curl_frac']:.3f}  cos(g,g*)={v['cos_g_gstar']:+.3f}")


if __name__ == "__main__":
    main()
