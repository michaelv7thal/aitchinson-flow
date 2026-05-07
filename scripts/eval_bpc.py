"""Bits-per-character on text8 test split (Phase E — TRAINING_PROTOCOL.md).

For each model family we compute a model-specific NLL upper bound and
divide by ``log(2)``. Discrete- and continuous-output models use
different surrogates (documented below); we report all four numbers in
one table for the writeup, with explicit honesty notes per row.

* **DFM** — Discrete ELBO. ``E_t[-log p_{1|t}(x | x_t)] / log 2``. Lou et al.
  SEDD-style upper bound; reproduces directly via ``DiscreteFlowMatching.bpd``.
* **LogitKLFlow** — Clean-logit denoiser CE. At random t draw
  ``l_t = (1−t)·l_0 + t·γ_l·onehot(ids)`` and report
  ``E_t[-log softmax(v̂(l_t,t))[token_i]]/log 2``. Reduces to log-loss of
  the predicted clean-logit softmax under the data tokens. Comparable to
  DFM in form (NLL on a t-conditional denoiser) and is a strict upper
  bound on the model's marginal data NLL.
* **EqM** — Implied-x1 CE on the conservative-gradient reconstruction.
  ``E_γ[-log softmax(x_γ − λ·grad_g)[token_i]] / log 2`` averaged over γ.
  Matches the training aux-CE term and is a meaningful surrogate, but
  **not** directly comparable to AR-LM BPC (the EqM energy is not a
  normalised density). Disclose explicitly.
* **FMonCLR** — Same surrogate as EqM but with raw velocity (``pred_x1 =
  x_γ − λ·v``). Same caveats apply.

The numbers land in ``runs/<name>/bpc.json`` and are also printed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import aitchinson_flow.models  # noqa: E402,F401 — populate registry
from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.data.transforms import token_ids_to_features  # noqa: E402
from aitchinson_flow.models import build_model  # noqa: E402
from aitchinson_flow.training import build_training_datamodule  # noqa: E402

from scripts.eval_full import _config_from_payload  # noqa: E402


@torch.no_grad()
def _bpc_dfm(model: Any, ids: torch.Tensor, *, n_mc: int = 8) -> float:
    """DFM ELBO via the existing model.bpd() — already returns bits/char."""
    return float(model.bpd(ids, n_mc=n_mc).item())


@torch.no_grad()
def _bpc_logitkl(model: Any, ids: torch.Tensor, *, n_mc: int = 8) -> float:
    """Clean-logit denoiser CE in bits/char."""
    K = model.cfg.text8_dataset.K
    gamma_l = model.cfg.logitkl.gamma_l
    sigma = model.cfg.logitkl.source_sigma
    B, L = ids.shape
    device = ids.device

    l_1 = gamma_l * F.one_hot(ids.long(), K).to(
        dtype=next(model.parameters()).dtype
    )
    total_nll = 0.0
    for _ in range(n_mc):
        l_0 = sigma * torch.randn_like(l_1)
        t = torch.rand(B, device=device, dtype=l_1.dtype)
        l_t = (1.0 - t)[:, None, None] * l_0 + t[:, None, None] * l_1
        v_hat = model.forward(l_t, t)
        log_p = v_hat.log_softmax(dim=-1)
        nll = -log_p.gather(-1, ids.unsqueeze(-1)).squeeze(-1).mean()
        total_nll += float(nll.item())
    return (total_nll / n_mc) / math.log(2)


@torch.no_grad()
def _bpc_eqm_like(
    model: Any, ids: torch.Tensor, *, K: int, label_smoothing: float, n_mc: int = 8
) -> float:
    """EqM/FMonCLR surrogate: implied-x1 CE in bits/char, averaged over γ.

    EqM uses the conservative gradient ``grad_g = ∇⟨x,f⟩`` to define
    ``pred_x1 = x_γ − λ·grad_g``; FMonCLR uses raw velocity ``v``. We
    select the right path automatically: if the model exposes
    ``position_uncertainty`` (EqM-only), we re-enable autograd to compute
    ``grad_g``; else we treat the velocity output directly.
    """
    s = model.cfg.eqm
    sigma = s.source_sigma
    gamma_power = s.gamma_power
    lam = s.gradient_lambda
    B, L = ids.shape
    device = ids.device
    dtype = next(model.parameters()).dtype

    x1 = token_ids_to_features(ids, K, label_smoothing=label_smoothing)
    is_eqm = hasattr(model, "position_uncertainty")

    total_nll = 0.0
    for _ in range(n_mc):
        x0 = sigma * torch.randn(B, L, K, device=device, dtype=dtype)
        x0 = x0 - x0.mean(dim=-1, keepdim=True)
        gamma = torch.rand(B, device=device, dtype=dtype).pow(gamma_power)
        x_g = (1.0 - gamma[:, None, None]) * x0 + gamma[:, None, None] * x1

        if is_eqm:
            with torch.enable_grad():
                x_req = x_g.detach().requires_grad_(True)
                v = model.forward(x_req, gamma)
                energy = (x_req * v).sum()
                grad_g = torch.autograd.grad(energy, x_req)[0]
            pred_x1 = (x_g - lam * grad_g).detach()
        else:
            v = model.forward(x_g, gamma)
            pred_x1 = x_g - lam * v

        log_p = pred_x1 - torch.logsumexp(pred_x1, dim=-1, keepdim=True)
        nll = -log_p.gather(-1, ids.unsqueeze(-1)).squeeze(-1).mean()
        total_nll += float(nll.item())
    return (total_nll / n_mc) / math.log(2)


def evaluate_bpc(
    ckpt_path: str | Path,
    *,
    n_chunks: int = 256,
    n_mc: int = 8,
    seed: int = 1234,
) -> dict[str, Any]:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = _config_from_payload(payload)
    device = cfg.training.device
    K = cfg.text8_dataset.K
    label_smoothing = cfg.transformation.label_smoothing
    model_name = cfg.training.model_name

    model = build_model(cfg).to(device)
    state = payload["model_state_dict"] if "model_state_dict" in payload else payload
    model.load_state_dict(state)
    model.eval()

    dm, _ = build_training_datamodule(cfg)
    test_ids = dm.splits.test.long()
    rng = torch.Generator().manual_seed(seed)
    pick = torch.randperm(test_ids.shape[0], generator=rng)[:n_chunks]
    ids = test_ids[pick].to(device)

    if model_name == "DFM":
        bpc = _bpc_dfm(model, ids, n_mc=n_mc)
        method = "discrete_elbo"
    elif model_name == "LogitKLFlow":
        bpc = _bpc_logitkl(model, ids, n_mc=n_mc)
        method = "clean_logit_ce"
    elif model_name in ("EqM", "FMonCLR"):
        bpc = _bpc_eqm_like(
            model, ids, K=K, label_smoothing=label_smoothing, n_mc=n_mc
        )
        method = "implied_x1_ce_surrogate"
    else:
        raise RuntimeError(f"BPC not defined for model {model_name!r}")

    return {
        "ckpt": str(ckpt_path),
        "model_name": model_name,
        "method": method,
        "bpc": bpc,
        "n_chunks": n_chunks,
        "n_mc": n_mc,
        "L": int(ids.shape[1]),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n", type=int, default=256, dest="n_chunks")
    p.add_argument("--n-mc", type=int, default=8)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args(argv)

    result = evaluate_bpc(
        args.ckpt, n_chunks=args.n_chunks, n_mc=args.n_mc, seed=args.seed
    )
    print(
        f"ckpt={Path(args.ckpt).parent.name}/{Path(args.ckpt).name}  "
        f"model={result['model_name']:<12s}  method={result['method']:<24s}  "
        f"BPC={result['bpc']:.4f}  (n_chunks={result['n_chunks']}, n_mc={result['n_mc']})"
    )
    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
