"""Helper used by ``scripts/cascade_audit.py`` to score the trained EqM
auditor for the Phase U cascade audit.

Lives in its own file so ``cascade_audit.py`` doesn't pull in the full
EqM build path when it's not asked for the auditor column.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

import aitchinson_flow.models  # noqa: E402,F401
from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.models import build_model  # noqa: E402

from scripts.eval_full import _config_from_payload  # noqa: E402


@torch.no_grad()
def _per_position_grad_sq(
    model: torch.nn.Module,
    x: torch.Tensor,
    h_ctx: torch.Tensor | None,
    *,
    gamma_value: float,
    time_conditioned: bool,
) -> torch.Tensor:
    """Per-position ``Σ_k (∇⟨x,f⟩)_k²`` — auditor's per-position score."""
    B, L = x.shape[:2]
    out = torch.zeros(B, L, device=x.device)
    # Process one sequence at a time to keep peak memory bounded — the
    # second-order autograd path is the memory hot-spot and the wiki cache
    # has small enough B that this is fine.
    for i in range(B):
        xi = x[i:i+1].clone().requires_grad_(True)
        hi = h_ctx[i:i+1] if h_ctx is not None else None
        if time_conditioned:
            gamma = torch.full((1,), float(gamma_value), device=x.device)
        else:
            gamma = None
        with torch.enable_grad():
            v = model(xi, gamma, hi)
            energy = (xi * v).sum()
            grad = torch.autograd.grad(energy, xi, create_graph=False)[0]
        out[i] = grad.pow(2).sum(dim=-1).squeeze(0)
    return out


def score_eqm_auditor(
    ckpt: str | Path, cache: dict[str, Any], device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the auditor checkpoint and emit per-position grad-norm² scores
    on the wiki cache's clean/invalid CLR features.

    Higher ``Σ‖∇⟨x,f⟩‖²`` → more "invalid". The hinge target is grad-norm²
    (margin² on invalid, ≈0 on clean), so this is the natural readout.
    """
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = _config_from_payload(payload)

    model = build_model(cfg).to(device)
    state = payload["model_state_dict"] if "model_state_dict" in payload else payload
    model.load_state_dict(state)
    model.eval()

    time_cond = getattr(cfg.eqm, "time_conditioning", "off") != "off"
    gamma_value = float(getattr(cfg.eqm, "auditor_gamma", 1.0))

    x_clean = cache["clean_clr"].float().to(device)
    x_invalid = cache["invalid_clr"].float().to(device)
    h_clean = cache.get("clean_h").float().to(device) if "clean_h" in cache else None
    h_invalid = cache.get("invalid_h").float().to(device) if "invalid_h" in cache else None

    s_clean = _per_position_grad_sq(
        model, x_clean, h_clean, gamma_value=gamma_value, time_conditioned=time_cond
    ).cpu()
    s_invalid = _per_position_grad_sq(
        model, x_invalid, h_invalid, gamma_value=gamma_value, time_conditioned=time_cond
    ).cpu()
    return s_clean, s_invalid
