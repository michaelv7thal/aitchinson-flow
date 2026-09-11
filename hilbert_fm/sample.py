"""Iterative re-prediction sampling along the log-linear path.

Run with::

    python -m hilbert_fm.sample <ckpt.pt>
"""
from __future__ import annotations

import sys

import torch

from .data import decode
from .model import HilbertFMTransformer
from .path import advance, make_log_p0


def load_model(ckpt_path: str, device: str | torch.device = "cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model = HilbertFMTransformer(
        K=cfg["K"], L=cfg["L"],
        d_model=cfg["d_model"], n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


@torch.no_grad()
def sample(
    model: HilbertFMTransformer,
    B: int,
    L: int,
    K: int,
    n_steps: int = 50,
    t_max: float = 0.99,
    device: str | torch.device = "cpu",
    return_trajectory: bool = False,
    source_kind: str = "random_token",
    eps_smooth: float = 0.01,
):
    """Walk from ``t=0`` to ``t=t_max`` and decode argmax tokens.

    The source distribution must match what the model was trained on; pass
    ``source_kind`` and ``eps_smooth`` from the saved config.

    Returns ``(B, L)`` token ids; if ``return_trajectory`` also returns the list
    of ``log p_t`` snapshots (length ``n_steps + 1``).
    """
    log_p_t = make_log_p0(source_kind, B, L, K, eps_smooth, device)
    ts = torch.linspace(0.0, t_max, n_steps + 1)
    traj = [log_p_t.clone()] if return_trajectory else None

    for k in range(n_steps):
        t_now = float(ts[k])
        t_next = float(ts[k + 1])
        t_b = torch.full((B,), t_now, device=device)
        log_p1_hat = model(log_p_t, t_b)
        log_p_t = advance(log_p_t, log_p1_hat, t_now, t_next)
        if return_trajectory:
            traj.append(log_p_t.clone())

    ids = log_p_t.argmax(dim=-1)
    return (ids, traj) if return_trajectory else ids


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "hilbert_fm/runs/default/final.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(ckpt, device)
    ids = sample(
        model, B=5, L=cfg["L"], K=cfg["K"], device=device,
        source_kind=cfg.get("source_kind", "uniform"),
        eps_smooth=cfg["eps_smooth"],
    )
    for i, row in enumerate(ids):
        print(f"{i}: {decode(row)!r}")
