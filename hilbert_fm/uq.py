"""Four uncertainty signals derived from a Hilbert FM model.

All four operate on the **reconstruction-from-`t_start`** setup: build the path
target from real tokens, walk to `t_start`, then re-sample to `t_max`, recording
trajectory quantities along the way. This gives us a labelled "is the prediction
correct" target per position, which we don't have when sampling from pure noise.

Signals (all returned as ``(B, L)`` per-position tensors unless noted):
  * ``U_spread``   — final ``max log p̂_1 − min log p̂_1``. Hilbert/projective
                     analog of entropy. **High = peaked = confident**, so this
                     is the only signal that's anti-correlated with error.
  * ``U_traj``     — mean Hilbert distance from intermediate ``p̂_1^(k)`` to the
                     final ``p̂_1^(K)``. Self-consistency analog of spilled
                     energy: high = the model kept changing its mind.
  * ``L_excess``   — total Hilbert path length minus the straight-line
                     geodesic from start to finish. High = curvy trajectory.
  * ``U_ensemble`` — mean pairwise Hilbert distance between ``M`` independent
                     trajectories' final predictions, varying only the random
                     source. High = predictions disagree across seeds.
"""
from __future__ import annotations

import torch

from .path import advance, hilbert_distance, log_p1_from_ids, log_pt, make_log_p0


@torch.no_grad()
def reconstruction_trajectory(
    model,
    batch_ids: torch.Tensor,
    cfg: dict,
    device: str | torch.device,
    t_start: float = 0.5,
    n_steps: int = 25,
) -> dict:
    """Run one reconstruction trajectory and compute three signals + final pred.

    Returns dict with keys:
      ``U_spread`` (B, L), ``U_traj`` (B, L), ``L_excess`` (B, L),
      ``log_p1_final`` (B, L, K), ``pred`` (B, L), ``correct`` (B, L).
    """
    model.eval()
    K_ = cfg["K"]
    L = cfg["L"]
    B = batch_ids.shape[0]
    eps_smooth = cfg["eps_smooth"]
    t_max = cfg.get("t_train_max", 0.99)
    source_kind = cfg.get("source_kind", "uniform")

    log_p1 = log_p1_from_ids(batch_ids, K_, eps_smooth)
    log_p0 = make_log_p0(source_kind, B, L, K_, eps_smooth, device)
    t0 = torch.full((B,), t_start, device=device)
    log_p_t = log_pt(log_p0, log_p1, t0)
    log_p_t_init = log_p_t.clone()

    ts = torch.linspace(t_start, t_max, n_steps + 1)
    p1_hats: list[torch.Tensor] = []
    L_path = torch.zeros(B, L, device=device)

    for k in range(n_steps):
        tn = float(ts[k])
        tx = float(ts[k + 1])
        tb = torch.full((B,), tn, device=device)
        log_p1_hat = model(log_p_t, tb)
        p1_hats.append(log_p1_hat)
        prev = log_p_t
        log_p_t = advance(log_p_t, log_p1_hat, tn, tx)
        L_path = L_path + hilbert_distance(prev, log_p_t)

    final = p1_hats[-1]
    U_spread = final.max(dim=-1).values - final.min(dim=-1).values

    if len(p1_hats) > 1:
        diffs = torch.stack(
            [hilbert_distance(ph, final) for ph in p1_hats[:-1]], dim=0
        )
        U_traj = diffs.mean(dim=0)
    else:
        U_traj = torch.zeros(B, L, device=device)

    L_geodesic = hilbert_distance(log_p_t_init, log_p_t)
    L_excess = L_path - L_geodesic

    pred = final.argmax(dim=-1)
    correct = (pred == batch_ids).float()

    return dict(
        U_spread=U_spread,
        U_traj=U_traj,
        L_excess=L_excess,
        log_p1_final=final,
        pred=pred,
        correct=correct,
    )


@torch.no_grad()
def ensemble_disagreement(
    model,
    batch_ids: torch.Tensor,
    cfg: dict,
    device: str | torch.device,
    t_start: float = 0.5,
    n_steps: int = 25,
    M: int = 4,
) -> torch.Tensor:
    """Run ``M`` reconstruction trajectories with independent random sources.

    Returns ``(B, L)`` mean pairwise Hilbert distance of the final ``log p̂_1``.
    """
    finals: list[torch.Tensor] = []
    for _ in range(M):
        out = reconstruction_trajectory(model, batch_ids, cfg, device, t_start, n_steps)
        finals.append(out["log_p1_final"])

    n_pairs = 0
    sumd = torch.zeros(batch_ids.shape, device=device, dtype=torch.float32)
    for i in range(M):
        for j in range(i + 1, M):
            sumd = sumd + hilbert_distance(finals[i], finals[j])
            n_pairs += 1
    return sumd / max(n_pairs, 1)


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """AUROC where higher ``scores`` should predict ``labels == 1``.

    Implemented from the Mann-Whitney U statistic so we don't need sklearn.
    Returns ``float('nan')`` if the labels are degenerate (all 0 or all 1).
    """
    s = scores.flatten().detach().cpu()
    y = labels.flatten().detach().cpu().long()
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(s)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(s) + 1, dtype=torch.float64)
    rank_pos_sum = float(ranks[y == 1].sum())
    return (rank_pos_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
