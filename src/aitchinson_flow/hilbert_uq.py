"""Hilbert-FM UQ student for cached LM top-K distributions.

This module is the LLM-UQ counterpart to ``hilbert_fm/`` (the text8 toy).
Same path geometry (log-linear / Aitchison geodesic on the simplex), same
soft-Hilbert loss, but:

* The simplex lives over the LM's **top-K** next-token vocabulary at each
  position, not the full character set.
* The student is a small **context-conditioned MLP** (not a transformer):
  it takes the LM's last hidden state ``h_{i-1}``, the current
  ``log p_t`` over top-K, and time ``t``, and predicts ``log p̂_1``.
* Targets come from the LM itself — the cached top-K log-probs are the
  "soft teacher" — so we're distilling the LM's calibrated next-token
  uncertainty into a small student trained only on **clean** answer
  rows. UQ signals derived from the student's sampling trajectory are
  then evaluated on **all** rows for hallucination detection.

The four UQ signals match :mod:`hilbert_fm.uq` exactly:

  ``U_spread``    — Hilbert/projective spread of final ``log p̂_1``.
  ``U_traj``      — mean Hilbert distance of intermediate predictions to
                    the final one.  Self-consistency analog of paper
                    spilled energy ``ΔE = E^ℓ − E^m``.
  ``L_excess``    — total Hilbert path length minus geodesic.
  ``U_ensemble``  — mean pairwise Hilbert distance of M trajectories
                    that differ only in the random source.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Path utilities (top-K simplex; functionally identical to hilbert_fm.path
# but K is per-call so we don't import the text8 module).
# ---------------------------------------------------------------------------


def label_smoothed_log_onehot(idx: torch.Tensor, K: int, eps: float) -> torch.Tensor:
    """``idx``: ``(...,)`` long in ``[0, K)`` -> ``(..., K)`` log-prob."""
    p = torch.full((*idx.shape, K), eps / (K - 1), device=idx.device, dtype=torch.float32)
    p.scatter_(-1, idx.unsqueeze(-1), 1.0 - eps)
    return p.log()


def random_log_p0(shape, K: int, eps: float, device, generator=None) -> torch.Tensor:
    """Per-element random label-smoothed log-one-hot over ``K`` coordinates."""
    if generator is not None:
        idx = torch.randint(0, K, shape, device=device, generator=generator)
    else:
        idx = torch.randint(0, K, shape, device=device)
    return label_smoothed_log_onehot(idx, K, eps)


def log_pt(log_p0: torch.Tensor, log_p1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """``log_softmax((1-t) log p0 + t log p1)`` along the last dim."""
    while t.dim() < log_p0.dim():
        t = t.unsqueeze(-1)
    return F.log_softmax((1.0 - t) * log_p0 + t * log_p1, dim=-1)


def soft_hilbert(log_p_hat, log_p_target, tau: float = 0.3) -> torch.Tensor:
    r = log_p_hat - log_p_target
    return (tau * torch.logsumexp(r / tau, dim=-1)
            + tau * torch.logsumexp(-r / tau, dim=-1)).mean()


def advance(log_pt_now, log_p1_hat, t_now: float, t_next: float) -> torch.Tensor:
    a = (1.0 - t_next) / (1.0 - t_now)
    b = (t_next - t_now) / (1.0 - t_now)
    return F.log_softmax(a * log_pt_now + b * log_p1_hat, dim=-1)


def hilbert_distance(log_p, log_q) -> torch.Tensor:
    r = log_p - log_q
    return r.max(dim=-1).values - r.min(dim=-1).values


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=t.device, dtype=t.dtype) / max(half - 1, 1)
    )
    args = t[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[-1] < dim:
        emb = torch.cat(
            [emb, torch.zeros(emb.shape[0], dim - emb.shape[-1], device=t.device, dtype=t.dtype)],
            dim=-1,
        )
    return emb


# ---------------------------------------------------------------------------
# Student model
# ---------------------------------------------------------------------------


@dataclass
class StudentConfig:
    K: int = 32
    H: int = 768                  # GPT-2 hidden size
    d_model: int = 256
    n_layers: int = 4
    eps_smooth: float = 0.01
    tau: float = 0.3
    t_train_max: float = 0.99
    t_bias_power: float = 0.5
    source_kind: str = "random_token"


class HilbertUQStudent(nn.Module):
    """Context-conditioned x_1-prediction MLP on the top-K simplex.

    Inputs:
      log_pt: ``(B, K)`` current path state in log-probabilities.
      h:      ``(B, H)`` LM context (hidden state of the predecessor token).
      t:      ``(B,)``  in ``[0, 1]``.
    Output:
      log_p1_hat: ``(B, K)``.
    """

    def __init__(self, cfg: StudentConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.in_logp = nn.Linear(cfg.K, d)
        self.in_ctx = nn.Linear(cfg.H, d)
        self.t_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))

        layers = []
        for _ in range(cfg.n_layers):
            layers.append(nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, 4 * d),
                nn.GELU(),
                nn.Linear(4 * d, d),
            ))
        self.blocks = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.K)

    def forward(self, log_pt: torch.Tensor, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.in_logp(log_pt) + self.in_ctx(h)
        t_emb = self.t_mlp(sinusoidal_time_embedding(t, self.cfg.d_model))
        x = x + t_emb
        for blk in self.blocks:
            x = x + blk(x)
        x = self.norm(x)
        return F.log_softmax(self.head(x), dim=-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def make_log_p0(shape, kind: str, K: int, eps: float, device) -> torch.Tensor:
    if kind == "random_token":
        return random_log_p0(shape, K, eps, device)
    if kind == "uniform":
        return torch.full((*shape, K), -math.log(K), device=device, dtype=torch.float32)
    raise ValueError(f"unknown source kind {kind!r}")


def train_student(
    model: HilbertUQStudent,
    h_train: torch.Tensor,        # (N, H)
    logp_train: torch.Tensor,     # (N, K) target soft distribution (LM top-K, renormalised)
    *,
    n_steps: int = 5_000,
    batch_size: int = 256,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    log_every: int = 100,
    device=None,
):
    cfg = model.cfg
    device = device or next(model.parameters()).device
    h_train = h_train.to(device)
    logp_train = logp_train.to(device)
    N = h_train.shape[0]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    for step in range(1, n_steps + 1):
        idx = torch.randint(0, N, (batch_size,), device=device)
        h_b = h_train[idx]
        log_p1 = logp_train[idx]
        u = torch.rand(batch_size, device=device)
        t = u.pow(cfg.t_bias_power).clamp(min=1e-4, max=cfg.t_train_max)
        log_p0 = make_log_p0((batch_size,), cfg.source_kind, cfg.K, cfg.eps_smooth, device)
        lpt = log_pt(log_p0, log_p1, t)
        lph = model(lpt, h_b, t)
        loss = soft_hilbert(lph, log_p1, cfg.tau)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0:
            history.append(dict(step=step, loss=float(loss.item())))
            print(f"[student] step={step:6d} loss={loss.item():.4f}")
    return history


# ---------------------------------------------------------------------------
# UQ signals (mirror hilbert_fm.uq, applied per token then pooled per row)
# ---------------------------------------------------------------------------


@torch.no_grad()
def trajectory_signals(
    model: HilbertUQStudent,
    h: torch.Tensor,         # (B, H)
    log_p1_target: torch.Tensor | None,   # optional, only used to set t_start>0
    *,
    n_steps: int = 25,
    t_start: float = 0.0,
    device=None,
) -> dict:
    """Run one trajectory per element. Returns signals + final prediction."""
    cfg = model.cfg
    device = device or next(model.parameters()).device
    B = h.shape[0]
    if log_p1_target is None or t_start == 0.0:
        log_p_t = make_log_p0((B,), cfg.source_kind, cfg.K, cfg.eps_smooth, device)
    else:
        log_p0 = make_log_p0((B,), cfg.source_kind, cfg.K, cfg.eps_smooth, device)
        t0 = torch.full((B,), t_start, device=device)
        log_p_t = log_pt(log_p0, log_p1_target, t0)
    log_p_t_init = log_p_t.clone()

    ts = torch.linspace(t_start, cfg.t_train_max, n_steps + 1, device=device)
    p1_hats: list[torch.Tensor] = []
    L_path = torch.zeros(B, device=device)

    for k in range(n_steps):
        tn = float(ts[k])
        tx = float(ts[k + 1])
        tb = torch.full((B,), tn, device=device)
        lph = model(log_p_t, h, tb)
        p1_hats.append(lph)
        prev = log_p_t
        log_p_t = advance(log_p_t, lph, tn, tx)
        L_path = L_path + hilbert_distance(prev, log_p_t)

    final = p1_hats[-1]
    U_spread = final.max(dim=-1).values - final.min(dim=-1).values
    if len(p1_hats) > 1:
        diffs = torch.stack(
            [hilbert_distance(ph, final) for ph in p1_hats[:-1]], dim=0
        )
        U_traj = diffs.mean(dim=0)
    else:
        U_traj = torch.zeros(B, device=device)
    L_geodesic = hilbert_distance(log_p_t_init, log_p_t)
    L_excess = L_path - L_geodesic
    return dict(
        U_spread=U_spread, U_traj=U_traj, L_excess=L_excess, log_p1_final=final,
    )


@torch.no_grad()
def ensemble_disagreement(
    model: HilbertUQStudent,
    h: torch.Tensor,
    *,
    n_steps: int = 25,
    M: int = 4,
    device=None,
) -> torch.Tensor:
    finals = []
    for _ in range(M):
        out = trajectory_signals(model, h, log_p1_target=None,
                                 n_steps=n_steps, t_start=0.0, device=device)
        finals.append(out["log_p1_final"])
    sumd = torch.zeros(h.shape[0], device=h.device)
    n_pairs = 0
    for i in range(M):
        for j in range(i + 1, M):
            sumd = sumd + hilbert_distance(finals[i], finals[j])
            n_pairs += 1
    return sumd / max(n_pairs, 1)


# ---------------------------------------------------------------------------
# AUROC and pooling
# ---------------------------------------------------------------------------


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    s = scores.flatten().detach().cpu().double()
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


def pool_per_row(values: torch.Tensor, mask: torch.Tensor, *, method: str) -> torch.Tensor:
    """Pool a ``(N, T)`` per-token signal down to ``(N,)`` via ``mask`` (N, T)."""
    m = mask.float()
    if method == "mean":
        denom = m.sum(dim=-1).clamp(min=1)
        return (values * m).sum(dim=-1) / denom
    if method == "max":
        masked = values.masked_fill(~mask, float("-inf"))
        return masked.max(dim=-1).values
    if method == "last":
        idx = (m * torch.arange(m.shape[-1], device=values.device).float()).max(dim=-1).indices
        return values[torch.arange(values.shape[0]), idx]
    raise ValueError(f"unknown pool {method!r}")


__all__ = [
    "StudentConfig", "HilbertUQStudent",
    "label_smoothed_log_onehot", "log_pt", "soft_hilbert", "advance",
    "hilbert_distance", "make_log_p0", "random_log_p0",
    "train_student",
    "trajectory_signals", "ensemble_disagreement",
    "auroc", "pool_per_row",
]
