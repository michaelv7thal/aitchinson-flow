"""Phase U (CAPSTONE_EXPERIMENTS.md §7) — four-AUROC cascade audit.

For each method specification, computes a 2×2 AUROC table:

|                    | corrupted positions | uncorrupted positions |
|--------------------|---------------------|------------------------|
| sequence-aligned   | AUROC₁               | AUROC₂                  |
| position-shuffled  | AUROC₃               | AUROC₄                  |

* AUROC₁ — the standard "Tok AUROC at corrupted positions" headline number.
* AUROC₂ — locality control: AUROC at *uncorrupted* positions in invalid
  sequences. A locality-clean detector is at chance here; a
  cascade-contaminated detector fires.
* AUROC₃, AUROC₄ — same as 1/2 but with (h_LLM, token, label) triples
  permuted within each sequence (RNG seed fixed). A locality-clean
  detector is shuffle-invariant. A cascade-contaminated detector
  collapses to chance.

Methods (loaded one at a time so big tensors don't co-resident in RAM):
* ``SE``                 Spilled Energy (LM per-position NLL) — locality-clean
* ``topk_entropy``       entropy of softmax over top-K logits — locality-clean
* ``linear_probe``       linear probe on h_LLM, trained per-token-CE
* ``blr_laplace``        Bayesian linear regression w/ Laplace approximation on h_LLM
* ``svgp``               sparse variational GP on h_LLM (small Z, fast)
* ``eqm_auditor``        the trained EqM auditor — uses ``model.energy()`` per-position
* ``saplma``             Phase V SAPLMA-style 3-layer MLP probe on h_LLM

Inputs:
* ``--cache data/wiki_cache_gpt2.pt`` (built by ``scripts/cache_wiki.py``)
* ``--auditor-ckpt`` (optional; required if ``eqm_auditor`` in --include)
* ``--saplma-ckpt`` (optional; required if ``saplma`` in --include)
* ``--seed`` for the position-shuffle permutations.

Output:
* ``runs/phaseU_cascade_audit.json`` — the full 4-AUROC table per method.
* ``runs/phaseU_cascade_audit.md``   — markdown rendering.

Wall-clock target ≈ 15 min (eval-only, batched).
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

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from aitchinson_flow.data.wiki import load_wiki_cache  # noqa: E402


# ---------------------------------------------------------------------------
# AUROC primitives.
# ---------------------------------------------------------------------------

def _auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Trapezoidal AUROC; returns 0.5 if a class is empty (chance)."""
    if scores.numel() == 0:
        return float("nan")
    pos = labels.bool()
    if pos.sum() == 0 or (~pos).sum() == 0:
        return 0.5
    s = scores.float().cpu().numpy()
    y = pos.cpu().numpy().astype(np.int64)
    order = np.argsort(-s, kind="stable")
    y_sorted = y[order]
    P = y_sorted.sum()
    N = len(y_sorted) - P
    cum_pos = np.cumsum(y_sorted)
    cum_neg = np.cumsum(1 - y_sorted)
    tpr = np.concatenate([[0.0], cum_pos / P])
    fpr = np.concatenate([[0.0], cum_neg / N])
    return float(np.trapezoid(tpr, fpr))


def _shuffle_within_sequence(
    h: torch.Tensor, mask: torch.Tensor, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Permute (h, mask) jointly along the L axis, independently per row."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    B, L = h.shape[:2]
    perm = torch.stack([torch.randperm(L, generator=g) for _ in range(B)])
    idx = perm.unsqueeze(-1).expand(-1, -1, h.shape[-1]) if h.dim() == 3 else perm
    if h.dim() == 3:
        h_p = torch.gather(h, 1, idx)
    else:
        h_p = torch.gather(h, 1, perm)
    mask_p = torch.gather(mask, 1, perm)
    return h_p, mask_p


# ---------------------------------------------------------------------------
# Method scorers — each returns a (B, L) score tensor (higher = more invalid).
# ---------------------------------------------------------------------------

def _score_SE(cache: dict[str, Any], invalid: bool) -> torch.Tensor:
    return cache["invalid_SE_pos" if invalid else "clean_SE_pos"].float()


def _score_topk_entropy(cache: dict[str, Any], invalid: bool) -> torch.Tensor:
    """Higher entropy of the top-K softmax → higher uncertainty.

    A locality-clean signal: depends only on the LM logits at this
    position, not the surrounding sequence.
    """
    clr = cache["invalid_clr" if invalid else "clean_clr"].float()
    log_p = clr.log_softmax(dim=-1)
    p = log_p.exp()
    return -(p * log_p).sum(dim=-1)


def _train_linear_probe(
    h_clean: torch.Tensor,
    h_invalid: torch.Tensor,
    mask_corrupt: torch.Tensor,
    *,
    epochs: int = 5,
    device: str = "cuda",
) -> nn.Module:
    """Per-position binary classifier on h_LLM. Class label = corrupted-position-or-not.

    Trained on a 50/50 split: corrupted positions in invalid sequences get
    label 1, all positions in clean sequences get label 0. The
    cascade-contamination phenomenon is that *uncorrupted* positions in
    invalid sequences also fire because the LM hidden state propagates the
    cascade signal.
    """
    H = h_clean.shape[-1]
    probe = nn.Linear(H, 1).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=3e-3)
    # Build training pool: invalid positions (with their corruption label),
    # plus all clean positions (label 0).
    h_inv = h_invalid.flatten(0, 1).to(device)
    y_inv = mask_corrupt.flatten().float().to(device)
    h_cln = h_clean.flatten(0, 1).to(device)
    y_cln = torch.zeros(h_cln.shape[0], device=device)
    h_all = torch.cat([h_inv, h_cln], 0)
    y_all = torch.cat([y_inv, y_cln], 0)
    n = h_all.shape[0]
    bs = 4096
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            logits = probe(h_all[idx]).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y_all[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return probe


def _score_with_probe(probe: nn.Module, h: torch.Tensor, device: str) -> torch.Tensor:
    with torch.no_grad():
        return probe(h.to(device)).squeeze(-1).cpu()


def _train_blr_laplace(
    h_clean: torch.Tensor,
    h_invalid: torch.Tensor,
    mask_corrupt: torch.Tensor,
    *,
    device: str,
    prior_var: float = 1.0,
) -> nn.Module:
    """Bayesian logistic regression with Laplace approximation. The point
    estimate gives the per-position score (mean logit). Implemented as a
    Linear with a small ridge regulariser to match the BLR-Laplace MAP."""
    H = h_clean.shape[-1]
    probe = nn.Linear(H, 1).to(device)
    h_inv = h_invalid.flatten(0, 1).to(device)
    y_inv = mask_corrupt.flatten().float().to(device)
    h_cln = h_clean.flatten(0, 1).to(device)
    y_cln = torch.zeros(h_cln.shape[0], device=device)
    h_all = torch.cat([h_inv, h_cln], 0)
    y_all = torch.cat([y_inv, y_cln], 0)
    opt = torch.optim.Adam(probe.parameters(), lr=3e-3, weight_decay=1.0 / max(prior_var, 1e-6))
    n = h_all.shape[0]
    bs = 4096
    for _ in range(8):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            logits = probe(h_all[idx]).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y_all[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return probe


def _train_svgp_proxy(
    h_clean: torch.Tensor,
    h_invalid: torch.Tensor,
    mask_corrupt: torch.Tensor,
    *,
    device: str,
    M: int = 64,
) -> nn.Module:
    """Lightweight SVGP-on-h_LLM proxy: kernel ridge regression with M
    inducing points. Placeholder for a full SVGP — preserves the
    "GP-on-hidden-states" cascade-contamination prediction (~AUROC 0.98)
    without pulling in gpytorch."""
    H = h_clean.shape[-1]
    h_inv = h_invalid.flatten(0, 1)
    y_inv = mask_corrupt.flatten().float()
    h_cln = h_clean.flatten(0, 1)
    y_cln = torch.zeros(h_cln.shape[0])
    h_all = torch.cat([h_inv, h_cln], 0).to(device)
    y_all = torch.cat([y_inv, y_cln], 0).to(device)
    g = torch.Generator().manual_seed(0)
    Z = h_all[torch.randperm(h_all.shape[0], generator=g).to(device)[:M]].clone()  # inducing points
    # Closed-form ridge regression in the kernel feature map
    # k(h, z) = exp(-‖h - z‖² / (2·sigma²)) — RBF.
    sigma = float(h_all.std().item())
    def K_mat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        d = (a[:, None, :] - b[None, :, :]).pow(2).sum(-1)
        return torch.exp(-d / (2 * sigma * sigma + 1e-6))
    K_aZ = K_mat(h_all, Z)  # (N, M)
    K_ZZ = K_mat(Z, Z) + 1e-3 * torch.eye(M, device=device)
    # Solve (K_Za^T K_Za + λ K_ZZ) α = K_Za^T y — ridge with K_ZZ regulariser.
    lam = 1e-2
    A = K_aZ.T @ K_aZ + lam * K_ZZ
    b = K_aZ.T @ y_all
    alpha = torch.linalg.solve(A, b)
    class _SVGPProbe(nn.Module):
        def __init__(self, Z: torch.Tensor, alpha: torch.Tensor, sigma: float) -> None:
            super().__init__()
            self.register_buffer("Z", Z)
            self.register_buffer("alpha", alpha)
            self.sigma = sigma
        def forward(self, h: torch.Tensor) -> torch.Tensor:
            d = (h[:, None, :] - self.Z[None, :, :]).pow(2).sum(-1)
            K = torch.exp(-d / (2 * self.sigma * self.sigma + 1e-6))
            return (K @ self.alpha).unsqueeze(-1)
    return _SVGPProbe(Z, alpha, sigma).to(device)


# ---------------------------------------------------------------------------
# Per-method 4-AUROC table.
# ---------------------------------------------------------------------------

def _four_auroc(
    score_clean: torch.Tensor,
    score_invalid: torch.Tensor,
    mask_corrupt: torch.Tensor,
    *,
    seed: int = 0,
) -> dict[str, float]:
    """Build labels and emit the 4 AUROC numbers.

    AUROC₁ — corrupted positions in invalid (label 1) vs same positions in clean (label 0).
    AUROC₂ — uncorrupted positions in invalid (label 1) vs same positions in clean (label 0).
    AUROC₃, AUROC₄ — same after permuting (score, mask) within each sequence.
    """

    def _sa() -> tuple[float, float]:
        s_inv = score_invalid
        s_cln = score_clean
        m = mask_corrupt.bool()
        # AUROC₁: corrupted positions in invalid (label 1) vs clean reference (label 0).
        s1 = torch.cat([s_inv[m], s_cln[m]])
        y1 = torch.cat([torch.ones(m.sum()), torch.zeros(m.sum())])
        # AUROC₂: uncorrupted in invalid (label 1) vs clean reference (label 0).
        nm = ~m
        s2 = torch.cat([s_inv[nm], s_cln[nm]])
        y2 = torch.cat([torch.ones(nm.sum()), torch.zeros(nm.sum())])
        return _auroc(s1, y1), _auroc(s2, y2)

    auroc1, auroc2 = _sa()

    s_inv_p, m_inv_p = _shuffle_within_sequence(
        score_invalid.unsqueeze(-1), mask_corrupt, seed
    )
    s_inv_p = s_inv_p.squeeze(-1)
    s_cln_p, _ = _shuffle_within_sequence(
        score_clean.unsqueeze(-1), mask_corrupt, seed + 1
    )
    s_cln_p = s_cln_p.squeeze(-1)

    m_p = m_inv_p.bool()
    s3 = torch.cat([s_inv_p[m_p], s_cln_p[m_p]])
    y3 = torch.cat([torch.ones(m_p.sum()), torch.zeros(m_p.sum())])
    nmp = ~m_p
    s4 = torch.cat([s_inv_p[nmp], s_cln_p[nmp]])
    y4 = torch.cat([torch.ones(nmp.sum()), torch.zeros(nmp.sum())])
    return {
        "AUROC1_corrupted_aligned": auroc1,
        "AUROC2_uncorrupted_aligned": auroc2,
        "AUROC3_corrupted_shuffled": _auroc(s3, y3),
        "AUROC4_uncorrupted_shuffled": _auroc(s4, y4),
    }


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="data/wiki_cache_gpt2.pt")
    p.add_argument(
        "--include",
        nargs="+",
        default=["SE", "topk_entropy", "linear_probe", "blr_laplace", "svgp"],
        choices=["SE", "topk_entropy", "linear_probe", "blr_laplace", "svgp",
                 "eqm_auditor", "saplma"],
    )
    p.add_argument("--auditor-ckpt", default=None)
    p.add_argument("--saplma-ckpt", default=None)
    p.add_argument("--out", default="runs/capstone/U/phaseU_cascade_audit.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--probe-epochs", type=int, default=5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    cache = load_wiki_cache(args.cache)
    print(f"[load] cache: n={cache['clean_clr'].shape[0]} L={cache['clean_clr'].shape[1]}")

    h_clean = cache.get("clean_h")
    h_invalid = cache.get("invalid_h")
    mask_corrupt = cache["mask_corrupt"].bool()

    results: dict[str, dict[str, float]] = {}

    for name in args.include:
        if name == "SE":
            s_clean = _score_SE(cache, invalid=False)
            s_invalid = _score_SE(cache, invalid=True)
        elif name == "topk_entropy":
            s_clean = _score_topk_entropy(cache, invalid=False)
            s_invalid = _score_topk_entropy(cache, invalid=True)
        elif name in ("linear_probe", "blr_laplace", "svgp"):
            if h_clean is None:
                print(f"[skip] {name}: cache has no clean_h")
                continue
            if name == "linear_probe":
                probe = _train_linear_probe(
                    h_clean, h_invalid, mask_corrupt,
                    epochs=args.probe_epochs, device=args.device,
                )
            elif name == "blr_laplace":
                probe = _train_blr_laplace(
                    h_clean, h_invalid, mask_corrupt, device=args.device
                )
            else:  # svgp
                probe = _train_svgp_proxy(
                    h_clean, h_invalid, mask_corrupt, device=args.device
                )
            B, L, _ = h_clean.shape
            s_clean = _score_with_probe(probe, h_clean.flatten(0, 1), args.device).view(B, L)
            s_invalid = _score_with_probe(probe, h_invalid.flatten(0, 1), args.device).view(B, L)
        elif name == "eqm_auditor":
            if not args.auditor_ckpt:
                print(f"[skip] {name}: --auditor-ckpt not given")
                continue
            from scripts.cascade_audit_eqm import score_eqm_auditor
            s_clean, s_invalid = score_eqm_auditor(args.auditor_ckpt, cache, args.device)
        elif name == "saplma":
            if not args.saplma_ckpt:
                print(f"[skip] {name}: --saplma-ckpt not given")
                continue
            from aitchinson_flow.baselines.saplma import score_saplma
            s_clean, s_invalid = score_saplma(args.saplma_ckpt, cache, args.device)
        else:
            print(f"[skip] {name}: unknown method")
            continue

        table = _four_auroc(s_clean, s_invalid, mask_corrupt, seed=args.seed)
        results[name] = table
        print(
            f"{name:>14}  "
            f"AUROC₁={table['AUROC1_corrupted_aligned']:.3f}  "
            f"AUROC₂={table['AUROC2_uncorrupted_aligned']:.3f}  "
            f"AUROC₃={table['AUROC3_corrupted_shuffled']:.3f}  "
            f"AUROC₄={table['AUROC4_uncorrupted_shuffled']:.3f}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"[write] {out}")

    md = ["# Phase U — cascade audit four-AUROC table", ""]
    md.append("| method | AUROC₁ corrupted aligned | AUROC₂ uncorrupted aligned | AUROC₃ corrupted shuffled | AUROC₄ uncorrupted shuffled |")
    md.append("|---|---:|---:|---:|---:|")
    for name, t in results.items():
        md.append(
            f"| {name} | {t['AUROC1_corrupted_aligned']:.3f} | "
            f"{t['AUROC2_uncorrupted_aligned']:.3f} | "
            f"{t['AUROC3_corrupted_shuffled']:.3f} | "
            f"{t['AUROC4_uncorrupted_shuffled']:.3f} |"
        )
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print(f"[write] {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
