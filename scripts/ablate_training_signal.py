"""E2a — the training-signal-class matrix (the 'money' causal experiment).

Fix the backbone (shared ``TransformerBackbone`` + ``VelocityHead``), the data
source (text8 token windows, or synthetic ids under ``--smoke``), and the
decode/recovery protocol; vary **only** the training target × the data recipe.
The point is a single-variable causal comparison: which *training signal* —
not which architecture — is what makes a continuous-simplex field collapse to
the unigram peak vs. escape into bigram-faithful structure.

Targets (the training loss the shared field regresses to):

  * fm_velocity_l2   — Equilibrium-FM target. ``v = f(x_γ)``, energy E=⟨x,v⟩,
                       conservative gradient ``g = ∇_{x_γ} E`` regressed onto the
                       FM velocity ``c(γ)·(x0 − x1)`` under L2. NO per-token CE
                       anchor (this is the *bare* FM target — the documented
                       mode-collapse regime, SESSION_SUMMARY.md). Second-order
                       autograd (create_graph=True), so MATH SDPA is mandatory.
  * x1_point_l2      — direct x1-prediction. The field predicts the clean point
                       ``x̂1 = x_γ − λ·g`` and regresses it onto x1 under L2 (no
                       distributional/CE term). Same conservative-gradient
                       machinery; the supervision is a single point, not a
                       distribution — also expected to collapse.
  * ce_distributional— per-token cross-entropy on the implied-x1 reconstruction
                       (the EqM anti-collapse anchor, lambda_ce path in
                       eqm.py:_eqm_loss). FM-velocity L2 + CE. Expected to
                       ESCAPE collapse.
  * dsm_eps          — denoising-score-matching ε-prediction. ``x_σ = x1 + σ·ε``;
                       the field predicts ε (no second-order autograd needed —
                       the velocity head output is read as ε̂ directly). A
                       distributional/score signal; expected to escape collapse
                       on generation, partially.

Recipes (how x1 — the clean simplex point — is encoded from token ids):

  * det_clr   — deterministic ``token_ids_to_features`` (one-hot → label-smooth
                → CLR). A single fixed vertex per token.
  * dirichlet — ``token_ids_to_features_dirichlet`` (fresh Dir(α_base+α_peak·e)
                draw each step → CLR). Thickens x1 off the vertex; the
                Dirichlet-FM-style data recipe (Stark et al. 2024). Expected to
                rescue *recovery* (the field sees a basin, not a point) but not
                *generation* (unconditional KL still poor without a CE anchor).

8 cells = 4 targets × 2 recipes. For each cell we train the shared field for a
few epochs and report KL_uni, KL_bi (n-gram generation, vs. the training
corpus), Δ@.50 (recovery: token_acc − token_acc_perturbed at α=0.5·feature-norm),
and a ``collapsed`` flag (KL_uni < 0.05 and KL_bi > 1.0 — the
low-unigram-KL/high-bigram-KL signature of the unigram-peak attractor).

PREDICTED PATTERN (documented as the expected result — NOT hardcoded; the
script reports whatever it measures):
  * fm_velocity_l2 & x1_point_l2  → collapse (KL_uni→0, KL_bi high, Δ≈0).
  * ce_distributional & dsm_eps   → escape (higher KL_uni, lower KL_bi, Δ>0).
  * dirichlet rescues recovery (Δ↑) but not generation (KL still poor without CE).

Usage:
    python scripts/ablate_training_signal.py --out runs/e2a/ablate_training_signal.json
    python scripts/ablate_training_signal.py --seeds 0,1 --epochs 4 --n 256
    # CPU self-test (no HF download; synthetic ids; <60s):
    python scripts/ablate_training_signal.py --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace as _replace
from pathlib import Path
from typing import Any


def _bootstrap() -> None:
    root = Path(__file__).resolve().parent.parent
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    if str(root) not in sys.path:
        sys.path.append(str(root))


_bootstrap()

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.data.char_window_dataset import CHAR2ID  # noqa: E402
from aitchinson_flow.data.transforms import (  # noqa: E402
    token_ids_to_features,
    token_ids_to_features_dirichlet,
)
from aitchinson_flow.transformer_backbone import (  # noqa: E402
    TransformerBackbone,
    VelocityHead,
)
from scripts.eval_full import ngram_kl, unigram_kl  # noqa: E402

ALPHABET = "".join(sorted(CHAR2ID, key=CHAR2ID.__getitem__))

TARGETS = ("fm_velocity_l2", "x1_point_l2", "ce_distributional", "dsm_eps")
RECIPES = ("det_clr", "dirichlet")

# Documented predicted pattern — printed alongside the measured numbers so the
# writeup can compare. NOT used to fabricate results.
PREDICTED = {
    "fm_velocity_l2": "collapse (KL_uni→0, KL_bi high, Δ≈0)",
    "x1_point_l2": "collapse (point target; KL_uni→0, KL_bi high, Δ≈0)",
    "ce_distributional": "escape (KL_uni up, KL_bi down, Δ>0)",
    "dsm_eps": "escape (score signal; generation improves)",
    "_recipe": "dirichlet rescues recovery (Δ↑) not generation (KL still poor w/o CE)",
}


# ─────────────────────────── shared field model ────────────────────────────


class SharedFieldModel(nn.Module):
    """Shared backbone + velocity head with four interchangeable training
    targets. The backbone/head are *identical* across cells (same constructor
    args from ``cfg``); only ``training_target`` and the data recipe change.

    All targets read the velocity head output ``v = f(x)``. The two FM-style
    targets (``fm_velocity_l2``, ``x1_point_l2``) and ``ce_distributional`` use
    the EqM conservative gradient ``g = ∇_x ⟨x, f(x)⟩`` (second-order autograd,
    MATH SDPA). ``dsm_eps`` reads ``v`` directly as the ε-prediction (no
    second-order autograd).
    """

    def __init__(self, cfg: Config, training_target: str) -> None:
        super().__init__()
        if training_target not in TARGETS:
            raise ValueError(f"unknown training_target={training_target!r}")
        self.cfg = cfg
        self.training_target = training_target
        self.backbone = TransformerBackbone(cfg=cfg)
        self.velocity_head = VelocityHead(cfg=cfg)
        self._K = int(cfg.text8_dataset.K)
        self._sigma = float(cfg.eqm.source_sigma)
        self._lambda = float(cfg.eqm.gradient_lambda)
        self._gamma_power = float(cfg.eqm.gamma_power)
        self._ce_min_gamma = float(cfg.eqm.ce_min_gamma)
        # DSM σ ladder (shared with cfg.dsm; small fixed range here).
        self._dsm_sigma_min = float(cfg.dsm.sigma_min)
        self._dsm_sigma_max = float(cfg.dsm.sigma_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.velocity_head(self.backbone(x))

    def _conservative_grad(
        self, x_in: torch.Tensor, *, create_graph: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (v, g) where g = ∇_x ⟨x, f(x)⟩ — the EqM conservative
        gradient. ``x_in`` must already require grad (or be detached and
        re-flagged here)."""
        x_req = x_in if x_in.requires_grad else x_in.detach().requires_grad_(True)
        v = self.forward(x_req)
        energy = (x_req * v).sum()
        g = torch.autograd.grad(
            energy, x_req, create_graph=create_graph, retain_graph=create_graph
        )[0]
        return v, g

    # ---- training -----------------------------------------------------------

    def training_loss(self, x1: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute the per-target training loss. ``x1`` is the (already
        encoded) clean CLR feature batch (B, L, K)."""
        B, L, K = x1.shape
        device, dt = x1.device, x1.dtype

        if self.training_target == "dsm_eps":
            return self._dsm_loss(x1)

        # FM-style targets (fm_velocity_l2 / x1_point_l2 / ce_distributional).
        x0 = self._sigma * torch.randn(B, L, K, device=device, dtype=dt)
        x0 = x0 - x0.mean(dim=-1, keepdim=True)  # stay in V_d
        gamma = torch.rand(B, device=device, dtype=dt).pow(self._gamma_power)
        x_gamma = (1.0 - gamma[:, None, None]) * x0 + gamma[:, None, None] * x1
        x_gamma.requires_grad_(True)
        # c(γ) = (1−γ) / gradient_lambda  (EqM linear decay).
        c_gamma = ((1.0 - gamma) / self._lambda)[:, None, None]
        u_tgt = c_gamma * (x0 - x1)

        v, g = self._conservative_grad(x_gamma, create_graph=True)

        if self.training_target == "fm_velocity_l2":
            loss = F.mse_loss(g, u_tgt)
            return {"loss": loss, "fm": loss.detach()}

        if self.training_target == "x1_point_l2":
            # Direct point target: implied x̂1 = x_γ − λ·g should equal x1.
            pred_x1 = x_gamma - self._lambda * g
            loss = F.mse_loss(pred_x1, x1)
            return {"loss": loss, "x1_l2": loss.detach()}

        # ce_distributional: FM-velocity L2 + per-token CE anchor on implied-x1.
        flow_loss = F.mse_loss(g, u_tgt)
        ce_mask = gamma >= self._ce_min_gamma
        ce = torch.zeros((), device=device, dtype=dt)
        if ce_mask.any():
            pred_x1 = x_gamma[ce_mask] - self._lambda * g[ce_mask]
            log_probs = pred_x1 - torch.logsumexp(pred_x1, dim=-1, keepdim=True)
            ids = self._features_to_ids(x1[ce_mask])
            ce = F.nll_loss(log_probs.reshape(-1, K), ids.reshape(-1))
        loss = flow_loss + self.cfg.eqm.lambda_ce * ce
        return {"loss": loss, "fm": flow_loss.detach(), "ce": ce.detach()}

    def _dsm_loss(self, x1: torch.Tensor) -> dict[str, torch.Tensor]:
        """Denoising-score-matching ε-prediction. No second-order autograd:
        the velocity head output is read directly as ε̂."""
        B, L, K = x1.shape
        device, dt = x1.device, x1.dtype
        log_lo, log_hi = (
            torch.log(torch.tensor(self._dsm_sigma_min)),
            torch.log(torch.tensor(self._dsm_sigma_max)),
        )
        log_sig = log_lo + (log_hi - log_lo) * torch.rand(B, device=device, dtype=dt)
        sigma = log_sig.exp()[:, None, None]
        eps = torch.randn(B, L, K, device=device, dtype=dt)
        eps = eps - eps.mean(dim=-1, keepdim=True)  # stay in V_d
        x_sigma = x1 + sigma * eps
        eps_hat = self.forward(x_sigma)
        loss = F.mse_loss(eps_hat, eps)
        return {"loss": loss, "dsm": loss.detach()}

    def _features_to_ids(self, feats: torch.Tensor) -> torch.Tensor:
        """Recover the underlying token ids from CLR features by argmax (CLR is
        monotone in the per-class one-hot mass, so argmax is exact)."""
        return feats.argmax(dim=-1).long()

    # ---- inference: unconditional sampling ----------------------------------

    @torch.no_grad()
    def _decode_ids(self, x: torch.Tensor) -> torch.Tensor:
        log_probs = x - torch.logsumexp(x, dim=-1, keepdim=True)
        return log_probs.argmax(-1).cpu()

    def sample(self, B: int, L: int, *, steps: int) -> torch.Tensor:
        """Unconditional sample → token ids (B, L). FM-style targets descend
        the conservative-gradient energy with NAG-GD (EqM sampler, simplified);
        dsm_eps runs a short ε-prediction denoising loop down the σ ladder."""
        device = next(self.parameters()).device
        K = self._K
        if self.training_target == "dsm_eps":
            return self._sample_dsm(B, L, steps=steps)
        # NAG-GD on the conservative gradient, matched train/sample σ.
        x = self._sigma * torch.randn(B, L, K, device=device)
        x = x - x.mean(dim=-1, keepdim=True)
        x_last = x.clone()
        eta, mu = float(self.cfg.eqm.sample_eta), float(self.cfg.eqm.sample_mu)
        grad = self._grad_at(x)
        for _ in range(steps):
            x_last = x
            x = x - eta * grad
            grad = self._grad_at(x + mu * (x - x_last))
        return self._decode_ids(x)

    def _grad_at(self, x: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            v = self.forward(x_req)
            energy = (x_req * v).sum()
            grad = torch.autograd.grad(energy, x_req, create_graph=False)[0]
        return grad.detach()

    def _sample_dsm(self, B: int, L: int, *, steps: int) -> torch.Tensor:
        """Short annealed-Langevin-ish denoising on the σ ladder for the ε
        model: x ← x − σ·ε̂(x) walking σ down from σ_max to σ_min."""
        device = next(self.parameters()).device
        K = self._K
        x = self._dsm_sigma_max * torch.randn(B, L, K, device=device)
        x = x - x.mean(dim=-1, keepdim=True)
        n_sig = max(2, steps)
        sigmas = torch.logspace(
            float(torch.log10(torch.tensor(self._dsm_sigma_max))),
            float(torch.log10(torch.tensor(self._dsm_sigma_min))),
            n_sig,
            device=device,
        )
        with torch.no_grad():
            for sigma in sigmas:
                eps_hat = self.forward(x)
                x = x - float(sigma) * eps_hat
                x = x - x.mean(dim=-1, keepdim=True)
        return self._decode_ids(x)

    # ---- inference: recovery (Δ@α) ------------------------------------------

    def recover_ids(
        self, x1: torch.Tensor, *, alpha: float, feat_norm: float, steps: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perturb encoded clean features by α·feat_norm Gaussian noise, then
        denoise. Returns (recovered_ids, perturbed_ids) — the two argmax decodes
        whose token-accuracy difference is Δ@α."""
        sig = alpha * feat_norm
        x_init = x1 + sig * torch.randn_like(x1)
        x_init = x_init - x_init.mean(dim=-1, keepdim=True)
        ids_pt = self._decode_ids(x_init)
        if self.training_target == "dsm_eps":
            # Single ε-step at the matching σ, then argmax.
            with torch.no_grad():
                eps_hat = self.forward(x_init)
                x_rec = x_init - sig * eps_hat
                x_rec = x_rec - x_rec.mean(dim=-1, keepdim=True)
            ids_rc = self._decode_ids(x_rec)
            return ids_rc, ids_pt
        # FM-style: NAG-GD from the perturbed init.
        device = next(self.parameters()).device
        x = x_init.to(device).detach()
        x_last = x.clone()
        eta, mu = float(self.cfg.eqm.sample_eta), float(self.cfg.eqm.sample_mu)
        grad = self._grad_at(x)
        for _ in range(steps):
            x_last = x
            x = x - eta * grad
            grad = self._grad_at(x + mu * (x - x_last))
        return self._decode_ids(x), ids_pt


# ─────────────────────────── data / encoding ───────────────────────────────


def _encode_x1(
    token_ids: torch.Tensor, K: int, recipe: str, *, label_smoothing: float
) -> torch.Tensor:
    """Token ids (B, L) → clean CLR features (B, L, K) under the chosen recipe.

    ``det_clr`` is deterministic; ``dirichlet`` draws a fresh Dirichlet simplex
    point per token each call (so re-encoding the SAME ids every train step
    thickens x1 off the vertex)."""
    if recipe == "det_clr":
        return token_ids_to_features(token_ids, K, label_smoothing=label_smoothing)
    if recipe == "dirichlet":
        return token_ids_to_features_dirichlet(
            token_ids,
            K,
            alpha_peak=10.0,
            alpha_base=0.1,
        )
    raise ValueError(f"unknown recipe={recipe!r}")


def _load_token_ids(
    cfg: Config, *, smoke: bool, n_windows: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (train_ids, eval_ids) integer windows (N, L).

    ``--smoke`` synthesises random ids (no HF download). Otherwise build the
    text8 datamodule with a small ``max_train_windows`` cap (cached windows
    under ~/.cache/huggingface; never triggers a fresh large download here)."""
    L = int(cfg.text8_dataset.L)
    K = int(cfg.text8_dataset.K)
    if smoke:
        g = torch.Generator().manual_seed(0)
        train_ids = torch.randint(0, K, (n_windows, L), generator=g)
        eval_ids = torch.randint(0, K, (max(8, n_windows // 2), L), generator=g)
        return train_ids, eval_ids
    from aitchinson_flow.training import build_training_datamodule

    dm, _ = build_training_datamodule(cfg)
    train_ids = dm.splits.train.long()
    eval_ids = dm.splits.val.long() if dm.splits.val is not None else train_ids
    return train_ids, eval_ids


# ─────────────────────────── one cell ──────────────────────────────────────


def _build_cfg(
    *, d_model: int, L: int, num_layers: int, nhead: int, n_windows: int
) -> Config:
    cfg = Config()
    K = cfg.text8_dataset.K
    cfg.training = _replace(cfg.training, model_name="EqM", L=L, K=K, device="cpu")
    cfg.text8_dataset = _replace(
        cfg.text8_dataset, L=L, K=K, max_train_windows=n_windows, max_eval_windows=n_windows
    )
    cfg.transformer = _replace(
        cfg.transformer, d_model=d_model, num_layers=num_layers, nhead=nhead
    )
    # Keep the EqM anti-collapse defaults (gamma_power=0.5, ce_min_gamma=0.5).
    return cfg


def run_cell(
    target: str,
    recipe: str,
    *,
    cfg: Config,
    train_ids: torch.Tensor,
    eval_ids: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    n_samples: int,
    sample_steps: int,
    recover_alpha: float,
    seed: int,
    device: str,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    K = int(cfg.text8_dataset.K)
    L = int(cfg.text8_dataset.L)
    ls = float(cfg.transformation.label_smoothing)

    model = SharedFieldModel(cfg, training_target=target).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n_train = train_ids.shape[0]
    model.train()
    for _ep in range(epochs):
        perm = torch.randperm(n_train)
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            ids_b = train_ids[idx].to(device)
            # Re-encode each step so the dirichlet recipe thickens x1 freshly.
            x1 = _encode_x1(ids_b, K, recipe, label_smoothing=ls)
            out = model.training_loss(x1)
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            if cfg.training.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip_norm)
            opt.step()

    model.eval()
    # ── generation: unconditional sample → n-gram KL vs the training corpus.
    gen_ids = model.sample(n_samples, L, steps=sample_steps)
    kl_u, _h_gen, _h_ref = unigram_kl(gen_ids, train_ids, K=K)
    kl_b = ngram_kl(gen_ids, train_ids, 2, K=K)
    collapsed = bool(kl_u < 0.05 and kl_b > 1.0)

    # ── recovery: Δ@α = token_acc − token_acc_perturbed at α=recover_alpha.
    ev = eval_ids[: min(n_samples, eval_ids.shape[0])].to(device)
    # Deterministic clean features set the perturbation scale (feat_norm) and the
    # ground-truth ids; the recipe under test still governs training.
    x1_clean = token_ids_to_features(ev, K, label_smoothing=ls)
    feat_norm = float(x1_clean.norm(dim=-1).mean().item())
    ids_rc, ids_pt = model.recover_ids(
        x1_clean, alpha=recover_alpha, feat_norm=feat_norm, steps=sample_steps
    )
    ev_cpu = ev.cpu()
    tok_acc = float((ids_rc == ev_cpu).float().mean())
    tok_acc_pt = float((ids_pt == ev_cpu).float().mean())
    delta50 = tok_acc - tok_acc_pt

    sample0 = "".join(ALPHABET[int(i) % len(ALPHABET)] for i in gen_ids[0])

    return {
        "target": target,
        "recipe": recipe,
        "seed": seed,
        "KL_uni": kl_u,
        "KL_bi": kl_b,
        "delta50": delta50,
        "token_acc": tok_acc,
        "token_acc_perturbed": tok_acc_pt,
        "collapsed": collapsed,
        "feat_norm": feat_norm,
        "sample0": sample0,
    }


# ─────────────────────────── matrix driver ─────────────────────────────────


def run_matrix(
    *,
    cfg: Config,
    train_ids: torch.Tensor,
    eval_ids: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    n_samples: int,
    sample_steps: int,
    recover_alpha: float,
    seeds: list[int],
    device: str,
) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for target in TARGETS:
        for recipe in RECIPES:
            seed_rows = [
                run_cell(
                    target,
                    recipe,
                    cfg=cfg,
                    train_ids=train_ids,
                    eval_ids=eval_ids,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    n_samples=n_samples,
                    sample_steps=sample_steps,
                    recover_alpha=recover_alpha,
                    seed=seed,
                    device=device,
                )
                for seed in seeds
            ]
            # Aggregate across seeds (mean of the scalar metrics; collapsed = any).
            agg = {
                "target": target,
                "recipe": recipe,
                "cell": f"{target}__{recipe}",
                "seeds": list(seeds),
                "KL_uni": float(sum(r["KL_uni"] for r in seed_rows) / len(seed_rows)),
                "KL_bi": float(sum(r["KL_bi"] for r in seed_rows) / len(seed_rows)),
                "delta50": float(sum(r["delta50"] for r in seed_rows) / len(seed_rows)),
                "collapsed": bool(any(r["collapsed"] for r in seed_rows)),
                "predicted": PREDICTED[target],
                "per_seed": seed_rows,
            }
            cells.append(agg)
            print(
                f"[{target:18s} | {recipe:9s}] "
                f"KL_uni={agg['KL_uni']:.4f}  KL_bi={agg['KL_bi']:.4f}  "
                f"Δ@.50={agg['delta50']:+.4f}  collapsed={agg['collapsed']}"
            )
    return {
        "experiment": "E2a_training_signal_matrix",
        "targets": list(TARGETS),
        "recipes": list(RECIPES),
        "predicted_pattern": PREDICTED,
        "config": {
            "d_model": cfg.transformer.d_model,
            "L": cfg.text8_dataset.L,
            "num_layers": cfg.transformer.num_layers,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "n_samples": n_samples,
            "sample_steps": sample_steps,
            "recover_alpha": recover_alpha,
            "n_train_windows": int(train_ids.shape[0]),
        },
        "cells": cells,
    }


def _validate_result(result: dict[str, Any]) -> None:
    cells = result["cells"]
    assert len(cells) == 8, f"expected 8 cells, got {len(cells)}"
    seen = set()
    for c in cells:
        for k in ("KL_uni", "KL_bi", "delta50", "collapsed"):
            assert k in c, f"cell {c.get('cell')!r} missing {k}"
        assert isinstance(c["collapsed"], bool), "collapsed must be a bool"
        seen.add((c["target"], c["recipe"]))
    expected = {(t, r) for t in TARGETS for r in RECIPES}
    assert seen == expected, f"cells {seen} != {expected}"


# ─────────────────────────── smoke / cli ───────────────────────────────────


def _smoke() -> int:
    """All 8 cells at d_model=32 / L=12 / 2 layers / 2 epochs / 64 windows on
    CPU; assert a well-formed json with 8 cells each carrying the required
    keys; print OK."""
    import tempfile

    torch.manual_seed(0)
    cfg = _build_cfg(d_model=32, L=12, num_layers=2, nhead=2, n_windows=64)
    train_ids, eval_ids = _load_token_ids(cfg, smoke=True, n_windows=64)
    result = run_matrix(
        cfg=cfg,
        train_ids=train_ids,
        eval_ids=eval_ids,
        epochs=2,
        batch_size=16,
        lr=1e-3,
        n_samples=16,
        sample_steps=4,
        recover_alpha=0.5,
        seeds=[0],
        device="cpu",
    )
    _validate_result(result)
    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / "ablate_training_signal.json"
        out_path.write_text(json.dumps(result, indent=2))
        loaded = json.loads(out_path.read_text())
        _validate_result(loaded)
    print(
        "OK ablate_training_signal smoke: 8 cells "
        f"({len(TARGETS)} targets × {len(RECIPES)} recipes), each with "
        "KL_uni/KL_bi/delta50/collapsed; json validated."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str, default=None, help="json output path")
    ap.add_argument("--seeds", type=str, default="0", help="comma-separated seeds")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n", type=int, default=256, dest="n_samples")
    ap.add_argument("--steps", type=int, default=100, dest="sample_steps")
    ap.add_argument("--recover-alpha", type=float, default=0.5)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--L", type=int, default=40, dest="L")
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument(
        "--n-windows",
        type=int,
        default=2000,
        help="max_train_windows cap (kept small; never triggers a fresh HF download)",
    )
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--smoke", action="store_true", help="CPU self-test (<60s)")
    args = ap.parse_args(argv)

    if args.smoke:
        return _smoke()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    cfg = _build_cfg(
        d_model=args.d_model,
        L=args.L,
        num_layers=args.num_layers,
        nhead=args.nhead,
        n_windows=args.n_windows,
    )
    train_ids, eval_ids = _load_token_ids(cfg, smoke=False, n_windows=args.n_windows)

    t0 = time.time()
    result = run_matrix(
        cfg=cfg,
        train_ids=train_ids,
        eval_ids=eval_ids,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_samples=args.n_samples,
        sample_steps=args.sample_steps,
        recover_alpha=args.recover_alpha,
        seeds=seeds,
        device=args.device,
    )
    result["wall_time_s"] = time.time() - t0
    _validate_result(result)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
