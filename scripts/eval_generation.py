"""Unified generation evaluation across the 5 generation arms used for the
capstone write-up:

  1. EqM_OneHot   — one-hot → CLR/ILR → EqM (label-smoothed CLR, no
                    Dirichlet thickening).
  2. EqMLatent    — EqM in a learned embedding space.
  3. EqM          — Dirichlet-thickened EqM (the project's tuned recipe;
                    ``cfg.transformation.dirichlet_sampling = True``).
  4. DirichletFM  — Stark et al. (2024) Dirichlet flow matching.
  5. SFLM         — time-conditioned hyperspherical flow.

For each arm computes, on a fresh seed for fairness:

  BPD                 — model.bpd() on held-out val (reconstruction NLL).
  KL_uni / KL_bi /    — n-gram KL of unconditional samples vs the train
  KL_tri / H_gen        corpus n-gram distribution.
  recover_acc(α)      — token-level recovery accuracy after perturbing
                        encoded val sequences with α·embed_norm·N(0,I)
                        and re-sampling (mirrors recovery_check.py).
                        Marked "—" for DirichletFM which has no x_init
                        sampler hook.

Writes runs/sflm_bench_<scale>/generation_eval.json + printed table.

Usage:
  python scripts/eval_generation.py --scale cluster --n 256 \
      --recover-alphas 0.1,0.3,0.5,1.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.append(str(_ROOT))

import aitchinson_flow.models  # noqa: E402,F401  populate REGISTRY
from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.data.char_window_dataset import CHAR2ID  # noqa: E402
from aitchinson_flow.data.transforms import token_ids_to_features  # noqa: E402
from aitchinson_flow.models import build_model  # noqa: E402
from aitchinson_flow.training import build_training_datamodule  # noqa: E402
from scripts.eval_full import _config_from_payload  # noqa: E402
from scripts._ensure_ckpt import ensure_checkpoint  # noqa: E402
from scripts.train_for_sflm_bench import SCALES  # noqa: E402  (L per scale)

# Eight-arm unified generation comparison.  SFLMEBM / SFLMEBM_FM are now
# *also* benchmarked here for a fair density column (the previous "OOD
# only" framing left half the arms with no published BPC); DFM is the only
# honest non-collapsing density baseline on text8 in this codebase so it
# anchors the column.  The OOD bench (bench_sflm_ebm.py) still owns the
# AUROC comparison.
GEN_ARMS = [
    "EqM_OneHot", "EqMLatent", "EqM",
    "DirichletFM", "SFLM",
    "SFLMEBM", "SFLMEBM_FM", "DFM",
]
ALPHABET = "".join(sorted(CHAR2ID, key=CHAR2ID.__getitem__))


def _load(arm: str, device, scale: str, *,
          epochs: int, auto_train: bool):
    ckpt = ensure_checkpoint(
        arm, scale=scale, epochs=epochs, auto_train=auto_train,
    )
    if ckpt is None:
        return None, None
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = _config_from_payload(payload)
    model = build_model(cfg).to(device)
    state = payload.get("model_state_dict", payload)
    model.load_state_dict(state)
    model.eval()
    return model, cfg


# --------------------------------------------------------------------------
# Corpus n-gram counts (cached per run; matches recovery_check.py logic).
# --------------------------------------------------------------------------
def _ngram_counts(ids2d: torch.Tensor, K: int, n: int) -> torch.Tensor:
    L = ids2d.shape[1]
    if L < n:
        return torch.zeros(K ** n)
    idx = torch.zeros(ids2d.shape[0], L - n + 1, dtype=torch.long)
    for i in range(n):
        idx = idx + ids2d[:, i:L - n + 1 + i].long() * (K ** (n - 1 - i))
    counts = torch.zeros(K ** n)
    counts.scatter_add_(0, idx.reshape(-1), torch.ones(idx.numel()))
    return counts


def _kl(gen: torch.Tensor, ref: torch.Tensor) -> float:
    smoothing = 1e-6
    Ksize = gen.numel()
    gp = (gen + smoothing) / (gen.sum() + Ksize * smoothing)
    rp = (ref + smoothing) / (ref.sum() + Ksize * smoothing)
    return float((gp * (gp.log() - rp.log())).sum())


# --------------------------------------------------------------------------
# Per-arm metric helpers.  Each arm exposes a slightly different generative
# API; this layer normalises {generate, bpd, recover} for the report.
# --------------------------------------------------------------------------
@torch.no_grad()
def _generate_ids(arm: str, model, cfg, n: int, L: int) -> torch.Tensor:
    """Unconditional generation → token ids (n, L) on cpu.

    For DirichletFM, ``sample`` already returns ids; for everything else
    we ``sample`` in latent space and argmax-decode."""
    if arm == "DirichletFM":
        return model.sample(n, L).cpu()
    z = model.sample(n, L)
    log_p = model.decode_to_logprobs(z)
    return log_p.argmax(dim=-1).cpu()


@torch.no_grad()
def _fair_bpc(arm: str, model, cfg, ids: torch.Tensor, n_mc: int = 4) -> float:
    """Held-out denoiser cross-entropy in bits/char — a single forward of
    each arm's training-time noised → denoise pipeline, averaged over
    ``n_mc`` independent noise draws (NO ``model.eval_step`` because EqM-
    family ``_eqm_loss`` uses second-order autograd with
    ``create_graph=True`` which is incompatible with this ``@torch.no_grad``
    context and would also OOM at this batch size).

    Replaces the legacy ``model.bpd()`` recovery-NLL path which for
    EqM / EqM_OneHot / EqMLatent / SFLM / SFLMEBM is a NAG-GD / SLERP
    integration from a *small* perturbation of the clean codebook entry —
    i.e. measures local-basin attraction rather than data NLL, and
    consequently reports bpc ≈ 0.  Each arm's denoiser CE is computed
    directly: noise the input via the training path, decode logits at
    that noised point, take −log p(ids).  This is the same "training-
    objective bits-per-char" used by SEDD / D3PM in their Table-2
    comparisons.  Published text8 reference floors: SEDD-Absorb ≈ 1.32,
    Transformer-XL (AR) ≈ 1.04.
    """
    import math
    K = cfg.text8_dataset.K
    device = next(model.parameters()).device
    ids = ids.to(device).long()
    B, L = ids.shape
    ls = getattr(cfg.transformation, "label_smoothing", 1e-4)
    total = 0.0
    for _ in range(n_mc):
        if arm == "DFM":
            t = torch.rand(B, device=device)
            x_t = model._corrupt(ids, t)
            logits = model.forward(x_t, t)
            log_p = logits.log_softmax(dim=-1)
        elif arm == "DirichletFM":
            t_max = float(model.t_max)
            t = torch.empty(B, device=device).uniform_(1.0, t_max)
            x_t = model._sample_xt(ids, t)
            logits = model.forward(x_t, t)
            log_p = logits.log_softmax(dim=-1)
        elif arm == "SFLM":
            # Match SFLM._loss: SLERP at α ∈ training schedule range.
            s_sflm = cfg.sflm
            z1 = model.encode(ids)
            z0 = torch.randn_like(z1)
            z0 = z0 / z0.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            alpha = (s_sflm.alpha_lo + (s_sflm.alpha_hi - s_sflm.alpha_lo)
                     * torch.rand(B, 1, device=device))
            cos_w = (z0 * z1).sum(-1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7)
            omega = torch.arccos(cos_w)
            s = torch.sin(omega).clamp(min=1e-7)
            z_a = torch.sin((1 - alpha) * omega) / s * z0 \
                + torch.sin(alpha * omega) / s * z1
            log_p = model.decode_to_logprobs(z_a, gamma=alpha.squeeze(-1))
        elif arm in ("SFLMEBM", "SFLMEBM_FM"):
            # Match SFLMEBM._loss: only the α ≥ ce_min_alpha range was
            # CE-trained — so we average BPC over the same window.
            s_ebm = cfg.sflm_ebm
            ce_lo = float(getattr(s_ebm, "ce_min_alpha", 0.5))
            z1 = model.encode(ids)
            z0 = torch.randn_like(z1)
            z0 = z0 / z0.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            alpha = ce_lo + (1.0 - ce_lo) * torch.rand(B, 1, device=device)
            cos_w = (z0 * z1).sum(-1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7)
            omega = torch.arccos(cos_w)
            s = torch.sin(omega).clamp(min=1e-7)
            z_a = torch.sin((1 - alpha) * omega) / s * z0 \
                + torch.sin(alpha * omega) / s * z1
            log_p = model.decode_to_logprobs(z_a)
        elif arm in ("EqM", "EqM_OneHot"):
            # EqM's decode_to_logprobs(x) is just log_softmax(x) — it does NOT
            # actually invoke the trained velocity field. To compute a
            # meaningful denoiser NLL we must mirror training: predict x1
            # from the *implied-x1* pathway, i.e. ``pred_x1 = x_γ − λ·grad_g``
            # where grad_g = ∇_x ⟨x, f(x)⟩ (first-order autograd; create_graph
            # not needed since we don't backprop further).
            # Sample γ ∈ [ce_min_gamma, 1] — the CE-trained range; outside it
            # the implied-x1 decoder isn't calibrated.
            ce_lo = float(getattr(cfg.eqm, "ce_min_gamma", 0.5))
            x1 = token_ids_to_features(ids, K, label_smoothing=ls)
            sigma = float(cfg.eqm.source_sigma)
            x0 = sigma * torch.randn_like(x1)
            x0 = x0 - x0.mean(dim=-1, keepdim=True)
            gamma = (ce_lo + (1.0 - ce_lo)
                     * torch.rand(B, 1, 1, device=device))
            x_g = (1.0 - gamma) * x0 + gamma * x1
            lam = float(getattr(cfg.eqm, "gradient_lambda", 1.0))
            with torch.enable_grad():
                x_req = x_g.detach().requires_grad_(True)
                v = model.forward(x_req)
                E = (x_req * v).sum()
                grad_g, = torch.autograd.grad(E, x_req)
            pred_x1 = (x_g - lam * grad_g).detach()
            log_p = pred_x1 - torch.logsumexp(pred_x1, dim=-1, keepdim=True)
        elif arm == "EqMLatent":
            # Same idea — train-time CE on `decode_to_logits(x_γ − λ·grad_g)`,
            # not on `decode_to_logits(x_γ)`.  See eqm_latent.py:293–299.
            ce_lo = float(getattr(cfg.eqm, "ce_min_gamma", 0.5))
            z1 = model.encode(ids)
            z_norm = z1.norm(dim=-1).mean().item()
            z0 = torch.randn_like(z1) * z_norm
            gamma = (ce_lo + (1.0 - ce_lo)
                     * torch.rand(B, 1, 1, device=device))
            z_g = (1.0 - gamma) * z0 + gamma * z1
            s_eqm = cfg.eqm
            lam = float(getattr(s_eqm, "gradient_lambda", 1.0))
            time_cond = getattr(s_eqm, "time_conditioning", "off") != "off"
            with torch.enable_grad():
                z_req = z_g.detach().requires_grad_(True)
                gamma_arg = (gamma.squeeze(-1).squeeze(-1)
                             if time_cond else None)
                v = model.forward(z_req, gamma_arg)
                E = (z_req * v).sum()
                grad_g, = torch.autograd.grad(E, z_req)
            pred_x1 = (z_g - lam * grad_g).detach()
            log_p = model.decode_to_logprobs(pred_x1)
        else:
            return float("nan")
        nll = -log_p.gather(-1, ids.unsqueeze(-1)).squeeze(-1).mean()
        total += float(nll)
    return total / n_mc / math.log(2)


def _infill_acc(
    arm: str, model, cfg, ids: torch.Tensor, infill_rate: float,
    steps: int, seed: int,
) -> dict | None:
    """Conditional generation by infilling: per-position mask M with
    ``infill_rate`` fraction True ⇒ "unknown / to be generated"; the rest
    are pinned to ground-truth latents in ``x_init``.  Recovery accuracy
    is reported separately for the masked (generation) and unmasked (kept)
    positions, plus an end-to-end NLL summary for the masked-only positions.

    Geometry-wise this is the same noise→data init as ``_recover_acc`` but
    with a *per-position* sigma_perturb that is α=1·embed_norm on the
    masked positions and α=0 on the kept positions.  All samplers in this
    codebase ingest ``x_init`` without an in-loop clamp, so the kept
    positions are stable only because they're already near a model
    attractor — not enforced.  Reported as ``infill_kept_acc`` so any
    drift in the "fixed" positions is visible alongside the metric of
    interest (``infill_acc``).
    """
    device = next(model.parameters()).device
    ids = ids.to(device).long()
    B, L = ids.shape
    K = cfg.text8_dataset.K
    g_cpu = torch.Generator().manual_seed(seed)
    mask = torch.rand((B, L), generator=g_cpu) < infill_rate    # (B, L) bool
    mask_dev = mask.to(device)
    n_masked = int(mask.sum())
    n_kept = int((~mask).sum())
    if n_masked == 0:
        return None

    if arm == "DirichletFM":
        t_max = float(model.t_max)
        beta = torch.ones(B, L, K, device=device, dtype=torch.float32)
        # Mirror DirichletFM._sample_xt convention: REPLACE beta[token_id]
        # by the per-position concentration.  Kept positions → t_max
        # (sharp delta on GT); masked positions → 1.0 (uniform Dirichlet).
        conc = torch.where(
            mask_dev,
            torch.ones_like(mask_dev, dtype=beta.dtype),
            torch.full_like(mask_dev, t_max, dtype=beta.dtype),
        )
        beta.scatter_(-1, ids.unsqueeze(-1), conc.unsqueeze(-1))
        with torch.no_grad():
            x_init = torch.distributions.Dirichlet(beta).sample()
            rec = model.sample(B, L, x_init=x_init, t_start=1.0,
                               nfe=steps).cpu()
    else:
        if hasattr(model, "encode"):
            z1 = model.encode(ids)
        else:
            ls = cfg.transformation.label_smoothing
            z1 = token_ids_to_features(ids, K, label_smoothing=ls)
        noise = torch.randn_like(z1)
        if arm == "SFLM":
            noise = noise / noise.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            z_noisy = noise
        else:
            z_noisy = z1.norm(dim=-1).mean().item() * noise
        z_init = torch.where(mask_dev.unsqueeze(-1), z_noisy, z1)
        with torch.no_grad():
            z = model.sample(ids.shape[0], ids.shape[1],
                             x_init=z_init, max_steps=steps)
            log_p = model.decode_to_logprobs(z)
            rec = log_p.argmax(dim=-1).cpu()

    ids_cpu = ids.cpu()
    agree = (rec == ids_cpu)
    infill_acc = float((agree & mask).sum() / max(n_masked, 1))
    kept_acc = (float((agree & ~mask).sum() / n_kept)
                if n_kept > 0 else float("nan"))
    return {"infill_acc": infill_acc, "kept_acc": kept_acc,
            "n_masked": n_masked, "n_kept": n_kept}


def _recover_acc(
    arm: str, model, cfg, ids: torch.Tensor, alpha: float, steps: int
) -> float | None:
    """Token-level recovery accuracy at *path-midpoint* perturbation α.

    α is now the **fraction of the way from data toward the noise source**
    along each arm's native noise path (α=0 ⇔ clean / identity, α=1 ⇔
    pure noise / unconditional).  This replaces the previous
    ``z_init = z1 + α·embed_norm·N(0,I)`` recipe which mixed dimensions
    incomparably across arms (embed_norm = 1 on the unit sphere vs ≈√K
    for CLR features, so the SAME α produced very different effective
    SNR per arm).  Geometry-aware initialisation:

      * **SFLM** (sphere)            — z_init = SLERP(z0=uniform_sphere, z1, 1−α)
      * **EqM / EqM_OneHot** (CLR)   — z_init = (1−α)·z1 + α·(σ_src·N(0,I) − mean)
      * **EqMLatent / SFLMEBM***    — z_init = (1−α)·z1 + α·N(0, ‖z1‖²·I/d)
      * **DirichletFM**              — partial-path: t_start = t_max − α·(t_max−1)

    All four put α on the same data↔noise axis, so a single α value picks
    out the *same* nominal signal level across arms.
    """
    device = next(model.parameters()).device
    if arm == "DirichletFM":
        ids_dev = ids.to(device).long()
        B, L = ids_dev.shape
        t_max = float(model.t_max)
        t_start = max(1.0, t_max - float(alpha) * (t_max - 1.0))
        t_b = torch.full((B,), t_start, device=device, dtype=torch.float32)
        with torch.no_grad():
            x_init = model._sample_xt(ids_dev, t_b)
            rec = model.sample(
                B, L, x_init=x_init, t_start=t_start, nfe=steps,
            ).cpu()
        return float((rec == ids.cpu()).float().mean())
    ids = ids.to(device).long()
    K = cfg.text8_dataset.K
    if hasattr(model, "encode"):
        z1 = model.encode(ids)
    else:
        ls = cfg.transformation.label_smoothing
        z1 = token_ids_to_features(ids, K, label_smoothing=ls)

    if arm == "SFLM":
        # Detect SFLM by unit-norm structure: every z1 row is on S^{d-1}.
        z0 = torch.randn_like(z1)
        z0 = z0 / z0.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        # Geodesic SLERP from data toward noise.
        cos_omega = (z0 * z1).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7)
        omega = torch.arccos(cos_omega).unsqueeze(-1)
        s = torch.sin(omega).clamp(min=1e-7)
        a = float(alpha)
        z_init = torch.sin(a * omega) / s * z0 + torch.sin((1 - a) * omega) / s * z1
    elif arm in ("EqM", "EqM_OneHot"):
        sigma = float(cfg.eqm.source_sigma)
        x0 = sigma * torch.randn_like(z1)
        x0 = x0 - x0.mean(dim=-1, keepdim=True)        # project to V_d
        z_init = (1.0 - alpha) * z1 + alpha * x0
    else:  # EqMLatent, SFLMEBM, SFLMEBM_FM — generic learned ℝ^d embedding
        z_norm = z1.norm(dim=-1).mean().item()
        z0 = torch.randn_like(z1) * z_norm
        z_init = (1.0 - alpha) * z1 + alpha * z0

    with torch.no_grad():
        z = model.sample(ids.shape[0], ids.shape[1],
                         x_init=z_init, max_steps=steps)
        log_p = model.decode_to_logprobs(z)
        rec = log_p.argmax(dim=-1).cpu()
    return float((rec == ids.cpu()).float().mean())


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale",
                    choices=["local", "cluster", "a100_20g", "a100_20g_L256"],
                    default="cluster",
                    help="a100_20g_L256 = d512/6L at L=256, text8 publication "
                         "convention (matches SEDD / D3PM / Transformer-XL)")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--steps", type=int, default=200,
                    help="recovery sampler max_steps")
    ap.add_argument(
        "--recover-alphas", type=str,
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="recovery-from-perturbation sweep (α·embed_norm·N(0,I)). "
             "α=0 ≡ identity, α=1 ≡ unconditional",
    )
    ap.add_argument(
        "--infill-rates", type=str,
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="conditional-infill sweep: fraction of positions masked and "
             "regenerated, with the rest pinned to the ground-truth latent "
             "in x_init.  rate=0.0 ≡ identity, rate=1.0 ≡ unconditional",
    )
    ap.add_argument("--bpc-mc", type=int, default=8,
                    help="Monte-Carlo draws for fair BPC (training-objective "
                         "denoiser CE averaged over the noise schedule)")
    ap.add_argument("--epochs", type=int, default=50,
                    help="epochs for any arm that needs auto-training")
    ap.add_argument("--no-auto-train", action="store_true",
                    help="revert to legacy 'skip if missing' behaviour")
    ap.add_argument("--out", type=str, default=None,
                    help="output JSON (default: <root>/generation_eval.json)")
    args = ap.parse_args()
    root = f"runs/sflm_bench_{args.scale}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    alphas = [float(a) for a in args.recover_alphas.split(",") if a.strip()]
    infill_rates = [float(a) for a in args.infill_rates.split(",") if a.strip()]

    # Build the data module once (text8 splits) for clean TEST ids + train-
    # corpus n-gram counts.  Test (not val) for publication numbers; val
    # is reserved for hyperparameter selection.  L follows the scale so
    # the data windows match the per-arm training-time L.
    from dataclasses import replace as _replace
    base_cfg = Config()
    L_eval = SCALES[args.scale].get("L", 40)
    base_cfg.training = _replace(base_cfg.training, L=L_eval)
    base_cfg.text8_dataset = _replace(base_cfg.text8_dataset, L=L_eval)
    dm, _ = build_training_datamodule(base_cfg)
    train_ids = dm.splits.train.long()
    eval_split = getattr(dm.splits, "test", None)
    if eval_split is None or eval_split.numel() == 0:
        eval_split = dm.splits.val
        print("[warn] dm.splits.test is empty — falling back to val.")
    eval_ids = eval_split.long()
    K = base_cfg.text8_dataset.K
    L = base_cfg.text8_dataset.L
    g = torch.Generator().manual_seed(args.seed)
    pick = torch.randperm(eval_ids.shape[0], generator=g)[: args.n]
    clean_val = eval_ids[pick].to(device)
    print(f"data: K={K}, L={L}, n_eval={args.n} (test split) | device={device}")

    ref_uni = _ngram_counts(train_ids, K, 1)
    ref_bi = _ngram_counts(train_ids, K, 2)
    ref_tri = _ngram_counts(train_ids, K, 3)

    results: dict = {}
    for arm in GEN_ARMS:
        model, cfg = _load(
            arm, device, args.scale,
            epochs=args.epochs, auto_train=not args.no_auto_train,
        )
        if model is None:
            print(f"[skip] {arm}: no checkpoint and auto-train failed/disabled")
            continue
        print(f"\n=== {arm} ===")
        torch.manual_seed(args.seed)
        bpc = _fair_bpc(arm, model, cfg, clean_val, n_mc=args.bpc_mc)
        ppl, bpb = float(2.0 ** bpc), bpc  # BPB ≡ BPC on text8 (1 byte/token)

        # Unconditional generation → n-gram KL.
        gen_ids = _generate_ids(arm, model, cfg, args.n, L)
        gen_uni = _ngram_counts(gen_ids, K, 1)
        gen_bi = _ngram_counts(gen_ids, K, 2)
        gen_tri = _ngram_counts(gen_ids, K, 3)
        kl_u, kl_b, kl_t = _kl(gen_uni, ref_uni), _kl(gen_bi, ref_bi), _kl(gen_tri, ref_tri)
        p = (gen_uni + 1e-9) / (gen_uni.sum() + K * 1e-9)
        H_gen = float(-(p * p.log()).sum())
        sample_text = "".join(ALPHABET[int(i)] for i in gen_ids[0])

        # Recovery curve.
        rec_curve = {}
        for a in alphas:
            acc = _recover_acc(arm, model, cfg, clean_val, a, args.steps)
            rec_curve[a] = acc
            tag = "—" if acc is None else f"{acc:.3f}"
            print(f"  α={a:.2f}  recover_acc={tag}")

        # Conditional infill: mask `rate` fraction of positions, pin the rest
        # to GT, sample, measure recovery on masked vs kept positions.
        infill_curve: dict[float, dict | None] = {}
        for ir in infill_rates:
            stats = _infill_acc(arm, model, cfg, clean_val, ir,
                                args.steps, args.seed)
            infill_curve[ir] = stats
            if stats is None:
                print(f"  infill_rate={ir:.2f}  (unavailable)")
            else:
                print(f"  infill_rate={ir:.2f}  "
                      f"masked_acc={stats['infill_acc']:.3f}  "
                      f"kept_acc={stats['kept_acc']:.3f}  "
                      f"(M={stats['n_masked']}, K={stats['n_kept']})")

        results[arm] = {
            "ppl": ppl, "bpb": bpb, "bpc": bpc,
            "KL_uni": kl_u, "KL_bi": kl_b, "KL_tri": kl_t,
            "H_gen": H_gen, "recover": rec_curve,
            "infill": infill_curve,
            "sample0": sample_text,
        }
        print(f"  PPL={ppl:7.3f}  BPB={bpb:.3f}  BPC={bpc:.3f}  "
              f"KL_uni={kl_u:.4f}  KL_bi={kl_b:.4f}  "
              f"KL_tri={kl_t:.4f}  H_gen={H_gen:.3f}")
        print(f"  sample[0]: {sample_text!r}")

    out_path = (Path(args.out) if args.out
                else Path(f"{root}/generation_eval.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str))

    # ---- printed comparison tables ----
    def _f(v, w=5, p=3):
        if v is None or (isinstance(v, float) and v != v):
            return f"{'—':>{w}}"
        return f"{v:>{w}.{p}f}"

    print("\n" + "=" * 110)
    print("(0) DENSITY + UNCONDITIONAL n-GRAM KL  (lower=better; PPL=2^BPC; "
          "H_gen close to corpus 2.73 nats = healthy unigram coverage)")
    print("    BPC = TRAINING-OBJECTIVE DENOISER CE averaged over each arm's "
          "native noise schedule (NOT the legacy .bpd() identity-recovery "
          "path).  This is the same metric SEDD/D3PM/Multinomial-Diffusion "
          "report; on text8 the published non-AR floor is SEDD-Absorb ≈ 1.32 "
          "BPC and the AR ceiling is Transformer-XL ≈ 1.04 BPC.")
    print("=" * 110)
    head = (f"{'arm':12s}  {'PPL':>8s}  {'BPB':>7s}  {'BPC':>7s}  "
            f"{'KL_uni':>8s}  {'KL_bi':>8s}  {'KL_tri':>8s}  {'H_gen':>6s}")
    print(head)
    for arm in GEN_ARMS:
        r = results.get(arm)
        if r is None:
            print(f"{arm:12s}  (missing checkpoint)")
            continue
        print(f"{arm:12s}  {r['ppl']:8.3f}  {r['bpb']:7.3f}  {r['bpc']:7.3f}  "
              f"{r['KL_uni']:8.4f}  {r['KL_bi']:8.4f}  {r['KL_tri']:8.4f}  "
              f"{r['H_gen']:6.3f}")

    print("\n" + "=" * 110)
    print("(1) RECOVERY-FROM-PERTURBATION GRID  (α·embed_norm·N(0,I); "
          "α=0≈identity, α=1≈unconditional; higher=better)")
    print("=" * 110)
    head = f"{'arm':12s}"
    for a in alphas:
        head += f"  α={a:.1f}"
    print(head)
    for arm in GEN_ARMS:
        r = results.get(arm)
        if r is None:
            print(f"{arm:12s}  (missing checkpoint)")
            continue
        row = f"{arm:12s}"
        for a in alphas:
            row += f"  {_f(r['recover'].get(a))}"
        print(row)

    print("\n" + "=" * 110)
    print("(2) CONDITIONAL-INFILL GRID  (mask `rate` fraction of positions, "
          "pin the rest to GT, then sample.  Reports accuracy on the MASKED "
          "positions — the conditionally generated ones — and on the KEPT "
          "positions for drift sanity.)")
    print("=" * 110)
    print(f"{'arm':12s}  metric          " +
          "".join(f"  r={r:.1f}" for r in infill_rates))
    for arm in GEN_ARMS:
        r = results.get(arm)
        if r is None:
            print(f"{arm:12s}  (missing checkpoint)")
            continue
        for key, label in (("infill_acc", "masked_acc "),
                           ("kept_acc",   "kept_acc   ")):
            row = f"{arm:12s}  {label}    "
            for ir in infill_rates:
                stats = r["infill"].get(ir)
                v = stats[key] if isinstance(stats, dict) else None
                row += f"  {_f(v)}"
            print(row)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
