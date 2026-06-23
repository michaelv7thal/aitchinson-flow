"""Recovery diagnostic — test EqMLatent's self-healing property.

Two evaluation modes given a trained EqMLatent checkpoint:

  (A) Unconditional generation: sample from N(0, σ²I) → run sampler → decode.
      Headline KL_uni / KL_bi / KL_tri / H_ratio against the training corpus
      tells whether the field generates text-like sequences.

  (B) Recovery from perturbation: take held-out test sequences, encode →
      add Gaussian perturbation of varying scale α ∈ {0.1, 0.3, 0.5, 1.0} ·
      embed_norm, run sampler, decode. Report token-level recovery accuracy.
      A healthy EqM should:
        * α small  → near-100 % recovery (samples descend back to original)
        * α large  → recovery degrades; samples are still valid text but
          may diverge from the source.

The "diverge but stay valid" failure mode is interesting: it means the
field has multiple basins (the desired EBM property) and the trajectory
reached a different basin than the original. KL of those samples vs corpus
should still be low.

Usage:
    python scripts/recovery_check.py \\
        --ckpt runs/latent_skipgram_d128_timecond_euler_ep20/epoch_final.pt \\
        --alphas 0.1,0.3,0.5,1.0 --n 256
"""

from __future__ import annotations

import argparse
import json
import math
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
from aitchinson_flow.data.char_window_dataset import CHAR2ID  # noqa: E402
from aitchinson_flow.models import build_model  # noqa: E402
from aitchinson_flow.training import build_training_datamodule  # noqa: E402
from scripts.eval_full import _config_from_payload  # noqa: E402

ALPHABET = "".join(sorted(CHAR2ID, key=CHAR2ID.__getitem__))


def _ngram_counts_flat(ids2d: torch.Tensor, K: int, n: int) -> torch.Tensor:
    L = ids2d.shape[1]
    if L < n:
        return torch.zeros(K**n)
    idx = torch.zeros(ids2d.shape[0], L - n + 1, dtype=torch.long)
    for i in range(n):
        idx = idx + ids2d[:, i : L - n + 1 + i].long() * (K ** (n - 1 - i))
    counts = torch.zeros(K**n)
    counts.scatter_add_(0, idx.reshape(-1), torch.ones(idx.numel()))
    return counts


def _kl_smoothed(gen: torch.Tensor, ref: torch.Tensor) -> float:
    smoothing = 1e-6
    Ksize = gen.numel()
    gp = (gen + smoothing) / (gen.sum() + Ksize * smoothing)
    rp = (ref + smoothing) / (ref.sum() + Ksize * smoothing)
    return float((gp * (gp.log() - rp.log())).sum())


def _score_stats(model, x: torch.Tensor, *, curv_samples: int = 4) -> dict:
    """Compute UQ scalars on a batch of latents: energy g(x), ‖∇g(x)‖, and a
    Hutchinson estimate of tr(∇²g(x)). Reports per-sample summary stats
    (mean / std / min / max) for each. Skips silently if the model doesn't
    expose the score_* API (older checkpoints, simplex EqM, etc.). Curvature
    is the only expensive piece; if it OOMs we drop just that field and
    emit the cheaper energy/grad_norm anyway."""
    if not hasattr(model, "score_energy"):
        return {}

    def _summary(t: torch.Tensor) -> dict:
        return {
            "mean": float(t.mean().item()),
            "std": float(t.std().item()) if t.numel() > 1 else 0.0,
            "min": float(t.min().item()),
            "max": float(t.max().item()),
        }

    energy = model.score_energy(x).cpu()
    grad_norm = model.score_gradient_norm(x).cpu()
    out = {"energy": _summary(energy), "grad_norm": _summary(grad_norm)}
    try:
        curvature = model.score_curvature(x, n_samples=curv_samples).cpu()
        out["curvature"] = _summary(curvature)
    except torch.cuda.OutOfMemoryError as e:
        print(f"  [warn] score_curvature OOM ({e}); skipping curvature")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except RuntimeError as e:
        # NVML / allocator-internal asserts from PyTorch surface as RuntimeError
        # in this environment; treat them as transient OOM-class failures.
        msg = str(e)
        if "CUDA" in msg or "NVML" in msg or "out of memory" in msg.lower():
            print(f"  [warn] score_curvature CUDA error: {msg}; skipping")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            raise
    return out


def _is_categorical_denoiser(model) -> bool:
    """True for DFM / DirichletFM — categorical denoisers whose ``sample``
    returns token IDs directly and which expose ``forward(x_t, t) -> logits``
    instead of the EqM-family ``decode_to_logprobs``. EqM/EqMLatent/SFLM keep
    ``decode_to_logprobs`` so they take the legacy (latent) path."""
    return not hasattr(model, "decode_to_logprobs") and hasattr(model, "forward")


def _dfm_uncond_ids(model, n: int, L: int, steps: int) -> torch.Tensor:
    """Unconditional sample for a categorical denoiser. ``sample`` already
    returns argmax token IDs; route ``steps`` to ``nfe`` (DFM) which is the
    NFE budget for both DFM and DirichletFM."""
    return model.sample(n, L, nfe=steps).cpu()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--alphas", type=str, default="0.1,0.3,0.5,0.7,1.0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument(
        "--sampler",
        type=str,
        default=None,
        help="Override cfg.eqm.sampler at load time. e.g. nag / euler / sde. "
             "Only applies to EqM-class models that dispatch on cfg.eqm.sampler.",
    )
    ap.add_argument(
        "--sde-alpha",
        type=float,
        default=None,
        help="Override cfg.eqm.sde_alpha (Langevin diffusion coefficient). "
             "Has no effect for non-SDE samplers.",
    )
    args = ap.parse_args()

    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = _config_from_payload(payload)
    # Optional per-run sampler override (no retraining needed). Applies to
    # EqM-class models whose ``sample()`` reads cfg.eqm.sampler.
    if args.sampler is not None or args.sde_alpha is not None:
        from dataclasses import replace as _replace
        eqm_overrides: dict = {}
        if args.sampler is not None:
            eqm_overrides["sampler"] = args.sampler
        if args.sde_alpha is not None:
            eqm_overrides["sde_alpha"] = float(args.sde_alpha)
        cfg.eqm = _replace(cfg.eqm, **eqm_overrides)
        print(f"[sampler override] {eqm_overrides}")
    device = cfg.training.device
    model = build_model(cfg).to(device)
    state = payload["model_state_dict"] if "model_state_dict" in payload else payload
    model.load_state_dict(state)
    model.eval()

    dm, _ = build_training_datamodule(cfg)
    train_ids = dm.splits.train.long()
    val_ids = dm.splits.val.long()
    K = cfg.text8_dataset.K
    L = cfg.text8_dataset.L
    sigma = cfg.eqm.source_sigma
    if hasattr(model, "embed"):
        embed = model.embed.weight.detach().to(device)
        embed_norm = embed.norm(dim=-1).mean().item()
    elif hasattr(model, "encode"):
        # Contextual encoder (e.g. EqMAE): perturbation scale comes from the
        # actual latent-norm distribution of held-out tokens — there is no
        # per-token embedding row to read.
        with torch.no_grad():
            sample_z = model.encode(val_ids[:64].to(device))
        embed_norm = sample_z.norm(dim=-1).mean().item()
        embed = None
    else:
        # Simplex EqM: use the data-feature L2 norm as the perturbation scale.
        # Compute on a held-out batch.
        from aitchinson_flow.data.transforms import token_ids_to_features

        ls = cfg.transformation.label_smoothing
        sample_z = token_ids_to_features(val_ids[:64].to(device), K, label_smoothing=ls)
        embed_norm = sample_z.norm(dim=-1).mean().item()
        embed = None

    ref_uni = _ngram_counts_flat(train_ids, K, 1)
    ref_bi = _ngram_counts_flat(train_ids, K, 2)
    ref_tri = _ngram_counts_flat(train_ids, K, 3)

    rows: list[dict] = []

    # ─── (0) Clean reference — calibration baseline for the UQ scalars ────
    # Encode held-out val tokens and score them: this is what "in-distribution"
    # looks like under the trained energy. Generated/recovered samples will be
    # compared against these reference statistics.
    if hasattr(model, "score_energy"):
        with torch.no_grad():
            z_ref = (
                model.encode(val_ids[: args.n].to(device))
                if hasattr(model, "encode")
                else None
            )
        if z_ref is not None:
            ref_scores = _score_stats(model, z_ref)
            print("=== Score reference (clean encoded val) ===")
            print(
                f"  energy:    mean={ref_scores['energy']['mean']:.3f}  std={ref_scores['energy']['std']:.3f}"
            )
            print(
                f"  grad_norm: mean={ref_scores['grad_norm']['mean']:.3f}  std={ref_scores['grad_norm']['std']:.3f}"
            )
            print(
                f"  curvature: mean={ref_scores['curvature']['mean']:.3f}  std={ref_scores['curvature']['std']:.3f}"
            )
            rows.append({"mode": "score_reference", **ref_scores})

    # ─── (A) Unconditional ────────────────────────────────────────────────
    categorical = _is_categorical_denoiser(model)
    # SFM (Fisher-Rao √μ-sphere FM) carries decode_to_logprobs for API parity
    # but its sample() returns ids (like DirichletFM) and it has no encode(), so
    # it takes neither the categorical-denoiser nor the EqM-family latent path;
    # it gets its own ids-sampling + Fisher-sphere partial-path recovery below.
    is_sfm = cfg.training.model_name == "SFM"
    torch.manual_seed(args.seed)
    print("=== Unconditional generation ===")
    if categorical or is_sfm:
        # DFM / DirichletFM / SFM: sample() returns token IDs directly; no latent
        # x to score and no usable decode_to_logprobs path.
        with torch.no_grad():
            ids = _dfm_uncond_ids(model, args.n, L, args.steps)
        uncond_scores = {}
    else:
        with torch.no_grad():
            x = model.sample(args.n, L, max_steps=args.steps)
            log_probs = model.decode_to_logprobs(x)
            ids = log_probs.argmax(-1).cpu()
        uncond_scores = _score_stats(model, x)

    gen_uni = _ngram_counts_flat(ids, K, 1)
    gen_bi = _ngram_counts_flat(ids, K, 2)
    gen_tri = _ngram_counts_flat(ids, K, 3)
    kl_u = _kl_smoothed(gen_uni, ref_uni)
    kl_b = _kl_smoothed(gen_bi, ref_bi)
    kl_t = _kl_smoothed(gen_tri, ref_tri)
    p = (gen_uni + 1e-9) / (gen_uni.sum() + K * 1e-9)
    H_gen = float(-(p * p.log()).sum())
    print(
        f"  KL_uni={kl_u:.4f}  KL_bi={kl_b:.4f}  KL_tri={kl_t:.4f}  H_gen={H_gen:.4f}"
    )
    print(f"  sample[0]: {''.join(ALPHABET[int(i)] for i in ids[0])!r}")
    print(f"  sample[1]: {''.join(ALPHABET[int(i)] for i in ids[1])!r}")
    if uncond_scores:
        print(
            f"  energy:    mean={uncond_scores['energy']['mean']:.3f}  std={uncond_scores['energy']['std']:.3f}"
        )
        print(
            f"  grad_norm: mean={uncond_scores['grad_norm']['mean']:.3f}  std={uncond_scores['grad_norm']['std']:.3f}"
        )
        print(
            f"  curvature: mean={uncond_scores['curvature']['mean']:.3f}  std={uncond_scores['curvature']['std']:.3f}"
        )
    rows.append(
        {
            "mode": "unconditional",
            "alpha": None,
            "KL_uni": kl_u,
            "KL_bi": kl_b,
            "KL_tri": kl_t,
            "H_gen": H_gen,
            "samples": [
                "".join(ALPHABET[int(i)] for i in ids[s]) for s in range(min(4, args.n))
            ],
            **uncond_scores,
        }
    )

    # ─── (B) Recovery ─────────────────────────────────────────────────────
    print("\n=== Recovery from perturbation ===")
    print(f"  embed_norm_mean={embed_norm:.4f}  σ_source={sigma:.4f}")
    # Headline metric is Δ = token_acc − token_acc_perturbed (the recovery work
    # done by the field/denoiser, not the raw accuracy). KL is secondary.
    print(
        f"{'α':>5} {'σ_perturb':>10} {'Δ':>8} {'tok_acc':>8} {'tok_acc_pt':>10} "
        f"{'KL_bi':>8}  sample[0]: gt / pt / rc"
    )
    val_pick = val_ids[: args.n].to(device)
    # Encode token_ids → data tensor. EqMLatent has model.encode; simplex EqM
    # uses CLR features via token_ids_to_features. Categorical denoisers
    # (DFM/DirichletFM) drive recovery from token_ids directly (no latent).
    if not categorical and not is_sfm:
        if hasattr(model, "encode"):
            z_clean = model.encode(val_pick)
        else:
            from aitchinson_flow.data.transforms import token_ids_to_features

            ls = cfg.transformation.label_smoothing
            z_clean = token_ids_to_features(val_pick, K, label_smoothing=ls)
    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    is_dirichlet = categorical and hasattr(model, "_sample_xt")
    for alpha in alphas:
        torch.manual_seed(args.seed + int(alpha * 1000))
        rc_scores: dict = {}
        if categorical and is_dirichlet:
            # ── DirichletFM: partial-path recovery ───────────────────────
            # α maps to a start-time on the native Dirichlet path:
            # t_start = t_max − α·(t_max−1)  (α=0 ⇔ clean@t_max, α=1 ⇔ noisy@1).
            # x_init = Dir(β(t_start, clean_ids)) is the perturbed simplex point;
            # integrate forward to recover. token_acc_pt is its argmax (the
            # pre-recovery state), token_acc is the recovered sample.
            t_max = float(model.t_max)
            t_start = max(1.0, t_max - float(alpha) * (t_max - 1.0))
            sig_perturb = float(alpha)
            t_b = torch.full((args.n,), t_start, device=device, dtype=torch.float32)
            with torch.no_grad():
                x_init = model._sample_xt(val_pick, t_b)
                ids_pt = x_init.argmax(dim=-1).cpu()
                ids = model.sample(
                    args.n, L, x_init=x_init, t_start=t_start, nfe=args.steps
                ).cpu()
        elif categorical:
            # ── DFM: uniform-corruption recovery ─────────────────────────
            # DFM has no x_init/partial-path sampler; recover by corrupting the
            # clean tokens at keep-prob κ=1−α (α=0 ⇔ clean, α=1 ⇔ pure noise),
            # then running the denoiser ``forward`` once at the matching flow-
            # time and taking argmax. ids_pt is the corrupted state x_t (the
            # pre-recovery accuracy); ids is the denoiser argmax (recovered).
            kappa = max(0.0, min(1.0, 1.0 - float(alpha)))
            sig_perturb = float(alpha)
            kappa_b = torch.full((args.n,), kappa, device=device, dtype=torch.float32)
            with torch.no_grad():
                x_t = model._corrupt_kappa(val_pick, kappa_b)
                ids_pt = x_t.cpu()
                # Invert the schedule κ=κ(t) to feed the matching flow-time:
                # quadratic κ=t² ⇒ t=√κ; linear κ=t ⇒ t=κ.
                if cfg.dfm.kappa_schedule == "quadratic":
                    t_in = math.sqrt(kappa)
                else:
                    t_in = kappa
                t_b = torch.full((args.n,), t_in, device=device, dtype=torch.float32)
                logits = model.forward(x_t, t_b)
                ids = logits.argmax(dim=-1).cpu()
        elif is_sfm:
            # ── SFM: partial-path recovery on the Fisher √μ-sphere ────────
            # Mirror eval_generation.py: x_init is the point a "data-ness"
            # fraction f=1−α along the geodesic from a uniform-sphere noise
            # point x0 toward the data vertex x1=√(one-hot); integrate the ODE
            # from t_start=f to 1. ids_pt is the argmax of the perturbed init's
            # μ=x² (pre-recovery state), ids is the integrated recovery.
            from aitchinson_flow.models.sfm import (
                _exp_map as _sfm_exp,
                _log_map as _sfm_log,
                _normalize as _sfm_norm,
            )

            sig_perturb = float(alpha)
            f = 1.0 - float(alpha)
            with torch.no_grad():
                x1 = model._sphere_target(val_pick)
                x0 = _sfm_norm(torch.randn_like(x1))
                x_init = _sfm_exp(x0, f * _sfm_log(x0, x1))
                ids_pt = (x_init * x_init).argmax(dim=-1).cpu()
                ids = model.sample(
                    args.n, L, x_init=x_init, t_start=f, nfe=args.steps
                ).cpu()
        else:
            # ── EqM / EqMLatent / SFLM (identity-path latents) ──────────
            sig_perturb = alpha * embed_norm
            z_init = z_clean + sig_perturb * torch.randn_like(z_clean)

            # Decode the perturbed (sampler-input) latents too. Argmax-decoding
            # `z_init` shows what tokens the noisy embedding nearest-neighbours.
            with torch.no_grad():
                log_probs_pt = model.decode_to_logprobs(z_init)
                ids_pt = log_probs_pt.argmax(-1).cpu()

                # NCSN-style annealed-Langevin samplers (ScoreDSM/EqMDSM) accept a
                # ``start_sigma`` kwarg that restricts the σ-ladder to values ≤
                # the perturbation magnitude — without it the sampler re-noises
                # ``z_init`` all the way back up to σ_max before annealing,
                # which destroys the recovery signal at small α. EqM/EqMAE
                # samplers don't accept this kwarg; fall back to the legacy call.
                import inspect as _inspect
                _sig = _inspect.signature(model.sample)
                sample_kwargs = {"x_init": z_init, "max_steps": args.steps}
                if "start_sigma" in _sig.parameters:
                    sample_kwargs["start_sigma"] = float(sig_perturb)
                x = model.sample(args.n, L, **sample_kwargs)
                log_probs = model.decode_to_logprobs(x)
                ids = log_probs.argmax(-1).cpu()
            rc_scores = _score_stats(model, x)

        gen_bi = _ngram_counts_flat(ids, K, 2)
        kl_b = _kl_smoothed(gen_bi, ref_bi)
        token_acc = float((ids == val_pick.cpu()).float().mean())
        token_acc_pt = float((ids_pt == val_pick.cpu()).float().mean())
        delta = token_acc - token_acc_pt
        sample_gt = "".join(ALPHABET[int(i)] for i in val_pick[0].cpu())
        sample_pt = "".join(ALPHABET[int(i)] for i in ids_pt[0])
        sample_rc = "".join(ALPHABET[int(i)] for i in ids[0])
        print(
            f"{alpha:>5.2f} {sig_perturb:>10.3f} {delta:>8.4f} {token_acc:>8.4f} "
            f"{token_acc_pt:>10.4f} {kl_b:>8.4f}  "
            f"gt={sample_gt!r}\n{'':>45}pt={sample_pt!r}"
            f"\n{'':>45}rc={sample_rc!r}"
        )
        rows.append(
            {
                "mode": "recovery",
                "alpha": alpha,
                "sigma_perturb": sig_perturb,
                "KL_bi": kl_b,
                "token_acc": token_acc,
                "token_acc_perturbed": token_acc_pt,
                "delta": delta,
                "gt_sample0": sample_gt,
                "perturbed_sample0": sample_pt,
                "rc_sample0": sample_rc,
                **rc_scores,
            }
        )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"rows": rows}, indent=2))
        print(f"\nWrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
