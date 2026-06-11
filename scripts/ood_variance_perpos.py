"""Per-position one-class VARIANCE OOD detector on DirichletFM (Dirichlet FM).

An *unsupervised* alternative to the energy-hinge SVGP. Instead of a trained
discriminative mean, fit a one-class SVGP on IN-DISTRIBUTION per-position
DirichletFM features (regress-to-constant, ARD Matern-5/2 kernel) and use the
predictive LATENT variance sigma^2(z) as the OOD score:

  * in-distribution token  -> feature near the inducing points -> LOW variance
  * OOD token (replaced, or shuffled into a wrong context)     -> off-manifold
                                                               -> HIGH variance

Per-position (no pooling) so a few corrupted tokens are not averaged away and we
get token-level localization. Sequence score = aggregate (max / mean) of the
per-position variance. Also reports a token-level localization AUROC (corrupted
vs clean POSITIONS *within* corrupted sequences).

Detector name: "PerPosVarGP" — deliberately distinct from the energy-hinge
detector ("HingeSVGP", scripts/fit_dfm_svgp_hinge.py). Different score (variance
vs trained mean/prob), different objective (one-class vs discriminative hinge),
different outputs (vargp_perpos_sweep.json vs svgp_corruption_sweep.json).

IMPORTANT: DirichletFM == DirichletFlowMatching (Stark et al. 2024). This is NOT
the Discrete-FM 'DFM' arm (DiscreteFlowMatching) — the loader guards against it.

Usage:
    python scripts/ood_variance_perpos.py \
        --ckpt runs/sflm_bench_a100_20g_L256/DirichletFM_ep30_d30k/epoch_final.pt \
        --out  runs/ood_vargp_perpos_dirichletfm/bigger_ep30_d30k/vargp_perpos_sweep.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


def _bootstrap() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))


_bootstrap()

import gpytorch  # noqa: E402
import aitchinson_flow.models  # noqa: E402,F401
from aitchinson_flow.models import build_model  # noqa: E402
from aitchinson_flow.training import build_training_datamodule  # noqa: E402
from aitchinson_flow.data.corruption import (  # noqa: E402
    corrupt_token_ids,
    partially_shuffle_token_ids,
)
from scripts.eval_full import _config_from_payload  # noqa: E402


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if (labels == 1).sum() < 2 or (labels == 0).sum() < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


class _OneClassSVGP(gpytorch.models.ApproximateGP):
    """Minimal SVGP: ConstantMean + ScaleKernel(Matern-5/2, ARD), learnable
    inducing locations. Fit by regressing in-distribution features to a constant
    so the inducing points + ARD lengthscales cover the in-distribution manifold;
    the LATENT predictive variance is then the novelty score."""

    def __init__(self, inducing: torch.Tensor, ard_dims: int,
                 lengthscale_init: float | None = None) -> None:
        from gpytorch.variational import (
            CholeskyVariationalDistribution,
            VariationalStrategy,
        )
        vdist = CholeskyVariationalDistribution(inducing.size(0))
        vstrat = VariationalStrategy(
            self, inducing, vdist, learn_inducing_locations=True
        )
        super().__init__(vstrat)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=ard_dims)
        )
        # Initialise the (ARD) lengthscale near the inter-point distance
        # ~sqrt(feat_dim) so the kernel starts in its SENSITIVE regime. With
        # the gpytorch default (~0.69) every standardised point is ~sqrt(2·d)
        # away -> kernel ~0 everywhere -> constant variance (concentration of
        # measure). Matching ℓ to the scale lets near/far inputs differ.
        if lengthscale_init is not None:
            self.covar_module.base_kernel.lengthscale = float(lengthscale_init)
        self.covar_module.outputscale = 1.0

    def forward(self, x):  # noqa: D401
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def _load_dirichletfm(ckpt: str, device: str):
    """Load a DirichletFM (or DirichletFMSvgp) checkpoint as a DirichletFMSvgp
    wrapper (for get_hidden_states / _sample_xt). Guards against Discrete FM."""
    from dataclasses import replace as _replace
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = _config_from_payload(payload)
    name = cfg.training.model_name
    from_plain = name != "DirichletFMSvgp"
    if from_plain and name != "DirichletFM":
        raise SystemExit(
            f"--ckpt is a '{name}' checkpoint. This detector only runs on "
            f"DirichletFM (DirichletFlowMatching). The 'DFM' arm is Discrete FM "
            f"(DiscreteFlowMatching) — a different model; refusing to proceed."
        )
    if from_plain:
        cfg.training = _replace(cfg.training, model_name="DirichletFMSvgp")
    cfg.training = _replace(cfg.training, device=device)
    model = build_model(cfg).to(device)
    state = payload.get("model_state_dict", payload)
    if from_plain:
        miss, unexp = model.dfm.load_state_dict(state, strict=False)
        if miss or unexp:
            raise SystemExit(
                f"DirichletFM weights did not load cleanly (missing={len(miss)} "
                f"unexpected={len(unexp)}) — not a DirichletFM checkpoint?"
            )
    else:
        model.load_state_dict(state, strict=False)
    model.eval()
    return model, cfg


@torch.no_grad()
def _perpos_feats(model, tok, t_eval, device, chunk=16):
    """token_ids (B, L) -> per-position backbone features (B, L, d_model) at t_eval.

    DETERMINISTIC input: feed the *mean* of Dir(beta(t, tokens)) — i.e.
    beta / sum(beta) — through the backbone, NOT a stochastic
    ``Dirichlet(beta).sample()``. The sampled x_t injects per-position noise
    that swamps the token identity (clean vs corrupted features collapse to the
    same noise — the cause of E_clean==E_invalid). The deterministic mean keeps
    the backbone as a clean text->representation encoder, so corrupted tokens
    produce a genuinely different representation (~3x larger clean-vs-corrupt
    feature shift measured at t=4.5).
    """
    K = model.K
    outs = []
    for i in range(0, tok.shape[0], chunk):
        tb = tok[i:i + chunk].to(device).long()
        B, L = tb.shape
        t = torch.full((B,), float(t_eval), device=device)
        beta = torch.ones(B, L, K, device=device)
        beta.scatter_(-1, tb.unsqueeze(-1), float(t_eval))
        x_t = beta / beta.sum(-1, keepdim=True)          # Dirichlet mean (deterministic)
        h = model.get_hidden_states(x_t, t)              # (B, L, d_model)
        outs.append(h.cpu())
    return torch.cat(outs, dim=0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--t-eval", type=float, default=None,
                    help="Dirichlet path time for feature extraction "
                         "(default cfg.dfm_svgp.t_eval, ~4.5)")
    ap.add_argument("--fit-seqs", type=int, default=256,
                    help="# in-distribution sequences to fit the one-class GP")
    ap.add_argument("--n", type=int, default=256,
                    help="# eval sequences per split (pos + each corruption)")
    ap.add_argument("--inducing", type=int, default=256)
    ap.add_argument("--fit-steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--lengthscale", type=float, default=None,
                    help="ARD lengthscale init (default sqrt(feat_dim) — matched "
                         "to the inter-point distance so the kernel is NOT "
                         "degenerate at init). The key knob vs concentration of "
                         "measure / variance saturation.")
    ap.add_argument("--pca-dim", type=int, default=0,
                    help="0 = raw d_model features + ARD (default). >0 projects "
                         "per-position features to this dim via PCA on the ID "
                         "fit set BEFORE the ARD GP (extra lever vs concentration "
                         "of measure).")
    ap.add_argument("--max-fit-pos", type=int, default=40000,
                    help="cap on # per-position vectors used to fit the GP")
    ap.add_argument("--schemes", type=str, default="replace,shuffle,both")
    ap.add_argument("--rates", type=str, default="0.1,0.3,0.5,0.7,1.0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device0 = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = _load_dirichletfm(args.ckpt, device0)
    device = next(model.parameters()).device
    K = cfg.text8_dataset.K
    t_eval = float(args.t_eval if args.t_eval is not None else cfg.dfm_svgp.t_eval)
    dm, _ = build_training_datamodule(cfg)
    val_loader = dm.val_dataloader() or dm.train_dataloader()

    # ---- collect in-distribution sequences (fit + eval-positive) -----------
    seqs = []
    for batch in val_loader:
        seqs.append(batch["token_ids"].long())
        if sum(s.shape[0] for s in seqs) >= args.fit_seqs + args.n:
            break
    seqs = torch.cat(seqs, dim=0)
    fit_tok = seqs[: args.fit_seqs]
    pos_tok = seqs[args.fit_seqs: args.fit_seqs + args.n]
    print(f"[var-perpos] t_eval={t_eval}  fit_seqs={fit_tok.shape}  "
          f"eval_pos={pos_tok.shape}  pca_dim={args.pca_dim}")

    # ---- per-position features on the fit set ------------------------------
    fit_feats = _perpos_feats(model, fit_tok, t_eval, device)            # (Nf,L,d)
    d_model = fit_feats.shape[-1]
    fit_flat = fit_feats.reshape(-1, d_model)                            # (Nf*L,d)
    if fit_flat.shape[0] > args.max_fit_pos:
        idx = torch.randperm(fit_flat.shape[0])[: args.max_fit_pos]
        fit_flat = fit_flat[idx]

    # standardise (per-dim); optional PCA on the ID fit set
    mu = fit_flat.mean(0)
    sigma = fit_flat.std(0).clamp_min(1e-6)
    Xs = (fit_flat - mu) / sigma
    pca_V = None
    if args.pca_dim and args.pca_dim < d_model:
        # principal axes of the ID per-position cloud (unsupervised)
        U, S, V = torch.pca_lowrank(Xs, q=min(args.pca_dim, Xs.shape[1]))
        pca_V = V[:, : args.pca_dim]                                     # (d, p)
        Xs = Xs @ pca_V
    feat_dim = Xs.shape[1]
    print(f"[var-perpos] fit points={Xs.shape[0]}  feat_dim={feat_dim}  "
          f"d_model={d_model}")

    def _project(h_flat):  # standardise (+pca) a raw per-position block
        z = (h_flat - mu) / sigma
        return z @ pca_V if pca_V is not None else z

    # ---- fit one-class SVGP (regress ID features to 0) ---------------------
    Xs = Xs.to(device)
    perm = torch.randperm(Xs.shape[0], device=device)[: args.inducing]
    ls_init = args.lengthscale if args.lengthscale is not None else math.sqrt(feat_dim)
    print(f"[var-perpos] ARD lengthscale init = {ls_init:.3f}")
    gp = _OneClassSVGP(Xs[perm].clone(), ard_dims=feat_dim,
                       lengthscale_init=ls_init).to(device)
    lik = gpytorch.likelihoods.GaussianLikelihood().to(device)
    gp.train(); lik.train()
    mll = gpytorch.mlls.VariationalELBO(lik, gp, num_data=Xs.shape[0])
    opt = torch.optim.Adam(
        [{"params": gp.parameters()}, {"params": lik.parameters()}], lr=args.lr
    )
    y0 = torch.zeros(Xs.shape[0], device=device)
    bs = 2048
    for step in range(args.fit_steps):
        bidx = torch.randint(0, Xs.shape[0], (bs,), device=device)
        opt.zero_grad()
        loss = -mll(gp(Xs[bidx]), y0[bidx])
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == args.fit_steps - 1:
            print(f"  fit step {step:4d}  elbo_loss={loss.item():.4f}")
    gp.eval(); lik.eval()

    @torch.no_grad()
    def perpos_var(tok, chunk=16):
        """token_ids (B,L) -> per-position latent variance (B,L)."""
        outs = []
        for i in range(0, tok.shape[0], chunk):
            tb = tok[i:i + chunk].to(device).long()
            B, L = tb.shape
            t = torch.full((B,), t_eval, device=device)
            beta = torch.ones(B, L, model.K, device=device)
            beta.scatter_(-1, tb.unsqueeze(-1), float(t_eval))
            x_t = beta / beta.sum(-1, keepdim=True)      # deterministic (match _perpos_feats)
            h = model.get_hidden_states(x_t, t).reshape(B * L, d_model)
            z = _project(h.cpu()).to(device)
            with gpytorch.settings.fast_pred_var():
                var = gp(z).variance.detach().cpu()                      # (B*L,)
            outs.append(var.reshape(B, L))
        return torch.cat(outs, dim=0)

    # ---- score positives + corruption ladder ------------------------------
    pos_var = perpos_var(pos_tok)                                        # (n,L)
    pos_seq_max = pos_var.max(dim=1).values.numpy()
    pos_seq_mean = pos_var.mean(dim=1).numpy()
    print(f"[var-perpos] positive per-pos var: mean={pos_var.mean():.4g} "
          f"seq-max mean={pos_seq_max.mean():.4g}")

    schemes = [s for s in args.schemes.split(",") if s.strip()]
    rates = [float(r) for r in args.rates.split(",") if r.strip()]
    rows = [{"scheme": None, "rate": 0.0, "n": int(pos_tok.shape[0]),
             "mean_var": float(pos_var.mean()),
             "seq_max_mean": float(pos_seq_max.mean())}]

    def _corrupt(tok, scheme, rate, seed):
        if scheme == "replace":
            return corrupt_token_ids(tok, vocab_size=K, corrupt_rate=rate, seed=seed)
        if scheme == "shuffle":
            return partially_shuffle_token_ids(tok, shuffle_rate=rate, seed=seed)
        out = corrupt_token_ids(tok, vocab_size=K, corrupt_rate=rate, seed=seed)
        return partially_shuffle_token_ids(out, shuffle_rate=rate, seed=seed + 1)

    print(f"\n{'scheme':>10} {'rate':>5} {'AUROC_max':>10} {'AUROC_mean':>11} "
          f"{'tokloc_AUROC':>13}")
    for scheme in schemes:
        for r in rates:
            ood_tok = _corrupt(pos_tok.clone(), scheme, r, args.seed + int(1000 * r))
            ood_var = perpos_var(ood_tok)                                # (n,L)
            ood_seq_max = ood_var.max(dim=1).values.numpy()
            ood_seq_mean = ood_var.mean(dim=1).numpy()
            lab = np.concatenate([np.zeros(len(pos_seq_max)), np.ones(len(ood_seq_max))])
            au_max = _auroc(np.concatenate([pos_seq_max, ood_seq_max]), lab)
            au_mean = _auroc(np.concatenate([pos_seq_mean, ood_seq_mean]), lab)
            # token-level localization: within corrupted seqs, do CHANGED
            # positions have higher var than UNCHANGED ones? (replace/both only;
            # shuffle moves tokens so 'changed mask' is position-wise inequality)
            changed = (ood_tok != pos_tok)                               # (n,L) bool
            tokloc = float("nan")
            if changed.any() and (~changed).any():
                pv = ood_var.numpy().reshape(-1)
                cm = changed.numpy().reshape(-1).astype(int)
                tokloc = _auroc(pv, cm)
            print(f"{scheme:>10} {r:>5.2f} {au_max:>10.4f} {au_mean:>11.4f} "
                  f"{tokloc:>13.4f}")
            rows.append({"scheme": scheme, "rate": r, "n": int(ood_tok.shape[0]),
                         "mean_var": float(ood_var.mean()),
                         "seq_max_mean": float(ood_seq_max.mean()),
                         "auroc_var_max": au_max, "auroc_var_mean": au_mean,
                         "auroc_token_localization": tokloc})

    out_path = Path(args.out or (Path(args.ckpt).parent / "vargp_perpos_sweep.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "ckpt": args.ckpt, "t_eval": t_eval, "d_model": d_model,
        "feat_dim": feat_dim, "pca_dim": args.pca_dim,
        "inducing": args.inducing, "fit_steps": args.fit_steps,
        # Distinct detector name vs the energy-hinge SVGP ("HingeSVGP").
        "detector": "PerPosVarGP",
        "detector_long": "per-position one-class variance GP (ARD), variance-as-OOD-score",
        "rows": rows,
    }, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
