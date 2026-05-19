"""Stage 2 (sequential, hinge-trained SVGP) for DirichletFMSvgp.

Loads a Stage-1 DFM checkpoint, freezes the backbone + DFM head, and
trains pooler + SVGP via a contrastive energy hinge using
batch['token_ids_invalid'] (from CorruptingCollate) as the negative
distribution. SVGP latent mean is the "energy"; the kernel-collapse
failure mode at init is avoided by re-seeding inducing points,
standardisation buffers, and lengthscale from a warmup pass.

After hinge training, also writes svgp_eval.json with AUROC(std) and
AUROC(-prob) on val × {scrambled, random-simplex} OOD, plus the per-step
history.

Usage:
    python scripts/fit_dfm_svgp_hinge.py \\
        --ckpt runs/dfm_svgp_poc/epoch_final.pt \\
        --n-epochs 5 --lr 1e-3 --eval-n 500
"""

from __future__ import annotations

import argparse
import json
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

import aitchinson_flow.models  # noqa: E402,F401
from aitchinson_flow.models import build_model  # noqa: E402
from aitchinson_flow.training import build_training_datamodule  # noqa: E402
from scripts.eval_full import _config_from_payload  # noqa: E402


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if (labels == 1).sum() < 2 or (labels == 0).sum() < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--n-epochs", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--margin", type=float, default=None,
                    help="override cfg.dfm_svgp.margin_energy")
    ap.add_argument("--t-eval", type=float, default=None,
                    help="override cfg.dfm_svgp.t_eval")
    ap.add_argument("--no-pooler", action="store_true",
                    help="freeze the pooler too (SVGP-only training)")
    ap.add_argument("--eval-n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--d-embed", type=int, default=None,
                    help="override cfg.dfm_svgp.d_embed; if it differs from "
                         "the checkpoint's d_embed, pooler/SVGP/energy_head "
                         "are reinitialised (backbone+DFMHead are kept).")
    ap.add_argument("--lengthscale", type=float, default=None,
                    help="initial Matern-5/2 lengthscale at start of hinge "
                         "training (default √d_embed). Use a smaller value "
                         "(e.g. 1.0) with small d_embed to get input-dependent "
                         "std for OOD detection.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir or Path(args.ckpt).parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = _config_from_payload(payload)
    # Optional CLI overrides on dfm_svgp section before model build.
    from dataclasses import replace as _replace
    if args.d_embed is not None:
        cfg.dfm_svgp = _replace(cfg.dfm_svgp, d_embed=int(args.d_embed))
    device = cfg.training.device
    torch.manual_seed(args.seed)

    model = build_model(cfg).to(device)
    state = payload["model_state_dict"] if "model_state_dict" in payload else payload
    # If d_embed differs from the saved checkpoint, drop pooler/SVGP/
    # energy_head keys so the rebuild gets a fresh init at the new size.
    if args.d_embed is not None:
        drop = tuple(k for k in state if k.startswith(("pooler.", "svgp.", "energy_head."))
                     and any(d in (args.d_embed,) for d in []))  # placeholder for shape check
    drop_prefixes = ("pooler.", "svgp.", "energy_head.")
    filtered_state = {}
    for k, v in state.items():
        if k.startswith(drop_prefixes):
            # Check size mismatch against current model param/buffer
            try:
                target = dict(model.state_dict())[k]
                if target.shape != v.shape:
                    if args.d_embed is not None:
                        continue  # silent drop in expected re-init case
                    raise RuntimeError(f"unexpected shape mismatch on {k}: "
                                       f"{tuple(v.shape)} vs {tuple(target.shape)}")
            except KeyError:
                continue
        filtered_state[k] = v
    model.load_state_dict(filtered_state, strict=False)

    dm, _ = build_training_datamodule(cfg)
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader() or train_loader

    print(f"[stage 2 hinge] training SVGP via contrastive hinge")
    print(f"  ckpt:      {args.ckpt}")
    print(f"  n_epochs:  {args.n_epochs}")
    print(f"  lr:        {args.lr}")
    print(f"  t_eval:    {args.t_eval if args.t_eval else cfg.dfm_svgp.t_eval}")
    print(f"  margin:    {args.margin if args.margin else cfg.dfm_svgp.margin_energy}")
    print(f"  train_pooler: {not args.no_pooler}")
    summary = model.fit_svgp_hinge(
        train_loader,
        t_eval=args.t_eval,
        n_epochs=args.n_epochs,
        max_steps=args.max_steps,
        lr=args.lr,
        margin_energy=args.margin,
        train_pooler=not args.no_pooler,
        verbose=True,
        lengthscale_init=args.lengthscale,
    )
    print(f"[stage 2 hinge] done. n_steps={summary['n_steps']}  "
          f"final_hinge={summary['final_hinge']:.4f}")

    # Save weights.
    torch.save(model.state_dict(), out_dir / "model_with_svgp_hinge.pt")

    # --- Eval: AUROC ---------------------------------------------------
    t_eval = float(args.t_eval if args.t_eval else cfg.dfm_svgp.t_eval)
    pos_std_all, pos_prob_all = [], []
    n_eval = 0
    for batch in val_loader:
        tok = batch["token_ids"].to(device).long()
        s = model.ood_score(tok, t_eval=t_eval)
        pos_std_all.append(s["std"].cpu()); pos_prob_all.append(s["prob"].cpu())
        n_eval += tok.shape[0]
        if n_eval >= args.eval_n:
            break
    pos_std = torch.cat(pos_std_all)[:args.eval_n]
    pos_prob = torch.cat(pos_prob_all)[:args.eval_n]
    print(f"  positive  mean(std)={pos_std.mean():.4f}  mean(prob)={pos_prob.mean():.3f}")

    rows = []
    rows.append({"split": "positive_val", "n": int(pos_std.numel()),
                 "mean_std": float(pos_std.mean()),
                 "mean_prob": float(pos_prob.mean())})
    from torch.distributions import Dirichlet as _D
    for neg_strategy in ("scrambled", "random_simplex"):
        ood_std, ood_prob = [], []
        n = 0
        for batch in val_loader:
            tok = batch["token_ids"].to(device).long()
            B, L = tok.shape
            t = torch.full((B,), t_eval, device=device)
            if neg_strategy == "scrambled":
                perms = torch.argsort(torch.rand(B, L, device=device), dim=-1)
                tok_o = tok.gather(1, perms)
                z = model.pool_features(tok_o, t)
            else:
                K = model.K
                x_t = _D(torch.ones(K, device=device)).sample((B, L))
                h = model.get_hidden_states(x_t, t)
                z = model.pooler(h)
            prob, mean, std = model.svgp(z)
            ood_std.append(std.cpu()); ood_prob.append(prob.cpu())
            n += B
            if n >= args.eval_n:
                break
        ood_std_v = torch.cat(ood_std)[:args.eval_n]
        ood_prob_v = torch.cat(ood_prob)[:args.eval_n]
        all_std = torch.cat([pos_std, ood_std_v]).numpy()
        all_prob = torch.cat([pos_prob, ood_prob_v]).numpy()
        labels = np.concatenate([np.zeros(pos_std.numel()), np.ones(ood_std_v.numel())])
        auroc_std = _auroc(all_std, labels)
        auroc_negprob = _auroc(-all_prob, labels)
        print(f"  [{neg_strategy:>15}] mean(std)={ood_std_v.mean():.4f}  "
              f"AUROC(std)={auroc_std:.4f}  AUROC(-prob)={auroc_negprob:.4f}")
        rows.append({"split": f"ood_{neg_strategy}", "n": int(ood_std_v.numel()),
                     "mean_std": float(ood_std_v.mean()),
                     "mean_prob": float(ood_prob_v.mean()),
                     "auroc_std": auroc_std,
                     "auroc_neg_prob": auroc_negprob,
                     # Hinge trains E_invalid > E_clean → prob > 0.5 for OOD,
                     # so `prob` itself is the natural OOD score; report both.
                     "auroc_prob_as_ood": 1.0 - auroc_negprob})

    out_path = out_dir / "svgp_hinge_eval.json"
    out_path.write_text(json.dumps(
        {"hinge_summary": summary, "rows": rows}, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
