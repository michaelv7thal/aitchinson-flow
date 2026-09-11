"""E2e (P1) — hyperparameter ablations: smoothing ε and the OOD feature-time t.

Two small single-variable sweeps (Prioritized experiment P5):

  --mode eps : the label-smoothing ε (cfg.transformation.label_smoothing) sets
               the embedded scale / simplex geometry. Train a tiny model per ε
               and report generation KL_uni/KL_bi. Expectation: ε too small →
               degenerate CLR (log blows up); ε too large → washed-out one-hots.

  --mode t   : FM/denoiser features are *time-dependent*; the OOD head reads
               them at a fixed timestep. Sweep the feature-extraction t and
               report the **shuffle**-axis AUROC of the DFM denoiser-NLL score
               (clean vs histogram-preserving permutation). Expectation: a
               mid-band t is most discriminative; t→1 (near clean) and t→0
               (pure noise) are weak.

Usage:
  python scripts/ablate_hyperparams.py --mode eps --epochs 5
  python scripts/ablate_hyperparams.py --mode t --ckpt runs/.../DFM/epoch_final.pt
  python scripts/ablate_hyperparams.py --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.append(str(_ROOT))

import torch  # noqa: E402

import aitchinson_flow.models  # noqa: E402,F401  populate REGISTRY
from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.models.factory import build_model  # noqa: E402
from aitchinson_flow.training import build_training_datamodule, seed_all, fit  # noqa: E402
from scripts.eval_full import ngram_kl, unigram_kl  # noqa: E402

DEFAULT_EPS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
DEFAULT_TS = (0.1, 0.25, 0.5, 0.75, 0.9)


def _auroc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """P(score(pos) > score(neg)); pos = OOD/corrupted, neg = clean."""
    p, n = pos.flatten(), neg.flatten()
    if p.numel() == 0 or n.numel() == 0:
        return float("nan")
    comp = (p[:, None] > n[None, :]).float().mean() + 0.5 * (p[:, None] == n[None, :]).float().mean()
    return float(comp)


def _shuffle(ids: torch.Tensor, seed: int) -> torch.Tensor:
    """Histogram-preserving permutation of each row (the discriminating axis)."""
    g = torch.Generator().manual_seed(seed)
    out = ids.clone()
    for i in range(ids.shape[0]):
        out[i] = ids[i][torch.randperm(ids.shape[1], generator=g)]
    return out


def _tiny_cfg(eps: float, *, model="EqM", d=32, L=12, epochs=1, windows=64) -> Config:
    cfg = Config()
    cfg.training = replace(cfg.training, model_name=model, L=L, K=27, B=8,
                           epochs=epochs, eval_every=epochs + 1,
                           sample_eval_every=None, checkpoint_every=epochs)
    cfg.transformer = replace(cfg.transformer, d_model=d, nhead=2, num_layers=2)
    cfg.text8_dataset = replace(cfg.text8_dataset, L=L, K=27,
                                max_train_windows=windows, max_eval_windows=32)
    cfg.transformation = replace(cfg.transformation, label_smoothing=eps)
    cfg.wandb = replace(cfg.wandb, enabled=False, mode="disabled")
    return cfg


def ablate_eps(epsilons, *, epochs, d, L, windows, n_sample, seed) -> dict:
    out = {}
    for eps in epsilons:
        seed_all(seed)
        cfg = _tiny_cfg(eps, d=d, L=L, epochs=epochs, windows=windows)
        dm, _ = build_training_datamodule(cfg)
        model = fit(cfg=cfg, datamodule=dm, wandb_logger=None)
        model.eval()
        with torch.no_grad():
            x = model.sample(n_sample, cfg.text8_dataset.L, max_steps=32)
            gen = model.decode_to_logprobs(x).argmax(-1).cpu() \
                if hasattr(model, "decode_to_logprobs") else x.cpu().long()
        train_ids = dm.splits.train.long()
        ku, *_ = unigram_kl(gen, train_ids, K=27)
        kb = ngram_kl(gen, train_ids, 2, K=27)
        out[f"{eps:g}"] = {"KL_uni": float(ku), "KL_bi": float(kb)}
        print(f"  ε={eps:g}  KL_uni={ku:.4f}  KL_bi={kb:.4f}")
    return out


@torch.no_grad()
def ablate_t(model, cfg, ids: torch.Tensor, ts, *, seed) -> dict:
    """DFM denoiser-NLL OOD AUROC (clean vs shuffle) as a function of the
    feature-extraction timestep t. The denoiser p_{1|t}(x_t) is read at the
    corrupted input x_t drawn at t; score = per-seq NLL of the placed token."""
    assert hasattr(model, "forward") and hasattr(model, "_corrupt"), \
        "t-mode needs a DFM-style denoiser (forward + _corrupt)"
    shuf = _shuffle(ids, seed)
    out = {}
    for t in ts:
        tb = torch.full((ids.shape[0],), float(t))

        def _nll(x):
            xt = model._corrupt(x, tb)
            logits = model.forward(xt, tb)
            lp = torch.log_softmax(logits, -1)
            return -lp.gather(-1, x.long().unsqueeze(-1)).squeeze(-1).mean(-1)  # (B,)
        clean_s, shuf_s = _nll(ids), _nll(shuf)
        out[f"{t:g}"] = {"shuffle_auroc": _auroc(shuf_s, clean_s)}
        print(f"  t={t:g}  shuffle_auroc={out[f'{t:g}']['shuffle_auroc']:.3f}")
    return out


def _smoke() -> None:
    res_eps = ablate_eps((1e-4, 1e-2), epochs=1, d=32, L=12, windows=32,
                         n_sample=16, seed=0)
    assert set(res_eps) == {"0.0001", "0.01"}, res_eps
    # t-mode on a tiny DFM
    cfg = _tiny_cfg(1e-4, model="DFM", L=12, windows=32)
    model = build_model(cfg).eval()
    ids = torch.randint(0, 27, (16, 12))
    res_t = ablate_t(model, cfg, ids, (0.25, 0.5, 0.75), seed=0)
    assert set(res_t) == {"0.25", "0.5", "0.75"}, res_t
    for v in res_t.values():
        assert 0.0 <= v["shuffle_auroc"] <= 1.0001
    print("OK ablate_hyperparams smoke: eps + t sweeps both valid")


def _load(ckpt: str):
    from scripts.eval_full import _config_from_payload
    p = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = _config_from_payload(p)
    model = build_model(cfg)
    model.load_state_dict(p["model_state_dict"])
    model.eval()
    return model, cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["eps", "t"], default="eps")
    ap.add_argument("--ckpt", type=str, default=None, help="DFM ckpt for --mode t")
    ap.add_argument("--epsilons", type=str, default=None)
    ap.add_argument("--ts", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--L", type=int, default=40)
    ap.add_argument("--windows", type=int, default=2000)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        _smoke()
        return
    if args.mode == "eps":
        eps = ([float(x) for x in args.epsilons.split(",")] if args.epsilons
               else list(DEFAULT_EPS))
        res = {"mode": "eps", "results": ablate_eps(
            eps, epochs=args.epochs, d=args.d, L=args.L, windows=args.windows,
            n_sample=args.n, seed=args.seed)}
    else:
        if not args.ckpt:
            ap.error("--mode t requires --ckpt (a DFM checkpoint)")
        model, cfg = _load(args.ckpt)
        ids = torch.randint(0, cfg.text8_dataset.K, (args.n, cfg.text8_dataset.L))
        ts = ([float(x) for x in args.ts.split(",")] if args.ts else list(DEFAULT_TS))
        res = {"mode": "t", "ckpt": args.ckpt,
               "results": ablate_t(model, cfg, ids, ts, seed=args.seed)}
    out = args.out or f"ablate_hyperparams_{args.mode}.json"
    Path(out).write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
