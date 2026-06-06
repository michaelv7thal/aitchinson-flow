"""Train the Logit-KL Flow Matching baseline (E5a, the geometry-lever upside).

`LogitKLFlow` (`src/aitchinson_flow/models/logitkl_flow.py`, registered
``"LogitKLFlow"``, config block ``cfg.logitkl``) is the one untested lever that
may close the generation gap (Sevriugov & Oseledets, arXiv:2411.16821): it works
in *logit* space, regresses the **clean logits** ``l_1`` (not a velocity) under
MSE, and samples with a hybrid scheme — deterministic ODE for ``t <
sampler_split_t`` (0.28), stochastic re-noising for ``t >= sampler_split_t``.

Thin trainer that mirrors ``scripts/train_for_sflm_bench.py``: build a Config at
the requested scale, set ``model_name="LogitKLFlow"``, build the datamodule and
``fit()``. Writes ``<out-dir>/epoch_final.pt``.

Usage:
  python scripts/train_logit_kl_flow.py --scale a100_20g_L256 --epochs 20 --seeds 42,43,44
  python scripts/train_logit_kl_flow.py --smoke         # tiny CPU 1-epoch sanity
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.append(str(_ROOT))

import aitchinson_flow.models  # noqa: E402,F401  populate REGISTRY
from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.training import (  # noqa: E402
    build_training_datamodule,
    seed_all,
    fit,
)
from scripts.train_for_sflm_bench import SCALES  # noqa: E402


def _cfg(scale: str, epochs: int, out_dir: str, *, seed: int,
         mtw: int | None = -1, lazy: bool = False) -> Config:
    s = SCALES[scale]
    L = s.get("L", 40)
    batch = s["batch"]
    cfg = Config()
    cfg.training = replace(
        cfg.training,
        model_name="LogitKLFlow",
        epochs=epochs,
        lr=3e-4,
        seed=seed,
        scheduler_warmup_epochs=1,
        cosine_t_max_epochs=epochs,
        B=batch,
        L=L,
        checkpoint_dir=out_dir,
        checkpoint_every=epochs,
        sample_eval_every=None,
        eval_every=epochs + 1,
        eval_bpd=False,
    )
    cfg.transformer = replace(
        cfg.transformer, d_model=s["d_model"], num_layers=s["n_layers"],
        nhead=s["n_heads"],
    )
    cfg.loader_settings = replace(cfg.loader_settings, batch_size=batch)
    cfg.text8_dataset = replace(
        cfg.text8_dataset, L=L, batch_size=batch,
        max_train_windows=(s["max_train_windows"] if mtw == -1 else mtw),
        max_eval_windows=s["max_eval_windows"],
        lazy_features=lazy,
    )
    cfg.wandb = replace(cfg.wandb, enabled=False, mode="disabled")
    return cfg


def _smoke() -> None:
    import torch
    SCALES["_tiny"] = {"d_model": 32, "n_layers": 2, "n_heads": 2, "batch": 4,
                       "max_train_windows": 32, "max_eval_windows": 16, "L": 12}
    import tempfile
    out = tempfile.mkdtemp()
    cfg = _cfg("_tiny", epochs=1, out_dir=out, seed=42)
    seed_all(42)
    dm, _ = build_training_datamodule(cfg)
    fit(cfg=cfg, datamodule=dm, wandb_logger=None)
    ckpt = Path(out) / "epoch_final.pt"
    assert ckpt.exists(), f"no checkpoint written to {ckpt}"
    # confirm it reloads + samples
    from aitchinson_flow.models.factory import build_model
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = build_model(cfg)
    m.load_state_dict(payload["model_state_dict"])
    x = m.sample(2, cfg.text8_dataset.L, nfe=4)
    assert tuple(x.shape) == (2, cfg.text8_dataset.L), x.shape
    print(f"OK train_logit_kl_flow smoke: trained 1 ep, ckpt at {ckpt}, "
          f"sample shape {tuple(x.shape)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--scale", choices=list(SCALES), default="local")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="default: runs/logitkl_<scale>[/seed<seed>]")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default=None,
                    help="comma-separated seeds (overrides --seed)")
    ap.add_argument("--max-train-windows", type=int, default=None)
    ap.add_argument("--full-split", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        _smoke()
        return

    seeds = ([int(x) for x in args.seeds.split(",") if x.strip()]
             if args.seeds else [args.seed])
    mtw = None if args.full_split else (
        args.max_train_windows if args.max_train_windows is not None else -1)
    lazy = bool(args.full_split)
    for seed in seeds:
        base = args.out_dir or f"runs/logitkl_{args.scale}"
        out = base if seed == 42 else f"{base}/seed{seed}"
        ckpt = Path(out) / "epoch_final.pt"
        if ckpt.exists() and not args.force:
            print(f"[skip] seed={seed}: {ckpt} exists (use --force)")
            continue
        Path(out).mkdir(parents=True, exist_ok=True)
        cfg = _cfg(args.scale, args.epochs, out, seed=seed, mtw=mtw, lazy=lazy)
        print(f"=== TRAIN LogitKLFlow [{args.scale}] seed={seed} "
              f"({args.epochs} ep) → {out} ===", flush=True)
        seed_all(seed)
        dm, _ = build_training_datamodule(cfg)
        fit(cfg=cfg, datamodule=dm, wandb_logger=None)
        print(f"=== DONE LogitKLFlow seed={seed} → {out} ===", flush=True)


if __name__ == "__main__":
    main()
