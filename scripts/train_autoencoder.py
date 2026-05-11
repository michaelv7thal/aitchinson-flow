"""Train TextAutoencoder on text8 — pretrain-pass for EqMAE.

Drives the standard ``fit()`` loop with cfg.training.model_name = "TextAE".
The unigram-KL probe is disabled because the AE has no ``sample`` method;
it is not a generator on its own. Trains denoising CE + small latent L2 to
keep ‖z‖ bounded so the downstream EqM can use a sensible σ for x0.

Usage:
    python scripts/train_autoencoder.py \
        --out runs/textae_d64 \
        --d-latent 64 --d-model 256 --num-layers 2 \
        --denoising-sigma 0.1 --latent-l2 1e-3 \
        --epochs 5 --windows 10000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path


def _bootstrap() -> None:
    repo = Path(__file__).resolve().parent.parent
    src = repo / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    if str(repo) not in sys.path:
        sys.path.append(str(repo))


_bootstrap()

import torch  # noqa: E402

import aitchinson_flow.models  # noqa: E402,F401
from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.training import (  # noqa: E402
    build_training_datamodule,
    fit,
    seed_all,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, help="run directory (checkpoints + history)")
    p.add_argument("--d-latent", type=int, default=64)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--denoising-schedule", type=str, default="relative_uniform",
                   choices=("fixed", "relative_uniform"))
    p.add_argument("--denoising-sigma", type=float, default=0.5)
    p.add_argument("--latent-l2", type=float, default=1e-3)
    p.add_argument("--mode", type=str, default="ae", choices=("ae", "vae"),
                   help="ae=deterministic AE, vae=variational AE with KL prior")
    p.add_argument("--vae-beta", type=float, default=0.1,
                   help="KL weight for VAE mode (β-VAE); ignored when --mode=ae")
    p.add_argument("--vae-beta-warmup-epochs", type=int, default=1)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--windows", type=int, default=10_000)
    p.add_argument("--eval-windows", type=int, default=2_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=None,
                   help="window length L (overrides cfg.training.L and "
                        "cfg.text8_dataset.L). For variable-L training this "
                        "should equal L_max (positional embedding size).")
    p.add_argument("--variable-length", action="store_true",
                   help="train with per-sample variable L drawn from "
                        "[--L-min, --L-max]")
    p.add_argument("--L-min", type=int, default=40)
    p.add_argument("--L-max", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    cfg.training = replace(
        cfg.training,
        model_name="TextAE",
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        checkpoint_dir=str(out),
        checkpoint_every=10**9,  # only save epoch_final.pt
        sample_eval_every=None,  # AE has no sample method
        eval_every=10**9,  # disable validation — the AE only needs to save
        # the final checkpoint. Variable-L AE validation has hit the
        # CUDA caching allocator's NVML_SUCCESS assert (PyTorch issue
        # under shape-varying allocation patterns); the simplest fix
        # is to skip val entirely. tok_acc is tracked per training
        # batch so we still see convergence.
        eval_bpd=False,
    )
    text8_overrides: dict[str, object] = {
        "max_train_windows": args.windows,
        "max_eval_windows": args.eval_windows,
        "batch_size": args.batch_size,
    }
    if args.seq_len is not None:
        text8_overrides["L"] = int(args.seq_len)
        cfg.training = replace(cfg.training, L=int(args.seq_len), B=int(args.batch_size))
    if args.variable_length:
        text8_overrides["variable_length"] = True
        text8_overrides["L_min"] = int(args.L_min)
        text8_overrides["L_max"] = int(args.L_max)
        # Pos embeddings sized to L_max; force training.L = L_max so the
        # model's nn.Embedding(L, d_model) is big enough.
        cfg.training = replace(cfg.training, L=int(args.L_max), B=int(args.batch_size))
        text8_overrides["L"] = int(args.L_max)
    cfg.text8_dataset = replace(cfg.text8_dataset, **text8_overrides)
    cfg.autoencoder = replace(
        cfg.autoencoder,
        d_latent=args.d_latent,
        d_model=args.d_model,
        num_layers=args.num_layers,
        nhead=args.nhead,
        dropout=args.dropout,
        denoising_schedule=args.denoising_schedule,
        denoising_sigma=args.denoising_sigma,
        latent_l2=args.latent_l2,
        mode=args.mode,
        vae_beta=args.vae_beta,
        vae_beta_warmup_epochs=args.vae_beta_warmup_epochs,
    )

    cfg_dict = asdict(cfg)
    cfg_dict["training"]["device"] = str(cfg.training.device)
    if "loader_settings" in cfg_dict:
        cfg_dict["loader_settings"]["device_type"] = str(cfg.loader_settings.device_type)
    (out / "config.json").write_text(json.dumps(cfg_dict, indent=2))

    seed_all(cfg.training.seed)
    dm, _ = build_training_datamodule(cfg)
    history: list[dict[str, float]] = []
    fit(cfg=cfg, datamodule=dm, history_out=history)
    (out / "history.jsonl").write_text("\n".join(json.dumps(h) for h in history))
    print(f"[done] AE trained → {out / 'epoch_final.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
