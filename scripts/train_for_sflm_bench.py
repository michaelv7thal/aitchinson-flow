"""Train SFLMEBM + EqM + EqMLatent + DFM on text8 at a *shared scale* (so
the spilled-energy benchmark is apples-to-apples — the existing runs/
checkpoints differ in width/epochs/data and would confound it), with each
model additionally given its own known-good knobs.

Two scale presets (--scale):
  local   — d512/6L, B=16: a serious run that fits an 8 GB GPU with all
            four models (SFLMEBM's 4-forward hinge + EqM/EqMLatent
            second-order autograd). For local development.
  cluster — d1024/8L, B=64: the DFM_SVGP_FINDINGS.md Stage-1 reference
            scale; ~16-18 GB, intended for the 20 GB cluster.

Writes runs/sflm_bench_<scale>/<model>/epoch_final.pt.

Usage:  python scripts/train_for_sflm_bench.py --scale local  --epochs 50
        python scripts/train_for_sflm_bench.py --scale cluster --epochs 50
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

# Scale presets. Each model also gets its *known-good knobs* (read off the
# best existing runs/ checkpoints) so the comparison is each method at its
# best, not a strawman.
#
#   local    — fits an 8 GB laptop GPU with all four models, incl.
#              SFLMEBM's 4-forward hinge and EqM/EqMLatent second-order
#              autograd. A serious run (vs the d256/4L/6ep smoke) for
#              local dev.
#   cluster  — d1024/8L, batch 64: the DFM_SVGP_FINDINGS.md Stage-1
#              reference scale (~85M params, the "paper-small" tier of
#              the diffusion-LM / Dirichlet-FM literature).
#   a100_20g — d1024/12L, batch 32: the *modern medium* benchmark tier
#              (~120M params, GPT-2-small-equivalent depth) that fits a
#              20 GB A100 MIG with headroom for the worst-case arm
#              (SFLMEBM_FM second-order autograd). Also bumps the data
#              budget to 50k windows — the cluster preset's 10k is the
#              actual bottleneck at ~100M params, not the model size.
SCALES = {
    "local":    {"d_model": 512,  "n_layers": 6,  "n_heads": 8,  "batch": 16,
                 "max_train_windows": 10_000, "max_eval_windows": 5_000,
                 "L": 40},
    "cluster":  {"d_model": 1024, "n_layers": 8,  "n_heads": 8,  "batch": 64,
                 "max_train_windows": 10_000, "max_eval_windows": 5_000,
                 "L": 40},
    "a100_20g": {"d_model": 1024, "n_layers": 12, "n_heads": 16, "batch": 32,
                 # NB: capped at 10k windows (matching cluster) to keep
                 # auto-train chain wallclock in single-digit hours; the
                 # original 50k bumped SFLMEBM alone to ~7h/arm @ 10 ep.
                 "max_train_windows": 10_000, "max_eval_windows": 5_000,
                 "L": 40},
    # L=256: matches the SEDD / D3PM / Multinomial-Diffusion / Transformer-
    # XL text8 evaluation convention so our BPC is directly comparable
    # to published numbers.  Memory at L=256 is attention-quadratic and
    # second-order autograd doubles it for EqM / SFLMEBM_FM; sizing here
    # is the largest d_model/n_layers/B triple that fits a 20 GB A100 MIG
    # with the MATH SDPA backend (Flash is incompatible with EqM's
    # create_graph=True conservative-gradient path — see CLAUDE.md).
    "a100_20g_L256": {
        # Sized to the GPT-2-small / modal diffusion-LM peer tier (~127M
        # params) so our text8 BPC is directly comparable to D3PM (~90M),
        # SEDD-Absorb (~85M), MDLM, Plaid (~120M), and the Transformer-XL
        # reference. d1024/10L/16H @ B8/L256 was measured to peak ~11 GB on
        # SFLMEBM's 4-forward hinge (the memory-binding arm) under the MATH
        # SDPA backend on a 20 GB A100 MIG — ~9 GB headroom. (The earlier
        # d768/8L ~57M sizing left the 20 GB slice mostly idle.)
        "d_model": 1024, "n_layers": 10, "n_heads": 16, "batch": 8,
        "max_train_windows": 10_000, "max_eval_windows": 2_000,
        "L": 256,
    },
}
D_EMBED = 128  # EqMLatent + SFLMEBM latent dim (matched; = best EqMLatent run)


def _base_cfg(epochs: int, out_dir: str, scale: str) -> Config:
    s = SCALES[scale]
    d_model, n_layers, n_head, batch = (
        s["d_model"], s["n_layers"], s["n_heads"], s["batch"],
    )
    L = s.get("L", 40)
    cfg = Config()
    cfg.training = replace(
        cfg.training,
        epochs=epochs,
        lr=3e-4,
        scheduler_warmup_epochs=1,
        cosine_t_max_epochs=epochs,
        B=batch,
        L=L,
        checkpoint_dir=out_dir,
        checkpoint_every=epochs,        # only the final checkpoint
        sample_eval_every=None,         # skip the (slow) KL probe during training
        eval_every=epochs + 1,          # disable val eval: SFLMEBM's 4-forward
                                        # hinge eval_step spikes memory enough to
                                        # force a CUDA allocator pool expansion,
                                        # which queries NVML and asserts on this
                                        # MIG slice (NVML restricted). Val loss is
                                        # not a deliverable; epoch_final.pt is
                                        # saved before eval anyway.
        eval_bpd=False,
    )
    cfg.transformer = replace(
        cfg.transformer, d_model=d_model, num_layers=n_layers, nhead=n_head
    )
    # LoaderSettings.batch_size is frozen at the class default (64); the
    # datamodule reads it, so it must be rebuilt for the batch override.
    cfg.loader_settings = replace(cfg.loader_settings, batch_size=batch)
    cfg.text8_dataset = replace(
        cfg.text8_dataset,
        L=L,
        batch_size=batch,
        max_train_windows=s["max_train_windows"],
        max_eval_windows=s["max_eval_windows"],
    )
    cfg.wandb = replace(cfg.wandb, enabled=False, mode="disabled")
    # Corruption on (default 0.15) → collate emits token_ids_invalid, which
    # the SFLMEBM hinge consumes; harmless for the others.
    return cfg


# Map from *arm* (experiment) name → registered model class name. Two
# arms share the SFLMEBM class but differ in cfg (CE+hinge vs CE+FM):
#   SFLMEBM       — CE + contrastive hinge (current).
#   SFLMEBM_FM    — CE + conservative-grad FM target (option 2,
#                   SFLM_EBM_FINDINGS.md); hinge OFF to isolate whether
#                   FM supervision alone gives both OOD *and* generation.
# SFLM is the Stage-1 generator (S-FLM proper, time-conditioned). Pair
# with a post-hoc SVGP head (mirror DirichletFMSvgp) for Stage 2 OOD.
ARM_TO_MODEL = {
    "SFLMEBM": "SFLMEBM",
    "SFLMEBM_FM": "SFLMEBM",
    "SFLM": "SFLM",
    # Two EqM variants exposed as separate arms so the unified generation
    # evaluator can compare *one-hot CLR* (label-smoothed only) vs the
    # *Dirichlet-thickened* CLR pipeline.  Same model class, different
    # data augmentation flag (cfg.transformation.dirichlet_sampling).
    "EqM_OneHot": "EqM",        # dirichlet_sampling=False
    "EqM": "EqM",               # dirichlet_sampling=True  (existing tuned recipe)
    "EqMLatent": "EqMLatent",
    "DirichletFM": "DirichletFM",
    "DFM": "DFM",
}
ARMS = list(ARM_TO_MODEL)


def _model_cfg(name: str, epochs: int, scale: str) -> Config:
    cfg = _base_cfg(epochs, f"runs/sflm_bench_{scale}/{name}", scale)
    model_name = ARM_TO_MODEL[name]
    cfg.training = replace(cfg.training, model_name=model_name)
    if name == "EqMLatent":
        # Best existing EqMLatent run: d128, tied, lambda_ce=1.0, euler.
        cfg.embedding = replace(cfg.embedding, enabled=True,
                                d_embed=D_EMBED, tie_decoder=True)
        cfg.eqm = replace(cfg.eqm, lambda_ce=1.0, sampler="euler")
    elif name == "SFLMEBM":
        cfg.sflm_ebm = replace(cfg.sflm_ebm, d_embed=D_EMBED)
    elif name == "SFLMEBM_FM":
        # Option 2: CE + Riemannian-FM regression, no hinge.
        cfg.sflm_ebm = replace(
            cfg.sflm_ebm, d_embed=D_EMBED,
            lambda_fm=1.0, lambda_hinge=0.0,
        )
    elif name == "SFLM":
        cfg.sflm = replace(cfg.sflm, d_embed=D_EMBED)
    elif name == "EqM":
        # Tuned simplex-EqM recipe (runs/dphase4_lambda_recalib): needs
        # gradient_lambda=3.0 + Dirichlet-sampled CLR data.
        cfg.eqm = replace(cfg.eqm, gradient_lambda=3.0)
        cfg.transformation = replace(cfg.transformation,
                                     dirichlet_sampling=True)
    elif name == "EqM_OneHot":
        # Same tuned recipe except Dirichlet thickening is OFF — uses
        # deterministic label-smoothed one-hot CLR features. Isolates the
        # contribution of the Dirichlet x_1 augmentation in the
        # unified generation comparison (the EqM arm minus its data side).
        cfg.eqm = replace(cfg.eqm, gradient_lambda=3.0)
        cfg.transformation = replace(cfg.transformation,
                                     dirichlet_sampling=False)
    elif name == "DirichletFM":
        # DirichletFlowMatching (Stark et al. 2024) — Dirichlet conditional
        # probability path on the simplex. Config defaults are the
        # known-good recipe (matches DFM_SVGP_FINDINGS Stage-1).
        pass
    # DFM: Config defaults are its known-good recipe.
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--scale", choices=list(SCALES), default="local",
                    help="local = fits 8 GB; cluster = d1024/8L for ~20 GB")
    ap.add_argument("--only", type=str, default=None,
                    help="train just this model name")
    ap.add_argument("--max-train-windows", type=int, default=None,
                    help="override the scale's training-data budget without "
                         "changing the scale name (so downstream bench/eval "
                         "discovery under runs/sflm_bench_<scale>/ is preserved)")
    args = ap.parse_args()
    s = SCALES[args.scale]
    if args.max_train_windows is not None:
        # Mutate the scale dict in-place: _base_cfg re-reads SCALES[scale].
        s["max_train_windows"] = args.max_train_windows
    d_model, n_layers, batch = s["d_model"], s["n_layers"], s["batch"]
    n_windows = s["max_train_windows"]

    names = ARMS
    if args.only:
        names = [a.strip() for a in args.only.split(",") if a.strip()]
        unknown = [n for n in names if n not in ARM_TO_MODEL]
        if unknown:
            raise SystemExit(f"unknown arm(s): {unknown}; choose from {ARMS}")

    for name in names:
        print(f"\n{'='*70}\n=== TRAIN {name} [{args.scale}] ({args.epochs} ep, "
              f"d_model={d_model}/{n_layers}L, B={batch}, "
              f"n_train_windows={n_windows}) ===\n{'='*70}",
              flush=True)
        cfg = _model_cfg(name, args.epochs, args.scale)
        Path(cfg.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        seed_all(cfg.training.seed)
        dm, _ = build_training_datamodule(cfg)
        fit(cfg=cfg, datamodule=dm, wandb_logger=None)
        print(f"=== DONE {name} → {cfg.training.checkpoint_dir} ===", flush=True)


if __name__ == "__main__":
    main()
