"""Train SFLMEBM + EqM + EqMLatent + DFM on text8 at a *shared scale* (so
the spilled-energy benchmark is apples-to-apples — the existing runs/
checkpoints differ in width/epochs/data and would confound it), with each
model additionally given its own known-good knobs.

Scale presets (--scale): local (d512/6L, 8 GB), cluster (d1024/8L), a100_20g,
and a100_20g_L256 (d1024/10L/16H/B8 @ L=256 — the publication-length tier).
SFLMEBM's contrastive-hinge training step runs ~5 backbone forwards (1 CE + 4
hinge after the diagnostic-reuse fix) and EqM/EqMLatent/SFLMEBM_FM use
second-order autograd, so the L=256 tier is memory-bound; see the
memory-fallback ladder below.

Robustness (added for the autonomous cloud run):
  * --seed / --seeds 42,43,44   — 3-seed headline policy (seed 42 keeps the
                                   canonical dir; others go to <arm>/seed<seed>/).
  * idempotent resume           — finished arms (epoch_final.pt present) are
                                   skipped unless --force.
  * memory-fallback ladder      — on a CUDA OOM or the MIG-NVML allocator
                                   assert, escalate grad-checkpointing → halve
                                   batch → L=128 → mark the arm failed and
                                   CONTINUE the queue (never aborts the chain).
  * --full-split                — train on the full afmck/text8 split with
                                   lazy CLR features (memory scales with batch).

Writes runs/sflm_bench_<scale>/<model>[/seed<seed>]/epoch_final.pt and a
train_meta.json (or FAILED.json) recording the fallback stage actually used.

Usage:  python scripts/train_for_sflm_bench.py --scale a100_20g_L256 --only DFM --epochs 20
        python scripts/train_for_sflm_bench.py --scale a100_20g_L256 --seeds 42,43,44 --full-split
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback
from dataclasses import replace
from pathlib import Path

import torch

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
#              SFLMEBM's contrastive hinge (~5 backbone forwards) and
#              EqM/EqMLatent second-order autograd. A serious run (vs the
#              d256/4L/6ep smoke) for local dev.
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
        # SFLMEBM's contrastive hinge (~5 fwd, the memory-binding arm) under the MATH
        # SDPA backend on a 20 GB A100 MIG — ~9 GB headroom. (The earlier
        # d768/8L ~57M sizing left the 20 GB slice mostly idle.)
        "d_model": 1024, "n_layers": 10, "n_heads": 16, "batch": 8,
        "max_train_windows": 10_000, "max_eval_windows": 2_000,
        "L": 256,
    },
}
D_EMBED = 128  # EqMLatent + SFLMEBM latent dim (matched; = best EqMLatent run)


def _base_cfg(
    epochs: int,
    out_dir: str,
    scale: str,
    *,
    seed: int = 42,
    mtw: int | None = -1,      # -1 sentinel = use the scale's value; None = full split
    L: int | None = None,      # override the scale's context length (ladder L=128 fallback)
    B: int | None = None,      # override the scale's batch (ladder halving)
    grad_ckpt: bool = False,   # activation checkpointing (ladder lever)
    val_eval: bool = False,    # periodic val eval (memory-safe after the eval_step fix)
    lazy: bool = False,        # lazy CLR features (full-split memory lever)
) -> Config:
    s = SCALES[scale]
    d_model, n_layers, n_head = s["d_model"], s["n_layers"], s["n_heads"]
    batch = B if B is not None else s["batch"]
    L = L if L is not None else s.get("L", 40)
    max_train_windows = s["max_train_windows"] if mtw == -1 else mtw
    cfg = Config()
    cfg.training = replace(
        cfg.training,
        epochs=epochs,
        lr=3e-4,
        seed=seed,
        scheduler_warmup_epochs=1,
        cosine_t_max_epochs=epochs,
        B=batch,
        L=L,
        checkpoint_dir=out_dir,
        checkpoint_every=epochs,        # only the final checkpoint
        sample_eval_every=None,         # skip the (slow) KL probe during training
        # Val eval is OFF by default. Historically it was disabled because
        # SFLMEBM's hinge eval_step built full autograd graphs and spiked the
        # 20 GB MIG into an NVML allocator assert; eval_step is now a no_grad
        # CE readout (memory-safe), so --val-eval can be re-enabled for the
        # runbook's val tracking. (Early stopping itself is not wired into the
        # fixed-epoch loop — see CLUSTER_RUNBOOK.)
        eval_every=max(1, epochs // 10) if val_eval else epochs + 1,
        eval_bpd=False,
    )
    cfg.transformer = replace(
        cfg.transformer, d_model=d_model, num_layers=n_layers, nhead=n_head,
        grad_checkpointing=grad_ckpt,
    )
    # LoaderSettings.batch_size is frozen at the class default (64); the
    # datamodule reads it, so it must be rebuilt for the batch override.
    cfg.loader_settings = replace(cfg.loader_settings, batch_size=batch)
    cfg.text8_dataset = replace(
        cfg.text8_dataset,
        L=L,
        batch_size=batch,
        max_train_windows=max_train_windows,
        max_eval_windows=s["max_eval_windows"],
        lazy_features=lazy,
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


def _model_cfg(name: str, epochs: int, scale: str, *, out_dir: str | None = None,
               **base_kw) -> Config:
    out_dir = out_dir or f"runs/sflm_bench_{scale}/{name}"
    cfg = _base_cfg(epochs, out_dir, scale, **base_kw)
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
        # gradient_lambda=3.0 + Dirichlet-sampled CLR data. gamma_power=0.5 is
        # pinned defensively (it is also the restored global default) so the
        # authoritative EqM arm never silently trains in the noise regime if
        # the global default drifts again (cf. commit 2e15b3f).
        cfg.eqm = replace(cfg.eqm, gradient_lambda=3.0, gamma_power=0.5)
        cfg.transformation = replace(cfg.transformation,
                                     dirichlet_sampling=True)
    elif name == "EqM_OneHot":
        # Same tuned recipe except Dirichlet thickening is OFF — uses
        # deterministic label-smoothed one-hot CLR features. Isolates the
        # contribution of the Dirichlet x_1 augmentation in the
        # unified generation comparison (the EqM arm minus its data side).
        cfg.eqm = replace(cfg.eqm, gradient_lambda=3.0, gamma_power=0.5)
        cfg.transformation = replace(cfg.transformation,
                                     dirichlet_sampling=False)
    elif name == "DirichletFM":
        # DirichletFlowMatching (Stark et al. 2024) — Dirichlet conditional
        # probability path on the simplex. Config defaults are the
        # known-good recipe (matches DFM_SVGP_FINDINGS Stage-1).
        pass
    # DFM: Config defaults are its known-good recipe.
    return cfg


def _is_oom(e: BaseException) -> bool:
    """True for a CUDA OOM or the MIG-restricted NVML allocator assert.

    The 20 GB MIG slice raises ``RuntimeError: NVML_SUCCESS == r INTERNAL
    ASSERT FAILED at CUDACachingAllocator.cpp`` when the caching allocator
    expands its pool and queries NVML — surfaces as a RuntimeError, not a
    clean OutOfMemoryError, so we match on the message too."""
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(e, RuntimeError):
        m = str(e)
        return any(k in m for k in (
            "out of memory", "OutOfMemory", "NVML",
            "CUDACachingAllocator", "CUDA error", "alloc",
        ))
    return False


# Memory-fallback ladder (runbook §1, mandatory for the second-order /
# multi-forward arms). Each stage is a cfg delta tried in order on OOM/NVML.
# grad-ckpt is applied BEFORE batch halving (it preserves the effective batch
# and hyperparameters — strictly better than the runbook's stated halve-first
# order) and L=128 is the last resort (records a length_fallback in the
# manifest). Hutchinson (the runbook's 4th rung) is not implemented; if the
# L=128 stage still OOMs the arm is marked failed and the queue continues.
_LADDER = [
    {},                                            # 0: as configured
    {"grad_ckpt": True},                           # 1: + activation checkpointing
    {"grad_ckpt": True, "B_factor": 0.5},          # 2: + halve batch
    {"grad_ckpt": True, "B_factor": 0.25},         # 3: + quarter batch
    {"grad_ckpt": True, "B_factor": 0.25, "L": 128},  # 4: + shorten context
]


def _train_arm(name: str, *, scale: str, epochs: int, seed: int, out_dir: str,
               mtw, full_split: bool, force_ckpt: bool, val_eval: bool) -> dict:
    """Train one (arm, seed) with the memory-fallback ladder. Writes
    ``train_meta.json`` recording the stage that succeeded (or the failure)
    and returns that record."""
    base_batch = SCALES[scale]["batch"]
    mtw_arg = None if full_split else (mtw if mtw is not None else -1)
    lazy = full_split or (mtw is not None and mtw > 200_000)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    last_tb = ""
    for stage_i, stage in enumerate(_LADDER):
        if stage_i == 0 and force_ckpt:
            continue  # user forced checkpointing → skip the no-ckpt stage
        B = stage.get("B_factor")
        B = max(1, int(base_batch * B)) if B is not None else None
        grad_ckpt = stage.get("grad_ckpt", False) or force_ckpt
        L = stage.get("L")
        cfg = _model_cfg(
            name, epochs, scale, out_dir=out_dir, seed=seed,
            mtw=mtw_arg, B=B, L=L, grad_ckpt=grad_ckpt,
            val_eval=val_eval, lazy=lazy,
        )
        tag = (f"stage{stage_i}: B={cfg.training.B} L={cfg.training.L} "
               f"grad_ckpt={grad_ckpt}")
        print(f"  [{name} seed={seed}] {tag}", flush=True)
        seed_all(seed)
        try:
            dm, _ = build_training_datamodule(cfg)
            fit(cfg=cfg, datamodule=dm, wandb_logger=None)
            meta = {
                "arm": name, "seed": seed, "scale": scale, "status": "done",
                "stage": stage_i, "batch": cfg.training.B, "L": cfg.training.L,
                "grad_checkpointing": grad_ckpt,
                "length_fallback": cfg.training.L if cfg.training.L != SCALES[scale].get("L", 40) else None,
                "max_train_windows": cfg.text8_dataset.max_train_windows,
                "lazy_features": cfg.text8_dataset.lazy_features,
            }
            (Path(out_dir) / "train_meta.json").write_text(json.dumps(meta, indent=2))
            print(f"=== DONE {name} (seed {seed}, {tag}) → {out_dir} ===", flush=True)
            return meta
        except Exception as e:  # noqa: BLE001 — record + continue per runbook policy
            last_tb = traceback.format_exc()
            del cfg
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if _is_oom(e):
                print(f"  [{name} seed={seed}] OOM/NVML at {tag} — escalating ladder", flush=True)
                continue
            # non-OOM failure: record failed + stop escalating this arm
            print(f"  [{name} seed={seed}] non-OOM failure: {type(e).__name__}: {e}", flush=True)
            break

    meta = {"arm": name, "seed": seed, "scale": scale, "status": "failed",
            "traceback": last_tb}
    (Path(out_dir) / "FAILED.json").write_text(json.dumps(meta, indent=2))
    print(f"=== FAILED {name} (seed {seed}) — see {out_dir}/FAILED.json ===", flush=True)
    return meta


def _out_dir(scale: str, name: str, seed: int) -> str:
    """Seed 42 keeps the canonical path (so existing bench/eval discovery of
    runs/sflm_bench_<scale>/<arm>/epoch_final.pt is preserved); other seeds go
    under a seed subdir."""
    base = f"runs/sflm_bench_{scale}/{name}"
    return base if seed == 42 else f"{base}/seed{seed}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--scale", choices=list(SCALES), default="local",
                    help="local = fits 8 GB; a100_20g_L256 = d1024/10L/16H/B8 @ L256")
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated arm(s) to train (default: all)")
    ap.add_argument("--seed", type=int, default=42, help="single seed (default 42)")
    ap.add_argument("--seeds", type=str, default=None,
                    help="comma-separated seeds (overrides --seed; e.g. 42,43,44 "
                         "for the 3-seed headline policy). seed 42 -> canonical "
                         "dir; others -> <arm>/seed<seed>/")
    ap.add_argument("--max-train-windows", type=int, default=None,
                    help="override the scale's training-data budget (downstream "
                         "discovery under runs/sflm_bench_<scale>/ is preserved)")
    ap.add_argument("--full-split", action="store_true",
                    help="train on the FULL afmck/text8 split (max_train_windows="
                         "None) with lazy CLR features (memory scales with batch)")
    ap.add_argument("--force", action="store_true",
                    help="retrain even if epoch_final.pt already exists "
                         "(default: skip finished arms — idempotent resume)")
    ap.add_argument("--grad-checkpointing", action="store_true",
                    help="force activation checkpointing on from the start")
    ap.add_argument("--val-eval", action="store_true",
                    help="enable periodic memory-safe val eval during training")
    args = ap.parse_args()

    if args.seeds:
        seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    else:
        seeds = [args.seed]

    names = ARMS
    if args.only:
        names = [a.strip() for a in args.only.split(",") if a.strip()]
        unknown = [n for n in names if n not in ARM_TO_MODEL]
        if unknown:
            raise SystemExit(f"unknown arm(s): {unknown}; choose from {ARMS}")

    s = SCALES[args.scale]
    for name in names:
        for seed in seeds:
            out_dir = _out_dir(args.scale, name, seed)
            ckpt = Path(out_dir) / "epoch_final.pt"
            if ckpt.exists() and not args.force:
                print(f"[skip] {name} seed={seed}: {ckpt} exists "
                      f"(use --force to retrain)", flush=True)
                continue
            print(f"\n{'='*70}\n=== TRAIN {name} [{args.scale}] seed={seed} "
                  f"({args.epochs} ep, d_model={s['d_model']}/{s['n_layers']}L/"
                  f"{s['n_heads']}H, B={s['batch']}, L={s.get('L', 40)}, "
                  f"full_split={args.full_split}) ===\n{'='*70}", flush=True)
            _train_arm(
                name, scale=args.scale, epochs=args.epochs, seed=seed,
                out_dir=out_dir, mtw=args.max_train_windows,
                full_split=args.full_split, force_ckpt=args.grad_checkpointing,
                val_eval=args.val_eval,
            )


if __name__ == "__main__":
    main()
