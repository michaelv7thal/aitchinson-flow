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
from dataclasses import fields, replace
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.append(str(_ROOT))

import aitchinson_flow.models  # noqa: E402,F401  populate REGISTRY
from aitchinson_flow.models import build_model  # noqa: E402  warm-start resume
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
    # Bigger-backbone twin of a100_20g_L256 (d1024/10L, ~127M). Same B8/L256 so
    # a DirichletFM trained here at the SAME 30k×30ep isolates the *capacity*
    # lever vs DirichletFM_ep30_d30k. ~2.2× params (~275M, GPT-2-medium tier);
    # first-order DirichletFM is light, and the memory ladder enables grad-ckpt
    # (which keeps batch 8) if it's tight — so the comparison stays matched.
    "a100_20g_L256_d1280L14": {
        "d_model": 1280, "n_layers": 14, "n_heads": 16, "batch": 8,
        "max_train_windows": 10_000, "max_eval_windows": 2_000,
        "L": 256,
    },
    # Same d1280/14L backbone but batch 16: DirichletFM is FIRST-ORDER (no
    # second-order autograd), so the 20 GB MIG has headroom the batch-8 (sized
    # for the second-order EqM/SFLMEBM arms) leaves idle. For the longer
    # DirichletFM-only push (50 ep × 50k). The memory ladder halves the batch if
    # it ever OOMs, so this is safe (a fallback is footnoted in train_meta).
    "a100_20g_L256_d1280L14_b16": {
        "d_model": 1280, "n_layers": 14, "n_heads": 16, "batch": 16,
        "max_train_windows": 10_000, "max_eval_windows": 2_000,
        "L": 256,
    },
    # Full-split convergence run (d1280/14L, L256). Same backbone as the _b16
    # 50k push; the "batch" here is only a fallback default — the driver picks
    # the throughput-optimal batch at runtime via --batch (probe_batch_throughput
    # .py) and scales --lr with it. Distinct scale dir keeps the full-data run
    # isolated from the 50k arm. Trains to convergence via --early-stop-patience.
    "a100_20g_L256_d1280L14_full": {
        "d_model": 1280, "n_layers": 14, "n_heads": 16, "batch": 16,
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
    early_stop_patience: int | None = None,  # val-eval stalls before stopping
    es_min_delta: float = 0.0,
    max_hours: float | None = None,          # wall-clock cap (runbook 36h)
    lr: float = 3e-4,                        # base LR; scale with batch for big-B runs
    warmup_epochs: int | None = None,        # LR warmup epochs; None = default (1)
) -> Config:
    s = SCALES[scale]
    d_model, n_layers, n_head = s["d_model"], s["n_layers"], s["n_heads"]
    batch = B if B is not None else s["batch"]
    L = L if L is not None else s.get("L", 40)
    max_train_windows = s["max_train_windows"] if mtw == -1 else mtw
    # Early stopping needs val eval to fire, so it implies val_eval.
    do_val = val_eval or (early_stop_patience is not None)
    cfg = Config()
    cfg.training = replace(
        cfg.training,
        epochs=epochs,
        lr=lr,
        seed=seed,
        # Warmup is normally 1 epoch (ramp 0.01·lr → lr). A warm-start
        # continuation can set warmup_epochs=0 so the cosine starts at the (low)
        # peak immediately: with no re-warmup AND a peak LR below the
        # oscillation onset, the schedule decays monotonically to 0 with no
        # val "bump" — and it doesn't burn a ~10 h epoch ramping at near-zero LR.
        scheduler_warmup_epochs=(1 if warmup_epochs is None else max(0, warmup_epochs)),
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
        # Early stopping needs a fine val cadence: full-split epochs are
        # multi-hour, so epochs//10 would stride past the convergence plateau.
        # Eval every epoch when patience is set; else the coarse 10% cadence.
        eval_every=(
            1 if early_stop_patience is not None
            else (max(1, epochs // 10) if do_val else epochs + 1)
        ),
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=es_min_delta,
        max_wall_clock_hours=max_hours,
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
    "EqMLatent": "EqMLatent",   # learned-embedding EqM (NOT a VAE; see EqMAE)
    "DirichletFM": "DirichletFM",
    # Warm-start continuation of the full-text8 DirichletFM convergence run.
    # Same model+recipe; separate arm name so it writes to its own run dir and
    # the original ep10 checkpoint/evals are not clobbered. Used with
    # --resume-weights <orig>/epoch_final.pt to keep training the ep10 weights.
    "DirichletFM_continue": "DirichletFM",
    # Second warm-start continuation, tuned to CONVERGE without the val "bump"
    # the first continuation showed: warm-start from the best ep5 ckpt (val
    # 0.8554), warmup OFF, a low peak LR (below the ~4e-4 oscillation onset)
    # that cosine-anneals to 0, early-stop on the plateau. Own dir so the
    # ep5/ep10 artifacts are preserved. See drive_fulltext8_converge2.sh.
    "DirichletFM_converge": "DirichletFM",
    # Extended-budget DirichletFM alias (same model + default recipe — the
    # `name == "DirichletFM"` cfg branch is a no-op `pass`). Separate arm name
    # so it writes to its own run dir (epochs/data encoded) without touching
    # the matched-budget DirichletFM results. Trained at 30 ep × 30k windows.
    "DirichletFM_ep30_d30k": "DirichletFM",
    # Longer DirichletFM push (50 ep × 50k windows). Same model+recipe; separate
    # arm name so it gets its own run dir without touching the other results.
    "DirichletFM_ep50_d50k": "DirichletFM",
    "DFM": "DFM",
    # Statistical Flow Matching (Cheng et al. 2024, arXiv:2405.16441): Fisher–Rao
    # √μ-sphere geodesic FM. Config defaults (cfg.sfm) are the known-good recipe.
    "SFM": "SFM",
    # Fisher-Flow (Davis et al. 2024, arXiv:2405.14664, davis2024fisherfm): SFM +
    # a Riemannian minibatch-OT coupling. Config defaults (cfg.fisher_fm) are the
    # recipe; use_ot=True is the delta over SFM.
    "FisherFM": "FisherFM",
    # The 7-model generation benchmark also needs these two (added):
    #   FMonCLR — Standard Flow Matching (Lipman): raw-velocity FM on CLR,
    #             the linear-FM control. No conservative-gradient step.
    #   EqMAE   — VAE+EqM: EqM in a *frozen* pretrained (V)AE latent. 2-stage:
    #             train the (V)AE first, then pass --ae-ckpt (see run notes).
    "FMonCLR": "FMonCLR",
    "EqMAE": "EqMAE",
}
ARMS = list(ARM_TO_MODEL)


def _model_cfg(name: str, epochs: int, scale: str, *, out_dir: str | None = None,
               ae_ckpt: str | None = None, gamma_power: float | None = None,
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
        # NB: the uniform/alpha_hi=1.0 schedule was TRIED (hypothesis: under-
        # supervised near-noise) and REGRESSED generation (KL_bi 1.33→1.74) —
        # the import schedule's near-clean concentration matters more. Reverted
        # to defaults; the real lever is training budget (epochs), as
        # DirichletFM showed (5ep 1.54 → 20ep 0.33).
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
    elif name == "SFM":
        # StatisticalFlowMatching (Cheng et al. 2024) — Fisher–Rao √μ-sphere
        # geodesic flow matching. Reuses the DirichletFM backbone (DFMBackbone +
        # DFMHead); only cfg.sfm (sample_nfe, t_eps) applies, and its defaults
        # are the recipe. No special sizing needed.
        pass
    elif name == "FisherFM":
        # FisherFlowMatching (Davis et al. 2024) — SFM + Riemannian minibatch-OT
        # coupling. Same DirichletFM backbone; only cfg.fisher_fm applies and its
        # defaults are the recipe. No special sizing needed.
        pass
    elif name == "FMonCLR":
        # Standard Flow Matching (Lipman et al. 2022) on CLR — the linear-FM
        # control: regress the raw velocity to the CONSTANT conditional-OT
        # target (x0−x1) under uniform t, then Euler-integrate the ODE (no
        # conservative-gradient step, no c(γ) decay — see models/fm_clr.py;
        # the aux CE IS kept, contrary to what this comment used to say: the
        # code and the trained checkpoint both carry it, and the paper's
        # tab:config row is right). gamma_power is irrelevant (t is uniform).
        # Use the canonical Lipman base p0=N(0,I): source_sigma=1.0 (decoupled
        # from EqM's NAG-tuned σ=0.1) so the source carries real sampling
        # entropy instead of starting every trajectory at a near-point mass.
        cfg.eqm = replace(cfg.eqm, time_conditioning="add", source_sigma=1.0)
    elif name == "EqMAE":
        # VAE+EqM: EqM flow over a FROZEN pretrained (V)AE latent. 2-stage —
        # the AE must be trained first (scripts/train_autoencoder.py --mode vae)
        # and its checkpoint passed via --ae-ckpt. We read the AE's own config
        # block from the checkpoint so cfg.autoencoder matches the frozen
        # weights exactly (a dim mismatch would silently load a broken AE).
        if not ae_ckpt:
            raise SystemExit(
                "EqMAE arm requires --ae-ckpt <frozen (V)AE checkpoint> "
                "(train it first with scripts/train_autoencoder.py --mode vae)"
            )
        payload = torch.load(ae_ckpt, map_location="cpu", weights_only=False)
        ae_cfg = (payload.get("cfg") or {}).get("autoencoder", {}) or {}
        if not ae_cfg:
            # A *reconstructed* VAE checkpoint is a bare state dict with no cfg
            # payload (see runs/vae_a100_20g_L256/RECONSTRUCTED.md), so the line
            # above silently yields {} and every field below keeps its dataclass
            # default — including mode="ae", which is precisely the failure the
            # comment below claims to guard against. Recover the block from the
            # config.json the AE's own training run wrote beside the checkpoint.
            sib = Path(ae_ckpt).parent / "config.json"
            if sib.exists():
                ae_cfg = (json.loads(sib.read_text()).get("autoencoder") or {})
                print(f"[EqMAE] ae_cfg recovered from {sib} "
                      f"(checkpoint carries no cfg payload)")
        # Copy *every* saved AE field that exists on the dataclass, not a
        # hardcoded dims-only subset. `mode` ('ae'|'vae') in particular changes
        # which submodules TextAutoencoder builds: omitting it silently loaded a
        # VAE checkpoint into an AE-mode module, leaving the `to_latent` head at
        # random init (the VAE ckpt has mu_head/logsig_head, no to_latent) and
        # feeding the frozen decoder a mismatched latent — i.e. EqM over a broken
        # latent space, not the pretrained VAE latent the arm intends.
        valid = {f.name for f in fields(cfg.autoencoder)}
        cfg.autoencoder = replace(cfg.autoencoder, **{
            k: v for k, v in ae_cfg.items() if k in valid})
        # Ground truth for `mode` is the WEIGHTS, not any config file: a VAE
        # checkpoint carries mu_head/logsig_head and no to_latent. If the
        # resolved mode disagrees, TextAutoencoder builds the wrong latent head
        # and load_state_dict(strict=False) leaves it at random init — EqM then
        # trains over a random projection and feeds the frozen decoder a
        # mismatched latent. Fail loudly rather than train a broken arm.
        _sd = payload.get("model_state_dict", payload)
        _keys = set(_sd) if isinstance(_sd, dict) else set()
        _inferred = "vae" if any(k.endswith("mu_head.weight") for k in _keys) else "ae"
        if cfg.autoencoder.mode != _inferred:
            print(f"[EqMAE] AE mode {cfg.autoencoder.mode!r} contradicts the "
                  f"checkpoint weights ({_inferred!r}) — using {_inferred!r}")
            cfg.autoencoder = replace(cfg.autoencoder, mode=_inferred)
        cfg.eqm_ae = replace(cfg.eqm_ae, ae_ckpt_path=ae_ckpt)
        cfg.eqm = replace(cfg.eqm, sampler="euler")  # latent-space Euler (as EqMLatent)
    # DFM: Config defaults are its known-good recipe.
    if gamma_power is not None:
        # γ-schedule override (γ = U(0,1)**p; 1.0 = uniform, 0.5 = the
        # published √u recipe). Applied AFTER the arm blocks so it beats the
        # defensive 0.5 pins above. Only the EqM-family arms read
        # cfg.eqm.gamma_power; for the others this is inert.
        cfg.eqm = replace(cfg.eqm, gamma_power=gamma_power)
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
               mtw, full_split: bool, force_ckpt: bool, val_eval: bool,
               early_stop_patience: int | None = None, es_min_delta: float = 0.0,
               max_hours: float | None = None, length: int | None = None,
               ae_ckpt: str | None = None, resume_weights: str | None = None,
               batch_override: int | None = None, lr: float = 3e-4,
               warmup_epochs: int | None = None,
               gamma_power: float | None = None) -> dict:
    """Train one (arm, seed) with the memory-fallback ladder. Writes
    ``train_meta.json`` recording the stage that succeeded (or the failure)
    and returns that record. ``batch_override`` replaces the scale's batch
    (the ladder still halves relative to it); ``lr`` overrides the base LR."""
    base_batch = batch_override if batch_override is not None else SCALES[scale]["batch"]
    mtw_arg = None if full_split else (mtw if mtw is not None else -1)
    lazy = full_split or (mtw is not None and mtw > 200_000)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    last_tb = ""
    for stage_i, stage in enumerate(_LADDER):
        if stage_i == 0 and force_ckpt:
            continue  # user forced checkpointing → skip the no-ckpt stage
        B = stage.get("B_factor")
        # No B_factor (stages 0/1) → run at base_batch (the --batch override, or
        # the scale default). Passing None here would make _model_cfg silently
        # fall back to the scale's batch, dropping --batch on the no-halving
        # stages — the bug that ran a --batch 48 request at the scale default 16.
        B = max(1, int(base_batch * B)) if B is not None else base_batch
        grad_ckpt = stage.get("grad_ckpt", False) or force_ckpt
        # stage L=128 (the last-resort fallback) wins; else the user --length
        # (E1b sweep) overrides the scale's L; else None → scale default.
        L = stage.get("L") or length
        cfg = _model_cfg(
            name, epochs, scale, out_dir=out_dir, seed=seed, ae_ckpt=ae_ckpt,
            mtw=mtw_arg, B=B, L=L, grad_ckpt=grad_ckpt,
            val_eval=val_eval, lazy=lazy,
            early_stop_patience=early_stop_patience, es_min_delta=es_min_delta,
            max_hours=max_hours, lr=lr, warmup_epochs=warmup_epochs,
            gamma_power=gamma_power,
        )
        tag = (f"stage{stage_i}: B={cfg.training.B} L={cfg.training.L} "
               f"grad_ckpt={grad_ckpt}")
        print(f"  [{name} seed={seed}] {tag}", flush=True)
        seed_all(seed)
        try:
            dm, _ = build_training_datamodule(cfg)
            # Warm-start continuation: load ONLY the model weights from a prior
            # checkpoint into a freshly built model, then run a normal fit
            # (fresh optimizer + full cosine over `epochs`). Deliberately not
            # fit(resume_from=...): that shifts start_epoch to the saved epoch+1
            # and rebuilds the scheduler from scratch, so the cosine would
            # warm-restart then under-anneal over the remaining epochs. A clean
            # full schedule on the loaded weights is the correct continuation.
            warm_model = None
            if resume_weights is not None:
                warm_model = build_model(cfg).to(cfg.training.device)
                # Load the checkpoint onto the CPU first, then copy into the
                # (already-on-GPU) model. Loading with map_location=device put a
                # full second copy of the ~1.1 GB state dict on the GPU *during*
                # load — on the tight 18.7 GB B48 footprint that transient spilled
                # the 20 GB MIG slice and forced the memory ladder down to B24
                # (the original, no-warm-start run fit B48 fine). CPU load avoids
                # the duplicate GPU allocation; load_state_dict copies in-place.
                payload = torch.load(
                    resume_weights, map_location="cpu", weights_only=False,
                )
                warm_model.load_state_dict(payload["model_state_dict"])
                saved_epoch = payload.get("epoch")
                del payload
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"  [{name} seed={seed}] warm-started weights from "
                      f"{resume_weights} (saved epoch={saved_epoch})",
                      flush=True)
            fit(cfg=cfg, datamodule=dm, model=warm_model, wandb_logger=None)
            meta = {
                "arm": name, "seed": seed, "scale": scale, "status": "done",
                "stage": stage_i, "batch": cfg.training.B, "L": cfg.training.L,
                "grad_checkpointing": grad_ckpt,
                "gamma_power": cfg.eqm.gamma_power,
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
    ap.add_argument("--early-stop-patience", type=int, default=None,
                    help="stop after N consecutive val-evals with no improvement "
                         "(implies --val-eval; restores the best ckpt as "
                         "epoch_final.pt). None = fixed-epoch loop.")
    ap.add_argument("--es-min-delta", type=float, default=0.0,
                    help="minimum val-loss improvement to reset the patience counter")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="hard per-run wall-clock cap in hours (runbook 36h)")
    ap.add_argument("--length", type=int, default=None,
                    help="override the scale's context length L (E1b length "
                         "sweep: L∈{40,128,256}). Output dir gets an L<n> suffix "
                         "so the sweep cells don't collide.")
    ap.add_argument("--ae-ckpt", type=str, default=None,
                    help="frozen (V)AE checkpoint for the EqMAE (VAE+EqM) arm; "
                         "train it first with scripts/train_autoencoder.py --mode vae")
    ap.add_argument("--resume-weights", type=str, default=None,
                    help="warm-start: load ONLY the model weights from this "
                         "checkpoint into a fresh model, then train a full new "
                         "schedule (fresh optimizer/cosine). For continuing a "
                         "capped run from its epoch_final.pt without clobbering "
                         "it — point --only at a distinct arm/out dir.")
    ap.add_argument("--batch", type=int, default=None,
                    help="override the scale's batch size (the memory-fallback "
                         "ladder still halves relative to it). Used by the "
                         "full-text8 driver with a probe-selected batch.")
    ap.add_argument("--lr", type=float, default=3e-4,
                    help="base learning rate (default 3e-4). Scale with batch "
                         "for large-B runs, e.g. 3e-4·sqrt(B/16).")
    ap.add_argument("--warmup-epochs", type=int, default=None,
                    help="LR warmup epochs (default 1). Set 0 for a warm-start "
                         "continuation so the cosine starts at the peak LR "
                         "immediately (no re-warmup bump, no wasted ramp epoch).")
    ap.add_argument("--gamma-power", type=float, default=None,
                    help="override eqm.gamma_power for the EqM-family arms "
                         "(γ = U(0,1)**p; 1.0 = uniform, 0.5 = the published "
                         "√u recipe; inert for the non-EqM arms). Output dir "
                         "gets a _gp<p> suffix so the canonical runs are "
                         "never overwritten.")
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
    eff_L = args.length if args.length is not None else s.get("L", 40)
    for name in names:
        for seed in seeds:
            out_dir = _out_dir(args.scale, name, seed)
            if args.length is not None and args.length != s.get("L", 40):
                out_dir = f"{out_dir}_L{args.length}"
            if args.gamma_power is not None:
                out_dir = f"{out_dir}_gp{str(args.gamma_power).replace('.', 'p')}"
            ckpt = Path(out_dir) / "epoch_final.pt"
            if ckpt.exists() and not args.force:
                print(f"[skip] {name} seed={seed}: {ckpt} exists "
                      f"(use --force to retrain)", flush=True)
                continue
            print(f"\n{'='*70}\n=== TRAIN {name} [{args.scale}] seed={seed} "
                  f"({args.epochs} ep, d_model={s['d_model']}/{s['n_layers']}L/"
                  f"{s['n_heads']}H, B={args.batch or s['batch']}, lr={args.lr}, "
                  f"L={eff_L}, full_split={args.full_split}) ===\n{'='*70}",
                  flush=True)
            _train_arm(
                name, scale=args.scale, epochs=args.epochs, seed=seed,
                out_dir=out_dir, mtw=args.max_train_windows,
                full_split=args.full_split, force_ckpt=args.grad_checkpointing,
                val_eval=args.val_eval,
                early_stop_patience=args.early_stop_patience,
                es_min_delta=args.es_min_delta, max_hours=args.max_hours,
                length=args.length, ae_ckpt=args.ae_ckpt,
                resume_weights=args.resume_weights,
                batch_override=args.batch, lr=args.lr,
                warmup_epochs=args.warmup_epochs,
                gamma_power=args.gamma_power,
            )


if __name__ == "__main__":
    main()
