"""Shared helper: ensure a Stage-1 or Stage-2 checkpoint exists for an arm,
auto-training it via the standard pipeline scripts when absent.

Used by ``scripts/bench_sflm_ebm.py`` and ``scripts/eval_generation.py`` so
that running an evaluator on a fresh runs/ directory doesn't silently skip
missing arms — instead it transparently trains them at the requested
``--scale`` / ``--epochs`` and returns the path once training is done.

Stage-1 arms (model_name in train_for_sflm_bench.ARM_TO_MODEL):
    runs/sflm_bench_<scale>/<arm>/epoch_final.pt
    ↳ produced by ``train_for_sflm_bench.py --scale <scale> --only <arm>
                                            --epochs <epochs>``

Stage-2 SVGP arms (DFM_SVGP, SFLM_SVGP):
    runs/sflm_bench_<scale>/<stage1>/model_with_svgp_hinge.pt
    ↳ produced by ``fit_{dfm|sflm}_svgp_hinge.py --ckpt <stage1_epoch_final>
                                                  --n-epochs <svgp_epochs>``
    Recursively ensures the Stage-1 ``epoch_final.pt`` first.

Legacy fallback (only when ``auto_train=False``): mapping from arm name to
a pre-existing checkpoint in runs/ (e.g. ``runs/dphase4_lambda_recalib`` for
EqM). Useful for local runs that prefer the existing tuned baselines over
spending hours retraining at matched scale.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Legacy well-trained checkpoints — only used when --no-auto-train.
LEGACY_BASELINES = {
    "EqM": "runs/dphase4_lambda_recalib/epoch_final.pt",
    "EqMLatent": "runs/latent_d128_trainable_tied_ce0_ep20/epoch_final.pt",
    "DFM_SVGP": "runs/dfm_svgp_pure50_lr3e4/model_with_svgp_hinge.pt",
}

# Which fit script wraps which Stage-1 arm.
SVGP_FIT_SCRIPTS = {
    "DFM_SVGP": ("DFM", "scripts/fit_dfm_svgp_hinge.py"),
    "SFLM_SVGP": ("SFLM", "scripts/fit_sflm_svgp_hinge.py"),
}


def _run(cmd: list[str], reason: str) -> int:
    print(f"\n[ensure_ckpt] {reason}\n             $ {' '.join(cmd)}",
          flush=True)
    proc = subprocess.run(cmd, cwd=str(_REPO_ROOT))
    return proc.returncode


def ensure_checkpoint(
    arm: str,
    *,
    scale: str,
    epochs: int,
    svgp_epochs: int = 5,
    auto_train: bool = True,
    svgp_lr: float = 1e-3,
) -> Path | None:
    """Return the path to a usable checkpoint for ``arm``, training or
    fitting it on the fly if missing (when ``auto_train=True``).

    Returns ``None`` if the artefact can't be produced (e.g. training
    failed) or if ``auto_train=False`` and there's no legacy fallback.
    """
    root = _REPO_ROOT / "runs" / f"sflm_bench_{scale}"

    # ---- Stage-2 SVGP arms ----
    if arm in SVGP_FIT_SCRIPTS:
        stage1_name, fit_script = SVGP_FIT_SCRIPTS[arm]
        ckpt = root / stage1_name / "model_with_svgp_hinge.pt"
        if ckpt.exists():
            return ckpt
        if not auto_train:
            legacy = LEGACY_BASELINES.get(arm)
            if legacy and (_REPO_ROOT / legacy).exists():
                print(f"[{arm}] reusing legacy SVGP checkpoint {legacy}")
                return _REPO_ROOT / legacy
            return None
        # Recursively ensure Stage-1 exists, then fit the SVGP head.
        stage1_ckpt = ensure_checkpoint(
            stage1_name, scale=scale, epochs=epochs,
            svgp_epochs=svgp_epochs, auto_train=True, svgp_lr=svgp_lr,
        )
        if stage1_ckpt is None:
            return None
        rc = _run(
            [sys.executable, fit_script,
             "--ckpt", str(stage1_ckpt),
             "--n-epochs", str(svgp_epochs),
             "--lr", str(svgp_lr)],
            reason=f"auto-fit Stage-2 SVGP for {arm} on {stage1_ckpt}",
        )
        return ckpt if (rc == 0 and ckpt.exists()) else None

    # ---- Stage-1 arms ----
    ckpt = root / arm / "epoch_final.pt"
    if ckpt.exists():
        return ckpt
    # A published arm may have lost its final while numbered checkpoints
    # survive (SFLM at L=256: only epoch_10/20.pt remain). Prefer the highest
    # surviving epoch over training a NEW model under the old arm's name --
    # auto-training here would silently replace a published row's weights.
    numbered = sorted(
        (p for p in (root / arm).glob("epoch_*.pt")
         if p.stem.split("_", 1)[1].isdigit()),
        key=lambda p: int(p.stem.split("_", 1)[1]),
    )
    if numbered:
        print(f"[{arm}] epoch_final.pt missing; using the highest surviving "
              f"checkpoint {numbered[-1].name} instead of auto-training "
              f"(a fresh train would not reproduce the published arm)")
        return numbered[-1]
    if not auto_train:
        legacy = LEGACY_BASELINES.get(arm)
        if legacy and (_REPO_ROOT / legacy).exists():
            print(f"[{arm}] reusing legacy checkpoint {legacy}")
            return _REPO_ROOT / legacy
        return None
    if "L256" in scale:
        print(f"[{arm}] refusing to auto-train at scale {scale}: the L=256 "
              f"arms back published numbers; train explicitly via "
              f"train_for_sflm_bench.py if a new model is really intended")
        return None
    rc = _run(
        [sys.executable, "scripts/train_for_sflm_bench.py",
         "--scale", scale,
         "--only", arm,
         "--epochs", str(epochs)],
        reason=f"auto-train missing arm {arm} (scale={scale}, "
               f"epochs={epochs}); this may take a while",
    )
    return ckpt if (rc == 0 and ckpt.exists()) else None
