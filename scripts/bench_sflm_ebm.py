"""Spilled-energy benchmark on text8.

**Spilled energy is the *universal* baseline**: ``SE(i) = logsumexp_v logits_i −
logits_i[token_i]``, the per-position NLL of the placed token under a *fixed
reference LM* (default: DFM, override with ``--ref-model``). The reference
is loaded once and scores every test arm with the same logits, so the SE
columns in the table are identical across arms by construction — any
difference between arms in the headline OOD comparison must come from their
*own* scores (``model.energy``, ``‖∇E‖``).

For an EBM the question this bench answers is: do the specialised readouts
beat the reference SE baseline, at sequence and per-position level?

Layout:

  (1) SEQ vs spilled energy — mean reference-LM SE clean vs corrupted.
      Identical across arms. For EBMs the ``model.energy`` AUROC sits next
      to it; Δ = energy − SE shows whether the EBM machinery beats the
      reference baseline.

  (2) PER-POSITION spilled-energy localisation — AUROC of corrupted vs
      unchanged positions, also identical across arms (reference). EqM
      and SFLMEBM family also report ``‖∇E‖`` per-position with the same
      delta framing.

Earlier versions computed SE under *each arm's* own per-position logits.
That conflates "is this arm's logit head sharp on the placed token?" with
"is this sequence anomalous under a fixed LM?" and, for arms with an
identity encode→decode path (SFLM), pegged seq_se_auroc at 1.0 trivially
because clean SE was identically 0. The reference-LM design closes both
confounds.

Reads runs/sflm_bench_<scale>/<model>/epoch_final.pt (see
scripts/train_for_sflm_bench.py) and falls back to the well-trained
existing baseline checkpoints in runs/ where a fresh one is absent.
Writes runs/sflm_bench_<scale>/bench.json + a printed table.

Usage:  python scripts/bench_sflm_ebm.py --scale {local,cluster} [--n 256]
                                          [--ref-model DFM]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.append(str(_ROOT))

import aitchinson_flow.models  # noqa: E402,F401
from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.data.corruption import (  # noqa: E402
    corrupt_token_ids,
    partially_shuffle_token_ids,
)
from aitchinson_flow.data.transforms import token_ids_to_features  # noqa: E402
from aitchinson_flow.models import build_model  # noqa: E402
from aitchinson_flow.training import build_training_datamodule  # noqa: E402
from scripts.eval_full import _config_from_payload  # noqa: E402
from scripts._ensure_ckpt import ensure_checkpoint  # noqa: E402
from scripts.train_for_sflm_bench import SCALES  # noqa: E402  (L per scale)

MODELS = [
    # EBMs with a native energy field (sequence + per-position readouts).
    "SFLMEBM", "SFLMEBM_FM",
    # EqM family on the simplex / latent space.
    "EqM", "EqM_OneHot", "EqMLatent",
    # Time-conditioned transport generators (no native energy; bench uses
    # spilled energy as the universal baseline).
    "SFLM", "DirichletFM", "DFM",
    # Two-stage SVGP detectors. Load model_with_svgp_hinge.pt (post-hoc
    # Stage-2 fit); SVGP latent mean → Bernoulli probability is the
    # sequence-level OOD score. Per-position remains SE (SVGP is
    # seq-level by design — `DFM_SVGP_FINDINGS.md`).
    "DFM_SVGP", "SFLM_SVGP",
]

# Existing baseline SVGP checkpoint (DFM_SVGP_FINDINGS Stage-2 run).
EXISTING_SVGP_BASELINES = {
    "DFM_SVGP": "runs/dfm_svgp_pure50_lr3e4/model_with_svgp_hinge.pt",
}


def _auroc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """AUROC with `pos` the class that should score higher."""
    pos, neg = pos.flatten().float().cpu(), neg.flatten().float().cpu()
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    comb = torch.cat([pos, neg])
    order = comb.argsort()
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, comb.numel() + 1, dtype=torch.float)
    return float(
        (ranks[: pos.numel()].sum() - pos.numel() * (pos.numel() + 1) / 2)
        / (pos.numel() * neg.numel())
    )


_NOISE_LEVEL = 0.7
"""Common signal level γ/α/t at which every arm is forwarded — high enough
that the denoiser/EBM is well-conditioned on the input, low enough that the
arm's training distribution is matched.  Used now only for the per-arm
``model.energy`` and ``position_uncertainty`` readouts (and for the
reference LM's own forward); the SE baseline is universal — see the
reference-LM SE block in ``main()``."""


def _uniform_sphere(shape, device, dtype=torch.float32):
    x = torch.randn(shape, device=device, dtype=dtype)
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def _slerp(p, q, alpha):
    omega = torch.arccos((p * q).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7)).unsqueeze(-1)
    s = torch.sin(omega).clamp(min=1e-7)
    a = alpha.unsqueeze(-1) if alpha.dim() == p.dim() - 1 else alpha
    return torch.sin((1.0 - a) * omega) / s * p + torch.sin(a * omega) / s * q


@torch.no_grad()
def _per_pos_logits(model, name: str, ids: torch.Tensor, cfg: Config):
    """(B, L, K) per-position logits via each model's natural path, and the
    sequence-level energy score (None ⇒ fall back to mean SE).

    All arms are scored at a *noised* input at a common signal level
    (``_NOISE_LEVEL`` ≈ data-end of the noise schedule).  This closes the
    encode→decode identity shortcut for SFLM/EqM/EqMLatent — without it
    those arms reported SE ≈ 0 on clean *and* corrupted inputs and the
    AUROC collapsed to chance or pegged at 1.0 depending on how the
    residual stream leaked.  DFM and DirichletFM already noised their
    inputs (this is just standardising the level)."""
    K = cfg.text8_dataset.K
    device = ids.device
    g = _NOISE_LEVEL
    # ---- 2-stage SVGP arms.
    if name == "DFM_SVGP":
        t_max = float(model.dfm.t_max)
        t = torch.full((ids.shape[0],), 1.0 + g * (t_max - 1.0),
                       device=device, dtype=torch.float32)
        x_t = model.dfm._sample_xt(ids.long(), t)
        return model.forward(x_t, t).log_softmax(dim=-1), None
    if name == "SFLM_SVGP":
        # Noised latent path mirrors SFLM's scoring (see below).
        z1 = model.encode(ids)
        z0 = _uniform_sphere(z1.shape, device, z1.dtype)
        alpha = torch.full((ids.shape[0],), g, device=device, dtype=z1.dtype)
        z_t = _slerp(z0, z1, alpha)
        # SFLM_SVGP doesn't accept a γ; decode_to_logprobs uses eval_gamma.
        return model.decode_to_logprobs(z_t), None
    # ---- Stage-1 generators / EBMs.
    if name == "DFM":
        t = torch.full((ids.shape[0],), g, device=device)
        # DFM's input space is token ids; the path is a Bernoulli corrupt
        # at rate (1−κ_t).  Forward(ids, t) on clean ids at the noised t
        # already matches the model's training distribution at that t.
        return model.forward(ids, t), None
    if name == "SFLM":
        z1 = model.encode(ids)
        z0 = _uniform_sphere(z1.shape, device, z1.dtype)
        alpha = torch.full((ids.shape[0],), g, device=device, dtype=z1.dtype)
        z_t = _slerp(z0, z1, alpha)
        gamma = alpha
        return model.decode_to_logprobs(z_t, gamma=gamma), None
    if name in ("EqM", "EqM_OneHot"):
        feats = token_ids_to_features(
            ids, K, label_smoothing=cfg.transformation.label_smoothing
        )
        # EqM training: x_γ = (1−γ)·x0 + γ·x1, x0 = source_sigma·randn on V_d.
        sigma = float(cfg.eqm.source_sigma)
        x0 = sigma * torch.randn_like(feats)
        x0 = x0 - x0.mean(dim=-1, keepdim=True)  # V_d projection
        x_t = (1.0 - g) * x0 + g * feats
        return model.decode_to_logprobs(x_t), model.energy(x_t)
    if name == "DirichletFM":
        t_max = float(model.t_max)
        t = torch.full((ids.shape[0],), 1.0 + g * (t_max - 1.0),
                       device=device, dtype=torch.float32)
        x_t = model._sample_xt(ids.long(), t)
        return model.forward(x_t, t).log_softmax(dim=-1), None
    # SFLMEBM / SFLMEBM_FM / EqMLatent: learned embedding lookup, noise
    # path mirrors training (SLERP on sphere for SFLMEBM*; linear-mix on
    # learned ℝ^d for EqMLatent).
    z1 = model.encode(ids)
    is_sphere = (z1 - z1 / z1.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                 ).abs().mean() < 1e-3
    z0 = _uniform_sphere(z1.shape, device, z1.dtype) if is_sphere \
        else torch.randn_like(z1) * z1.norm(dim=-1).mean()
    if is_sphere:
        alpha = torch.full((ids.shape[0],), g, device=device, dtype=z1.dtype)
        z_t = _slerp(z0, z1, alpha)
    else:
        z_t = (1.0 - g) * z0 + g * z1
    return model.decode_to_logprobs(z_t), model.energy(z_t)


@torch.no_grad()
def _svgp_score(model, name: str, ids: torch.Tensor) -> torch.Tensor | None:
    """Per-sequence SVGP latent-mean Bernoulli probability — the calibrated
    OOD score from the 2-stage detectors. Higher = more OOD.

    Returns ``None`` if the arm has no SVGP head or the SVGP hasn't been
    fit yet (the shared ``_SVGPHead.forward`` raises
    ``"called before fit()"`` until ``fit_svgp_hinge`` has run).  In that
    case the bench falls back to spilled-energy as the seq score for that
    arm — matching the behaviour for any other no-SVGP arm.
    """
    if not name.endswith("_SVGP") or not hasattr(model, "ood_score"):
        return None
    try:
        out = model.ood_score(ids)
        return out["prob"].detach()
    except RuntimeError as e:
        print(f"[{name}] SVGP score unavailable: {e}")
        return None


@torch.no_grad()
def _spilled_energy(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """SE(i) = logsumexp_v logits_i − logits_i[token_i].  (B, L)."""
    lse = torch.logsumexp(logits, dim=-1)
    picked = logits.gather(-1, ids.long().unsqueeze(-1)).squeeze(-1)
    return lse - picked


def _position_uncertainty(model, name: str, ids: torch.Tensor, cfg: Config):
    # Models without a native per-position uncertainty field:
    # SFLM (generator), DFM (categorical denoiser), DirichletFM (categorical
    # denoiser), and the 2-stage SVGP wrappers (sequence-level only).
    if (name in ("DFM", "SFLM", "DirichletFM")
            or name.endswith("_SVGP")
            or not hasattr(model, "position_uncertainty")):
        return None
    # Simplex EqM arms have no .encode(); use CLR features instead.
    if name in ("EqM", "EqM_OneHot"):
        feats = token_ids_to_features(
            ids, cfg.text8_dataset.K,
            label_smoothing=cfg.transformation.label_smoothing,
        )
        return model.position_uncertainty(feats)
    # EqMLatent / SFLMEBM / SFLMEBM_FM — learned-embedding lookup.
    return model.position_uncertainty(model.encode(ids))


# Reusable, well-trained baseline checkpoints (d1024). No fresh retrain
# needed — these are *stronger* than a local d512 retrain, so a local
# SFLMEBM win against them is a tougher, more honest result. Scale is not
# matched (caveat); the matched apples-to-apples is `--scale cluster`,
# where train_for_sflm_bench.py trains all four at d1024. plain "DFM" has
# no reusable text8 checkpoint in runs/ → only available from the cluster
# run (runs/sflm_bench_cluster/DFM).
EXISTING_BASELINES = {
    "EqM": "runs/dphase4_lambda_recalib/epoch_final.pt",
    "EqMLatent": "runs/latent_d128_trainable_tied_ce0_ep20/epoch_final.pt",
}


def _load(name: str, device, scale: str, *,
          epochs: int, svgp_epochs: int, auto_train: bool):
    """Load Stage-1 or Stage-2 checkpoints, auto-training/-fitting any
    missing arm via :func:`ensure_checkpoint`.  Pass ``--no-auto-train``
    to revert to the legacy "skip if missing / fall back to existing
    repo baselines" behaviour."""
    ckpt = ensure_checkpoint(
        name, scale=scale, epochs=epochs, svgp_epochs=svgp_epochs,
        auto_train=auto_train,
    )
    if ckpt is None:
        return None, None
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = _config_from_payload(payload)
    model = build_model(cfg).to(device)
    state = payload.get("model_state_dict", payload)
    # SVGP arms have a Stage-1 sub-module → strict=False so unknown keys
    # from a Stage-1 fallback don't error.
    model.load_state_dict(state, strict=not name.endswith("_SVGP"))
    model.eval()
    return model, cfg


def _parse_levels(spec: str) -> list[float]:
    return [float(x) for x in spec.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--scale",
                    choices=["local", "cluster", "a100_20g", "a100_20g_L256"],
                    default="local",
                    help="which sflm_bench_<scale> checkpoints to benchmark; "
                         "a100_20g = d1024/12L medium-tier on the A100 MIG; "
                         "a100_20g_L256 = d512/6L at L=256 (text8 publication "
                         "convention — matches SEDD / D3PM / TXL)")
    ap.add_argument("--epochs", type=int, default=50,
                    help="epochs for any arm that needs auto-training")
    ap.add_argument("--svgp-epochs", type=int, default=5,
                    help="epochs for any SVGP arm that needs auto-fitting")
    ap.add_argument("--no-auto-train", action="store_true",
                    help="revert to legacy 'skip / use existing baseline' "
                         "behaviour when a checkpoint is missing")
    ap.add_argument("--ref-model", default="DFM", choices=MODELS,
                    help="model used as the fixed reference LM for computing "
                         "spilled energy. Loaded once; its per-position "
                         "logits are reused to score every other arm, so "
                         "SE columns are identical across arms (universal "
                         "baseline). Default: DFM (well-trained char-level "
                         "denoiser; closest thing to a calibrated text8 LM "
                         "in the repo).")
    ap.add_argument("--subst-levels", type=str,
                    default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
                    help="substitution corruption rates to sweep")
    ap.add_argument("--shuffle-levels", type=str,
                    default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
                    help="partial-shuffle rates to sweep")
    ap.add_argument("--out", type=str, default=None,
                    help="output JSON path (default: <root>/bench.json)")
    args = ap.parse_args()
    root = f"runs/sflm_bench_{args.scale}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    subst_levels = _parse_levels(args.subst_levels)
    shuffle_levels = _parse_levels(args.shuffle_levels)

    # Held-out text8 TEST split — disjoint from train (which is used to
    # fit the models) AND from val (which we reserve for hyperparameter
    # selection / dev iteration).  Publication-quality numbers must come
    # from a split the model never touched at training or selection time;
    # val violates that if anyone ever picked a checkpoint by val metric.
    base_cfg = Config()
    from dataclasses import replace
    L_eval = SCALES[args.scale].get("L", 40)
    base_cfg.training = replace(base_cfg.training, L=L_eval)
    base_cfg.text8_dataset = replace(base_cfg.text8_dataset, L=L_eval)
    dm, _ = build_training_datamodule(base_cfg)
    eval_split = getattr(dm.splits, "test", None)
    if eval_split is None or eval_split.numel() == 0:
        eval_split = dm.splits.val
        print("[warn] dm.splits.test is empty — falling back to val.")
    eval_split = eval_split.long()
    g = torch.Generator().manual_seed(args.seed)
    pick = torch.randperm(eval_split.shape[0], generator=g)[: args.n]
    clean = eval_split[pick].to(device)
    K = base_cfg.text8_dataset.K

    # Corruption ladder. `mask` marks changed positions (per-position GT).
    # Subst at r=1.0 with the "avoid-same-token" guard is a derangement (all
    # positions different); rand is the true uniform-random control kept
    # alongside as a sanity check.
    corruptions: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for r in subst_levels:
        c = corrupt_token_ids(clean, vocab_size=K, corrupt_rate=r,
                              seed=args.seed)
        corruptions[f"subst_{r:.1f}"] = (c, c != clean)
    for r in shuffle_levels:
        sh = partially_shuffle_token_ids(clean, shuffle_rate=r,
                                         seed=args.seed + 7)
        corruptions[f"shuffle_{r:.1f}"] = (sh, sh != clean)
    rnd = torch.randint(0, K, clean.shape, device=device,
                         generator=torch.Generator(device=device).manual_seed(args.seed))
    corruptions["rand"] = (rnd, torch.ones_like(rnd, dtype=torch.bool))

    # --- Reference LM for spilled energy (universal baseline) --------------
    # Load the reference model ONCE and cache its per-position logits on
    # clean + every corruption.  Every test arm is then scored against
    # these reference logits, so seq_se_auroc / pospair_se_auroc are
    # identical across arms.  The arm-specific readouts (model.energy,
    # position_uncertainty) remain per-arm.
    print(f"[ref] loading reference LM '{args.ref_model}' "
          f"for spilled-energy baseline …")
    ref_model, ref_cfg = _load(
        args.ref_model, device, args.scale,
        epochs=args.epochs, svgp_epochs=args.svgp_epochs,
        auto_train=not args.no_auto_train,
    )
    if ref_model is None:
        raise RuntimeError(
            f"Reference LM '{args.ref_model}' could not be loaded — "
            "spilled-energy baseline requires a working reference. "
            "Pass --ref-model to choose a different arm, or ensure the "
            "checkpoint exists / auto-train succeeds."
        )
    ref_cl_logits, _ = _per_pos_logits(ref_model, args.ref_model, clean, ref_cfg)
    ref_cl_SE = _spilled_energy(ref_cl_logits, clean)         # (B, L)
    ref_cl_seq_se = ref_cl_SE.mean(-1)                        # (B,)
    ref_co_SE: dict[str, torch.Tensor] = {}
    for cname, (cids, _cmask) in corruptions.items():
        co_logits, _ = _per_pos_logits(ref_model, args.ref_model, cids, ref_cfg)
        ref_co_SE[cname] = _spilled_energy(co_logits, cids)   # (B, L)
    ref_se_clean_mean = float(ref_cl_SE.mean())
    print(f"[ref] clean SE mean = {ref_se_clean_mean:.4f} "
          f"(reference: {args.ref_model})")
    # Free the reference forward graph; the cached SE tensors are all we need.
    del ref_model

    results: dict = {}
    for name in MODELS:
        model, cfg = _load(
            name, device, args.scale,
            epochs=args.epochs, svgp_epochs=args.svgp_epochs,
            auto_train=not args.no_auto_train,
        )
        if model is None:
            print(f"[skip] {name}: no checkpoint and auto-train failed/disabled")
            continue

        # Arm-specific energy field (used only for the model.energy AUROC
        # below; we discard the arm's own logits — SE comes from the fixed
        # reference LM).
        _, cl_E = _per_pos_logits(model, name, clean, cfg)
        cl_pu = _position_uncertainty(model, name, clean, cfg)
        # Universal spilled-energy baseline — reference-LM SE, identical
        # across arms by construction.
        cl_seq_se = ref_cl_seq_se

        # Generation metric — reconstruction NLL on held-out clean ids.
        # Each model exposes ``.bpd()`` in its natural geometry (EqM:
        # NAG-GD recovery NLL; SFLMEBM/SFLM: SLERP recovery NLL;
        # DirichletFM/DFM: pure denoiser NLL at high t). We report PPL,
        # BPB, and BPC — all derived from the same per-token NLL.  On
        # text8 each token is one lowercase-ASCII byte, so BPB ≡ BPC;
        # both columns are shown for cross-paper comparability (BPC is
        # the legacy text8 number; BPB is the modern LM convention).
        try:
            bpc_val = float(model.bpd(clean, max_steps=64))   # = NLL/log(2)
            ppl_val = float(2.0 ** bpc_val)                    # 2^BPC
            bpb_val = bpc_val                                   # text8: byte == char
        except Exception as e:  # noqa: BLE001 — eval-time best-effort
            print(f"[{name}] bpd unavailable: {type(e).__name__}: {e}")
            bpc_val = ppl_val = bpb_val = float("nan")

        # Reference-LM SE baseline + (optional) model-natural energy for EBMs
        # + (optional) SVGP Bernoulli probability for 2-stage detectors.
        cl_svgp = _svgp_score(model, name, clean)
        # Diagnostics — clean_se_mean is now the *reference* LM's clean SE
        # (identical across arms; useful as a sanity check that the same
        # reference loaded successfully).  clean_energy_mean stays per-arm
        # so the sign convention of each EBM's energy head is still visible.
        cl_se_mean = ref_se_clean_mean
        cl_energy_mean = (
            float(cl_E.mean()) if cl_E is not None else None
        )
        res = {
            "ppl": ppl_val,
            "bpb": bpb_val,
            "bpc": bpc_val,
            "clean_se_mean": cl_se_mean,
            "clean_energy_mean": cl_energy_mean,
            "seq_se_auroc": {},        # universal baseline (every model)
            "seq_se_clean": cl_se_mean,
            "seq_se_corrupt": {},      # mean SE on each corruption type
            "seq_energy_auroc": {},    # model.energy if available (EBMs)
            "seq_energy_corrupt": {},  # mean energy on each corruption
            "seq_svgp_auroc": {},      # SVGP prob if available (2-stage)
            "pospair_se_auroc": {},    # universal per-position baseline
            "pospair_pu_auroc": {},    # ‖∇E‖ if available (EBMs)
        }
        # Initialise sign-agnostic energy reporting (filled per corruption).
        res["seq_energy_auroc_signfree"] = {}
        res["seq_energy_sign"] = {}
        for cname, (cids, cmask) in corruptions.items():
            # Arm-specific energy field; logits discarded — SE comes from
            # the cached reference-LM logits computed once above.
            _, co_E = _per_pos_logits(model, name, cids, cfg)
            co_SE = ref_co_SE[cname]
            res["seq_se_corrupt"][cname] = float(co_SE.mean())
            # (1a) BASELINE — mean reference-LM spilled energy clean vs corrupted.
            res["seq_se_auroc"][cname] = _auroc(co_SE.mean(-1), cl_seq_se)
            # (1b) Model-natural energy (EBMs only) — what beats baseline?
            if cl_E is not None and co_E is not None:
                raw = _auroc(co_E, cl_E)
                res["seq_energy_auroc"][cname] = raw
                # Sign-agnostic: corrupt vs clean is invariant under sign
                # flip of the energy head, so max(auroc, 1-auroc) recovers
                # the discriminative power even when the energy was
                # trained with the "wrong" sign.  Sign = "+" if higher
                # energy ⇒ more OOD (the expected EBM convention), "−"
                # if the model learned the inverted convention.
                sign_free = max(raw, 1.0 - raw)
                res["seq_energy_auroc_signfree"][cname] = sign_free
                res["seq_energy_sign"][cname] = "+" if raw >= 0.5 else "−"
                res["seq_energy_corrupt"][cname] = float(co_E.mean())
            # (1c) SVGP probability (2-stage only).
            if cl_svgp is not None:
                co_svgp = _svgp_score(model, name, cids)
                res["seq_svgp_auroc"][cname] = _auroc(co_svgp, cl_svgp)
            # (2a) Per-position SE localisation (universal baseline).
            if cmask.any() and (~cmask).any():
                res["pospair_se_auroc"][cname] = _auroc(
                    co_SE[cmask], co_SE[~cmask]
                )
                # (2b) Per-position ‖∇E‖ localisation (EBMs).
                if cl_pu is not None:
                    co_pu = _position_uncertainty(model, name, cids, cfg)
                    res["pospair_pu_auroc"][cname] = _auroc(
                        co_pu[cmask], co_pu[~cmask]
                    )
        results[name] = res

    out_path = Path(args.out) if args.out else Path(f"{root}/bench.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    # ---- report ----
    def _fmt(x, w=6, p=3):
        return f"{x:>{w}.{p}f}" if isinstance(x, float) else f"{'—':>{w}}"

    subst_names = [f"subst_{r:.1f}" for r in subst_levels]
    shuffle_names = [f"shuffle_{r:.1f}" for r in shuffle_levels]
    rand_names = ["rand"]

    def _section(title, cnames, key, models_iter):
        print("\n" + "=" * (14 + 7 * len(cnames)))
        print(title)
        print("=" * (14 + 7 * len(cnames)))
        head = f"{'model':12s}"
        for c in cnames:
            tag = c.replace("subst_", "s").replace("shuffle_", "sh").replace("rand", "rnd")
            head += f"  {tag:>5s}"
        print(head)
        for n, r in models_iter:
            row = f"{n:12s}"
            for c in cnames:
                row += f"  {_fmt(r[key].get(c, float('nan')), 5)}"
            print(row)

    print("\n" + "=" * 110)
    print("(0) GENERATION  —  reconstruction NLL (per-arm) + reference SE_clean "
          "(identical across rows).  ⚠ bpc≈0 for arms whose bpd() reads the "
          "input through .encode()→.decode_to_logprobs() (SFLM/EqM/EqM_OneHot/"
          f"EqMLatent) — that path is essentially identity. SE_clean is the "
          f"fixed reference LM '{args.ref_model}' (universal baseline); "
          "E_clean is each EBM's own energy head.")
    print("    (text8: BPB ≡ BPC since each token is one ASCII byte)")
    print("=" * 110)
    print(f"{'model':12s}  {'PPL':>8s}  {'BPB':>8s}  {'BPC':>8s}  "
          f"{'SE_clean':>10s}  {'E_clean':>10s}")
    for n, r in results.items():
        e = r.get("clean_energy_mean")
        e_s = _fmt(e, 10) if isinstance(e, float) else f"{'—':>10s}"
        print(f"{n:12s}  {_fmt(r['ppl'], 8)}  {_fmt(r['bpb'], 8)}  "
              f"{_fmt(r['bpc'], 8)}  {_fmt(r['clean_se_mean'], 10)}  {e_s}")

    items = list(results.items())

    # Sequence-level — one section per corruption family.
    _section(
        "(1a) SEQ-LEVEL SE AUROC (clean vs subst_*; 0.5=chance, 1=perfect)",
        subst_names, "seq_se_auroc", items,
    )
    _section(
        "(1b) SEQ-LEVEL SE AUROC (clean vs shuffle_*)",
        shuffle_names, "seq_se_auroc", items,
    )
    _section(
        "(1c) SEQ-LEVEL SE AUROC (clean vs rand uniform)",
        rand_names, "seq_se_auroc", items,
    )

    ebm_items = [(n, r) for n, r in items if r["seq_energy_auroc"]]
    if ebm_items:
        _section(
            "(1d) SEQ-LEVEL ENERGY AUROC RAW (EBMs; subst_*).  "
            "Rows <0.5 ⇒ the energy head learned an inverted sign "
            "convention — see (1d') for the sign-agnostic version.",
            subst_names, "seq_energy_auroc", ebm_items,
        )
        _section(
            "(1d') SEQ-LEVEL ENERGY AUROC SIGN-AGNOSTIC = max(raw, 1−raw); "
            "this is the discriminative power independent of sign convention.",
            subst_names, "seq_energy_auroc_signfree", ebm_items,
        )
        _section(
            "(1e) SEQ-LEVEL ENERGY AUROC RAW (EBMs; shuffle_*)",
            shuffle_names, "seq_energy_auroc", ebm_items,
        )
        _section(
            "(1e') SEQ-LEVEL ENERGY AUROC SIGN-AGNOSTIC (shuffle_*)",
            shuffle_names, "seq_energy_auroc_signfree", ebm_items,
        )
        _section(
            "(1f) SEQ-LEVEL ENERGY AUROC RAW (EBMs; rand)",
            rand_names, "seq_energy_auroc", ebm_items,
        )
        _section(
            "(1f') SEQ-LEVEL ENERGY AUROC SIGN-AGNOSTIC (rand)",
            rand_names, "seq_energy_auroc_signfree", ebm_items,
        )

    svgp_items = [(n, r) for n, r in items if r["seq_svgp_auroc"]]
    if svgp_items:
        _section(
            "(1g) SVGP Bernoulli-prob AUROC (subst_*)",
            subst_names, "seq_svgp_auroc", svgp_items,
        )
        _section(
            "(1h) SVGP Bernoulli-prob AUROC (shuffle_*)",
            shuffle_names, "seq_svgp_auroc", svgp_items,
        )

    # Per-position localisation. rand has no clean positions ⇒ skipped.
    _section(
        "(2a) PER-POSITION SE AUROC (changed vs unchanged inside corrupted; "
        "subst_*).  Subst at rate 1.0 has no clean positions ⇒ entry absent.",
        subst_names, "pospair_se_auroc", items,
    )
    _section(
        "(2b) PER-POSITION SE AUROC (shuffle_*)",
        shuffle_names, "pospair_se_auroc", items,
    )

    pu_items = [(n, r) for n, r in items if r["pospair_pu_auroc"]]
    if pu_items:
        _section(
            "(2c) PER-POSITION ‖∇E‖ AUROC (EBMs; subst_*)",
            subst_names, "pospair_pu_auroc", pu_items,
        )
        _section(
            "(2d) PER-POSITION ‖∇E‖ AUROC (EBMs; shuffle_*)",
            shuffle_names, "pospair_pu_auroc", pu_items,
        )

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
