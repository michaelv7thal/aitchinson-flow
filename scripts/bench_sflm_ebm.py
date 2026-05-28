"""Spilled-energy benchmark on text8.

Two orthogonal axes:

  - **Per-arm OOD readouts** — every model reports OOD AUROCs computed from
    its own per-position logits / energy field: spilled energy SE under its
    own decoder, ``model.energy`` (EBMs), Riemannian ``‖∇E‖`` (EBMs),
    SVGP Bernoulli probability (2-stage detectors).

  - **External LLM reference** (``--ref-lm gpt2``) — spilled energy under a
    fixed pretrained LM, computed once on the same clean/corrupted text and
    reported as the ``gpt2_baseline`` row in the SE tables. This is the
    apples-to-apples external baseline the per-arm OOD scores should match
    or beat.  GPT-2 BPE doesn't align 1:1 with text8 chars, so per-char SE
    is obtained by distributing each BPE token's NLL across the characters
    it spans (offset-mapping).

Layout:

  (0) Generation — each arm's reconstruction NLL (PPL / BPB / BPC) and
      diagnostic ``clean_se_mean`` / ``clean_energy_mean``.

  (1) SEQ-LEVEL AUROC — for each arm, AUROC of corrupted vs clean
      *sequence-mean spilled energy* (per-arm SE). The ``gpt2_baseline``
      row alongside is the GPT-2 SE AUROC at the same task. Δ vs the
      gpt2_baseline row tells us whether the per-arm SE score is on par
      with the LLM reference.  EBM-only sections (1d/1e/1f) add
      ``model.energy`` AUROC; SVGP arms add ``ood_score`` AUROC.

  (2) PER-POSITION AUROC — AUROC of corrupted-position SE vs unchanged-
      position SE *inside* corrupted sequences, again per-arm with the
      ``gpt2_baseline`` row alongside.  EBMs add ``‖∇E‖`` per-position.

Reads runs/sflm_bench_<scale>/<model>/epoch_final.pt (see
scripts/train_for_sflm_bench.py) and falls back to the well-trained
existing baseline checkpoints in runs/ where a fresh one is absent.
Writes runs/sflm_bench_<scale>/bench.json + a printed table.

Usage:  python scripts/bench_sflm_ebm.py --scale {local,cluster} [--n 256]
                                          [--ref-lm gpt2]
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
    a = alpha
    while a.dim() < p.dim():
        a = a.unsqueeze(-1)
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


# --- External LLM reference (GPT-2) -------------------------------------------
# Per-position SE under a fixed pretrained LM.  Computed once on the same
# clean / corrupted text the arms see and reported as the universal baseline.
# Implementation: BPE-tokenise the decoded char string with offset mapping,
# forward GPT-2 once, get per-BPE-token NLL, then distribute each token's NLL
# uniformly across the characters it spans.
_TEXT8_ALPHABET = "abcdefghijklmnopqrstuvwxyz "


def _ids_to_text(char_ids: torch.Tensor) -> list[str]:
    """(B, L) text8 char ids → list of B strings of length L."""
    table = _TEXT8_ALPHABET
    out: list[str] = []
    for row in char_ids.detach().cpu().tolist():
        out.append("".join(table[c] for c in row))
    return out


@torch.no_grad()
def _gpt2_per_char_SE(
    model, tokenizer, char_ids: torch.Tensor, *, chunk: int = 32
) -> torch.Tensor:
    """Per-character spilled energy under GPT-2.

    Returns ``(B, L)`` of per-char NLL under the reference LM.  Each BPE
    token's NLL is divided uniformly across the chars it spans; this is
    the standard cross-tokenizer byte/char NLL convention (e.g. how
    text8 BPC is reported for GPT-2 in the literature).

    ``chunk`` controls the GPT-2 batch size — at L=256 the (B, T, V≈50k)
    log-softmax tensor alone is ~4 GB at B=256, which fragments the
    MIG-20GB allocator across repeated calls.  Process in chunks and let
    the caching allocator reuse the slab.
    """
    device = next(model.parameters()).device
    B, L = char_ids.shape
    SE = torch.zeros((B, L), dtype=torch.float32, device=device)
    bos = tokenizer.eos_token_id
    for s in range(0, B, chunk):
        e = min(s + chunk, B)
        sub_ids = char_ids[s:e]
        texts = _ids_to_text(sub_ids)
        enc = tokenizer(
            texts,
            return_offsets_mapping=True,
            return_attention_mask=True,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)            # (b, T)
        attn_mask = enc["attention_mask"].to(device)       # (b, T)
        offsets_l = enc["offset_mapping"].tolist()         # (b, T, 2)
        attn_l = attn_mask.tolist()
        # Prepend BOS = eos for conditional NLL on the first BPE token.
        b_size = input_ids.shape[0]
        bos_col = torch.full((b_size, 1), bos, dtype=input_ids.dtype, device=device)
        pad_col = torch.ones((b_size, 1), dtype=attn_mask.dtype, device=device)
        padded = torch.cat([bos_col, input_ids], dim=1)
        mask = torch.cat([pad_col, attn_mask], dim=1)
        logits = model(input_ids=padded, attention_mask=mask).logits  # (b, T+1, V)
        # logits[:, t, :] predicts padded[:, t+1] for t in 0..T-1.
        # Compute NLL without materialising a full (b, T, V) log-softmax tensor:
        # nll = log_sum_exp(logits) - logits_at(input_ids).
        pred_logits = logits[:, :-1, :].float()
        lse = torch.logsumexp(pred_logits, dim=-1)         # (b, T)
        picked = pred_logits.gather(
            -1, input_ids.unsqueeze(-1)
        ).squeeze(-1)                                       # (b, T)
        nll = (lse - picked).cpu()
        for j in range(b_size):
            for t, ((start, end), valid) in enumerate(
                zip(offsets_l[j], attn_l[j])
            ):
                if not valid or end <= start:
                    continue
                n_chars = end - start
                SE[s + j, start:end] = nll[j, t] / n_chars
        del logits, pred_logits, lse, picked, padded, mask, input_ids, attn_mask
        torch.cuda.empty_cache()
    return SE


def _load_gpt2(name: str, device):
    """Load a HuggingFace GPT-2 LM and its fast tokenizer.

    Cached under ``transformers``' default HF cache (set ``HF_HOME`` to
    relocate).  ``add_prefix_space=True`` because text8 char strings can
    start with non-space chars; without it the first BPE token would not
    get an offset that starts at 0.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, add_prefix_space=False, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
    model.to(device).eval()
    return model, tok


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
    ap.add_argument("--ref-lm", default="gpt2",
                    help="HuggingFace causal LM used as the external spilled-"
                         "energy reference (e.g. gpt2, gpt2-medium, distilgpt2). "
                         "Loaded once; per-char NLL is computed by distributing "
                         "each BPE token's NLL uniformly across the chars it "
                         "spans. Pass an empty string to disable the reference.")
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

    # --- External LLM reference (GPT-2) ----------------------------------
    # Spilled energy under a fixed pretrained LM, computed once on the same
    # clean + every corruption.  Becomes the ``gpt2_baseline`` row in the
    # AUROC tables — the universal external baseline the per-arm OOD
    # scores should match or beat.
    gpt2_baseline: dict | None = None
    if args.ref_lm:
        print(f"[ref-lm] loading HuggingFace '{args.ref_lm}' for "
              f"spilled-energy reference …")
        ref_model, ref_tok = _load_gpt2(args.ref_lm, device)
        gpt2_cl_SE = _gpt2_per_char_SE(ref_model, ref_tok, clean)        # (B, L)
        gpt2_cl_seq_se = gpt2_cl_SE.mean(-1)
        gpt2_co_SE: dict[str, torch.Tensor] = {}
        for cname, (cids, _cmask) in corruptions.items():
            gpt2_co_SE[cname] = _gpt2_per_char_SE(ref_model, ref_tok, cids)
        gpt2_cl_mean = float(gpt2_cl_SE.mean())
        print(f"[ref-lm] {args.ref_lm} clean per-char NLL mean = "
              f"{gpt2_cl_mean:.4f}")
        del ref_model
        # Pre-compute the reference AUROCs in the same dict-shape as a real arm
        # so the report code can iterate over it uniformly.
        gpt2_baseline = {
            "ppl": float("nan"), "bpb": float("nan"), "bpc": float("nan"),
            "clean_se_mean": gpt2_cl_mean, "clean_energy_mean": None,
            "seq_se_clean": gpt2_cl_mean,
            "seq_se_corrupt": {},
            "seq_se_auroc": {},
            "seq_energy_auroc": {}, "seq_energy_corrupt": {},
            "seq_svgp_auroc": {},
            "pospair_se_auroc": {},
            "pospair_pu_auroc": {},
            "seq_energy_auroc_signfree": {}, "seq_energy_sign": {},
        }
        for cname, (cids, cmask) in corruptions.items():
            co = gpt2_co_SE[cname]
            gpt2_baseline["seq_se_corrupt"][cname] = float(co.mean())
            gpt2_baseline["seq_se_auroc"][cname] = _auroc(
                co.mean(-1), gpt2_cl_seq_se
            )
            if cmask.any() and (~cmask).any():
                gpt2_baseline["pospair_se_auroc"][cname] = _auroc(
                    co[cmask], co[~cmask]
                )

    results: dict = {}
    if gpt2_baseline is not None:
        results[f"{args.ref_lm}_baseline"] = gpt2_baseline
    for name in MODELS:
        model, cfg = _load(
            name, device, args.scale,
            epochs=args.epochs, svgp_epochs=args.svgp_epochs,
            auto_train=not args.no_auto_train,
        )
        if model is None:
            print(f"[skip] {name}: no checkpoint and auto-train failed/disabled")
            continue

        # Per-arm native OOD readouts: sequence-level ``model.energy``
        # and per-position ``position_uncertainty``/‖∇E‖.  Spilled energy
        # is NOT per-arm — it lives only on the gpt2_baseline row.
        _, cl_E = _per_pos_logits(model, name, clean, cfg)
        cl_pu = _position_uncertainty(model, name, clean, cfg)

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

        # Native model-energy / SVGP readouts only — no per-arm SE.
        cl_svgp = _svgp_score(model, name, clean)
        cl_energy_mean = (
            float(cl_E.mean()) if cl_E is not None else None
        )
        res = {
            "ppl": ppl_val,
            "bpb": bpb_val,
            "bpc": bpc_val,
            "clean_energy_mean": cl_energy_mean,
            "seq_energy_auroc": {},    # model.energy if available (EBMs)
            "seq_energy_corrupt": {},  # mean energy on each corruption
            "seq_svgp_auroc": {},      # SVGP prob if available (2-stage)
            "pospair_pu_auroc": {},    # ‖∇E‖ if available (EBMs)
            "seq_energy_auroc_signfree": {},
            "seq_energy_sign": {},
        }
        for cname, (cids, cmask) in corruptions.items():
            _, co_E = _per_pos_logits(model, name, cids, cfg)
            # (1) Model-natural energy (EBMs only) — what beats the GPT-2 baseline?
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
            # (2) SVGP probability (2-stage only).
            if cl_svgp is not None:
                co_svgp = _svgp_score(model, name, cids)
                res["seq_svgp_auroc"][cname] = _auroc(co_svgp, cl_svgp)
            # (3) Per-position ‖∇E‖ localisation (EBMs).
            if cmask.any() and (~cmask).any() and cl_pu is not None:
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
    print("(0) GENERATION  —  reconstruction NLL (per-arm).  ⚠ bpc≈0 for "
          "arms whose bpd() reads the input through "
          ".encode()→.decode_to_logprobs() (SFLM/EqM/EqM_OneHot/EqMLatent) — "
          "that path is essentially identity; the calibrated reference "
          "for text8 char-NLL is the gpt2_baseline row below.  "
          "E_clean is each EBM's own energy head.")
    print("    (text8: BPB ≡ BPC since each token is one ASCII byte)")
    print("=" * 110)
    print(f"{'model':12s}  {'PPL':>8s}  {'BPB':>8s}  {'BPC':>8s}  "
          f"{'E_clean':>10s}")
    for n, r in results.items():
        e = r.get("clean_energy_mean")
        e_s = _fmt(e, 10) if isinstance(e, float) else f"{'—':>10s}"
        print(f"{n:12s}  {_fmt(r['ppl'], 8)}  {_fmt(r['bpb'], 8)}  "
              f"{_fmt(r['bpc'], 8)}  {e_s}")

    items = list(results.items())
    # SE rows come only from the gpt2_baseline entry (one row).
    se_items = [(n, r) for n, r in items if r.get("seq_se_auroc")]

    if se_items:
        _section(
            "(1a) SEQ-LEVEL spilled-energy AUROC under reference LM "
            f"({args.ref_lm}) — clean vs subst_*; 0.5=chance, 1=perfect.  "
            "This is THE external OOD baseline; the per-arm sections below "
            "(model.energy, SVGP) should be compared against this row.",
            subst_names, "seq_se_auroc", se_items,
        )
        _section(
            "(1b) SEQ-LEVEL spilled-energy AUROC under reference LM "
            f"({args.ref_lm}) — clean vs shuffle_*",
            shuffle_names, "seq_se_auroc", se_items,
        )
        _section(
            "(1c) SEQ-LEVEL spilled-energy AUROC under reference LM "
            f"({args.ref_lm}) — clean vs rand uniform",
            rand_names, "seq_se_auroc", se_items,
        )

    ebm_items = [(n, r) for n, r in items if r.get("seq_energy_auroc")]
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

    svgp_items = [(n, r) for n, r in items if r.get("seq_svgp_auroc")]
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
    if se_items:
        _section(
            "(2a) PER-POSITION spilled-energy AUROC under reference LM "
            f"({args.ref_lm}) — changed vs unchanged positions, subst_*.  "
            "Subst at rate 1.0 has no clean positions ⇒ entry absent.",
            subst_names, "pospair_se_auroc", se_items,
        )
        _section(
            "(2b) PER-POSITION spilled-energy AUROC under reference LM "
            f"({args.ref_lm}) — shuffle_*",
            shuffle_names, "pospair_se_auroc", se_items,
        )

    pu_items = [(n, r) for n, r in items if r.get("pospair_pu_auroc")]
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
