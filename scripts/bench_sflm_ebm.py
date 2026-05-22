"""Spilled-energy benchmark on text8.

**Spilled energy is the baseline** ``SE(i) = logsumexp_v logits_i −
logits_i[token_i]`` — per-position NLL of the placed token, defined
uniformly for every model (all expose per-position logits). It's the
trivial single-forward OOD signal. The competing question for an EBM
is whether its specialised readouts (sequence-level ``model.energy``,
per-position Riemannian ``‖∇E‖``) *beat plain SE* on the same model
and corpus, at sequence and per-position level.

Layout:

  (1) SEQ vs spilled energy — mean SE clean vs corrupted, **reported
      for every model** (the universal baseline). For EBMs we also
      report ``model.energy`` AUROC next to it; Δ = energy − SE shows
      whether the EBM machinery beats the baseline.

  (2) PER-POSITION spilled-energy localisation — AUROC of corrupted
      vs unchanged positions inside corrupted seqs (universal
      baseline). EqM-family models also report ``‖∇E‖`` per-position
      with the same delta framing.

Reads runs/sflm_bench_<scale>/<model>/epoch_final.pt (see
scripts/train_for_sflm_bench.py) and falls back to the well-trained
existing baseline checkpoints in runs/ where a fresh one is absent.
Writes runs/sflm_bench_<scale>/bench.json + a printed table.

Usage:  python scripts/bench_sflm_ebm.py --scale {local,cluster} [--n 256]
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


@torch.no_grad()
def _per_pos_logits(model, name: str, ids: torch.Tensor, cfg: Config):
    """(B, L, K) per-position logits via each model's natural path, and the
    sequence-level energy score (None ⇒ fall back to mean SE)."""
    K = cfg.text8_dataset.K
    # ---- 2-stage SVGP arms: delegate logits to the underlying generator,
    #      sequence-energy is the SVGP probability (handled separately
    #      in main()).
    if name == "DFM_SVGP":
        t = torch.full((ids.shape[0],), 0.99, device=ids.device)
        return model.forward(ids, t), None
    if name == "SFLM_SVGP":
        z = model.encode(ids)
        return model.decode_to_logprobs(z), None
    # ---- Stage-1 generators / EBMs as before.
    if name == "DFM":
        t = torch.full((ids.shape[0],), 0.99, device=ids.device)
        return model.forward(ids, t), None
    if name == "SFLM":
        z = model.encode(ids)
        return model.decode_to_logprobs(z), None
    if name in ("EqM", "EqM_OneHot"):
        feats = token_ids_to_features(
            ids, K, label_smoothing=cfg.transformation.label_smoothing
        )
        return model.decode_to_logprobs(feats), model.energy(feats)
    if name == "DirichletFM":
        t = torch.full((ids.shape[0],), 0.99 * model.t_max,
                       device=ids.device, dtype=torch.float32)
        # DFM-family: input must be a simplex draw conditioned on ids.
        x_t = model._sample_xt(ids.long(), t)
        return model.forward(x_t, t).log_softmax(dim=-1), None
    # SFLMEBM / SFLMEBM_FM / EqMLatent: learned embedding lookup.
    z = model.encode(ids)
    return model.decode_to_logprobs(z), model.energy(z)


@torch.no_grad()
def _svgp_score(model, name: str, ids: torch.Tensor) -> torch.Tensor | None:
    """Per-sequence SVGP latent-mean Bernoulli probability — the calibrated
    OOD score from the 2-stage detectors. Higher = more OOD."""
    if not name.endswith("_SVGP") or not hasattr(model, "ood_score"):
        return None
    out = model.ood_score(ids)
    return out["prob"].detach()


@torch.no_grad()
def _spilled_energy(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """SE(i) = logsumexp_v logits_i − logits_i[token_i].  (B, L)."""
    lse = torch.logsumexp(logits, dim=-1)
    picked = logits.gather(-1, ids.long().unsqueeze(-1)).squeeze(-1)
    return lse - picked


def _position_uncertainty(model, name: str, ids: torch.Tensor, cfg: Config):
    if name in ("DFM", "SFLM") or not hasattr(model, "position_uncertainty"):
        return None
    if name == "EqM":
        feats = token_ids_to_features(
            ids, cfg.text8_dataset.K,
            label_smoothing=cfg.transformation.label_smoothing,
        )
        return model.position_uncertainty(feats)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--scale",
                    choices=["local", "cluster", "a100_20g"],
                    default="local",
                    help="which sflm_bench_<scale> checkpoints to benchmark; "
                         "a100_20g = d1024/12L medium-tier on the A100 MIG")
    ap.add_argument("--epochs", type=int, default=50,
                    help="epochs for any arm that needs auto-training")
    ap.add_argument("--svgp-epochs", type=int, default=5,
                    help="epochs for any SVGP arm that needs auto-fitting")
    ap.add_argument("--no-auto-train", action="store_true",
                    help="revert to legacy 'skip / use existing baseline' "
                         "behaviour when a checkpoint is missing")
    args = ap.parse_args()
    root = f"runs/sflm_bench_{args.scale}"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Clean held-out text8 windows (datamodule is model-independent here).
    base_cfg = Config()
    dm, _ = build_training_datamodule(base_cfg)
    val = dm.splits.val.long()
    g = torch.Generator().manual_seed(args.seed)
    pick = torch.randperm(val.shape[0], generator=g)[: args.n]
    clean = val[pick].to(device)
    K = base_cfg.text8_dataset.K

    # Corruption ladder. `mask` marks changed positions (per-position GT).
    corruptions = {}
    for r in (0.1, 0.3):
        c = corrupt_token_ids(clean, vocab_size=K, corrupt_rate=r, seed=args.seed)
        corruptions[f"subst_{r}"] = (c, c != clean)
    sh = partially_shuffle_token_ids(clean, shuffle_rate=1.0, seed=args.seed + 7)
    corruptions["shuffle"] = (sh, sh != clean)
    rnd = torch.randint(0, K, clean.shape, device=device,
                         generator=torch.Generator(device=device).manual_seed(args.seed))
    corruptions["rand"] = (rnd, torch.ones_like(rnd, dtype=torch.bool))

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

        cl_logits, cl_E = _per_pos_logits(model, name, clean, cfg)
        cl_SE = _spilled_energy(cl_logits, clean)               # (B, L)
        cl_seq_se = cl_SE.mean(-1)                              # baseline (B,)
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

        # Universal SE baseline + (optional) model-natural energy for EBMs
        # + (optional) SVGP Bernoulli probability for 2-stage detectors.
        cl_svgp = _svgp_score(model, name, clean)
        res = {
            "ppl": ppl_val,
            "bpb": bpb_val,
            "bpc": bpc_val,
            "seq_se_auroc": {},        # universal baseline (every model)
            "seq_energy_auroc": {},    # model.energy if available (EBMs)
            "seq_svgp_auroc": {},      # SVGP prob if available (2-stage)
            "pospair_se_auroc": {},    # universal per-position baseline
            "pospair_pu_auroc": {},    # ‖∇E‖ if available (EBMs)
        }
        for cname, (cids, cmask) in corruptions.items():
            co_logits, co_E = _per_pos_logits(model, name, cids, cfg)
            co_SE = _spilled_energy(co_logits, cids)
            # (1a) BASELINE — mean spilled energy clean vs corrupted.
            res["seq_se_auroc"][cname] = _auroc(co_SE.mean(-1), cl_seq_se)
            # (1b) Model-natural energy (EBMs only) — what beats baseline?
            if cl_E is not None and co_E is not None:
                res["seq_energy_auroc"][cname] = _auroc(co_E, cl_E)
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

    Path(root).mkdir(parents=True, exist_ok=True)
    Path(f"{root}/bench.json").write_text(json.dumps(results, indent=2))

    # ---- report ----
    def _fmt(x, w=7):
        return f"{x:>{w}.3f}" if isinstance(x, float) else f"{'—':>{w}}"

    def _delta(a, b, w=7):
        if not (isinstance(a, float) and isinstance(b, float)):
            return f"{'—':>{w}}"
        d = a - b
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:>{w-1}.3f}"

    cs = list(corruptions)
    loc = [c for c in cs if c != "rand"]   # rand changes all positions

    print("\n" + "=" * 90)
    print("(0) GENERATION  —  reconstruction perplexity / bits-per-byte / "
          "bits-per-char on val")
    print("    (text8: BPB ≡ BPC since each token is one ASCII byte;"
          " BPC kept as legacy column)")
    print("=" * 90)
    print(f"{'model':12s}  {'PPL':>8s}  {'BPB':>8s}  {'BPC':>8s}")
    for n, r in results.items():
        print(f"{n:12s}  {_fmt(r['ppl'], 8)}  {_fmt(r['bpb'], 8)}  "
              f"{_fmt(r['bpc'], 8)}")

    print("\n" + "=" * 90)
    print("(1) SEQUENCE-level  AUROC  (clean vs corrupted; 0.5=chance, 1=perfect)")
    print("    BASELINE = mean spilled energy (universal).  EBMs also report "
          "model.energy.")
    print("    Δ = energy − SE  →  positive means EBM beats the baseline.")
    print("=" * 90)
    header = f"{'model':12s}"
    for c in cs:
        header += f"  {c+'/SE':>10s} {c+'/E':>8s} {'Δ':>8s}"
    print(header)
    for n, r in results.items():
        row = f"{n:12s}"
        for c in cs:
            se = r["seq_se_auroc"].get(c, float("nan"))
            e = r["seq_energy_auroc"].get(c)
            row += f"  {_fmt(se, 10)} {_fmt(e, 8)} {_delta(e, se, 8)}"
        print(row)

    # 2-stage SVGP table — only relevant for arms that have a fitted head.
    svgp_arms = [n for n, r in results.items() if r["seq_svgp_auroc"]]
    if svgp_arms:
        print("\n" + "=" * 90)
        print("(1b) 2-STAGE SVGP DETECTORS — Bernoulli prob AUROC")
        print("     Δ-vs-baseline = SVGP_prob_AUROC − spilled_energy_AUROC.")
        print("=" * 90)
        header = f"{'arm':12s}"
        for c in cs:
            header += f"  {c+'/SE':>10s} {c+'/SVGP':>10s} {'Δ':>8s}"
        print(header)
        for n in svgp_arms:
            r = results[n]
            row = f"{n:12s}"
            for c in cs:
                se = r["seq_se_auroc"].get(c, float("nan"))
                sv = r["seq_svgp_auroc"].get(c)
                row += f"  {_fmt(se, 10)} {_fmt(sv, 10)} {_delta(sv, se, 8)}"
            print(row)

    print("\n" + "=" * 90)
    print("(2) PER-POSITION  AUROC  (changed vs unchanged inside corrupted seqs)")
    print("    BASELINE = spilled energy per position.  EBMs also report "
          "Riemannian ‖∇E‖.")
    print("    Δ = ‖∇E‖ − SE  →  positive means EBM machinery beats the baseline.")
    print("=" * 90)
    header = f"{'model':12s}"
    for c in loc:
        header += f"  {c+'/SE':>10s} {c+'/∇E':>8s} {'Δ':>8s}"
    print(header)
    for n, r in results.items():
        row = f"{n:12s}"
        for c in loc:
            se = r["pospair_se_auroc"].get(c, float("nan"))
            pu = r["pospair_pu_auroc"].get(c)
            row += f"  {_fmt(se, 10)} {_fmt(pu, 8)} {_delta(pu, se, 8)}"
        print(row)
    print(f"\nWrote {root}/bench.json")


if __name__ == "__main__":
    main()
