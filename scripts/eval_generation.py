"""Unified generation evaluation across the 5 generation arms used for the
capstone write-up:

  1. EqM_OneHot   — one-hot → CLR/ILR → EqM (label-smoothed CLR, no
                    Dirichlet thickening).
  2. EqMLatent    — EqM in a learned embedding space.
  3. EqM          — Dirichlet-thickened EqM (the project's tuned recipe;
                    ``cfg.transformation.dirichlet_sampling = True``).
  4. DirichletFM  — Stark et al. (2024) Dirichlet flow matching.
  5. SFLM         — time-conditioned hyperspherical flow.

For each arm computes, on a fresh seed for fairness:

  BPD                 — model.bpd() on held-out val (reconstruction NLL).
  KL_uni / KL_bi /    — n-gram KL of unconditional samples vs the train
  KL_tri / H_gen        corpus n-gram distribution.
  recover_acc(α)      — token-level recovery accuracy after perturbing
                        encoded val sequences with α·embed_norm·N(0,I)
                        and re-sampling (mirrors recovery_check.py).
                        Marked "—" for DirichletFM which has no x_init
                        sampler hook.

Writes runs/sflm_bench_<scale>/generation_eval.json + printed table.

Usage:
  python scripts/eval_generation.py --scale cluster --n 256 \
      --recover-alphas 0.1,0.3,0.5,1.0
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

import aitchinson_flow.models  # noqa: E402,F401  populate REGISTRY
from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.data.char_window_dataset import CHAR2ID  # noqa: E402
from aitchinson_flow.data.transforms import token_ids_to_features  # noqa: E402
from aitchinson_flow.models import build_model  # noqa: E402
from aitchinson_flow.training import build_training_datamodule  # noqa: E402
from scripts.eval_full import _config_from_payload  # noqa: E402

# Five-arm unified comparison. SFLMEBM* arms are *not* in this list — they
# live in the OOD bench; this file is the *generation* comparator.
GEN_ARMS = ["EqM_OneHot", "EqMLatent", "EqM", "DirichletFM", "SFLM"]
ALPHABET = "".join(sorted(CHAR2ID, key=CHAR2ID.__getitem__))

# Existing tuned baseline checkpoints (re-used when a fresh sflm_bench_<scale>
# run is absent — same fallback policy as bench_sflm_ebm.py).
EXISTING_BASELINES = {
    "EqM": "runs/dphase4_lambda_recalib/epoch_final.pt",
    "EqMLatent": "runs/latent_d128_trainable_tied_ce0_ep20/epoch_final.pt",
}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _load(arm: str, device, root: str):
    ckpt = Path(f"{root}/{arm}/epoch_final.pt")
    if not ckpt.exists() and arm in EXISTING_BASELINES:
        ckpt = Path(EXISTING_BASELINES[arm])
        print(f"[{arm}] reusing existing checkpoint {ckpt}")
    if not ckpt.exists():
        return None, None
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = _config_from_payload(payload)
    model = build_model(cfg).to(device)
    state = payload.get("model_state_dict", payload)
    model.load_state_dict(state)
    model.eval()
    return model, cfg


# --------------------------------------------------------------------------
# Corpus n-gram counts (cached per run; matches recovery_check.py logic).
# --------------------------------------------------------------------------
def _ngram_counts(ids2d: torch.Tensor, K: int, n: int) -> torch.Tensor:
    L = ids2d.shape[1]
    if L < n:
        return torch.zeros(K ** n)
    idx = torch.zeros(ids2d.shape[0], L - n + 1, dtype=torch.long)
    for i in range(n):
        idx = idx + ids2d[:, i:L - n + 1 + i].long() * (K ** (n - 1 - i))
    counts = torch.zeros(K ** n)
    counts.scatter_add_(0, idx.reshape(-1), torch.ones(idx.numel()))
    return counts


def _kl(gen: torch.Tensor, ref: torch.Tensor) -> float:
    smoothing = 1e-6
    Ksize = gen.numel()
    gp = (gen + smoothing) / (gen.sum() + Ksize * smoothing)
    rp = (ref + smoothing) / (ref.sum() + Ksize * smoothing)
    return float((gp * (gp.log() - rp.log())).sum())


# --------------------------------------------------------------------------
# Per-arm metric helpers.  Each arm exposes a slightly different generative
# API; this layer normalises {generate, bpd, recover} for the report.
# --------------------------------------------------------------------------
@torch.no_grad()
def _generate_ids(arm: str, model, cfg, n: int, L: int) -> torch.Tensor:
    """Unconditional generation → token ids (n, L) on cpu.

    For DirichletFM, ``sample`` already returns ids; for everything else
    we ``sample`` in latent space and argmax-decode."""
    if arm == "DirichletFM":
        return model.sample(n, L).cpu()
    z = model.sample(n, L)
    log_p = model.decode_to_logprobs(z)
    return log_p.argmax(dim=-1).cpu()


@torch.no_grad()
def _bpd(arm: str, model, cfg, ids: torch.Tensor) -> float:
    """Bits-per-character.  For the latent / sphere / CLR families this is
    the model's own recovery-NLL BPD.  For DirichletFM (no bpd method) we
    compute the denoiser NLL at t close to ``t_max`` evaluated on a
    Dirichlet draw conditioned on the clean ids — the natural BPD analogue.
    """
    if hasattr(model, "bpd"):
        return float(model.bpd(ids, max_steps=64))
    if arm == "DirichletFM":
        import math
        device = next(model.parameters()).device
        ids = ids.to(device).long()
        B, L = ids.shape
        K = cfg.text8_dataset.K
        t = torch.full((B,), 0.95 * model.t_max, device=device)
        beta = torch.ones(B, L, K, device=device, dtype=t.dtype)
        beta.scatter_(-1, ids.unsqueeze(-1),
                      t[:, None, None].expand(B, L, 1).to(beta.dtype))
        x_t = torch.distributions.Dirichlet(beta).sample()
        logits = model.forward(x_t, t)
        log_p = logits.log_softmax(dim=-1)
        nll = -log_p.gather(-1, ids.unsqueeze(-1)).squeeze(-1).mean()
        return float(nll / math.log(2))
    return float("nan")


def _recover_acc(
    arm: str, model, cfg, ids: torch.Tensor, alpha: float, steps: int
) -> float | None:
    """Token-level recovery accuracy at perturbation scale α (per
    recovery_check.py).  Returns None when the arm has no x_init hook."""
    if arm == "DirichletFM":
        return None  # no x_init pathway in the marginal-velocity sampler
    device = next(model.parameters()).device
    ids = ids.to(device).long()
    K = cfg.text8_dataset.K
    # Encode → latent z1 (per arm's geometry).
    if hasattr(model, "encode"):
        z1 = model.encode(ids)
    else:  # simplex EqM
        ls = cfg.transformation.label_smoothing
        z1 = token_ids_to_features(ids, K, label_smoothing=ls)
    embed_norm = z1.norm(dim=-1).mean().item()
    sigma_perturb = alpha * embed_norm
    z_init = z1 + sigma_perturb * torch.randn_like(z1)
    with torch.no_grad():
        z = model.sample(ids.shape[0], ids.shape[1],
                         x_init=z_init, max_steps=steps)
        log_p = model.decode_to_logprobs(z)
        rec = log_p.argmax(dim=-1).cpu()
    return float((rec == ids.cpu()).float().mean())


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=["local", "cluster"], default="cluster")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--steps", type=int, default=200,
                    help="recovery sampler max_steps")
    ap.add_argument("--recover-alphas", type=str, default="0.1,0.3,0.5,1.0")
    args = ap.parse_args()
    root = f"runs/sflm_bench_{args.scale}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    alphas = [float(a) for a in args.recover_alphas.split(",") if a.strip()]

    # Build the data module once (text8 splits) for clean val ids + corpus
    # n-gram counts.  These don't depend on the model.
    base_cfg = Config()
    dm, _ = build_training_datamodule(base_cfg)
    train_ids = dm.splits.train.long()
    val_ids = dm.splits.val.long()
    K = base_cfg.text8_dataset.K
    L = base_cfg.text8_dataset.L
    g = torch.Generator().manual_seed(args.seed)
    pick = torch.randperm(val_ids.shape[0], generator=g)[: args.n]
    clean_val = val_ids[pick].to(device)
    print(f"data: K={K}, L={L}, n_eval={args.n} | device={device}")

    ref_uni = _ngram_counts(train_ids, K, 1)
    ref_bi = _ngram_counts(train_ids, K, 2)
    ref_tri = _ngram_counts(train_ids, K, 3)

    results: dict = {}
    for arm in GEN_ARMS:
        model, cfg = _load(arm, device, root)
        if model is None:
            print(f"[skip] {arm}: no checkpoint at {root}/{arm}/")
            continue
        print(f"\n=== {arm} ===")
        torch.manual_seed(args.seed)
        bpd = _bpd(arm, model, cfg, clean_val)

        # Unconditional generation → n-gram KL.
        gen_ids = _generate_ids(arm, model, cfg, args.n, L)
        gen_uni = _ngram_counts(gen_ids, K, 1)
        gen_bi = _ngram_counts(gen_ids, K, 2)
        gen_tri = _ngram_counts(gen_ids, K, 3)
        kl_u, kl_b, kl_t = _kl(gen_uni, ref_uni), _kl(gen_bi, ref_bi), _kl(gen_tri, ref_tri)
        p = (gen_uni + 1e-9) / (gen_uni.sum() + K * 1e-9)
        H_gen = float(-(p * p.log()).sum())
        sample_text = "".join(ALPHABET[int(i)] for i in gen_ids[0])

        # Recovery curve.
        rec_curve = {}
        for a in alphas:
            acc = _recover_acc(arm, model, cfg, clean_val, a, args.steps)
            rec_curve[a] = acc
            tag = "—" if acc is None else f"{acc:.3f}"
            print(f"  α={a:.2f}  recover_acc={tag}")

        results[arm] = {
            "bpd": bpd, "KL_uni": kl_u, "KL_bi": kl_b, "KL_tri": kl_t,
            "H_gen": H_gen, "recover": rec_curve,
            "sample0": sample_text,
        }
        print(f"  BPD={bpd:.3f}  KL_uni={kl_u:.4f}  KL_bi={kl_b:.4f}  "
              f"KL_tri={kl_t:.4f}  H_gen={H_gen:.3f}")
        print(f"  sample[0]: {sample_text!r}")

    out_path = Path(f"{root}/generation_eval.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    # ---- printed comparison table ----
    print("\n" + "=" * 100)
    print("UNIFIED GENERATION COMPARISON  (5 arms; lower BPD/KL = better;"
          " higher recover_acc = better)")
    print("=" * 100)
    head = (f"{'arm':12s}  {'BPD':>7s}  {'KL_uni':>8s}  {'KL_bi':>8s}  "
            f"{'KL_tri':>8s}  {'H_gen':>7s}")
    head += "".join(f"  {'rec@'+str(a):>7s}" for a in alphas)
    print(head)
    for arm in GEN_ARMS:
        r = results.get(arm)
        if r is None:
            print(f"{arm:12s}  (missing checkpoint)")
            continue
        row = (f"{arm:12s}  {r['bpd']:7.3f}  {r['KL_uni']:8.4f}  "
               f"{r['KL_bi']:8.4f}  {r['KL_tri']:8.4f}  {r['H_gen']:7.3f}")
        for a in alphas:
            v = r["recover"].get(a)
            row += f"  {'—':>7s}" if v is None else f"  {v:7.3f}"
        print(row)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
