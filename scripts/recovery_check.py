"""Recovery diagnostic — test EqMLatent's self-healing property.

Two evaluation modes given a trained EqMLatent checkpoint:

  (A) Unconditional generation: sample from N(0, σ²I) → run sampler → decode.
      Headline KL_uni / KL_bi / KL_tri / H_ratio against the training corpus
      tells whether the field generates text-like sequences.

  (B) Recovery from perturbation: take held-out test sequences, encode →
      add Gaussian perturbation of varying scale α ∈ {0.1, 0.3, 0.5, 1.0} ·
      embed_norm, run sampler, decode. Report token-level recovery accuracy.
      A healthy EqM should:
        * α small  → near-100 % recovery (samples descend back to original)
        * α large  → recovery degrades; samples are still valid text but
          may diverge from the source.

The "diverge but stay valid" failure mode is interesting: it means the
field has multiple basins (the desired EBM property) and the trajectory
reached a different basin than the original. KL of those samples vs corpus
should still be low.

Usage:
    python scripts/recovery_check.py \\
        --ckpt runs/latent_skipgram_d128_timecond_euler_ep20/epoch_final.pt \\
        --alphas 0.1,0.3,0.5,1.0 --n 256
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def _bootstrap() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))


_bootstrap()

import aitchinson_flow.models  # noqa: E402,F401
from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.data.char_window_dataset import CHAR2ID  # noqa: E402
from aitchinson_flow.models import build_model  # noqa: E402
from aitchinson_flow.training import build_training_datamodule  # noqa: E402
from scripts.eval_full import _config_from_payload  # noqa: E402

ALPHABET = "".join(sorted(CHAR2ID, key=CHAR2ID.__getitem__))


def _ngram_counts_flat(ids2d: torch.Tensor, K: int, n: int) -> torch.Tensor:
    L = ids2d.shape[1]
    if L < n:
        return torch.zeros(K**n)
    idx = torch.zeros(ids2d.shape[0], L - n + 1, dtype=torch.long)
    for i in range(n):
        idx = idx + ids2d[:, i : L - n + 1 + i].long() * (K ** (n - 1 - i))
    counts = torch.zeros(K**n)
    counts.scatter_add_(0, idx.reshape(-1), torch.ones(idx.numel()))
    return counts


def _kl_smoothed(gen: torch.Tensor, ref: torch.Tensor) -> float:
    smoothing = 1e-6
    Ksize = gen.numel()
    gp = (gen + smoothing) / (gen.sum() + Ksize * smoothing)
    rp = (ref + smoothing) / (ref.sum() + Ksize * smoothing)
    return float((gp * (gp.log() - rp.log())).sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--alphas", type=str, default="0.1,0.3,0.5,1.0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = _config_from_payload(payload)
    device = cfg.training.device
    model = build_model(cfg).to(device)
    state = payload["model_state_dict"] if "model_state_dict" in payload else payload
    model.load_state_dict(state)
    model.eval()

    dm, _ = build_training_datamodule(cfg)
    train_ids = dm.splits.train.long()
    val_ids = dm.splits.val.long()
    K = cfg.text8_dataset.K
    L = cfg.text8_dataset.L
    sigma = cfg.eqm.source_sigma
    if hasattr(model, "embed"):
        embed = model.embed.weight.detach().to(device)
        embed_norm = embed.norm(dim=-1).mean().item()
    else:
        # Simplex EqM: use the data-feature L2 norm as the perturbation scale.
        # Compute on a held-out batch.
        from aitchinson_flow.data.transforms import token_ids_to_features

        ls = cfg.transformation.label_smoothing
        sample_z = token_ids_to_features(val_ids[:64].to(device), K, label_smoothing=ls)
        embed_norm = sample_z.norm(dim=-1).mean().item()
        embed = None

    ref_uni = _ngram_counts_flat(train_ids, K, 1)
    ref_bi = _ngram_counts_flat(train_ids, K, 2)
    ref_tri = _ngram_counts_flat(train_ids, K, 3)

    rows: list[dict] = []

    # ─── (A) Unconditional ────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    print("=== Unconditional generation ===")
    with torch.no_grad():
        x = model.sample(args.n, L, max_steps=args.steps)
        log_probs = model.decode_to_logprobs(x)
        ids = log_probs.argmax(-1).cpu()

    gen_uni = _ngram_counts_flat(ids, K, 1)
    gen_bi = _ngram_counts_flat(ids, K, 2)
    gen_tri = _ngram_counts_flat(ids, K, 3)
    kl_u = _kl_smoothed(gen_uni, ref_uni)
    kl_b = _kl_smoothed(gen_bi, ref_bi)
    kl_t = _kl_smoothed(gen_tri, ref_tri)
    p = (gen_uni + 1e-9) / (gen_uni.sum() + K * 1e-9)
    H_gen = float(-(p * p.log()).sum())
    print(
        f"  KL_uni={kl_u:.4f}  KL_bi={kl_b:.4f}  KL_tri={kl_t:.4f}  H_gen={H_gen:.4f}"
    )
    print(f"  sample[0]: {''.join(ALPHABET[int(i)] for i in ids[0])!r}")
    print(f"  sample[1]: {''.join(ALPHABET[int(i)] for i in ids[1])!r}")
    rows.append(
        {
            "mode": "unconditional",
            "alpha": None,
            "KL_uni": kl_u,
            "KL_bi": kl_b,
            "KL_tri": kl_t,
            "H_gen": H_gen,
            "samples": [
                "".join(ALPHABET[int(i)] for i in ids[s]) for s in range(min(4, args.n))
            ],
        }
    )

    # ─── (B) Recovery ─────────────────────────────────────────────────────
    print("\n=== Recovery from perturbation ===")
    print(f"  embed_norm_mean={embed_norm:.4f}  σ_source={sigma:.4f}")
    print(
        f"{'α':>5} {'σ_perturb':>10} {'KL_bi':>8} {'tok_acc':>8}  sample[0]: gt / pt / rc"
    )
    val_pick = val_ids[: args.n].to(device)
    # Encode token_ids → data tensor. EqMLatent has model.encode; simplex EqM
    # uses CLR features via token_ids_to_features.
    if hasattr(model, "encode"):
        z_clean = model.encode(val_pick)
    else:
        from aitchinson_flow.data.transforms import token_ids_to_features

        ls = cfg.transformation.label_smoothing
        z_clean = token_ids_to_features(val_pick, K, label_smoothing=ls)
    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    for alpha in alphas:
        torch.manual_seed(args.seed + int(alpha * 1000))
        sig_perturb = alpha * embed_norm
        z_init = z_clean + sig_perturb * torch.randn_like(z_clean)

        # Decode the perturbed (sampler-input) latents too. Argmax-decoding
        # `z_init` shows what tokens the noisy embedding nearest-neighbours.
        with torch.no_grad():
            log_probs_pt = model.decode_to_logprobs(z_init)
            ids_pt = log_probs_pt.argmax(-1).cpu()

            x = model.sample(args.n, L, max_steps=args.steps, x_init=z_init)
            log_probs = model.decode_to_logprobs(x)
            ids = log_probs.argmax(-1).cpu()

        gen_uni = _ngram_counts_flat(ids, K, 1)
        gen_bi = _ngram_counts_flat(ids, K, 2)
        kl_b = _kl_smoothed(gen_bi, ref_uni if False else ref_bi)
        token_acc = float((ids == val_pick.cpu()).float().mean())
        token_acc_pt = float((ids_pt == val_pick.cpu()).float().mean())
        sample_gt = "".join(ALPHABET[int(i)] for i in val_pick[0].cpu())
        sample_pt = "".join(ALPHABET[int(i)] for i in ids_pt[0])
        sample_rc = "".join(ALPHABET[int(i)] for i in ids[0])
        print(
            f"{alpha:>5.2f} {sig_perturb:>10.3f} {kl_b:>8.4f} {token_acc:>8.4f}  "
            f"gt={sample_gt!r}\n{'':>34}pt={sample_pt!r}  (pt_acc={token_acc_pt:.3f})"
            f"\n{'':>34}rc={sample_rc!r}"
        )
        rows.append(
            {
                "mode": "recovery",
                "alpha": alpha,
                "sigma_perturb": sig_perturb,
                "KL_bi": kl_b,
                "token_acc": token_acc,
                "token_acc_perturbed": token_acc_pt,
                "gt_sample0": sample_gt,
                "perturbed_sample0": sample_pt,
                "rc_sample0": sample_rc,
            }
        )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"rows": rows}, indent=2))
        print(f"\nWrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
