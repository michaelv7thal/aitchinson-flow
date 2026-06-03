"""Recovery diagnostic for PerTokenBayesianAuditorWiki — product-GP variant.

Mirrors ``scripts/recovery_check.py`` but reads the wiki/GPT-2 cache directly
(no Text8DataModule), perturbs the cached CLR features by α·feature_norm,
runs the sampler with the cached GPT-2 hidden states as context, and reports
slot-recovery accuracy (argmax over the K top-K slots, since vocab-id
recovery would require remapping through ``clean_topk_idx`` anyway).

Usage:
    python scripts/recovery_check_product_gp.py \\
        --ckpt runs/.../bayes_auditor_gpt2_poc/epoch_final.pt \\
        --out  runs/.../bayes_auditor_gpt2_poc/recovery.json
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
from aitchinson_flow.models import build_model  # noqa: E402
from scripts.eval_full import _config_from_payload  # noqa: E402


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
    ap.add_argument("--cache", type=str, default=None,
                    help="override; defaults to cfg.auditor.cache_path")
    ap.add_argument("--n", type=int, default=64,
                    help="number of held-out sequences to perturb")
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

    cache_path = args.cache or cfg.auditor.cache_path
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    K = int(cache["K"])
    L = int(cache["L"])
    train_frac = float(cfg.auditor.train_frac)
    n_total = int(cache["clean_clr"].shape[0])
    n_train = max(1, int(train_frac * n_total))
    # Use held-out tail of the cache as val.
    clean_clr = cache["clean_clr"][n_train:][: args.n].to(device)
    clean_h = cache["clean_h"][n_train:][: args.n].to(device)
    # Slot-id labels are the argmax over the K top-K slots of the clean CLR;
    # the actual vocab id would be ``clean_topk_idx[i, j, slot_argmax]`` but
    # we measure slot accuracy here for a faithful "did the field move us
    # back to the right ranking" diagnostic.
    slot_ids = clean_clr.argmax(dim=-1).cpu()

    # Reference n-grams for KL diagnostics use *all* val slots, treating
    # slot index as the categorical alphabet (K=64).
    val_slot_ids_all = cache["clean_clr"][n_train:].argmax(dim=-1)
    ref_bi = _ngram_counts_flat(val_slot_ids_all, K, 2)

    # Feature norm — how far perturbation pushes the CLR vector.
    feature_norm = clean_clr.norm(dim=-1).mean().item()

    rows: list[dict] = []
    print("=== Recovery from perturbation (product-GP, slot-level) ===")
    print(f"  K={K}  L={L}  n_val={clean_clr.shape[0]}  feature_norm={feature_norm:.4f}")
    print(f"  α   σ_perturb    KL_bi    slot_acc  slot_acc_perturbed")

    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    for alpha in alphas:
        torch.manual_seed(args.seed + int(alpha * 1000))
        sig_perturb = alpha * feature_norm
        z_init = clean_clr + sig_perturb * torch.randn_like(clean_clr)
        # Keep on V_d (zero-mean across slot axis) to match the model's
        # training-time CLR hyperplane.
        z_init = z_init - z_init.mean(dim=-1, keepdim=True)

        with torch.no_grad():
            slot_ids_pt = z_init.argmax(dim=-1).cpu()
            x_rc = model.sample(
                clean_clr.shape[0], L,
                x_init=z_init, h_ctx=clean_h, max_steps=args.steps,
            )
            slot_ids_rc = x_rc.argmax(dim=-1).cpu()

        slot_acc = float((slot_ids_rc == slot_ids).float().mean())
        slot_acc_pt = float((slot_ids_pt == slot_ids).float().mean())
        gen_bi = _ngram_counts_flat(slot_ids_rc, K, 2)
        kl_b = _kl_smoothed(gen_bi, ref_bi)

        print(
            f"  {alpha:>5.2f}  {sig_perturb:>9.3f}  {kl_b:>7.4f}  "
            f"{slot_acc:>7.4f}   {slot_acc_pt:>7.4f}"
        )

        rows.append({
            "mode": "recovery",
            "alpha": alpha,
            "sigma_perturb": sig_perturb,
            "KL_bi_slots": kl_b,
            "slot_acc": slot_acc,
            "slot_acc_perturbed": slot_acc_pt,
        })

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"rows": rows}, indent=2))
        print(f"\nWrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
