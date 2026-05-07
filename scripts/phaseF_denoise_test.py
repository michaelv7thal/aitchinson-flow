"""Phase F denoising test — does the auditor's −∇E direction reduce
distance to the clean simplex?

A trained linear probe on h_LLM gives a *scalar score* per position; it
has no notion of "which way to move x to make it more clean-like". The
EqM auditor, by virtue of being a velocity field, does have a gradient
direction. If the auditor learned a useful energy landscape (and not
just a discriminator dressed in flow-matching clothes), running
``x ← x − η·∇E(x)`` from an invalid simplex point should reduce the
L2 distance to the corresponding *clean* simplex point.

Setup:

* Take invalid simplex x_inv from the held-out 60 chunks.
* Fix h_LLM = h_invalid (the LM's hidden state on the invalid input).
  This isolates the auditor's energy contribution from any LM
  re-computation.
* Run K Euler-style gradient-descent steps with step size η on
  E(x) = Σ ⟨x, f(x; γ=1, h_invalid)⟩.
* Compare:
    d_before = ‖x_inv − x_clean‖²     (per position)
    d_after  = ‖x_descent − x_clean‖²  (per position)
* Report Δd at corrupted positions vs uncorrupted positions.

A positive result is **Δd < 0 at corrupted positions and Δd ≈ 0 at
uncorrupted** — the auditor identifies *which* positions to fix and
moves them toward clean.

A linear probe has no analog of this direction; this is the EqM
auditor's structural contribution.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import aitchinson_flow.models  # noqa: E402,F401
from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.data.wiki import WikiAuditorDataset, load_wiki_cache  # noqa: E402
from aitchinson_flow.models import build_model  # noqa: E402

from scripts.eval_full import _config_from_payload  # noqa: E402


def _energy(model, x, gamma, h_ctx):
    time_cond = getattr(model.cfg.eqm, "time_conditioning", "off")
    B = x.shape[0]
    if time_cond != "off":
        gamma_t = torch.full((B,), float(gamma), device=x.device, dtype=x.dtype)
    else:
        gamma_t = None
    v = model.forward(x, gamma_t, h_ctx)
    return (x * v).sum(dim=(-1, -2)), v


def _grad_E(model, x, gamma, h_ctx):
    time_cond = getattr(model.cfg.eqm, "time_conditioning", "off")
    B = x.shape[0]
    if time_cond != "off":
        gamma_t = torch.full((B,), float(gamma), device=x.device, dtype=x.dtype)
    else:
        gamma_t = None
    with torch.enable_grad():
        x_req = x.detach().requires_grad_(True)
        v = model.forward(x_req, gamma_t, h_ctx)
        e = (x_req * v).sum()
        g = torch.autograd.grad(e, x_req, create_graph=False)[0].detach()
    return g


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cache", default="data/wiki_cache_gpt2.pt")
    p.add_argument("--n-eval", type=int, default=60)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--K", type=int, default=20, help="gradient-descent steps")
    p.add_argument("--eta", type=float, default=0.05, help="step size")
    p.add_argument("--out-md", default="runs/phaseF_denoise.md")
    p.add_argument("--out-png", default="runs/phaseF_denoise.png")
    args = p.parse_args(argv)

    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = _config_from_payload(payload)
    device = cfg.training.device
    gamma = float(cfg.eqm.auditor_gamma)
    ctx_mode = getattr(cfg.eqm, "context_features", "off")
    needs_h = ctx_mode != "off"

    model = build_model(cfg).to(device)
    model.load_state_dict(payload.get("model_state_dict", payload))
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)

    cache = load_wiki_cache(args.cache)
    n_total = int(cache["clean_clr"].shape[0])
    n_train = int(args.train_frac * n_total)
    val_start = n_train
    val_end = min(n_train + args.n_eval, n_total)

    x_clean = cache["clean_clr"][val_start:val_end].to(device).float()
    x_invalid = cache["invalid_clr"][val_start:val_end].to(device).float()
    mask = cache["mask_corrupt"][val_start:val_end].to(device)
    if needs_h:
        h_clean = cache["clean_h"][val_start:val_end].to(device).float()
        h_invalid = cache["invalid_h"][val_start:val_end].to(device).float()
    else:
        h_clean = h_invalid = None

    n = x_clean.shape[0]
    print(f"[denoise] n={n}, K={args.K}, η={args.eta}, ctx={ctx_mode}")

    # Baseline: distance from invalid to clean.
    d_before = (x_invalid - x_clean).pow(2).sum(dim=-1)  # (n, L)
    e_before, _ = _energy(model, x_invalid, gamma, h_invalid)

    # Gradient descent on E(x; h_invalid). h_LLM stays fixed at the
    # invalid hidden state (the realistic "OOD-correction" setting:
    # we don't get to recompute the LM after each step).
    x_descent = x_invalid.clone().detach()
    e_path: list[torch.Tensor] = [e_before.cpu().clone()]
    d_path: list[torch.Tensor] = [d_before.cpu().clone()]
    for k in range(args.K):
        g = _grad_E(model, x_descent, gamma, h_invalid)
        x_descent = x_descent - args.eta * g
        # Center to V_d (zero-mean across K) — same constraint as training.
        x_descent = x_descent - x_descent.mean(dim=-1, keepdim=True)
        e_now, _ = _energy(model, x_descent, gamma, h_invalid)
        d_now = (x_descent - x_clean).pow(2).sum(dim=-1)
        e_path.append(e_now.cpu().clone())
        d_path.append(d_now.cpu().clone())

    d_after = d_path[-1].to(device)
    e_after = e_path[-1].to(device)

    delta_d_corrupt = (d_after - d_before)[mask]
    delta_d_uncorrupt = (d_after - d_before)[~mask]

    # Per-position L2 reduction percentages.
    pct_corrupt = (delta_d_corrupt / d_before[mask].clamp(min=1e-9) * 100).cpu()
    pct_uncorrupt = (delta_d_uncorrupt / d_before[~mask].clamp(min=1e-9) * 100).cpu()

    # Argmax-token-level: did the descent argmax flip toward the clean
    # token's slot among the top-K? We can't reach the original GPT-2
    # vocab from the simplex alone (top-K indices vary), so we proxy
    # by checking whether the *top-1 argmax slot* matched x_clean's
    # top-1 argmax before vs after. (The simplex-slot indices are
    # consistent across clean and invalid because both are top-K
    # extractions from GPT-2 logits, but the slot-to-vocab map is
    # position-specific — so this only tells us whether the simplex
    # *shape* moved toward clean, which is exactly what we want.)
    arg_before = x_invalid.argmax(dim=-1)
    arg_after = x_descent.argmax(dim=-1)
    arg_clean = x_clean.argmax(dim=-1)
    flip_to_clean = (
        (arg_before != arg_clean) & (arg_after == arg_clean)
    )  # (n, L)
    flip_away = (
        (arg_before == arg_clean) & (arg_after != arg_clean)
    )
    # Net: at corrupted positions, how often does descent flip toward clean?
    n_flip_to_clean_corrupt = int(flip_to_clean[mask].sum().item())
    n_flip_away_corrupt = int(flip_away[mask].sum().item())
    n_flip_to_clean_unc = int(flip_to_clean[~mask].sum().item())
    n_flip_away_unc = int(flip_away[~mask].sum().item())
    n_corrupt = int(mask.sum().item())
    n_uncorrupt = int((~mask).sum().item())

    summary = {
        "ckpt": str(args.ckpt),
        "ctx_mode": ctx_mode,
        "n_eval": int(n),
        "n_corrupt_positions": n_corrupt,
        "n_uncorrupt_positions": n_uncorrupt,
        "K_steps": int(args.K),
        "eta": args.eta,
        "energy_mean_before": float(e_before.mean()),
        "energy_mean_after":  float(e_after.mean()),
        "d_before_mean_corrupt": float(d_before[mask].mean()),
        "d_after_mean_corrupt":  float(d_after[mask].mean()),
        "d_before_mean_uncorrupt": float(d_before[~mask].mean()),
        "d_after_mean_uncorrupt":  float(d_after[~mask].mean()),
        "pct_change_d_corrupt_mean": float(pct_corrupt.mean()),
        "pct_change_d_uncorrupt_mean": float(pct_uncorrupt.mean()),
        "fraction_d_decreased_corrupt": float((delta_d_corrupt < 0).float().mean()),
        "fraction_d_decreased_uncorrupt": float((delta_d_uncorrupt < 0).float().mean()),
        "argmax_flip_to_clean_corrupt":   {"count": n_flip_to_clean_corrupt, "of": n_corrupt},
        "argmax_flip_away_corrupt":       {"count": n_flip_away_corrupt,    "of": n_corrupt},
        "argmax_flip_to_clean_uncorrupt": {"count": n_flip_to_clean_unc,    "of": n_uncorrupt},
        "argmax_flip_away_uncorrupt":     {"count": n_flip_away_unc,        "of": n_uncorrupt},
    }
    print(json.dumps(summary, indent=2))

    # ---- Render plot: per-step energy + per-step distance, split by
    # corrupted vs uncorrupted. ----
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e_arr = torch.stack(e_path).numpy()  # (K+1, n)
    d_arr = torch.stack(d_path).numpy()  # (K+1, n, L)
    mask_cpu = mask.cpu().numpy()
    K = args.K
    steps = np.arange(K + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel 1: Energy trajectory.
    ax = axes[0]
    ax.plot(steps, e_arr.mean(axis=1), color="tab:blue", linewidth=2,
            label="mean across sequences")
    ax.fill_between(
        steps,
        np.percentile(e_arr, 25, axis=1),
        np.percentile(e_arr, 75, axis=1),
        color="tab:blue", alpha=0.2, label="IQR",
    )
    ax.set_xlabel("descent step")
    ax.set_ylabel("E(x) = Σ ⟨x, f(x; γ=1, h_invalid)⟩")
    ax.set_title("Energy trajectory under −∇E descent on x_invalid")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    # Panel 2: Distance to clean, split by mask.
    ax = axes[1]
    d_corrupt = d_arr.reshape(K + 1, -1)[:, mask_cpu.flatten()]
    d_uncorrupt = d_arr.reshape(K + 1, -1)[:, ~mask_cpu.flatten()]
    ax.plot(steps, d_corrupt.mean(axis=1), color="tab:red", linewidth=2,
            label=f"at corrupted positions (n={n_corrupt})")
    ax.fill_between(
        steps,
        np.percentile(d_corrupt, 25, axis=1),
        np.percentile(d_corrupt, 75, axis=1),
        color="tab:red", alpha=0.2,
    )
    ax.plot(steps, d_uncorrupt.mean(axis=1), color="tab:gray", linewidth=2,
            label=f"at UNcorrupted positions (n={n_uncorrupt})")
    ax.fill_between(
        steps,
        np.percentile(d_uncorrupt, 25, axis=1),
        np.percentile(d_uncorrupt, 75, axis=1),
        color="tab:gray", alpha=0.2,
    )
    ax.set_xlabel("descent step")
    ax.set_ylabel("‖x − x_clean‖² (per position)")
    ax.set_title("Distance to clean simplex during −∇E descent")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Phase F denoising test — does −∇E(x) point toward clean?",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)

    md = []
    md.append(f"# Phase F denoising test — {Path(args.ckpt).parent.name}")
    md.append("")
    md.append(f"- ckpt: `{args.ckpt}`")
    md.append(f"- ctx mode: {ctx_mode}")
    md.append(f"- n_eval: {n}, n_corrupt: {n_corrupt}, n_uncorrupt: {n_uncorrupt}")
    md.append(f"- K (descent steps): {args.K}, η: {args.eta}, γ: {gamma}")
    md.append("")
    md.append("## Distance to clean simplex (mean per-position L2²)")
    md.append("")
    md.append("|   region   | before | after | Δ% mean | % positions where d ↓ |")
    md.append("|------------|-------:|------:|--------:|----------------------:|")
    md.append(
        f"| **corrupted** | "
        f"{summary['d_before_mean_corrupt']:.3f} | "
        f"{summary['d_after_mean_corrupt']:.3f} | "
        f"{summary['pct_change_d_corrupt_mean']:+.1f}% | "
        f"{summary['fraction_d_decreased_corrupt']*100:.1f}% |"
    )
    md.append(
        f"| uncorrupted | "
        f"{summary['d_before_mean_uncorrupt']:.3f} | "
        f"{summary['d_after_mean_uncorrupt']:.3f} | "
        f"{summary['pct_change_d_uncorrupt_mean']:+.1f}% | "
        f"{summary['fraction_d_decreased_uncorrupt']*100:.1f}% |"
    )
    md.append("")
    md.append("## Argmax-slot flips during descent")
    md.append("")
    md.append("|   region   | flips toward clean | flips away from clean |")
    md.append("|------------|-------------------:|----------------------:|")
    md.append(
        f"| **corrupted** ({n_corrupt}) | "
        f"{n_flip_to_clean_corrupt} | "
        f"{n_flip_away_corrupt} |"
    )
    md.append(
        f"| uncorrupted ({n_uncorrupt}) | "
        f"{n_flip_to_clean_unc} | "
        f"{n_flip_away_unc} |"
    )
    md.append("")
    md.append("## Verdict")
    md.append("")
    if summary["pct_change_d_corrupt_mean"] < -2.0:
        md.append(
            "**The auditor's −∇E direction *does* point toward clean** at "
            "corrupted positions. The energy field has learned useful "
            "directionality beyond classification — a linear probe has no "
            "such direction."
        )
    else:
        md.append(
            "**The auditor's −∇E direction does NOT point toward clean** "
            "at corrupted positions. The energy field acts as a "
            "discriminator (small E at clean, large E at invalid) but its "
            "gradient does not give a useful denoising signal. The "
            "auditor's structural advantage over a linear probe is therefore "
            "*not* in the energy gradient itself."
        )
    md.append("")
    Path(args.out_md).write_text("\n".join(md))
    print(f"[denoise] wrote {args.out_md} and {args.out_png}")


if __name__ == "__main__":
    main()
