"""Phase F+ — distance-to-corruption AUROC profile.

Tests the hypothesis: per-position discriminator AUROC should be
≈0.5 *before* the first corruption (h_LLM is prefix-deterministic in
an autoregressive LM, so clean and invalid hidden states are identical
when no upstream corruption has happened yet); peak at the corrupted
position itself; then decay back toward 0.5 as the LM recovers from
the local error.

For each chunk and each position k, we compute:

* ``d_prev = k − max{j ≤ k : mask[j] = True}``  (or ``None`` if no such j)
* ``d_next = min{j ≥ k : mask[j] = True} − k``  (or ``None`` if no such j)

We bucket positions by ``d_prev`` (0, 1, 2, ...) and compute AUROC at
each bucket separately, comparing clean h vs invalid h at those
positions only.

Discriminators tested:

1. Linear probe on h_LLM (closed-form ridge).
2. SVGP latent mean.
3. SVGP latent std (UQ).
4. EqM ctx auditor (per-position grad-norm).
5. Spilled Energy (the locality-clean baseline).

Output: ``runs/phaseF_distance.png`` with line plots vs d_prev, plus
``runs/phaseF_distance.{md,json}`` with the raw numbers.
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
from scripts.eval_auditor_wiki import (  # noqa: E402
    _grad_norm_per_pos, _roc_auc,
)
from scripts.phaseF_uq import (  # noqa: E402
    linear_probe_ridge, predict_linear, train_svgp, predict_svgp,
)


def _distances_to_prev(mask: torch.Tensor) -> torch.Tensor:
    """For each position k, distance to the *nearest preceding* True
    in the mask (incl. k itself). Returns int tensor of shape mask.shape;
    use sentinel ``-1`` for "no preceding corruption".

    mask: (n, L) bool.
    """
    n, L = mask.shape
    out = -torch.ones(n, L, dtype=torch.long)
    for i in range(n):
        last = -1
        for k in range(L):
            if mask[i, k]:
                last = 0
            else:
                last = last + 1 if last >= 0 else -1
            out[i, k] = last
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/aud_gpt2_ctx/epoch_final.pt")
    p.add_argument("--cache", default="data/wiki_cache_gpt2.pt")
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--max-d", type=int, default=20,
                   help="report AUROC up to this distance to prev corruption")
    p.add_argument("--svgp-iters", type=int, default=200)
    p.add_argument("--out-md", default="runs/phaseF_distance.md")
    p.add_argument("--out-json", default="runs/phaseF_distance.json")
    p.add_argument("--out-png", default="runs/phaseF_distance.png")
    args = p.parse_args(argv)

    cache = load_wiki_cache(args.cache)
    n_total = int(cache["clean_clr"].shape[0])
    n_train = int(args.train_frac * n_total)
    train_idx = list(range(n_train))
    val_idx = list(range(n_train, n_total))

    # Compute distance-to-prev arrays on the val split.
    mask_val = cache["mask_corrupt"][n_train:]                       # (n_val, L)
    d_prev_val = _distances_to_prev(mask_val)                        # (n_val, L)
    h_clean_val = cache["clean_h"][n_train:].float()                 # (n_val, L, H)
    h_invalid_val = cache["invalid_h"][n_train:].float()
    se_clean_val = cache["clean_SE_pos"][n_train:].float()           # (n_val, L)
    se_invalid_val = cache["invalid_SE_pos"][n_train:].float()
    L = mask_val.shape[1]

    # ---- Train linear probe + SVGP on training-split corrupted-position pairs ----
    print("[dist] preparing train pairs (only at corrupted positions in train) …")
    h_pos_train = []   # invalid h at corrupted positions
    h_neg_train = []   # clean h at the same positions
    for i in train_idx:
        m = cache["mask_corrupt"][i]
        h_pos_train.append(cache["invalid_h"][i][m].float())
        h_neg_train.append(cache["clean_h"][i][m].float())
    h_pos_train = torch.cat(h_pos_train); h_neg_train = torch.cat(h_neg_train)
    print(f"  train: {h_pos_train.shape[0]} pos, {h_neg_train.shape[0]} neg")

    print("[dist] training linear probe …")
    w, mu, sigma = linear_probe_ridge(h_pos_train, h_neg_train)

    print("[dist] training SVGP …")
    sv_model, sv_lik, sv_mu, sv_sigma = train_svgp(
        h_pos_train, h_neg_train, n_inducing=64, n_iters=args.svgp_iters,
    )

    print("[dist] loading EqM auditor …")
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = _config_from_payload(payload)
    device = cfg.training.device
    eqm = build_model(cfg).to(device)
    eqm.load_state_dict(payload.get("model_state_dict", payload))
    eqm.eval()
    for pp in eqm.parameters():
        pp.requires_grad_(False)
    gamma = float(cfg.eqm.auditor_gamma)

    # ---- Score every val position with each discriminator ----
    print("[dist] scoring val positions with each discriminator …")
    n_val = h_clean_val.shape[0]
    H_dim = h_clean_val.shape[2]

    # Linear probe scores per position.
    flat_clean = h_clean_val.reshape(-1, H_dim)
    flat_invalid = h_invalid_val.reshape(-1, H_dim)
    s_lin_c = predict_linear(w, mu, sigma, flat_clean).reshape(n_val, L)
    s_lin_i = predict_linear(w, mu, sigma, flat_invalid).reshape(n_val, L)

    # SVGP scores per position.
    p_svgp_c, m_svgp_c, st_svgp_c = predict_svgp(sv_model, sv_lik, sv_mu, sv_sigma, flat_clean)
    p_svgp_i, m_svgp_i, st_svgp_i = predict_svgp(sv_model, sv_lik, sv_mu, sv_sigma, flat_invalid)
    m_svgp_c = m_svgp_c.reshape(n_val, L); m_svgp_i = m_svgp_i.reshape(n_val, L)
    st_svgp_c = st_svgp_c.reshape(n_val, L); st_svgp_i = st_svgp_i.reshape(n_val, L)

    # EqM ctx auditor per-position grad norm.
    print("    (forwarding val through EqM auditor) …")
    x_clean_val = cache["clean_clr"][n_train:].float().to(device)
    x_invalid_val = cache["invalid_clr"][n_train:].float().to(device)
    h_c_dev = h_clean_val.to(device)
    h_i_dev = h_invalid_val.to(device)
    bs = 16
    gn_c_chunks: list[torch.Tensor] = []
    gn_i_chunks: list[torch.Tensor] = []
    for s in range(0, n_val, bs):
        e = min(s + bs, n_val)
        gn_c_chunks.append(_grad_norm_per_pos(eqm, x_clean_val[s:e], gamma, h_c_dev[s:e]).cpu())
        gn_i_chunks.append(_grad_norm_per_pos(eqm, x_invalid_val[s:e], gamma, h_i_dev[s:e]).cpu())
    gn_c = torch.cat(gn_c_chunks); gn_i = torch.cat(gn_i_chunks)

    # ---- AUROC at each d_prev bucket ----
    print("[dist] computing AUROC per d_prev bucket …")
    methods = {
        "Linear probe":      (s_lin_c,  s_lin_i),
        "SVGP latent mean":  (m_svgp_c, m_svgp_i),
        "SVGP latent std":   (st_svgp_c, st_svgp_i),
        "EqM ctx (grad-norm)": (gn_c, gn_i),
        "Spilled Energy":    (se_clean_val, se_invalid_val),
    }
    d_prev_flat = d_prev_val.flatten()
    results: dict[str, dict[int, dict[str, float]]] = {}

    # bucket: d = -1 (no prior corruption), d = 0, 1, 2, ..., max-d, "≥max-d+1"
    for name, (sc, si) in methods.items():
        results[name] = {}
        sc_flat = sc.flatten(); si_flat = si.flatten()
        # No prior corruption: positions where d_prev == -1.
        m_no = d_prev_flat == -1
        if m_no.sum() > 0:
            n0 = sc_flat[m_no]; p0 = si_flat[m_no]
            auroc = _roc_auc(n0, p0)
            # Also report whether the two are *literally identical* (they
            # should be, for prefix-identical autoregressive LMs).
            max_diff = float((p0 - n0).abs().max().item())
            results[name][-1] = {
                "auroc": float(auroc), "n": int(m_no.sum().item()),
                "max_abs_diff_p_minus_n": max_diff,
            }
        for d in range(0, args.max_d + 1):
            md = d_prev_flat == d
            if md.sum() == 0:
                continue
            auroc = _roc_auc(sc_flat[md], si_flat[md])
            results[name][d] = {
                "auroc": float(auroc), "n": int(md.sum().item()),
            }
        # Tail bucket: ≥ max_d+1 if any.
        m_tail = d_prev_flat > args.max_d
        if m_tail.sum() > 0:
            auroc = _roc_auc(sc_flat[m_tail], si_flat[m_tail])
            results[name][9999] = {
                "auroc": float(auroc), "n": int(m_tail.sum().item()),
            }

    # ---- Print summary ----
    print()
    print(f"{'d_prev':>8s} | {'n':>6s} | " + " | ".join(f"{k:>22s}" for k in methods))
    print("-" * (10 + 8 + sum(25 for _ in methods)))
    for d in [-1] + list(range(0, args.max_d + 1)) + [9999]:
        if all(d not in results[k] for k in methods):
            continue
        n_here = results[next(iter(methods))][d]["n"]
        d_label = "no prev" if d == -1 else (f">{args.max_d}" if d == 9999 else str(d))
        row = f"{d_label:>8s} | {n_here:>6d} | " + " | ".join(
            f"{results[k][d]['auroc']:>22.4f}" if d in results[k] else f"{'—':>22s}"
            for k in methods
        )
        print(row)

    # Also report identity check at d = -1.
    if -1 in results.get("Linear probe", {}):
        max_diff = float(
            (h_invalid_val.flatten(0, 1)[d_prev_flat == -1] -
             h_clean_val.flatten(0, 1)[d_prev_flat == -1]).abs().max().item()
        )
        print(f"\nIdentity check at d=-1: max |h_invalid - h_clean| = {max_diff:.6f}")
        print("  (should be ≈0 — prefix-deterministic for autoregressive GPT-2)")
    print()

    # ---- Plot ----
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ds = sorted(set().union(*(results[k].keys() for k in methods)))
    # Display the no-prev-corruption bucket as d = -1, hidden bucket d = 9999 as "tail".
    x_display = []
    for d in ds:
        if d == 9999:
            x_display.append(args.max_d + 2)
        else:
            x_display.append(d)
    color_cycle = ["tab:blue", "tab:red", "tab:purple", "tab:green", "tab:orange"]
    markers = ["o", "s", "^", "D", "v"]
    for (name, _), color, marker in zip(methods.items(), color_cycle, markers):
        ys = []
        ns = []
        for d in ds:
            r = results[name].get(d)
            if r is None:
                ys.append(np.nan); ns.append(0)
            else:
                ys.append(r["auroc"]); ns.append(r["n"])
        ax.plot(x_display, ys, marker=marker, label=name, color=color,
                linewidth=2, markersize=7)
        # Annotate n at each bucket with small text below
        for x, y, n in zip(x_display, ys, ns):
            if not np.isnan(y) and n < 50:
                ax.annotate(f"n={n}", (x, y), fontsize=6, alpha=0.6,
                            xytext=(0, -10), textcoords="offset points",
                            ha="center", va="top")

    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.5, label="chance")
    ax.axvline(0, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.set_xlabel(
        "Distance to *nearest preceding* corrupted position\n"
        "(−1 = no preceding corruption,  0 = the corrupted position itself,  "
        f"{args.max_d}+ = tail)"
    )
    ax.set_ylabel("Tok AUROC clean vs invalid at this position")
    ax.set_title(
        "Per-position discriminator AUROC by distance to upstream corruption\n"
        "Hypothesis: ≈0.5 before any corruption, peak at d=0, decay as LM recovers"
    )
    # Custom xtick labels for −1 and tail.
    xticks = [-1, 0] + list(range(1, args.max_d + 1, 2)) + [args.max_d + 2]
    xticklabels = ["no prev"] + [str(x) for x in [0] + list(range(1, args.max_d + 1, 2))] + [f">{args.max_d}"]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=0, fontsize=9)
    ax.set_ylim(0.35, 1.02)
    ax.legend(loc="center right", fontsize=9)
    ax.grid(alpha=0.3)

    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=130, bbox_inches="tight")
    print(f"[dist] wrote {args.out_png}")

    Path(args.out_json).write_text(json.dumps({
        "ckpt": str(args.ckpt),
        "cache": str(args.cache),
        "n_val": n_val, "L": L, "max_d": args.max_d,
        "results": {
            name: {str(d): r for d, r in by_d.items()}
            for name, by_d in results.items()
        },
    }, indent=2))
    # Markdown
    md = []
    md.append("# Phase F+ — Per-position AUROC by distance to upstream corruption")
    md.append("")
    md.append("Tests whether the AR cascade follows the user's predicted profile:")
    md.append("≈0.5 before any corruption, peak at d=0, decay as LM recovers.")
    md.append("")
    md.append("| d_prev | n | " + " | ".join(methods.keys()) + " |")
    md.append("|" + "|".join(["---"] + ["---:"] * (1 + len(methods))) + "|")
    for d in ds:
        n_here = results[next(iter(methods))].get(d, {}).get("n", 0)
        d_label = "**no prev**" if d == -1 else (f">{args.max_d}" if d == 9999 else str(d))
        cells = [
            f"{results[name][d]['auroc']:.4f}" if d in results[name] else "—"
            for name in methods
        ]
        md.append(f"| {d_label} | {n_here} | " + " | ".join(cells) + " |")
    Path(args.out_md).write_text("\n".join(md))
    print(f"[dist] wrote {args.out_md}, {args.out_json}")


if __name__ == "__main__":
    main()
