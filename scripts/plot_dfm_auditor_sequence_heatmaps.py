"""Per-sequence heatmaps: 10 rows × tokens, energy + SVGP variance.

For each architecture (Transformer Arch B, MLP Arch B), produces 4
heatmaps as a 2×2 grid:

  +------------------+------------------+
  | clean (energy)   | halluc (energy)  |
  +------------------+------------------+
  | clean (variance) | halluc (variance)|
  +------------------+------------------+

Each panel: y-axis = 10 sequences, x-axis = token positions (centered
on the answer span), color = either DFM EBM energy ``-log p_t(x|h)`` or
SVGP predictive variance per position. Hallucinated answer-mask
positions are marked with white "×" overlays — these are the "faulty"
positions the auditor should flag.

The SVGP is fitted online (on encoder features from this architecture's
val-train split) using the same machinery as ``scripts/phaseF_uq.py``
and ``scripts/run_dfm_auditor_archA.py``, so the variance channel
reflects an SVGP head trained on this exact encoder.

Usage::

    python scripts/plot_dfm_auditor_sequence_heatmaps.py \\
        --ckpt-transformer runs/dfm_auditor_archB_d512/best.pt \\
        --ckpt-mlp         runs/dfm_archB_d512_mlp/best.pt \\
        --out runs/dfm_auditor_plots/sequence_heatmaps
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import aitchinson_flow.models  # noqa: E402,F401
from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.data.hallueval_dfm import HalluevalDFMDataModule  # noqa: E402
from aitchinson_flow.models.factory import build_model  # noqa: E402

sys.path.insert(0, str(REPO / "scripts"))
from phaseF_uq import train_svgp, predict_svgp  # type: ignore # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def load_archB(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg: Config = ckpt["cfg"]
    K = ckpt.get("K", 32)
    L = ckpt.get("L", 160)
    model = build_model(cfg)
    model.set_dims(K=K, L=L)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, cfg, K, L


@torch.no_grad()
def gather_per_position(model, loader, device, energy_t):
    """Per-position encoder features, EBM energy, halluc-head probability,
    paper E_logit / E_marg / ΔE, answer mask, label, full_ids, attn_mask.
    """
    z_all = []; E_all = []; P_all = []
    El_all = []; Em_all = []; DE_all = []
    ans_all = []; lab_all = []; ids_all = []; attn_all = []
    for batch in loader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        topk = batch["topk_logp"].float()
        x = topk.exp()
        x = x / x.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        h = batch["hidden"].float()
        B, L, _ = x.shape
        t = torch.full((B,), float(energy_t), device=device, dtype=x.dtype)
        z = model.forward_features(x, t, h_ctx=h)
        E = model.energy_at_lm_distribution(batch, t=energy_t)
        z_all.append(z.cpu().numpy())
        E_all.append(E["E_pos"].cpu().numpy())
        if getattr(model, "halluc_head", None) is not None:
            H = model.halluc_score_at_lm(batch, t=energy_t)
            P_all.append(H["prob_pos"].cpu().numpy())
        else:
            P_all.append(np.full((B, L), np.nan, dtype=np.float32))
        El_all.append(batch["E_logit"].cpu().numpy())
        Em_all.append(batch["E_marg"].cpu().numpy())
        DE_all.append(batch["DeltaE"].cpu().numpy())
        ans_all.append(batch["answer_mask"].bool().cpu().numpy())
        lab_all.append(batch["label"].long().cpu().numpy())
        ids_all.append(batch["full_ids"].long().cpu().numpy())
        attn_all.append(batch["full_ids"].cpu().numpy() != 50256)
    return dict(
        z=np.concatenate(z_all, axis=0),
        E=np.concatenate(E_all, axis=0),
        P=np.concatenate(P_all, axis=0),
        E_logit=np.concatenate(El_all, axis=0),
        E_marg=np.concatenate(Em_all, axis=0),
        DeltaE=np.concatenate(DE_all, axis=0),
        ans=np.concatenate(ans_all, axis=0),
        label=np.concatenate(lab_all, axis=0),
        ids=np.concatenate(ids_all, axis=0),
        attn=np.concatenate(attn_all, axis=0),
    )


def fit_svgp_on_encoder(features_train, labels_train, ans_train, *,
                       n_inducing=128, n_iters=400, device="cuda"):
    """Train an SVGP on (encoder feature, row label) over answer-mask positions."""
    # labels_train is (N,); broadcast to (N, L) before masking by ans_train.
    labels_per_pos = np.broadcast_to(labels_train[:, None], ans_train.shape)
    z_pos_lab = features_train[ans_train]  # (M, d)
    y_pos_lab = labels_per_pos[ans_train]
    X_pos = torch.from_numpy(z_pos_lab[y_pos_lab == 1]).float()
    X_neg = torch.from_numpy(z_pos_lab[y_pos_lab == 0]).float()
    print(f"   [svgp] training on {X_pos.shape[0]} pos + {X_neg.shape[0]} neg, d={X_pos.shape[1]}")
    return train_svgp(
        X_pos, X_neg,
        n_inducing=n_inducing, n_iters=n_iters, lr=1e-2,
        device=device, seed=0,
    )


def svgp_variance_per_position(svgp_pack, z_per_pos):
    """Compute SVGP predictive variance per position. Returns array (N, L)."""
    svgp, lik, mu, sigma = svgp_pack
    N, L, d = z_per_pos.shape
    flat = torch.from_numpy(z_per_pos.reshape(-1, d)).float()
    _, _, std = predict_svgp(svgp, lik, mu, sigma, flat, batch=2048)
    return std.numpy().reshape(N, L)


def select_rows(label, ans, n_per_class=10, min_ans_len=4, seed=0):
    rng = np.random.default_rng(seed)
    halluc = []
    clean = []
    for r in range(label.shape[0]):
        if int(ans[r].sum()) < min_ans_len:
            continue
        (halluc if label[r] == 1 else clean).append(r)
    halluc = np.array(halluc)
    clean = np.array(clean)
    halluc_pick = rng.choice(halluc, n_per_class, replace=False)
    clean_pick = rng.choice(clean, n_per_class, replace=False)
    return clean_pick.tolist(), halluc_pick.tolist()


def per_row_aligned(rows_idx, mat, ans, total_L, window=80):
    """Extract a per-row window centered on each row's answer-span START.

    Returns (aligned, ans_aligned) of shape (n_rows, window). For row r
    we take ``mat[r, start_r : start_r + window]`` where ``start_r`` is
    chosen so the row's first answer-mask token sits at column index
    ``window // 4`` (so most of the answer span lies in the right half).

    Rows whose answer-span doesn't fit are zero-padded; the corresponding
    ans_aligned entries are False so no × is drawn there.
    """
    half = window // 2
    pre = window // 4  # tokens shown BEFORE the answer-span start
    aligned = np.zeros((len(rows_idx), window), dtype=mat.dtype)
    aligned_ans = np.zeros((len(rows_idx), window), dtype=bool)
    for i, r in enumerate(rows_idx):
        ap = np.where(ans[r])[0]
        if len(ap) == 0:
            start_r = 0
        else:
            start_r = max(0, ap[0] - pre)
        end_r = min(total_L, start_r + window)
        actual = end_r - start_r
        aligned[i, :actual] = mat[r, start_r:end_r]
        aligned_ans[i, :actual] = ans[r, start_r:end_r]
    return aligned, aligned_ans


def heatmap_panel(ax, mat, ans_subset, title, *, cmap, vmin, vmax,
                  token_labels=None, xticks_every=8):
    """Single 10×L heatmap with ×-marks on the answer-span positions of
    hallucinated rows. ``mat`` is (10, L), ``ans_subset`` is (10, L) bool.

    NaN cells are rendered as neutral grey so padding columns don't
    dominate the colour scale.
    """
    n, L = mat.shape
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("#cfcfcf")
    masked = np.ma.masked_invalid(mat)
    im = ax.imshow(masked, aspect="auto", cmap=cmap_obj, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    # Mark faulty positions with × — only present where ans_subset is True
    # AND we're in the "halluc" panel (caller's responsibility — pass an
    # all-False mask to suppress).
    yy, xx = np.where(ans_subset)
    if len(xx) > 0:
        ax.scatter(xx, yy, marker="x", s=22, color="white",
                   linewidths=1.2, alpha=0.95, zorder=3)
    ax.set_title(title, fontsize=10)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([f"seq {i}" for i in range(n)], fontsize=7)
    if token_labels is not None and len(token_labels) == L:
        # label every `xticks_every` token to keep readable
        idx = np.arange(0, L, xticks_every)
        ax.set_xticks(idx)
        ax.set_xticklabels(
            [token_labels[i] for i in idx], fontsize=6, rotation=80, ha="right"
        )
    else:
        ax.set_xticks(np.arange(0, L, xticks_every))
    return im


def render_method_panels(
    method_name,
    score_per_pos,
    var_per_pos,
    val_pack,
    *,
    score_label, score_cmap, score_vlim,
    var_label, var_cmap, var_vlim,
    n_per_class, window, out_dir,
    seed=0,
):
    """Render the 2×2 heatmap grid for one method.

    score_per_pos : (N, L) float — top row colour
    var_per_pos   : (N, L) float — bottom row colour (use NaN to skip)
    *_vlim        : (vmin, vmax) tuple OR None for percentile auto-scale
    """
    val_ans = val_pack["ans"]
    val_label = val_pack["label"]
    val_attn = val_pack["attn"]
    L = score_per_pos.shape[1]
    pre = window // 4

    clean_idx, halluc_idx = select_rows(
        val_label, val_ans, n_per_class=n_per_class, seed=seed,
    )
    print(f"[{method_name}] clean rows: {clean_idx}")
    print(f"[{method_name}] halluc rows: {halluc_idx}")

    S_clean, ans_clean = per_row_aligned(clean_idx, score_per_pos, val_ans, L, window)
    S_halluc, ans_halluc = per_row_aligned(halluc_idx, score_per_pos, val_ans, L, window)
    V_clean, _ = per_row_aligned(clean_idx, var_per_pos, val_ans, L, window)
    V_halluc, _ = per_row_aligned(halluc_idx, var_per_pos, val_ans, L, window)
    attn_clean, _ = per_row_aligned(clean_idx, val_attn.astype(np.float32),
                                     val_ans, L, window)
    attn_halluc, _ = per_row_aligned(halluc_idx, val_attn.astype(np.float32),
                                      val_ans, L, window)
    attn_clean = attn_clean.astype(bool)
    attn_halluc = attn_halluc.astype(bool)

    def mask_padding(mat, attn):
        out = mat.astype(np.float64).copy()
        out[~attn] = np.nan
        return out

    S_clean = mask_padding(S_clean, attn_clean)
    S_halluc = mask_padding(S_halluc, attn_halluc)
    V_clean = mask_padding(V_clean, attn_clean)
    V_halluc = mask_padding(V_halluc, attn_halluc)

    def pct(arrays, lo=2, hi=98):
        flat = np.concatenate(
            [a[~np.isnan(a)] for a in arrays if a is not None]
        )
        if len(flat) == 0:
            return 0.0, 1.0
        return float(np.percentile(flat, lo)), float(np.percentile(flat, hi))

    s_vmin, s_vmax = score_vlim if score_vlim else pct([S_clean, S_halluc])
    v_vmin, v_vmax = var_vlim if var_vlim else pct([V_clean, V_halluc])

    token_labels = [
        f"ans{(i - pre):+d}" if i % 8 == 0 else "" for i in range(window)
    ]

    fig, axes = plt.subplots(2, 2, figsize=(window * 0.18, 7.5))
    heatmap_panel(axes[0, 0], S_clean, np.zeros_like(S_clean, dtype=bool),
                  f"{score_label}  (clean rows)",
                  cmap=score_cmap, vmin=s_vmin, vmax=s_vmax,
                  token_labels=token_labels, xticks_every=8)
    heatmap_panel(axes[0, 1], S_halluc, ans_halluc,
                  f"{score_label}  (hallucinated rows; × = faulty)",
                  cmap=score_cmap, vmin=s_vmin, vmax=s_vmax,
                  token_labels=token_labels, xticks_every=8)
    heatmap_panel(axes[1, 0], V_clean, np.zeros_like(V_clean, dtype=bool),
                  f"{var_label}  (clean rows)",
                  cmap=var_cmap, vmin=v_vmin, vmax=v_vmax,
                  token_labels=token_labels, xticks_every=8)
    heatmap_panel(axes[1, 1], V_halluc, ans_halluc,
                  f"{var_label}  (hallucinated rows; × = faulty)",
                  cmap=var_cmap, vmin=v_vmin, vmax=v_vmax,
                  token_labels=token_labels, xticks_every=8)
    for ax in axes.flatten():
        ax.axvline(pre - 0.5, color="black", lw=1.0, alpha=0.5)
    # Colorbars (one per channel since clean/halluc share their colour scale).
    fig.colorbar(axes[0, 1].images[0], ax=axes[0, :].tolist(),
                 fraction=0.025, pad=0.01)
    fig.colorbar(axes[1, 1].images[0], ax=axes[1, :].tolist(),
                 fraction=0.025, pad=0.01)
    fig.suptitle(
        f"DFM auditor — {method_name}.  {len(clean_idx)} sequences × "
        f"{window} tokens, per-row aligned (vertical line = answer-span start). "
        f"× marks the faulty (answer-span) positions in hallucinated rows.\n"
        f"Top: {score_label}.   Bottom: {var_label}.",
        fontsize=10, y=1.03,
    )
    out_path = out_dir / f"sequence_heatmaps_{method_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{method_name}] wrote {out_path}")
    return {
        "method": method_name,
        "out_path": str(out_path),
        "window_size": int(window),
        "alignment": "per-row, answer-span start at relative offset {}".format(pre),
        "clean_rows": [int(r) for r in clean_idx],
        "halluc_rows": [int(r) for r in halluc_idx],
        "score_vmin": float(s_vmin), "score_vmax": float(s_vmax),
        "var_vmin": float(v_vmin), "var_vmax": float(v_vmax),
    }


def render_for_method(arch_name, model, val_pack, train_pack, tokenizer, *,
                      device, n_per_class=10, window=80, out_dir):
    """Legacy: render with halluc-head probability + SVGP variance.

    Kept for back-compat with callers that don't supply a multi-method
    config.
    """
    print(f"[{arch_name}] fitting SVGP on encoder features ...")
    svgp_pack = fit_svgp_on_encoder(
        train_pack["z"], train_pack["label"], train_pack["ans"], device=device,
    )
    print(f"[{arch_name}] computing SVGP variance per position ...")
    val_var = svgp_variance_per_position(svgp_pack, val_pack["z"])
    return render_method_panels(
        arch_name,
        val_pack["P"],
        val_var,
        val_pack,
        score_label="Halluc-head probability", score_cmap="Reds",
        score_vlim=(0.0, 1.0),
        var_label="SVGP variance", var_cmap="magma", var_vlim=None,
        n_per_class=n_per_class, window=window, out_dir=out_dir,
    )


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-transformer", default="runs/dfm_auditor_archB_d512/best.pt")
    p.add_argument("--ckpt-mlp",         default="runs/dfm_archB_d512_mlp/best.pt")
    p.add_argument("--ckpt-slot-only",   default="runs/dfm_auditor_d512_hidden/best.pt",
                   help="DFM EBM slot-only checkpoint (for the 'DFM EBM' method panels)")
    p.add_argument("--out", default="runs/dfm_auditor_plots/sequence_heatmaps")
    p.add_argument(
        "--out-suffix", default="",
        help="appended to --out so successive sweeps don't overwrite each "
             "other (e.g. --out-suffix _v3 -> .../sequence_heatmaps_v3)",
    )
    p.add_argument("--max-rows", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--energy-t", type=float, default=4.0)
    p.add_argument("--n-per-class", type=int, default=10)
    p.add_argument("--window", type=int, default=50)
    p.add_argument("--methods", nargs="+",
                   choices=("paper_dE", "dfm_ebm", "archB_transformer", "archB_mlp"),
                   default=["paper_dE", "dfm_ebm", "archB_transformer", "archB_mlp"],
                   help="which methods to plot (default: all four)")
    args = p.parse_args()

    out_dir = Path(args.out + args.out_suffix)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[heatmaps] writing to: {out_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[heatmaps] loading data ...")
    cfg_dummy = Config()
    cfg_dummy.hallueval_dfm_auditor = replace(
        cfg_dummy.hallueval_dfm_auditor,
        enabled=True, batch_size=args.batch_size, max_rows=args.max_rows,
    )
    dm = HalluevalDFMDataModule(cfg_dummy.hallueval_dfm_auditor)

    summaries = []

    # Each method needs (a) one trained model whose features we extract
    # (some methods share — paper_dE uses the cache directly), (b) an SVGP
    # fitted on those features for the variance channel.
    # We minimise compute by gathering each model's val_pack ONCE.

    common_kwargs = dict(
        n_per_class=args.n_per_class, window=args.window, out_dir=out_dir,
    )

    if "paper_dE" in args.methods or "dfm_ebm" in args.methods:
        print("\n=== loading slot-only DFM checkpoint (used for paper_dE + dfm_ebm) ===")
        model_S, _, _, _ = load_archB(args.ckpt_slot_only, device)
        print("[slot-only] gathering train features ...")
        train_pack_S = gather_per_position(
            model_S, dm.train_dataloader(), device, args.energy_t,
        )
        print("[slot-only] gathering val features ...")
        val_pack_S = gather_per_position(
            model_S, dm.val_dataloader(), device, args.energy_t,
        )

        if "paper_dE" in args.methods:
            print("\n--- method: paper ΔE (training-free) ---")
            # Score = ΔE per position (high = surprising token)
            # Variance = E_marg per position (proxy for LM marginal entropy:
            #   low E_marg = sharply peaked LM, high E_marg = uncertain LM).
            #   Strictly, this is a marginal energy not a variance, but it's
            #   the only per-position uncertainty channel naturally available
            #   for a training-free per-position score.
            s = render_method_panels(
                "paper_dE",
                val_pack_S["DeltaE"],
                val_pack_S["E_marg"],
                val_pack_S,
                score_label="paper ΔE (E_logit − E_marg)",
                score_cmap="Reds", score_vlim=None,
                var_label="LM marginal energy E_marg  (proxy uncertainty)",
                var_cmap="magma", var_vlim=None,
                **common_kwargs,
            )
            summaries.append(s)

        if "dfm_ebm" in args.methods:
            print("\n--- method: DFM EBM (slot-only, unsupervised) ---")
            # Energy = -log p_t(x|h), the closed-form mixture-of-Dirichlets
            # density. Variance = SVGP fitted to the slot-only encoder.
            print("[dfm_ebm] fitting SVGP on slot-only encoder ...")
            svgp_pack = fit_svgp_on_encoder(
                train_pack_S["z"], train_pack_S["label"], train_pack_S["ans"],
                device=device,
            )
            val_var = svgp_variance_per_position(svgp_pack, val_pack_S["z"])
            s = render_method_panels(
                "dfm_ebm",
                val_pack_S["E"],
                val_var,
                val_pack_S,
                score_label="DFM EBM  −log p_t(x|h)",
                score_cmap="Blues", score_vlim=None,
                var_label="SVGP variance (slot-only encoder)",
                var_cmap="magma", var_vlim=None,
                **common_kwargs,
            )
            summaries.append(s)

        del model_S, train_pack_S, val_pack_S
        torch.cuda.empty_cache()

    for arch_name, ckpt, label_cap in (
        ("archB_transformer", args.ckpt_transformer, "Transformer"),
        ("archB_mlp",         args.ckpt_mlp,         "MLP"),
    ):
        if arch_name not in args.methods:
            continue
        print(f"\n=== {arch_name} ===")
        model, cfg, _, _ = load_archB(ckpt, device)
        print(f"[{arch_name}] gathering train features (for SVGP fit) ...")
        train_pack = gather_per_position(
            model, dm.train_dataloader(), device, args.energy_t,
        )
        print(f"[{arch_name}] gathering val features ...")
        val_pack = gather_per_position(
            model, dm.val_dataloader(), device, args.energy_t,
        )
        print(f"[{arch_name}] fitting SVGP on encoder ...")
        svgp_pack = fit_svgp_on_encoder(
            train_pack["z"], train_pack["label"], train_pack["ans"],
            device=device,
        )
        val_var = svgp_variance_per_position(svgp_pack, val_pack["z"])
        s = render_method_panels(
            arch_name,
            val_pack["P"],
            val_var,
            val_pack,
            score_label=f"Halluc-head probability ({label_cap})",
            score_cmap="Reds", score_vlim=(0.0, 1.0),
            var_label=f"SVGP variance ({label_cap} encoder)",
            var_cmap="magma", var_vlim=None,
            **common_kwargs,
        )
        summaries.append(s)
        del model, train_pack, val_pack
        torch.cuda.empty_cache()

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summaries, f, indent=2, default=float)
    print(f"\n[heatmaps] wrote {out_dir / 'summary.json'}")
    print(f"[heatmaps] {len(summaries)} method figures in {out_dir}")


if __name__ == "__main__":
    main()
