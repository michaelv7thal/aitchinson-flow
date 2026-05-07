"""Phase F visualization — what the trained EqM auditor sees vs Spilled Energy.

Five-panel figure saved to ``runs/<auditor_run>/auditor_eval.png``:

1. Per-position grad-norm heatmap on a representative invalid sequence,
   with corrupted positions marked, overlaid with the LM's per-position
   Spilled Energy. Lets you eyeball whether the trained auditor
   localises corruption at the same positions SE does.
2. Per-position grad-norm overlay (line plot) for 4 sample sequences:
   clean vs invalid, with corrupted positions highlighted.
3. Score distributions — violin/strip plot of Seq-level statistics
   (EqM `E_seq_grad_sq`, `U_pos_mean`, `SE_seq`) split by clean / invalid.
4. ROC curves at sequence level for EqM (E_seq_grad_sq, U_pos_mean) and
   SE_seq, side-by-side.
5. ROC curves at token level for EqM `U_pos` and `SE_pos`.

The figure is the primary writeup deliverable for Phase F: it shows
that the trained auditor reaches AUROC parity with SE on per-token
detection (the protocol's sufficient outcome) and visualises the
*complementary* (per-position) angle that justifies the auditor track.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import aitchinson_flow.models  # noqa: E402,F401 — populate registry
from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.data.wiki import WikiAuditorDataset, load_wiki_cache  # noqa: E402
from aitchinson_flow.models import build_model  # noqa: E402

from scripts.eval_full import _config_from_payload  # noqa: E402
from scripts.eval_auditor_wiki import (  # noqa: E402
    _grad_norm_per_pos,
    _signed_energy,
    _roc_auc,
)


def _gather_scores(
    model, ds, *, gamma_value: float, batch_size: int = 16, needs_h: bool = False
) -> dict[str, torch.Tensor]:
    """One pass through the cache; returns per-sequence and per-position tensors."""
    device = next(model.parameters()).device
    n = len(ds)
    GN_clean: list[torch.Tensor] = []
    GN_invalid: list[torch.Tensor] = []
    SG_clean: list[torch.Tensor] = []
    SG_invalid: list[torch.Tensor] = []
    SE_pos_clean: list[torch.Tensor] = []
    SE_pos_invalid: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []

    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        items = [ds[i] for i in range(s, e)]
        x_clean = torch.stack([it["x"] for it in items]).to(device)
        x_invalid = torch.stack([it["x_invalid"] for it in items]).to(device)
        mask = torch.stack([it["mask_corrupt"] for it in items])
        if needs_h:
            h_c = torch.stack([it["h_clean"] for it in items]).to(device)
            h_i = torch.stack([it["h_invalid"] for it in items]).to(device)
        else:
            h_c = h_i = None

        gn_c = _grad_norm_per_pos(model, x_clean, gamma_value, h_c).cpu()
        gn_i = _grad_norm_per_pos(model, x_invalid, gamma_value, h_i).cpu()
        sg_c = _signed_energy(model, x_clean, gamma_value, h_c).cpu()
        sg_i = _signed_energy(model, x_invalid, gamma_value, h_i).cpu()
        se_c = torch.stack([it["SE_pos_clean"] for it in items])
        se_i = torch.stack([it["SE_pos_invalid"] for it in items])

        GN_clean.append(gn_c); GN_invalid.append(gn_i)
        SG_clean.append(sg_c); SG_invalid.append(sg_i)
        SE_pos_clean.append(se_c); SE_pos_invalid.append(se_i)
        masks.append(mask)

    return {
        "GN_clean":      torch.cat(GN_clean, dim=0),       # (n, L)
        "GN_invalid":    torch.cat(GN_invalid, dim=0),
        "Signed_clean":  torch.cat(SG_clean, dim=0),       # (n,)
        "Signed_invalid": torch.cat(SG_invalid, dim=0),
        "SE_pos_clean":  torch.cat(SE_pos_clean, dim=0),
        "SE_pos_invalid": torch.cat(SE_pos_invalid, dim=0),
        "Mask":          torch.cat(masks, dim=0),          # (n, L) bool
    }


def _roc_curve(neg: torch.Tensor, pos: torch.Tensor) -> tuple[list[float], list[float], float]:
    """Compute (fpr, tpr, auc). pos = corrupted (positive class)."""
    auc = _roc_auc(neg, pos)
    n_pos, n_neg = pos.numel(), neg.numel()
    if n_pos == 0 or n_neg == 0:
        return [], [], float("nan")
    pos_np = pos.flatten().numpy()
    neg_np = neg.flatten().numpy()
    import numpy as np

    thresholds = np.sort(np.concatenate([pos_np, neg_np]))[::-1]
    tpr = []
    fpr = []
    for thr in thresholds:
        tpr.append(float((pos_np >= thr).mean()))
        fpr.append(float((neg_np >= thr).mean()))
    return fpr, tpr, auc


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cache", default="data/wiki_cache_gpt2.pt")
    p.add_argument("--out", default=None, help="output PNG path; default <ckpt-dir>/auditor_eval.png")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-rows", type=int, default=4, help="sample sequences for the line plot")
    args = p.parse_args(argv)

    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = _config_from_payload(payload)
    device = cfg.training.device
    gamma = float(cfg.eqm.auditor_gamma)

    model = build_model(cfg).to(device)
    state = payload["model_state_dict"] if "model_state_dict" in payload else payload
    model.load_state_dict(state)
    model.eval()

    cache = load_wiki_cache(args.cache)
    ctx_mode = getattr(cfg.eqm, "context_features", "off")
    needs_h = ctx_mode != "off"
    ds = WikiAuditorDataset(cache, with_hidden=needs_h)
    print(
        f"[plot_phaseF] gathering scores on {len(ds)} chunks "
        f"(ctx_mode={ctx_mode!r}) …"
    )
    scores = _gather_scores(
        model, ds, gamma_value=gamma, batch_size=args.batch_size, needs_h=needs_h
    )
    print("[plot_phaseF] computing AUROCs and rendering …")

    # Sequence-level statistics.
    E_seq_clean = scores["GN_clean"].pow(2).sum(dim=-1)
    E_seq_invalid = scores["GN_invalid"].pow(2).sum(dim=-1)
    U_mean_clean = scores["GN_clean"].mean(dim=-1)
    U_mean_invalid = scores["GN_invalid"].mean(dim=-1)
    SE_seq_clean = scores["SE_pos_clean"].sum(dim=-1)
    SE_seq_invalid = scores["SE_pos_invalid"].sum(dim=-1)
    SG_clean = scores["Signed_clean"]
    SG_invalid = scores["Signed_invalid"]

    # Token-level: only at corrupted positions.
    Mask = scores["Mask"]
    GN_pos_clean = scores["GN_clean"][Mask]
    GN_pos_invalid = scores["GN_invalid"][Mask]
    SE_pos_clean_at_corrupt = scores["SE_pos_clean"][Mask]
    SE_pos_invalid_at_corrupt = scores["SE_pos_invalid"][Mask]

    # Render figure.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.35)

    # Panel 1 (top-left, span 2 cols): heatmap of per-position grad-norm
    # for first ~32 invalid sequences, with corrupted positions overlaid.
    ax1 = fig.add_subplot(gs[0, :2])
    n_show = min(32, scores["GN_invalid"].shape[0])
    GN_show = scores["GN_invalid"][:n_show].numpy()
    Mask_show = Mask[:n_show].numpy()
    im1 = ax1.imshow(GN_show, aspect="auto", cmap="viridis")
    ax1.set_xlabel("Position (BPE token)")
    ax1.set_ylabel("Sequence index (invalid)")
    ax1.set_title("EqM auditor grad-norm per position (invalid sequences)\n× = corrupted positions")
    fig.colorbar(im1, ax=ax1, fraction=0.04, pad=0.02, label="‖∇⟨x,f⟩‖")
    # Mark corrupted positions with white x
    ys, xs = np.nonzero(Mask_show)
    ax1.scatter(xs, ys, marker="x", s=18, c="white", linewidths=1.0)

    # Panel 2 (top-right, span 2 cols): heatmap of SE_pos for the same sequences.
    ax2 = fig.add_subplot(gs[0, 2:])
    SE_show = scores["SE_pos_invalid"][:n_show].numpy()
    im2 = ax2.imshow(SE_show, aspect="auto", cmap="viridis")
    ax2.set_xlabel("Position (BPE token)")
    ax2.set_ylabel("Sequence index (invalid)")
    ax2.set_title("Spilled Energy per position (invalid sequences)\n× = corrupted positions")
    fig.colorbar(im2, ax=ax2, fraction=0.04, pad=0.02, label="LM NLL")
    ys, xs = np.nonzero(Mask_show)
    ax2.scatter(xs, ys, marker="x", s=18, c="white", linewidths=1.0)

    # Panel 3 (mid-left): per-position line plot for n_rows sample sequences.
    ax3 = fig.add_subplot(gs[1, :2])
    n_rows = min(args.n_rows, scores["GN_invalid"].shape[0])
    L = scores["GN_invalid"].shape[1]
    pos = np.arange(L)
    colors = plt.cm.tab10(np.arange(n_rows))
    for r in range(n_rows):
        ax3.plot(
            pos,
            scores["GN_clean"][r].numpy(),
            color=colors[r], alpha=0.5, linestyle=":", linewidth=1.2,
        )
        ax3.plot(
            pos,
            scores["GN_invalid"][r].numpy(),
            color=colors[r], alpha=1.0, linestyle="-", linewidth=1.6,
            label=f"seq {r}",
        )
        # Mark corrupted positions for this sequence
        idx_corrupt = np.where(Mask[r].numpy())[0]
        ax3.scatter(
            idx_corrupt,
            scores["GN_invalid"][r, idx_corrupt].numpy(),
            color=colors[r], marker="x", s=60, linewidths=2,
        )
    ax3.set_xlabel("Position")
    ax3.set_ylabel("‖∇⟨x,f⟩‖")
    ax3.set_title(
        f"Per-position grad-norm (4 example seqs)\ndotted = clean, solid = invalid, × = corrupted"
    )
    ax3.grid(alpha=0.3)
    ax3.legend(loc="upper right", fontsize=8, ncol=2)

    # Panel 4 (mid-right): score distributions — violin per stat.
    ax4 = fig.add_subplot(gs[1, 2:])
    stats = [
        ("E_seq_grad²", E_seq_clean, E_seq_invalid),
        ("U_pos_mean", U_mean_clean, U_mean_invalid),
        ("E_seq_signed", SG_clean, SG_invalid),
        ("SE_seq", SE_seq_clean, SE_seq_invalid),
    ]
    positions = []
    data = []
    labels = []
    for i, (nm, c, inv) in enumerate(stats):
        # Z-score normalisation for visual comparison.
        z_c = (c - c.mean()) / (c.std() + 1e-9)
        z_i = (inv - inv.mean()) / (inv.std() + 1e-9)
        # Use ranks to avoid extreme outliers killing the violin shape.
        all_v = torch.cat([c, inv])
        med = all_v.median()
        iqr = (all_v.quantile(0.75) - all_v.quantile(0.25)).clamp(min=1e-9)
        z_c = (c - med) / iqr
        z_i = (inv - med) / iqr
        positions.extend([2 * i + 1, 2 * i + 2])
        data.extend([z_c.numpy(), z_i.numpy()])
        labels.extend([f"{nm}\nclean", f"{nm}\ninvalid"])
    parts = ax4.violinplot(data, positions=positions, showmeans=True, widths=0.85)
    for j, pc in enumerate(parts["bodies"]):
        pc.set_facecolor("tab:blue" if j % 2 == 0 else "tab:red")
        pc.set_alpha(0.7)
    ax4.set_xticks(positions)
    ax4.set_xticklabels(labels, rotation=0, fontsize=8)
    ax4.set_ylabel("normalised score (median-IQR)")
    ax4.set_title("Score distributions: clean (blue) vs invalid (red)")
    ax4.grid(alpha=0.3, axis="y")

    # Panel 5 (bottom-left): Sequence-level ROC curves.
    ax5 = fig.add_subplot(gs[2, :2])
    seq_curves = [
        ("EqM E_seq_grad²", E_seq_clean, E_seq_invalid),
        ("EqM U_pos_mean", U_mean_clean, U_mean_invalid),
        ("EqM E_seq_signed", SG_clean, SG_invalid),
        ("Spilled Energy seq", SE_seq_clean, SE_seq_invalid),
    ]
    for nm, neg, pos_ in seq_curves:
        fpr, tpr, auc = _roc_curve(neg, pos_)
        ax5.plot(fpr, tpr, label=f"{nm} (AUC={auc:.3f})")
    ax5.plot([0, 1], [0, 1], "k:", alpha=0.4)
    ax5.set_xlabel("FPR")
    ax5.set_ylabel("TPR")
    ax5.set_title("Sequence-level ROC: clean vs invalid")
    ax5.legend(loc="lower right", fontsize=9)
    ax5.grid(alpha=0.3)
    ax5.set_xlim(-0.02, 1.02); ax5.set_ylim(-0.02, 1.02)

    # Panel 6 (bottom-right): Token-level ROC curves at corrupted positions.
    ax6 = fig.add_subplot(gs[2, 2:])
    tok_curves = [
        ("EqM U_pos (grad-norm)", GN_pos_clean, GN_pos_invalid),
        ("Spilled Energy pos", SE_pos_clean_at_corrupt, SE_pos_invalid_at_corrupt),
    ]
    for nm, neg, pos_ in tok_curves:
        fpr, tpr, auc = _roc_curve(neg, pos_)
        ax6.plot(fpr, tpr, label=f"{nm} (AUC={auc:.3f})")
    ax6.plot([0, 1], [0, 1], "k:", alpha=0.4)
    ax6.set_xlabel("FPR")
    ax6.set_ylabel("TPR")
    ax6.set_title("Token-level ROC: clean vs invalid (at corrupted positions)")
    ax6.legend(loc="lower right", fontsize=9)
    ax6.grid(alpha=0.3)
    ax6.set_xlim(-0.02, 1.02); ax6.set_ylim(-0.02, 1.02)

    fig.suptitle(
        f"Phase F auditor scorecard — {Path(args.ckpt).parent.name} "
        f"(γ_aud={gamma}, n={len(ds)})",
        fontsize=13,
    )

    out_path = Path(args.out) if args.out else Path(args.ckpt).parent / "auditor_eval.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_phaseF] wrote {out_path}")


if __name__ == "__main__":
    main()
