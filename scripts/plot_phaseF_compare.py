"""Phase F multi-cell comparison — Seq + Tok AUROC across all auditor cells.

Reads ``runs/<ckpt>/auditor_eval.json`` for every auditor cell and
plots:

1. Bar chart of Seq AUROC and Tok AUROC across cells, with the
   protocol's F1 thresholds (0.99 / 0.95) and the SE baseline drawn.
2. ROC overlay at sequence level for the headline statistic per cell
   (best of E_seq_grad², U_pos_mean, signed-flipped energy).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

import aitchinson_flow.models  # noqa: E402,F401 — populate registry
from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.data.wiki import WikiAuditorDataset, load_wiki_cache  # noqa: E402
from aitchinson_flow.models import build_model  # noqa: E402

from scripts.eval_full import _config_from_payload  # noqa: E402
from scripts.eval_auditor_wiki import (  # noqa: E402
    _grad_norm_per_pos, _signed_energy, _roc_auc,
)


CELLS = [
    ("aud_gpt2_logit",      "logit-only,  d256/L4"),
    ("aud_gpt2_ctx",        "+ context,   d256/L4"),
    ("aud_gpt2_logit_d512", "logit-only,  d512/L6"),
    ("aud_gpt2_ctx_d512",   "+ context,   d512/L6"),
]


def _roc_curve(neg: torch.Tensor, pos: torch.Tensor) -> tuple[list[float], list[float], float]:
    n_pos, n_neg = pos.numel(), neg.numel()
    if n_pos == 0 or n_neg == 0:
        return [], [], float("nan")
    pos_np = pos.flatten().numpy()
    neg_np = neg.flatten().numpy()
    thresholds = np.sort(np.concatenate([pos_np, neg_np]))[::-1]
    tpr = []
    fpr = []
    for thr in thresholds:
        tpr.append(float((pos_np >= thr).mean()))
        fpr.append(float((neg_np >= thr).mean()))
    auc = _roc_auc(neg, pos)
    return fpr, tpr, auc


def _gather_seq_scores(model, ds, *, gamma_value, batch_size, needs_h):
    device = next(model.parameters()).device
    n = len(ds)
    GN_clean: list[torch.Tensor] = []
    GN_invalid: list[torch.Tensor] = []
    SE_clean: list[torch.Tensor] = []
    SE_invalid: list[torch.Tensor] = []
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        items = [ds[i] for i in range(s, e)]
        x_clean = torch.stack([it["x"] for it in items]).to(device)
        x_invalid = torch.stack([it["x_invalid"] for it in items]).to(device)
        if needs_h:
            h_c = torch.stack([it["h_clean"] for it in items]).to(device)
            h_i = torch.stack([it["h_invalid"] for it in items]).to(device)
        else:
            h_c = h_i = None
        gn_c = _grad_norm_per_pos(model, x_clean, gamma_value, h_c).cpu()
        gn_i = _grad_norm_per_pos(model, x_invalid, gamma_value, h_i).cpu()
        GN_clean.append(gn_c)
        GN_invalid.append(gn_i)
        SE_clean.append(torch.stack([it["SE_pos_clean"] for it in items]).sum(dim=-1))
        SE_invalid.append(torch.stack([it["SE_pos_invalid"] for it in items]).sum(dim=-1))
    return (
        torch.cat(GN_clean).pow(2).sum(dim=-1),
        torch.cat(GN_invalid).pow(2).sum(dim=-1),
        torch.cat(SE_clean),
        torch.cat(SE_invalid),
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="data/wiki_cache_gpt2.pt")
    p.add_argument("--out", default="runs/phaseF_compare.png")
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args(argv)

    cache = load_wiki_cache(args.cache)

    summaries: dict[str, dict] = {}
    seq_curves: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    seq_curves_se: tuple[torch.Tensor, torch.Tensor] | None = None
    for ck, _ in CELLS:
        path = ROOT / "runs" / ck / "auditor_eval.json"
        if not path.exists():
            print(f"[plot] missing {path}, skipping {ck}")
            continue
        summaries[ck] = json.loads(path.read_text())

        # Re-derive ROC curve points: forward pass through the model.
        ckpt = ROOT / "runs" / ck / "epoch_final.pt"
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = _config_from_payload(payload)
        model = build_model(cfg).to(cfg.training.device)
        model.load_state_dict(payload.get("model_state_dict", payload))
        model.eval()
        ctx_mode = getattr(cfg.eqm, "context_features", "off")
        needs_h = ctx_mode != "off"
        ds = WikiAuditorDataset(cache, with_hidden=needs_h)
        gamma = float(cfg.eqm.auditor_gamma)
        E_c, E_i, SE_c, SE_i = _gather_seq_scores(
            model, ds, gamma_value=gamma, batch_size=args.batch_size, needs_h=needs_h
        )
        seq_curves[ck] = (E_c, E_i)
        if seq_curves_se is None:
            seq_curves_se = (SE_c, SE_i)
        del model
        torch.cuda.empty_cache()

    # Render figure.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1.1, 1.0]})

    # Panel 1: Seq + Tok AUROC bars.
    ax = axes[0]
    labels = []
    seq_aucs = []
    tok_aucs = []
    for ck, label in CELLS:
        if ck not in summaries:
            continue
        s = summaries[ck]
        seq = s["seq_auc"]
        # Headline seq AUC: best of E_seq_grad_sq / U_pos_mean / |E_seq_signed|
        candidates = [
            ("E_seq_grad²", seq.get("E_seq_grad_sq", float("nan"))),
            ("U_pos_mean",  seq.get("U_pos_mean", float("nan"))),
        ]
        signed = seq.get("E_seq_signed", float("nan"))
        if not np.isnan(signed):
            candidates.append((
                "|E_seq_signed|",
                signed if signed >= 0.5 else 1.0 - signed,
            ))
        best_seq = max(candidates, key=lambda kv: kv[1])
        tok_u = s["tok_auc"].get("U_pos", float("nan"))
        labels.append(label)
        seq_aucs.append(best_seq[1])
        tok_aucs.append(tok_u)

    x = np.arange(len(labels))
    w = 0.35
    bars1 = ax.bar(x - w/2, seq_aucs, w, label="Seq AUROC (best stat)", color="tab:blue")
    bars2 = ax.bar(x + w/2, tok_aucs, w, label="Tok AUROC (U_pos)",     color="tab:orange")
    # Threshold lines
    se_seq = float(summaries[next(iter(summaries))]["seq_auc"].get("SE_seq", float("nan")))
    se_tok = float(summaries[next(iter(summaries))]["tok_auc"].get("SE_pos", float("nan")))
    ax.axhline(0.99, color="darkgreen", linestyle="--", linewidth=0.8, label="F1 Seq target (0.99)")
    ax.axhline(0.95, color="green", linestyle=":", linewidth=0.8, label="F1 Tok target (0.95)")
    ax.axhline(se_seq, color="tab:purple", linestyle="-", linewidth=0.8, alpha=0.6,
               label=f"SE_seq baseline ({se_seq:.3f})")
    ax.axhline(se_tok, color="tab:red", linestyle="-", linewidth=0.8, alpha=0.6,
               label=f"SE_pos baseline ({se_tok:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right", fontsize=9)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.92, 1.005)
    ax.set_title("Phase F — Seq + Tok AUROC across auditor cells (n=300)")
    for r, v in zip(bars1, seq_aucs):
        ax.text(r.get_x() + r.get_width()/2, v + 0.001, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    for r, v in zip(bars2, tok_aucs):
        ax.text(r.get_x() + r.get_width()/2, v + 0.001, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # Panel 2: Sequence-level ROC overlay.
    ax = axes[1]
    for ck, label in CELLS:
        if ck not in seq_curves:
            continue
        E_c, E_i = seq_curves[ck]
        fpr, tpr, auc = _roc_curve(E_c, E_i)
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
    if seq_curves_se is not None:
        SE_c, SE_i = seq_curves_se
        fpr, tpr, auc = _roc_curve(SE_c, SE_i)
        ax.plot(fpr, tpr, label=f"Spilled Energy seq (AUC={auc:.3f})", color="tab:purple",
                linewidth=2.0, linestyle="--")
    ax.plot([0, 1], [0, 1], "k:", alpha=0.4)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("Sequence-level ROC: clean vs invalid (E_seq_grad²)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.02, 0.4)
    ax.set_ylim(0.6, 1.02)

    fig.suptitle(
        "Phase F — context conditioning + backbone scaling on the GPT-2 auditor",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"[plot] wrote {out_path}")


if __name__ == "__main__":
    main()
