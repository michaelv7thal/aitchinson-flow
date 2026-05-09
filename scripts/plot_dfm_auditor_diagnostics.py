"""Diagnostic plots for the DFM auditor results.

Produces four figures into ``runs/dfm_auditor_plots/``:

* ``fig1_auroc_locality.png`` — bar chart comparing row-level AUROC,
  per-token AUROC at answer span, per-token AUROC at non-answer
  (cascade), and the locality gap (ans − non-ans) across all signals
  in the auditor track. Visualizes the headline cascade-vs-locality
  trade-off.

* ``fig2_score_distributions.png`` — class-conditional histograms of
  row-level scores (paper ΔE, DFM EBM, transformer Arch B halluc,
  MLP Arch B halluc) plus reliability diagrams.

* ``fig3_position_trace.png`` — mean halluc score by position relative
  to the answer-span start, for clean vs hallucinated rows, transformer
  vs MLP backbones. Visually shows the cascade: transformer scores
  stay high *after* the answer span (cross-position info leakage),
  while MLP scores fall back to chance.

* ``fig4_token_heatmaps.png`` — per-token annotated heatmaps for a
  small set of example rows. Each row shows decoded token text +
  answer-mask + transformer halluc + MLP halluc + EBM energy + paper
  ΔE, side-by-side. The "which tokens are faulty" view.

Run with the existing trained GPT-2 checkpoints (`runs/dfm_auditor_*`).
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

import numpy as np  # noqa: E402
import torch  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import aitchinson_flow.models  # noqa: E402,F401
from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.data.hallueval_dfm import HalluevalDFMDataModule  # noqa: E402
from aitchinson_flow.models.factory import build_model  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    order = np.argsort(-s)
    ys = y[order]
    npos, nneg = int(y.sum()), int(len(y) - y.sum())
    cum = np.cumsum(ys == 1)
    auc = float((cum * (ys == 0)).sum() / (npos * nneg))
    return max(auc, 1.0 - auc)


def load_model(ckpt_path: str, device):
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
def gather_scores(model, loader, device, *, energy_t):
    """Per-row + per-position scores from a single trained model.

    Returns a dict of arrays:
      label, prob_pos (B, L) [if halluc_head], E_pos (B, L), E_seq_mean (B,),
      DeltaE_pos (B, L), DeltaE_seq (B,), answer_mask (B, L), full_ids (B, L).
    """
    out = {
        "label": [], "answer_mask": [], "full_ids": [], "topk_idx": [],
        "prob_pos": [], "E_pos": [], "E_seq_mean": [], "DeltaE_pos": [], "DeltaE_seq": [],
    }
    has_halluc = getattr(model, "halluc_head", None) is not None
    for batch in loader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        E = model.energy_at_lm_distribution(batch, t=energy_t)
        out["E_pos"].append(E["E_pos"].cpu().numpy())
        out["E_seq_mean"].append(E["E_seq_mean"].cpu().numpy())
        out["DeltaE_pos"].append(batch["DeltaE"].cpu().numpy())
        out["DeltaE_seq"].append(E["DeltaE_seq"].cpu().numpy())
        out["label"].append(batch["label"].cpu().numpy().astype(int))
        out["answer_mask"].append(batch["answer_mask"].cpu().numpy().astype(bool))
        out["full_ids"].append(batch["full_ids"].cpu().numpy())
        out["topk_idx"].append(batch["topk_idx"].cpu().numpy())
        if has_halluc:
            H = model.halluc_score_at_lm(batch, t=energy_t)
            out["prob_pos"].append(H["prob_pos"].cpu().numpy())
        else:
            B, L, _ = batch["topk_logp"].shape
            out["prob_pos"].append(np.full((B, L), np.nan, dtype=np.float32))
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def row_pool(prob_pos, answer_mask):
    denom = answer_mask.sum(-1).clip(1)
    return (prob_pos * answer_mask).sum(-1) / denom


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def fig1_auroc_locality(rows, out_path):
    """Bar chart: row AUROC, the two per-token row-discrimination metrics,
    AND the *within-row localisation* AUROC (the truer "does the model
    identify which token is faulty?" measure)."""
    methods = [r["label"] for r in rows]
    metrics = [
        "row AUROC",
        "per-tok (answer)\n(is from halluc row?)",
        "per-tok (non-answer)\n(is from halluc row?)",
        "within-row localisation\n(is this position the faulty one?)",
    ]
    vals = np.array([
        [r["auc_row"], r["auc_tok_ans"], r["auc_tok_non"],
         r.get("auc_within_row", float("nan"))]
        for r in rows
    ])
    fig, axes = plt.subplots(1, 4, figsize=(17, 5), sharey=False)
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b3", "#937860", "#da8bc3"]
    for j, (ax, met) in enumerate(zip(axes, metrics)):
        bars = ax.bar(range(len(methods)), vals[:, j],
                      color=colors[: len(methods)], edgecolor="black", linewidth=0.6)
        ax.set_title(met, fontsize=10)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
        ax.set_ylim(min(0.4, np.nanmin(vals[:, j]) - 0.05),
                    max(1.0, np.nanmax(vals[:, j]) + 0.05))
        for bar, v in zip(bars, vals[:, j]):
            if np.isnan(v):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(
        "DFM auditor — AUROC across methods (HaluEval-QA, GPT-2, n_val=800)\n"
        "Note: per-tok (answer/non-answer) AUROC measures *row* "
        "discrimination, dominated by cross-position signal; "
        "within-row localisation is the honest 'which token is faulty?' metric.",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def within_row_localization_auroc(prob_pos, answer_mask, label, attn_mask):
    """For halluc rows only, compute per-row AUROC of (prob ranks
    answer-positions above non-answer-positions). Returns mean across
    halluc rows. NaN values (e.g. uniform probability rows) are ignored.
    """
    # Reuse the AUROC helper defined elsewhere in this file
    aucs = []
    for r in np.where(label == 1)[0]:
        is_ans = answer_mask[r] & attn_mask[r]
        is_non = (~answer_mask[r]) & attn_mask[r]
        if is_ans.sum() == 0 or is_non.sum() == 0:
            continue
        valid = is_ans | is_non
        a = auroc(prob_pos[r, valid], is_ans[valid].astype(int))
        if not np.isnan(a):
            aucs.append(a)
    if not aucs:
        return float("nan")
    return float(np.mean(aucs))


def fig2_score_distributions(buckets, out_path):
    """Class-conditional score histograms for the row-level signals."""
    fig, axes = plt.subplots(1, len(buckets), figsize=(5 * len(buckets), 4))
    if len(buckets) == 1:
        axes = [axes]
    for ax, (title, scores, labels) in zip(axes, buckets):
        s_clean = scores[labels == 0]
        s_halluc = scores[labels == 1]
        bins = np.linspace(min(scores.min(), 0.0), scores.max(), 30)
        ax.hist(s_clean, bins=bins, alpha=0.6, color="#4c72b0",
                label=f"clean (n={len(s_clean)})", density=True, edgecolor="black",
                linewidth=0.4)
        ax.hist(s_halluc, bins=bins, alpha=0.6, color="#c44e52",
                label=f"halluc (n={len(s_halluc)})", density=True, edgecolor="black",
                linewidth=0.4)
        auc = auroc(scores, labels)
        ax.set_title(f"{title}  (AUROC = {auc:.3f})", fontsize=11)
        ax.set_xlabel("score"); ax.set_ylabel("density")
        ax.legend(fontsize=8)
    fig.suptitle("Row-level score distributions by class", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig3_position_trace(per_position_traces, out_path):
    """Mean halluc score vs position-relative-to-answer-span-start, by class.

    The cascade signature: transformer scores stay high *after* the answer
    span (positive offsets); MLP scores fall back to chance.
    """
    fig, axes = plt.subplots(1, len(per_position_traces),
                             figsize=(8 * len(per_position_traces), 6),
                             sharey=True)
    if len(per_position_traces) == 1:
        axes = [axes]
    for ax, (title, traces) in zip(axes, per_position_traces):
        offsets = traces["offsets"]
        ax.plot(offsets, traces["clean_mean"], color="#4c72b0", lw=2.5,
                label="clean rows")
        ax.fill_between(offsets,
                        traces["clean_mean"] - traces["clean_se"],
                        traces["clean_mean"] + traces["clean_se"],
                        color="#4c72b0", alpha=0.2)
        ax.plot(offsets, traces["halluc_mean"], color="#c44e52", lw=2.5,
                label="hallucinated rows")
        ax.fill_between(offsets,
                        traces["halluc_mean"] - traces["halluc_se"],
                        traces["halluc_mean"] + traces["halluc_se"],
                        color="#c44e52", alpha=0.2)
        ax.axvspan(0, traces["answer_span_mean"], color="gold", alpha=0.20,
                   label=f"typical answer span (μ≈{traces['answer_span_mean']:.0f} tokens)")
        ax.axhline(0.5, color="gray", linestyle=":", lw=1.0, label="chance")
        ax.set_xlabel("position − answer-span start", fontsize=11)
        ax.set_ylabel("mean halluc probability", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right")
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(-50, 50)
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        "Per-position halluc score — cascade signature\n"
        "Halluc curve high *after* gold band = cross-position info leakage; "
        "curve dropping back to chance = honest locality.",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig4_token_heatmaps(rows, tokenizer, out_path, max_tokens=40):
    """Per-token heatmaps with decoded text labels.

    For each example row plot 4 channels stacked: transformer halluc prob,
    MLP halluc prob, EBM energy (z-scored), paper ΔE per position.
    Token strings appear as the x-tick labels; answer mask is shaded.
    """
    n = len(rows)
    fig, axes = plt.subplots(n, 1, figsize=(max_tokens * 0.38, 3.6 * n))
    if n == 1:
        axes = [axes]

    channels = [
        ("Transformer halluc prob", "prob_T", (0.0, 1.0), "Reds"),
        ("MLP halluc prob",        "prob_M", (0.0, 1.0), "Greens"),
        ("EBM energy (z)",         "E_z",    (-2.5, 2.5), "PuOr_r"),
        ("paper ΔE (z)",           "DE_z",   (-2.5, 2.5), "RdBu_r"),
    ]

    for ax, row in zip(axes, rows):
        # Slice token range — center on answer-span if it's narrow
        ans_pos = np.where(row["answer_mask"])[0]
        if len(ans_pos) == 0:
            start, end = 0, max_tokens
        else:
            mid = int(np.mean(ans_pos))
            start = max(0, mid - max_tokens // 2)
            end = min(len(row["full_ids"]), start + max_tokens)
            start = max(0, end - max_tokens)
        toks = row["full_ids"][start:end]
        ans_mask = row["answer_mask"][start:end]
        prob_T = row["prob_T"][start:end]
        prob_M = row["prob_M"][start:end]
        E_z = row["E_z"][start:end]
        DE_z = row["DE_z"][start:end]
        labels = [tokenizer.decode([int(t)]).replace("\n", "↵") for t in toks]

        # Build a grid: 4 channels × len(toks)
        grid = np.stack([prob_T, prob_M, E_z, DE_z], axis=0)
        # Mask of clamps for display range
        ax.set_facecolor("#fafafa")
        for i, (name, key, (vmin, vmax), cmap) in enumerate(channels):
            row_data = grid[i:i + 1]
            ax.imshow(
                row_data,
                aspect="auto",
                interpolation="nearest",
                cmap=cmap, vmin=vmin, vmax=vmax,
                extent=[-0.5, len(toks) - 0.5, n - i - 0.5, n - i + 0.5],
            )
        # Annotate each cell with the value
        for i, (_, _, _, _) in enumerate(channels):
            for j in range(len(toks)):
                v = grid[i, j]
                if not np.isfinite(v):
                    continue
                if i < 2:
                    txt = f"{v:.2f}"
                else:
                    txt = f"{v:+.1f}"
                ax.text(j, n - i, txt, ha="center", va="center",
                        fontsize=6, color="black")
        # Token labels as x-ticks at the bottom
        ax.set_xticks(np.arange(len(toks)))
        ax.set_xticklabels(labels, fontsize=7, rotation=80, ha="right")
        ax.set_yticks([n - i for i in range(len(channels))])
        ax.set_yticklabels([c[0] for c in channels], fontsize=8)
        ax.set_xlim(-0.5, len(toks) - 0.5)
        ax.set_ylim(0.5, n + 0.5)
        # Shade answer span
        in_ans = np.where(ans_mask)[0]
        if len(in_ans) > 0:
            ax.axvspan(in_ans[0] - 0.5, in_ans[-1] + 0.5, color="gold", alpha=0.25,
                       zorder=-1)
        title = (
            f"row {row['row_idx']}  ·  label = "
            f"{'HALLUC' if row['label'] == 1 else 'CLEAN'}  ·  pair_id = {row['pair_id']}"
        )
        ax.set_title(title, fontsize=10, loc="left")

    fig.suptitle(
        "Per-token diagnostics — gold band = answer span. "
        "Rows: transformer halluc, MLP halluc, EBM energy (z), paper ΔE (z).",
        fontsize=11, y=1.005,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-transformer-archB",
                   default="runs/dfm_auditor_archB_d512/best.pt")
    p.add_argument("--ckpt-mlp-archB",
                   default="runs/dfm_archB_d512_mlp/best.pt")
    p.add_argument("--ckpt-slot-only",
                   default="runs/dfm_auditor_d512_hidden/best.pt",
                   help="DFM EBM checkpoint (slot-only training, no halluc head)")
    p.add_argument("--out", default="runs/dfm_auditor_plots")
    p.add_argument("--max-rows", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--energy-t", type=float, default=4.0)
    p.add_argument("--n-example-rows", type=int, default=4,
                   help="how many example rows for fig4 heatmaps")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[plot] loading data ...")
    cfg_dummy = Config()
    cfg_dummy.hallueval_dfm_auditor = replace(
        cfg_dummy.hallueval_dfm_auditor,
        enabled=True, batch_size=args.batch_size, max_rows=args.max_rows,
    )
    dm = HalluevalDFMDataModule(cfg_dummy.hallueval_dfm_auditor)

    print("[plot] loading transformer Arch B ...")
    model_T, _, _, _ = load_model(args.ckpt_transformer_archB, device)
    print("[plot] loading MLP Arch B ...")
    model_M, _, _, _ = load_model(args.ckpt_mlp_archB, device)
    print("[plot] loading slot-only (DFM EBM) ...")
    model_S, _, _, _ = load_model(args.ckpt_slot_only, device)

    print("[plot] gathering scores from val loader ...")
    s_T = gather_scores(model_T, dm.val_dataloader(), device, energy_t=args.energy_t)
    s_M = gather_scores(model_M, dm.val_dataloader(), device, energy_t=args.energy_t)
    # Slot-only EBM has no halluc head; we only use its E_pos / E_seq_mean.
    s_S = gather_scores(model_S, dm.val_dataloader(), device, energy_t=args.energy_t)

    label = s_T["label"]
    answer_mask = s_T["answer_mask"]

    # ------------------ Figure 1: AUROC + locality bars ------------------- #
    # Real-token attn mask (drop GPT-2 pad token positions)
    attn_mask = (s_T["full_ids"] != 50256)
    rows_summary = []
    label_pos = np.broadcast_to(label[:, None], answer_mask.shape)

    # paper ΔE
    DE = s_T["DeltaE_seq"]; DE_pos = s_T["DeltaE_pos"]
    rows_summary.append(dict(
        label="paper ΔE",
        auc_row=auroc(DE, label),
        auc_tok_ans=auroc(DE_pos[answer_mask], label_pos[answer_mask]),
        auc_tok_non=auroc(DE_pos[~answer_mask], label_pos[~answer_mask]),
        auc_within_row=within_row_localization_auroc(
            DE_pos, answer_mask, label, attn_mask
        ),
    ))
    # DFM EBM (slot-only): per-token signal is E_pos (negate so larger = more anomalous)
    E_S = -s_S["E_pos"]                # negate: larger = more out-of-distribution
    E_S_seq = -s_S["E_seq_mean"]
    rows_summary.append(dict(
        label="DFM EBM (slot only)",
        auc_row=auroc(E_S_seq, label),
        auc_tok_ans=auroc(E_S[answer_mask], label_pos[answer_mask]),
        auc_tok_non=auroc(E_S[~answer_mask], label_pos[~answer_mask]),
        auc_within_row=within_row_localization_auroc(
            E_S, answer_mask, label, attn_mask
        ),
    ))
    # Transformer halluc head
    P_T = s_T["prob_pos"]; P_T_row = row_pool(P_T, answer_mask)
    rows_summary.append(dict(
        label="Arch B (Transformer)",
        auc_row=auroc(P_T_row, label),
        auc_tok_ans=auroc(P_T[answer_mask], label_pos[answer_mask]),
        auc_tok_non=auroc(P_T[~answer_mask], label_pos[~answer_mask]),
        auc_within_row=within_row_localization_auroc(
            P_T, answer_mask, label, attn_mask
        ),
    ))
    # MLP halluc head
    P_M = s_M["prob_pos"]; P_M_row = row_pool(P_M, answer_mask)
    rows_summary.append(dict(
        label="Arch B (MLP)",
        auc_row=auroc(P_M_row, label),
        auc_tok_ans=auroc(P_M[answer_mask], label_pos[answer_mask]),
        auc_tok_non=auroc(P_M[~answer_mask], label_pos[~answer_mask]),
        auc_within_row=within_row_localization_auroc(
            P_M, answer_mask, label, attn_mask
        ),
    ))
    fig1_path = out / "fig1_auroc_locality.png"
    fig1_auroc_locality(rows_summary, fig1_path)
    print(f"[plot] wrote {fig1_path}")

    # ------------------ Figure 2: row-score distributions ----------------- #
    buckets = [
        ("paper ΔE",                DE,        label),
        ("DFM EBM (−E_seq_mean)",   E_S_seq,   label),
        ("Transformer halluc row",  P_T_row,   label),
        ("MLP halluc row",          P_M_row,   label),
    ]
    fig2_path = out / "fig2_score_distributions.png"
    fig2_score_distributions(buckets, fig2_path)
    print(f"[plot] wrote {fig2_path}")

    # ------------------ Figure 3: per-position trace ---------------------- #
    def per_pos_trace(prob_pos, label, answer_mask, max_offset=80):
        # offset = position - first answer-mask token of the row
        out_clean = np.zeros(2 * max_offset + 1)
        out_halluc = np.zeros(2 * max_offset + 1)
        cnt_clean = np.zeros(2 * max_offset + 1)
        cnt_halluc = np.zeros(2 * max_offset + 1)
        ans_lens = []
        for r in range(prob_pos.shape[0]):
            ap = np.where(answer_mask[r])[0]
            if len(ap) == 0:
                continue
            start = ap[0]
            ans_lens.append(len(ap))
            for k in range(prob_pos.shape[1]):
                off = k - start
                if -max_offset <= off <= max_offset:
                    if label[r] == 0:
                        out_clean[off + max_offset] += prob_pos[r, k]
                        cnt_clean[off + max_offset] += 1
                    else:
                        out_halluc[off + max_offset] += prob_pos[r, k]
                        cnt_halluc[off + max_offset] += 1
        cnt_clean = np.clip(cnt_clean, 1, None)
        cnt_halluc = np.clip(cnt_halluc, 1, None)
        clean_mean = out_clean / cnt_clean
        halluc_mean = out_halluc / cnt_halluc
        # crude SE (not strictly correct because no variance is tracked, but
        # shows roughly where there are few samples)
        clean_se = 1.0 / np.sqrt(cnt_clean)
        halluc_se = 1.0 / np.sqrt(cnt_halluc)
        return dict(
            offsets=np.arange(-max_offset, max_offset + 1),
            clean_mean=clean_mean,
            halluc_mean=halluc_mean,
            clean_se=clean_se,
            halluc_se=halluc_se,
            answer_span_mean=float(np.mean(ans_lens)) if ans_lens else 0.0,
        )

    traces = [
        ("Transformer halluc head", per_pos_trace(P_T, label, answer_mask)),
        ("MLP halluc head",          per_pos_trace(P_M, label, answer_mask)),
    ]
    fig3_path = out / "fig3_position_trace.png"
    fig3_position_trace(traces, fig3_path)
    print(f"[plot] wrote {fig3_path}")

    # ------------------ Figure 4: token-level heatmaps -------------------- #
    print("[plot] selecting example rows ...")
    # z-score E and ΔE per position so the heatmap colors are comparable
    def zscore(x):
        x = np.asarray(x, dtype=np.float64)
        mu, sd = x.mean(), x.std() + 1e-9
        return (x - mu) / sd

    n_rows = label.shape[0]
    rng = np.random.default_rng(seed=0)
    # Pick a couple of clearly-hallucinated rows where Arch B fires strongly
    score_T = P_T_row
    score_M = P_M_row
    # Pair rows: prefer (clean, halluc) pairs that have non-trivial answer length
    candidates = []
    for r in range(n_rows):
        if int(answer_mask[r].sum()) >= 4:
            candidates.append(r)
    candidates = np.array(candidates)
    # Stratify: 1 clean + 3 halluc + adjacent pair if we can
    halluc_idx = candidates[label[candidates] == 1]
    clean_idx = candidates[label[candidates] == 0]
    n_h = min(args.n_example_rows - 1, len(halluc_idx))
    n_c = max(1, args.n_example_rows - n_h)
    chosen = list(rng.choice(clean_idx, n_c, replace=False)) + \
             list(rng.choice(halluc_idx, n_h, replace=False))

    print(f"[plot] example row indices: {chosen}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    rows = []
    for r in chosen:
        rows.append(dict(
            row_idx=int(r),
            pair_id=int(r // 2),  # rows are paired clean/halluc
            label=int(label[r]),
            full_ids=s_T["full_ids"][r],
            answer_mask=s_T["answer_mask"][r],
            prob_T=s_T["prob_pos"][r],
            prob_M=s_M["prob_pos"][r],
            E_z=zscore(-s_T["E_pos"][r]),
            DE_z=zscore(s_T["DeltaE_pos"][r]),
        ))
    fig4_path = out / "fig4_token_heatmaps.png"
    fig4_token_heatmaps(rows, tokenizer, fig4_path)
    print(f"[plot] wrote {fig4_path}")

    # ------------------ Save numbers used ---------------------------------- #
    summary = {"figures": ["fig1_auroc_locality.png", "fig2_score_distributions.png",
                            "fig3_position_trace.png", "fig4_token_heatmaps.png"],
               "auroc_table": rows_summary}
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[plot] wrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
