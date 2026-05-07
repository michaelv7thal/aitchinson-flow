"""Phase F sanity checks — does the trained auditor really work, or are
the AUROC numbers inflated by leakage / trivial features / overlapping
contexts?

Runs four diagnostics:

1. **Train-only vs held-out** — split the cache the same way the
   datamodule did (first 80% train, last 20% val) and report Seq+Tok
   AUROC for each split separately. If the held-out AUROC drops
   sharply, the auditor is overfitting.

2. **Trivial-feature baselines** — zero-train (or near-zero) discrimin-
   ators that ignore the trained EqM:
     a) Top-K simplex *entropy* per position (high entropy = LM
        unsure = OOD).
     b) Top-K simplex *peak gap* (top1 − top2 logit).
     c) GPT-2 hidden-state L2 norm per position.
     d) Linear logistic-regression probe on h_LLM (per-position).
   If any of these alone matches the trained auditor, the auditor's
   contribution is just absorbing pre-existing LM signal.

3. **Uncorrupted-position Tok AUROC** — measure Tok AUROC at
   *non-corrupted* positions in invalid sequences. The invalid context
   bleeds into nearby hidden states (the LM is autoregressive), so a
   little signal here is expected; lots of signal here means the
   discriminator isn't actually localising the corruption.

4. **Cross-seed corruption** — re-corrupt the held-out chunks with a
   *different* RNG seed and re-evaluate. AUROC should hold up; if it
   drops sharply, the auditor memorised the specific corruption draw.

Outputs a markdown report to ``runs/phaseF_sanity.md`` and writes
``runs/phaseF_sanity.json`` with the raw numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import aitchinson_flow.models  # noqa: E402,F401
from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.data.wiki import (  # noqa: E402
    WikiAuditorDataset, load_wiki_cache, span_corrupt, _topk_clr,
    _spilled_energy_per_pos,
)
from aitchinson_flow.models import build_model  # noqa: E402

from scripts.eval_full import _config_from_payload  # noqa: E402
from scripts.eval_auditor_wiki import (  # noqa: E402
    _grad_norm_per_pos, _signed_energy, _roc_auc,
)


def _run_auditor(
    model, ds, *, gamma_value: float, batch_size: int, needs_h: bool,
    indices: list[int],
) -> dict[str, torch.Tensor]:
    """Forward the auditor on a subset of dataset indices."""
    device = next(model.parameters()).device
    GN_clean: list[torch.Tensor] = []
    GN_invalid: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    SE_pos_clean: list[torch.Tensor] = []
    SE_pos_invalid: list[torch.Tensor] = []
    for s in range(0, len(indices), batch_size):
        e = min(s + batch_size, len(indices))
        items = [ds[i] for i in indices[s:e]]
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
        GN_clean.append(gn_c); GN_invalid.append(gn_i)
        SE_pos_clean.append(torch.stack([it["SE_pos_clean"] for it in items]))
        SE_pos_invalid.append(torch.stack([it["SE_pos_invalid"] for it in items]))
        masks.append(mask)
    return {
        "GN_clean":      torch.cat(GN_clean),
        "GN_invalid":    torch.cat(GN_invalid),
        "Mask":          torch.cat(masks),
        "SE_pos_clean":  torch.cat(SE_pos_clean),
        "SE_pos_invalid": torch.cat(SE_pos_invalid),
    }


def _seq_tok_aurocs(out: dict[str, torch.Tensor]) -> dict[str, float]:
    GN_c, GN_i = out["GN_clean"], out["GN_invalid"]
    Mask = out["Mask"]
    SE_c_pos, SE_i_pos = out["SE_pos_clean"], out["SE_pos_invalid"]
    seq_grad_sq_c = GN_c.pow(2).sum(dim=-1)
    seq_grad_sq_i = GN_i.pow(2).sum(dim=-1)
    SE_seq_c = SE_c_pos.sum(dim=-1)
    SE_seq_i = SE_i_pos.sum(dim=-1)
    return {
        "Seq_EqM_Egrad²": _roc_auc(seq_grad_sq_c, seq_grad_sq_i),
        "Seq_SE":         _roc_auc(SE_seq_c, SE_seq_i),
        "Tok_EqM_Upos":   _roc_auc(GN_c[Mask], GN_i[Mask]),
        "Tok_SE":         _roc_auc(SE_c_pos[Mask], SE_i_pos[Mask]),
    }


def _entropy_topk(clr: torch.Tensor) -> torch.Tensor:
    """Entropy of the top-K log-simplex per position. clr (..., K) → (...,)."""
    log_p = clr - torch.logsumexp(clr, dim=-1, keepdim=True)
    p = log_p.exp()
    return -(p * log_p).sum(dim=-1)


def _peak_gap_topk(clr: torch.Tensor) -> torch.Tensor:
    """top1 − top2 logit gap. clr (..., K) → (...,)."""
    sorted_vals, _ = clr.sort(dim=-1, descending=True)
    return sorted_vals[..., 0] - sorted_vals[..., 1]


def _h_norm(h: torch.Tensor) -> torch.Tensor:
    return h.norm(dim=-1)


def _logreg_probe_auroc(
    h_clean: torch.Tensor, h_invalid: torch.Tensor, mask: torch.Tensor,
    *, train_frac: float = 0.8, lam: float = 1e-2,
) -> dict[str, float]:
    """Train a simple per-position logistic-regression probe on h_LLM
    and report its Tok AUROC at corrupted positions.

    Inputs:
      h_clean: (n, L, H)  — clean hidden states (label = 0)
      h_invalid: (n, L, H) — invalid hidden states (label = 1, only at
                              corrupted positions)
      mask: (n, L) bool — True at corrupted positions
    """
    n, L, H = h_clean.shape
    n_train = int(train_frac * n)

    # Positives = invalid hiddens at corrupted positions. Negatives =
    # clean hiddens at the same positions.
    h_pos_train = h_invalid[:n_train][mask[:n_train]]
    h_neg_train = h_clean[:n_train][mask[:n_train]]
    h_pos_test = h_invalid[n_train:][mask[n_train:]]
    h_neg_test = h_clean[n_train:][mask[n_train:]]

    X_train = torch.cat([h_neg_train, h_pos_train])
    y_train = torch.cat([
        torch.zeros(h_neg_train.shape[0]),
        torch.ones(h_pos_train.shape[0]),
    ])
    X_test = torch.cat([h_neg_test, h_pos_test])
    y_test = torch.cat([
        torch.zeros(h_neg_test.shape[0]),
        torch.ones(h_pos_test.shape[0]),
    ])
    # Standardise.
    mu, sigma = X_train.mean(dim=0), X_train.std(dim=0).clamp(min=1e-6)
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma

    # Closed-form ridge logistic via single Newton step on the logits;
    # for a sanity-check probe, use linear-regression-on-y as a stand-in
    # (its decision function gives a valid AUROC).
    Xtx = X_train.T @ X_train + lam * torch.eye(X_train.shape[1])
    Xty = X_train.T @ y_train
    w = torch.linalg.solve(Xtx, Xty)
    pred_train = X_train @ w
    pred_test = X_test @ w
    return {
        "train_auroc": float(_roc_auc(pred_train[y_train == 0], pred_train[y_train == 1])),
        "test_auroc":  float(_roc_auc(pred_test[y_test == 0],   pred_test[y_test == 1])),
        "n_train_pos": int(h_pos_train.shape[0]),
        "n_test_pos":  int(h_pos_test.shape[0]),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cache", default="data/wiki_cache_gpt2.pt")
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out-md", default="runs/phaseF_sanity.md")
    p.add_argument("--out-json", default="runs/phaseF_sanity.json")
    args = p.parse_args(argv)

    print(f"[sanity] loading {args.ckpt} …")
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = _config_from_payload(payload)
    device = cfg.training.device
    gamma = float(cfg.eqm.auditor_gamma)
    ctx_mode = getattr(cfg.eqm, "context_features", "off")
    needs_h = ctx_mode != "off"

    model = build_model(cfg).to(device)
    model.load_state_dict(payload.get("model_state_dict", payload))
    model.eval()

    cache = load_wiki_cache(args.cache)
    ds = WikiAuditorDataset(cache, with_hidden=True)
    n = len(ds)
    n_train = int(args.train_frac * n)
    train_idx = list(range(n_train))
    val_idx = list(range(n_train, n))
    print(f"[sanity] n={n}, train_idx=[0..{n_train-1}], val_idx=[{n_train}..{n-1}]")

    # ---- (1) Train-only vs held-out auditor AUROC ----
    print("[sanity] (1) running auditor on train and val splits …")
    out_train = _run_auditor(
        model, ds, gamma_value=gamma, batch_size=args.batch_size,
        needs_h=needs_h, indices=train_idx,
    )
    out_val = _run_auditor(
        model, ds, gamma_value=gamma, batch_size=args.batch_size,
        needs_h=needs_h, indices=val_idx,
    )
    train_aucs = _seq_tok_aurocs(out_train)
    val_aucs = _seq_tok_aurocs(out_val)
    print(f"  train: {train_aucs}")
    print(f"  val:   {val_aucs}")

    # ---- (2) Trivial-feature baselines on val split ----
    print("[sanity] (2) trivial-feature baselines on val …")
    val_clr_clean    = cache["clean_clr"][n_train:]
    val_clr_invalid  = cache["invalid_clr"][n_train:]
    val_h_clean      = cache["clean_h"][n_train:]
    val_h_invalid    = cache["invalid_h"][n_train:]
    val_mask         = cache["mask_corrupt"][n_train:]
    val_se_c         = cache["clean_SE_pos"][n_train:]
    val_se_i         = cache["invalid_SE_pos"][n_train:]

    # Per-position trivial features.
    H_c = _entropy_topk(val_clr_clean)
    H_i = _entropy_topk(val_clr_invalid)
    PG_c = _peak_gap_topk(val_clr_clean)
    PG_i = _peak_gap_topk(val_clr_invalid)
    HN_c = _h_norm(val_h_clean)
    HN_i = _h_norm(val_h_invalid)
    trivial_seq = {
        "topk_entropy_seq":    _roc_auc(H_c.sum(dim=-1),  H_i.sum(dim=-1)),
        "topk_peak_gap_seq":   _roc_auc(PG_c.sum(dim=-1), PG_i.sum(dim=-1)),
        "h_norm_seq":          _roc_auc(HN_c.sum(dim=-1), HN_i.sum(dim=-1)),
    }
    trivial_tok = {
        "topk_entropy_tok":    _roc_auc(H_c[val_mask],  H_i[val_mask]),
        "topk_peak_gap_tok":   _roc_auc(PG_c[val_mask], PG_i[val_mask]),
        "h_norm_tok":          _roc_auc(HN_c[val_mask], HN_i[val_mask]),
        "SE_tok":              _roc_auc(val_se_c[val_mask], val_se_i[val_mask]),
    }
    print(f"  trivial seq: {trivial_seq}")
    print(f"  trivial tok: {trivial_tok}")

    # Linear-regression probe on full hidden state.
    h_clean_full = cache["clean_h"]
    h_invalid_full = cache["invalid_h"]
    mask_full = cache["mask_corrupt"]
    print("[sanity] training linear probe on h_LLM …")
    probe = _logreg_probe_auroc(
        h_clean_full, h_invalid_full, mask_full, train_frac=args.train_frac
    )
    print(f"  linear probe: {probe}")

    # ---- (3) Uncorrupted-position Tok AUROC on val split ----
    print("[sanity] (3) Tok AUROC at *uncorrupted* positions in invalid sequences …")
    not_mask = ~out_val["Mask"]
    if not_mask.any():
        unc_eqm = _roc_auc(out_val["GN_clean"][not_mask], out_val["GN_invalid"][not_mask])
        unc_se = _roc_auc(
            out_val["SE_pos_clean"][not_mask], out_val["SE_pos_invalid"][not_mask],
        )
    else:
        unc_eqm = unc_se = float("nan")
    print(f"  Tok AUROC at UNcorrupted positions: EqM={unc_eqm:.4f} SE={unc_se:.4f}")

    # ---- (4) Re-corrupt with new seed on val split ----
    print("[sanity] (4) re-corrupting val split with new seed …")
    val_clean_ids = cache["clean_ids"][n_train:]
    new_invalid_ids, new_mask = span_corrupt(
        val_clean_ids, vocab_size=int(cache["V"]), corrupt_rate=float(cache["corrupt_rate"]),
        seed=999_999,
    )
    # Re-run the LM forward on the new invalid_ids to produce features.
    print("    (loading GPT-2 to re-cache new-seed invalid features …)")
    from transformers import AutoModelForCausalLM
    lm = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32).to(device).eval()
    bs_lm = 8
    new_logits_chunks: list[torch.Tensor] = []
    new_h_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for s in range(0, new_invalid_ids.shape[0], bs_lm):
            e = min(s + bs_lm, new_invalid_ids.shape[0])
            ids_b = new_invalid_ids[s:e].to(device)
            out_lm = lm(ids_b, output_hidden_states=True)
            new_logits_chunks.append(out_lm.logits.cpu().float())
            new_h_chunks.append(out_lm.hidden_states[-1].cpu().float())
    del lm
    torch.cuda.empty_cache()
    new_logits = torch.cat(new_logits_chunks, dim=0)
    new_h = torch.cat(new_h_chunks, dim=0)
    new_clr, _ = _topk_clr(new_logits, int(cache["K"]))
    new_se = _spilled_energy_per_pos(new_logits, new_invalid_ids)

    # Build a temporary dataset slice with the new-seed invalid features.
    K = cfg.text8_dataset.K
    # Forward the auditor on the new pairs.
    val_clean_clr = cache["clean_clr"][n_train:]
    val_clean_h = cache["clean_h"][n_train:]
    val_clean_se = cache["clean_SE_pos"][n_train:]

    GN_clean_new: list[torch.Tensor] = []
    GN_invalid_new: list[torch.Tensor] = []
    masks_new: list[torch.Tensor] = []
    SE_pos_clean_new: list[torch.Tensor] = []
    SE_pos_invalid_new: list[torch.Tensor] = []
    n_val = val_clean_clr.shape[0]
    for s in range(0, n_val, args.batch_size):
        e = min(s + args.batch_size, n_val)
        x_c = val_clean_clr[s:e].to(device).float()
        x_i = new_clr[s:e].to(device).float()
        if needs_h:
            h_c = val_clean_h[s:e].to(device).float()
            h_i = new_h[s:e].to(device).float()
        else:
            h_c = h_i = None
        gn_c = _grad_norm_per_pos(model, x_c, gamma, h_c).cpu()
        gn_i = _grad_norm_per_pos(model, x_i, gamma, h_i).cpu()
        GN_clean_new.append(gn_c)
        GN_invalid_new.append(gn_i)
        masks_new.append(new_mask[s:e])
        SE_pos_clean_new.append(val_clean_se[s:e])
        SE_pos_invalid_new.append(new_se[s:e])
    out_new = {
        "GN_clean":       torch.cat(GN_clean_new),
        "GN_invalid":     torch.cat(GN_invalid_new),
        "Mask":           torch.cat(masks_new),
        "SE_pos_clean":   torch.cat(SE_pos_clean_new),
        "SE_pos_invalid": torch.cat(SE_pos_invalid_new),
    }
    new_aucs = _seq_tok_aurocs(out_new)
    print(f"  cross-seed val: {new_aucs}")

    # Mask overlap with original
    mask_overlap = (cache["mask_corrupt"][n_train:] & new_mask).sum().item()
    mask_orig_only = (cache["mask_corrupt"][n_train:] & ~new_mask).sum().item()
    mask_new_only = (~cache["mask_corrupt"][n_train:] & new_mask).sum().item()
    print(f"  mask overlap (same position corrupted both seeds): {mask_overlap}")
    print(f"  mask in original only: {mask_orig_only},  new only: {mask_new_only}")

    # Aggregate report.
    report: dict[str, Any] = {
        "ckpt": args.ckpt,
        "n": int(n),
        "n_train": int(n_train),
        "ctx_mode": ctx_mode,
        "1_train_vs_val": {"train": train_aucs, "val": val_aucs},
        "2_trivial_baselines_val": {
            "seq": trivial_seq, "tok": trivial_tok, "linear_probe_h": probe,
        },
        "3_uncorrupted_pos_val": {
            "EqM_U_pos": float(unc_eqm), "SE_pos": float(unc_se),
            "n_uncorrupt": int(not_mask.sum().item()),
        },
        "4_cross_seed_val": new_aucs,
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2))

    # Markdown writeup.
    md = []
    md.append(f"# Phase F sanity checks — {Path(args.ckpt).parent.name}")
    md.append("")
    md.append(f"- ckpt: `{args.ckpt}`")
    md.append(f"- n_total={n}, n_train={n_train}, n_val={n-n_train}, ctx_mode={ctx_mode}")
    md.append("")
    md.append("## 1. Train vs held-out AUROC")
    md.append("")
    md.append("|   stat   |  train  |   val   |  gap  |")
    md.append("|----------|--------:|--------:|------:|")
    for k in train_aucs:
        t, v = train_aucs[k], val_aucs[k]
        md.append(f"| {k:<22s} | {t:.4f} | {v:.4f} | {t-v:+.4f} |")
    md.append("")
    md.append(
        f"**Verdict:** train and val AUROCs are very close — the auditor "
        f"is not memorising the train set."
        if max(abs(train_aucs[k] - val_aucs[k]) for k in train_aucs) < 0.02
        else
        f"**Verdict:** there is a noticeable train-vs-val gap; the AUROC "
        f"on the full cache is inflated by training-set leakage."
    )
    md.append("")
    md.append("## 2. Trivial-feature baselines on val split")
    md.append("")
    md.append("| Feature | Tok AUROC | Note |")
    md.append("|---|---:|---|")
    for k, v in trivial_tok.items():
        md.append(f"| {k} | {v:.4f} | per-position |")
    md.append(f"| linear probe on h_LLM (test) | {probe['test_auroc']:.4f} | "
              f"trained on first {n_train} chunks |")
    md.append(f"| linear probe on h_LLM (train) | {probe['train_auroc']:.4f} | "
              f"sanity — should be ≥ test |")
    md.append("")
    md.append("|  Seq feature  | AUROC |")
    md.append("|---|---:|")
    for k, v in trivial_seq.items():
        md.append(f"| {k} | {v:.4f} |")
    md.append("")
    md.append(
        "**Verdict:** the trained EqM auditor's contribution is the gap "
        "between its AUROC and the *best* trivial baseline. If the gap "
        "is small the headline number is mostly recovering pre-existing "
        "LM signal."
    )
    md.append("")
    md.append("## 3. Uncorrupted-position Tok AUROC (val split)")
    md.append("")
    md.append(f"- EqM `U_pos` at **un**corrupted positions: {unc_eqm:.4f}")
    md.append(f"- SE     at **un**corrupted positions: {unc_se:.4f}")
    md.append(f"- n positions: {int(not_mask.sum().item())}")
    md.append("")
    md.append(
        "Both should be ≈ 0.5 if the discriminator localises corruption. "
        "Significant deviation means the *invalid context* (cascade from "
        "the AR LM) bleeds into nearby hidden states and the discriminator "
        "picks that up indirectly."
    )
    md.append("")
    md.append("## 4. Cross-seed corruption (val split)")
    md.append("")
    md.append("|   stat   | original-seed | new-seed | Δ |")
    md.append("|---|---:|---:|---:|")
    for k in val_aucs:
        a, b = val_aucs[k], new_aucs[k]
        md.append(f"| {k:<22s} | {a:.4f} | {b:.4f} | {b-a:+.4f} |")
    md.append("")
    md.append(
        f"Mask overlap with original seed: "
        f"both={mask_overlap}, orig only={mask_orig_only}, new only={mask_new_only}. "
        f"AUROC should hold up across seeds; a sharp drop means the auditor "
        f"memorised the specific corruption draw."
    )
    md.append("")
    Path(args.out_md).write_text("\n".join(md))
    print(f"[sanity] wrote {args.out_md} and {args.out_json}")


if __name__ == "__main__":
    main()
