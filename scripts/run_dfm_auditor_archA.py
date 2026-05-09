"""Architecture A — supervised SVGP on the frozen DFM denoiser's encoder.

Sanity check before the dual-head retrain: take a trained DFM auditor
checkpoint, extract per-position encoder features
``z[r, k] = encoder(x_LM[r, k], t, h_LLM[r, k])``, fit a supervised SVGP
(RBF kernel, M inducing points) on the per-position features with the
row's hallucination label propagated to every answer-mask token, then
report:

* Row-level AUROC (mean-pool of per-token sigmoid → label).
* Per-token AUROC at answer-mask positions (locality).
* Per-token AUROC at non-answer-mask positions (cascade contamination
  check; should be ≈ 0.5 on this corpus).
* ECE on the SVGP predictive probability.

Decision rule: if row-level AUROC ≥ 0.92 the encoder carries the
supervised signal and Architecture B (joint dual-head retrain) is
worth doing. Below ~0.85 the encoder doesn't have it and only the
retrain unlocks the supervised ceiling.

Usage::

    python scripts/run_dfm_auditor_archA.py \\
        --ckpt runs/dfm_auditor_d512_hidden/best.pt \\
        --out  runs/dfm_auditor_archA_hidden \\
        --inducing 128 --iters 400
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

import torch  # noqa: E402

import aitchinson_flow.models  # noqa: E402,F401
from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.data.hallueval_dfm import HalluevalDFMDataModule  # noqa: E402
from aitchinson_flow.models.factory import build_model  # noqa: E402


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    s = scores.detach().cpu().double()
    y = labels.detach().cpu().long()
    if (y.sum() == 0) or (y.sum() == len(y)):
        return float("nan")
    order = torch.argsort(s, descending=True)
    ys = y[order]
    npos = int(y.sum())
    nneg = int(len(y) - npos)
    cum = torch.cumsum((ys == 1).double(), 0)
    auc = float(((cum * (ys == 0).double()).sum() / (npos * nneg)).item())
    return max(auc, 1.0 - auc)


def expected_calibration_error(p: torch.Tensor, y: torch.Tensor, n_bins: int = 15) -> float:
    p = p.detach().cpu().double().clamp(0.0, 1.0)
    y = y.detach().cpu().double()
    if len(p) == 0:
        return float("nan")
    bins = torch.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(p)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        if not in_bin.any():
            continue
        bin_acc = float(y[in_bin].mean())
        bin_conf = float(p[in_bin].mean())
        ece += (in_bin.sum().item() / n) * abs(bin_acc - bin_conf)
    return float(ece)


@torch.no_grad()
def extract_features(
    model,
    loader,
    *,
    device,
    energy_t: float,
):
    """Return per-position encoder features over answer-mask positions.

    Stacks (z, label, is_answer, is_train) for every position. The caller
    splits/standardises afterwards.

    Returns (z (N_pos, d), label (N_pos,), row_id (N_pos,), is_answer (N_pos,)).
    """
    z_all, label_all, row_all, ans_all = [], [], [], []
    row_offset = 0
    for batch in loader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        topk_logp = batch["topk_logp"]
        x = topk_logp.exp()
        x = x / x.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        h = batch["hidden"].float()
        B, L, _ = x.shape
        t = torch.full((B,), float(energy_t), device=device, dtype=x.dtype)
        z = model.forward_features(x, t, h_ctx=h)  # (B, L, d_model)
        d = z.shape[-1]

        ans = batch["answer_mask"].bool()
        # Stack everything per-position; we'll filter by ans downstream.
        z_all.append(z.reshape(-1, d).cpu())
        label_b = batch["label"].long().unsqueeze(-1).expand(B, L).reshape(-1).cpu()
        label_all.append(label_b)
        row_id = (
            torch.arange(B, device=device).unsqueeze(-1).expand(B, L).reshape(-1).cpu()
            + row_offset
        )
        row_all.append(row_id)
        ans_all.append(ans.reshape(-1).cpu())
        row_offset += B

    z = torch.cat(z_all, dim=0)
    label = torch.cat(label_all, dim=0)
    row = torch.cat(row_all, dim=0)
    ans = torch.cat(ans_all, dim=0)
    return z, label, row, ans


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--energy-t", type=float, default=4.0)
    parser.add_argument("--inducing", type=int, default=128)
    parser.add_argument("--iters", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-rows", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg: Config = ckpt["cfg"]
    K = ckpt.get("K", 32)
    L = ckpt.get("L", 160)

    cfg.hallueval_dfm_auditor = replace(
        cfg.hallueval_dfm_auditor,
        enabled=True,
        batch_size=args.batch_size,
        max_rows=args.max_rows,
    )
    dm = HalluevalDFMDataModule(cfg.hallueval_dfm_auditor)
    print(f"[archA] data: {dm.meta}")

    model = build_model(cfg)
    model.set_dims(K=K, L=L)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    print("[archA] extracting train features ...")
    z_tr, y_tr, row_tr, ans_tr = extract_features(
        model, dm.train_dataloader(), device=device, energy_t=args.energy_t
    )
    print("[archA] extracting val features ...")
    z_va, y_va, row_va, ans_va = extract_features(
        model, dm.val_dataloader(), device=device, energy_t=args.energy_t
    )

    d = z_tr.shape[-1]
    print(
        f"[archA] z_tr: {tuple(z_tr.shape)} (d={d})  "
        f"answer-mask train pos: {int(ans_tr.sum())}  "
        f"val pos: {int(ans_va.sum())}"
    )

    # SVGP train: only on answer-mask positions, separated into pos/neg by row label
    Xs_tr_ans = z_tr[ans_tr]
    ys_tr_ans = y_tr[ans_tr]
    X_pos = Xs_tr_ans[ys_tr_ans == 1]
    X_neg = Xs_tr_ans[ys_tr_ans == 0]
    print(
        f"[archA] SVGP training: {X_pos.shape[0]} hallucinated pos, "
        f"{X_neg.shape[0]} clean pos"
    )

    sys.path.insert(0, str(REPO / "scripts"))
    from phaseF_uq import train_svgp, predict_svgp  # type: ignore

    svgp, lik, mu, sigma = train_svgp(
        X_pos.float(),
        X_neg.float(),
        n_inducing=args.inducing,
        n_iters=args.iters,
        lr=args.lr,
        device=device,
        seed=args.seed,
    )

    # Predictions on val (all positions), then split by ans / non-ans.
    print("[archA] SVGP predictions on val ...")
    p_va, mu_va, std_va = predict_svgp(svgp, lik, mu, sigma, z_va.float())

    # Per-token AUROC on answer span (the meaningful locality measure)
    auc_tok_ans = auroc(p_va[ans_va], y_va[ans_va])
    auc_tok_non = auroc(p_va[~ans_va], y_va[~ans_va])
    auc_tok_std_ans = auroc(std_va[ans_va], y_va[ans_va])

    # Row-level: average per-token p over answer-mask positions per row.
    # row_va is the global row index; we need per-row mean of p_va[ans_va].
    rows_uniq = torch.unique(row_va)
    row_p = torch.zeros(rows_uniq.shape[0])
    row_y = torch.zeros(rows_uniq.shape[0], dtype=torch.long)
    for ri, r in enumerate(rows_uniq.tolist()):
        m = (row_va == r) & ans_va
        if m.any():
            row_p[ri] = p_va[m].mean()
            row_y[ri] = y_va[m][0]
        else:
            # No answer-mask positions — use mean over all positions as fallback
            mall = row_va == r
            row_p[ri] = p_va[mall].mean()
            row_y[ri] = y_va[mall][0]
    auc_row = auroc(row_p, row_y)
    ece_row = expected_calibration_error(row_p, row_y)

    # ECE per-token at answer span
    ece_tok_ans = expected_calibration_error(p_va[ans_va], y_va[ans_va])

    summary = {
        "ckpt": str(args.ckpt),
        "energy_t": args.energy_t,
        "n_inducing": args.inducing,
        "iters": args.iters,
        "n_train_pos": int(X_pos.shape[0]),
        "n_train_neg": int(X_neg.shape[0]),
        "auc_token_answer_span": auc_tok_ans,
        "auc_token_non_answer": auc_tok_non,  # cascade contamination check
        "auc_token_std_answer": auc_tok_std_ans,  # SVGP variance as score
        "auc_row_pool_mean": auc_row,
        "ece_token_answer_span": ece_tok_ans,
        "ece_row_pool_mean": ece_row,
    }
    with open(out_dir / "archA_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
