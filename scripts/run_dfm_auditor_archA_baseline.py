"""Architecture A baseline: SVGP directly on raw h_LLM (no DFM encoder).

Sanity check that our SVGP harness reaches the Phase K ceiling on the
exact same train/val split. If this hits row AUROC ≥ 0.99 then the gap
between Architecture A (≈0.84) and the supervised ceiling is genuinely
in the DFM encoder, not in our SVGP code or data split.
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

from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.data.hallueval_dfm import HalluevalDFMDataModule  # noqa: E402

sys.path.insert(0, str(REPO / "scripts"))
from phaseF_uq import train_svgp, predict_svgp  # type: ignore # noqa: E402
from run_dfm_auditor_archA import auroc, expected_calibration_error  # type: ignore # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/dfm_auditor_archA_baseline_hLLM")
    p.add_argument("--inducing", type=int, default=128)
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--max-rows", type=int, default=4000)
    args = p.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = Config()
    cfg.hallueval_dfm_auditor = replace(
        cfg.hallueval_dfm_auditor, enabled=True, batch_size=16, max_rows=args.max_rows
    )
    dm = HalluevalDFMDataModule(cfg.hallueval_dfm_auditor)

    def stack(loader):
        h, y, row, ans = [], [], [], []
        off = 0
        for b in loader:
            B, L, _ = b["hidden"].shape
            h.append(b["hidden"].float().reshape(-1, b["hidden"].shape[-1]))
            y.append(b["label"].long().unsqueeze(-1).expand(B, L).reshape(-1))
            row.append(
                torch.arange(B).unsqueeze(-1).expand(B, L).reshape(-1) + off
            )
            ans.append(b["answer_mask"].bool().reshape(-1))
            off += B
        return (
            torch.cat(h),
            torch.cat(y),
            torch.cat(row),
            torch.cat(ans),
        )

    print("[baseline] stacking h_LLM features ...")
    h_tr, y_tr, _, ans_tr = stack(dm.train_dataloader())
    h_va, y_va, row_va, ans_va = stack(dm.val_dataloader())

    H_pos = h_tr[ans_tr][y_tr[ans_tr] == 1]
    H_neg = h_tr[ans_tr][y_tr[ans_tr] == 0]
    print(f"[baseline] per-position training: {H_pos.shape[0]} pos, {H_neg.shape[0]} neg, dim={h_tr.shape[-1]}")

    # Row-pooled variant: mean h_LLM over answer-span tokens → one feature per row.
    # This matches the Phase K setup (meanpool over answer span) and is the
    # apples-to-apples comparison for the "supervised ceiling" claim.
    def row_pool(h, ans, label):
        # h, ans: (N_pos, *). Group by virtual row index = floor index / L.
        # We don't have row IDs here, but train_loader was built from a Subset
        # so rows are contiguous L-sized blocks.
        Lloc = 160
        N_pos = h.shape[0]
        n_rows = N_pos // Lloc
        h = h[: n_rows * Lloc].reshape(n_rows, Lloc, -1)
        ans = ans[: n_rows * Lloc].reshape(n_rows, Lloc)
        label = label[: n_rows * Lloc].reshape(n_rows, Lloc)
        denom = ans.sum(-1).clamp_min(1).float().unsqueeze(-1)
        pooled = (h * ans.unsqueeze(-1)).sum(1) / denom
        # row label: take label of any answer position (they all match)
        row_labels = label[:, 0]
        return pooled, row_labels

    h_tr_pooled, y_tr_pooled = row_pool(h_tr, ans_tr, y_tr)
    h_va_pooled, y_va_pooled = row_pool(h_va, ans_va, y_va)
    print(
        f"[baseline] row-pooled training: {int((y_tr_pooled==1).sum())} pos, "
        f"{int((y_tr_pooled==0).sum())} neg"
    )
    svgp_p, lik_p, mu_p, sigma_p = train_svgp(
        h_tr_pooled[y_tr_pooled == 1],
        h_tr_pooled[y_tr_pooled == 0],
        n_inducing=args.inducing,
        n_iters=args.iters,
        lr=1e-2,
        device=device,
        seed=0,
    )
    p_pooled, _, _ = predict_svgp(svgp_p, lik_p, mu_p, sigma_p, h_va_pooled)
    auc_row_pooled_first = auroc(p_pooled, y_va_pooled)
    ece_row_pooled_first = expected_calibration_error(p_pooled, y_va_pooled)
    print(f"[baseline] row-pooled SVGP row AUROC = {auc_row_pooled_first:.4f}")

    svgp, lik, mu, sigma = train_svgp(
        H_pos, H_neg, n_inducing=args.inducing, n_iters=args.iters, lr=1e-2,
        device=device, seed=0,
    )
    print("[baseline] predict val ...")
    p_va, _, _ = predict_svgp(svgp, lik, mu, sigma, h_va)

    auc_tok_ans = auroc(p_va[ans_va], y_va[ans_va])
    auc_tok_non = auroc(p_va[~ans_va], y_va[~ans_va])
    rows = torch.unique(row_va)
    rp = torch.zeros(rows.shape[0]); ry = torch.zeros(rows.shape[0], dtype=torch.long)
    for i, r in enumerate(rows.tolist()):
        m = (row_va == r) & ans_va
        if m.any():
            rp[i] = p_va[m].mean(); ry[i] = y_va[m][0]
    auc_row = auroc(rp, ry)
    summary = {
        "feature": "raw_h_LLM (768-d)",
        "n_inducing": args.inducing,
        "iters": args.iters,
        "per_position_then_pool": {
            "auc_token_answer_span": auc_tok_ans,
            "auc_token_non_answer": auc_tok_non,
            "auc_row_pool_mean": auc_row,
            "ece_token_answer_span": expected_calibration_error(p_va[ans_va], y_va[ans_va]),
        },
        "pool_first_then_svgp": {
            "auc_row": auc_row_pooled_first,
            "ece_row": ece_row_pooled_first,
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
