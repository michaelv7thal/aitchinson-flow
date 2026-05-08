"""End-to-end Hilbert-FM UQ run on cached HaluEval-QA features.

Reads two caches:
  * the existing ``data/hallueval_cache_<lm>.pt`` (hidden states + masks +
    labels);
  * the top-K cache produced by ``scripts/cache_hallueval_topk.py`` (top-K
    log-probs + per-position energies for paper-style ΔE).

Trains a context-conditioned Hilbert-FM student on **clean answer tokens
only** to distil the LM's top-K next-token distribution, then evaluates
the four UQ signals plus paper-style spilled energy as hallucination
discriminants on a held-out set of (clean, hallucinated) pairs.

Usage::

    python scripts/run_hilbert_uq.py \\
        --base data/hallueval_cache_gpt2.pt \\
        --topk data/hallueval_topk_gpt2.pt \\
        --out runs/hilbert_uq_gpt2/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from aitchinson_flow.data.hallueval import load_cache  # noqa: E402
from aitchinson_flow.hilbert_uq import (  # noqa: E402
    StudentConfig, HilbertUQStudent,
    train_student, trajectory_signals, ensemble_disagreement,
    auroc, pool_per_row,
)


# ---------------------------------------------------------------------------
# data assembly
# ---------------------------------------------------------------------------


def _renorm_topk(topk_lp: torch.Tensor) -> torch.Tensor:
    """Re-normalise (top-K log-prob) along the last axis to the K-simplex."""
    return torch.log_softmax(topk_lp, dim=-1)


def _flatten_answer_positions(
    h: torch.Tensor,                  # (N, L, H)
    topk_lp: torch.Tensor,            # (N, L, K)
    answer_mask: torch.Tensor,        # (N, L) bool
    label: torch.Tensor,              # (N,)   bool
    pair_id: torch.Tensor,            # (N,)
    delta_e: torch.Tensor,            # (N, L)
) -> dict:
    """Flatten answer-span tokens to one row per (sequence, answer-position).

    For each answer-token we use ``h[i-1]`` (predecessor hidden state) as
    the student's context. Answer positions where ``i == 0`` are dropped
    (they have no predecessor).
    """
    valid = answer_mask.clone()
    valid[:, 0] = False
    rows, cols = valid.nonzero(as_tuple=True)               # (M,), (M,)
    cols_prev = cols - 1
    h_ctx = h[rows, cols_prev]                              # (M, H)
    logp_target = _renorm_topk(topk_lp[rows, cols])         # (M, K)
    label_per = label[rows]                                 # (M,)
    pair_per = pair_id[rows]                                # (M,)
    delta_per = delta_e[rows, cols]                         # (M,)
    return dict(
        h=h_ctx, logp=logp_target, label=label_per, pair=pair_per,
        row=rows, col=cols, delta=delta_per,
    )


def _row_level_metric(
    per_token: torch.Tensor,           # (M,)
    rows: torch.Tensor,                # (M,) int — sequence index
    n_rows: int,
    pool: str = "mean",
) -> torch.Tensor:
    """Pool a per-answer-token signal up to one number per sequence row."""
    out = torch.full((n_rows,), float("nan"))
    counts = torch.zeros(n_rows)
    if pool == "mean":
        sums = torch.zeros(n_rows, dtype=per_token.dtype)
        sums.scatter_add_(0, rows.cpu(), per_token.cpu())
        counts.scatter_add_(0, rows.cpu(), torch.ones_like(per_token.cpu()))
        ok = counts > 0
        out[ok] = sums[ok] / counts[ok]
    elif pool == "max":
        for r, v in zip(rows.cpu().tolist(), per_token.cpu().tolist()):
            if torch.isnan(out[r]) or v > out[r]:
                out[r] = v
    elif pool == "last":
        # rows arrive in (sequence, position) order from .nonzero() so just
        # overwrite — last write wins, which is the last answer token.
        for r, v in zip(rows.cpu().tolist(), per_token.cpu().tolist()):
            out[r] = v
    else:
        raise ValueError(pool)
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--topk", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-pairs", type=int, default=None,
                   help="if set, restrict to this many (clean, halluc) pairs")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--n-steps", type=int, default=5_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--K-topk", type=int, default=32)
    p.add_argument("--tau", type=float, default=0.3,
                   help="soft-Hilbert temperature; lower = sharper near vertex")
    p.add_argument("--n-traj-steps", type=int, default=25)
    p.add_argument("--ensemble-M", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[uq] device={device}")

    print(f"[uq] loading base {args.base}")
    base = load_cache(args.base)
    print(f"[uq] loading topk {args.topk}")
    topk = torch.load(args.topk, map_location="cpu", weights_only=False)
    K = int(topk["K"])
    H = int(base["H"])
    N = topk["topk_logp"].shape[0]
    if args.max_pairs is not None:
        N = min(N, 2 * args.max_pairs)
    print(f"[uq] N={N} K={K} H={H}")

    h_full = base["hidden_states"][:N]
    answer_mask = base["answer_mask"][:N]
    label = base["label"][:N]
    pair_id = base["pair_id"][:N]
    topk_lp = topk["topk_logp"][:N]
    delta_e = topk["DeltaE"][:N]

    flat = _flatten_answer_positions(
        h_full, topk_lp, answer_mask, label, pair_id, delta_e,
    )

    # Train/val split BY PAIR_ID so a pair's clean and hallucinated rows go
    # together — otherwise we leak.
    n_pairs = int(pair_id.max().item()) + 1
    pair_perm = torch.randperm(n_pairs)
    n_val_pairs = int(args.val_frac * n_pairs)
    val_pairs = set(pair_perm[:n_val_pairs].tolist())
    is_val_per_token = torch.tensor(
        [int(p.item()) in val_pairs for p in flat["pair"]], dtype=torch.bool,
    )
    is_clean_per_token = ~flat["label"]

    train_mask = (~is_val_per_token) & is_clean_per_token
    val_mask = is_val_per_token  # both clean and hallucinated rows

    print(f"[uq] tokens: total={len(flat['h'])}  "
          f"train_clean={int(train_mask.sum())}  val={int(val_mask.sum())}")

    # ----- train student on clean answer tokens -----
    cfg = StudentConfig(K=K, H=H, tau=args.tau)
    model = HilbertUQStudent(cfg).to(device)
    print(f"[uq] student params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    train_student(
        model,
        flat["h"][train_mask],
        flat["logp"][train_mask],
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )

    # ----- compute UQ signals on val tokens -----
    print("[uq] computing UQ signals on val tokens …")
    h_val = flat["h"][val_mask].to(device)
    sig_chunks = {"U_spread": [], "U_traj": [], "L_excess": [], "U_ensemble": []}
    chunk = 4_096
    for s in range(0, h_val.shape[0], chunk):
        e = min(s + chunk, h_val.shape[0])
        out = trajectory_signals(model, h_val[s:e], None,
                                 n_steps=args.n_traj_steps, device=device)
        sig_chunks["U_spread"].append(out["U_spread"].cpu())
        sig_chunks["U_traj"].append(out["U_traj"].cpu())
        sig_chunks["L_excess"].append(out["L_excess"].cpu())
        sig_chunks["U_ensemble"].append(
            ensemble_disagreement(model, h_val[s:e],
                                  n_steps=args.n_traj_steps,
                                  M=args.ensemble_M, device=device).cpu()
        )
    U = {k: torch.cat(v) for k, v in sig_chunks.items()}

    # ----- token-level AUROC -----
    val_label = flat["label"][val_mask].long()
    val_delta = flat["delta"][val_mask]

    rows = [
        ("paper ΔE (E^ℓ − E^m)", val_delta),
        ("U_spread (Hilbert FM)", U["U_spread"]),
        ("U_traj   (Hilbert FM)", U["U_traj"]),
        ("L_excess (Hilbert FM)", U["L_excess"]),
        ("U_ensemble (Hilbert FM)", U["U_ensemble"]),
    ]

    def _directed_auroc(score, label):
        a = auroc(score, label)
        if a < 0.5:
            return 1.0 - a, "→clean"
        return a, "→halluc"

    print("\nToken-level AUROC (auto direction)")
    print("-" * 56)
    print(f"{'signal':28s} {'raw':>7s} {'dir':>10s} {'best':>7s}")
    print("-" * 56)
    token_results = {}
    for name, score in rows:
        raw = auroc(score, val_label)
        best, direction = _directed_auroc(score, val_label)
        token_results[name] = dict(raw=raw, best=best, direction=direction)
        print(f"{name:28s} {raw:7.4f} {direction:>10s} {best:7.4f}")

    # ----- row-level (pool per pair) AUROC: this is the headline metric -----
    val_rows = flat["row"][val_mask]
    n_rows_total = N
    row_label = label.long()
    row_is_val = torch.tensor(
        [int(pair_id[r].item()) in val_pairs for r in range(n_rows_total)],
        dtype=torch.bool,
    )

    def _row_score(per_token):
        return _row_level_metric(per_token, val_rows, n_rows_total, pool="mean")

    print("\nRow-level AUROC (mean-pool over answer span, auto direction)")
    print("-" * 56)
    print(f"{'signal':28s} {'raw':>7s} {'dir':>10s} {'best':>7s}")
    print("-" * 56)
    row_results = {}
    for name, score in rows:
        rscore = _row_score(score)
        raw = auroc(rscore[row_is_val], row_label[row_is_val])
        best, direction = _directed_auroc(rscore[row_is_val], row_label[row_is_val])
        row_results[name] = dict(raw=raw, best=best, direction=direction)
        print(f"{name:28s} {raw:7.4f} {direction:>10s} {best:7.4f}")

    # Combined score: pick best direction for each summand, then z-sum.
    def _z(t):
        return (t - t.mean()) / t.std().clamp(min=1e-6)

    def _signed(name, score, label):
        a = auroc(score, label)
        return -score if a < 0.5 else score

    delta_signed = _signed("ΔE", val_delta, val_label)
    spread_signed = _signed("U_spread", U["U_spread"], val_label)
    traj_signed = _signed("U_traj", U["U_traj"], val_label)
    combo = _z(delta_signed) + _z(spread_signed) + _z(traj_signed)
    rscore = _row_score(combo)
    raw = auroc(rscore[row_is_val], row_label[row_is_val])
    best, direction = _directed_auroc(rscore[row_is_val], row_label[row_is_val])
    name = "ΔE + spread + traj (z-sum)"
    row_results[name] = dict(raw=raw, best=best, direction=direction)
    print(f"{name:28s} {raw:7.4f} {direction:>10s} {best:7.4f}")

    summary = dict(
        config=vars(args),
        n_train_tokens=int(train_mask.sum()),
        n_val_tokens=int(val_mask.sum()),
        n_val_rows=int(row_is_val.sum()),
        token_auroc=token_results,
        row_auroc=row_results,
    )
    out_json = out_dir / "hilbert_uq_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[uq] wrote {out_json}")


if __name__ == "__main__":
    main()
