"""Plot top-K softmax probabilities per token for one valid and one invalid sequence.

Loads one char-level window from text8, tokenizes it with the frozen LLM, runs
a forward pass to obtain top-K softmax probabilities at every token position,
and plots them as a heatmap side-by-side (valid vs. corrupted).

A secondary plot shows the sorted top-K distribution for a single chosen token
position to make the peaked-vs-flat contrast explicit.

Usage:
    python scripts/plot_topk_probs.py --out-dir plots/topk_probs_inspect

Optional flags (all have sensible defaults matching the datamodule):
    --lm-key        LLM registry key (default: hf_causal)
    --top-k         K vocabulary slots to retain (default: 64)
    --seq-length    LLM token length L (default: 30)
    --char-window   Char window length (default: 128)
    --corrupt-rate  Fraction of tokens corrupted for the invalid copy (default: 0.15)
    --window-idx    Which text8 window to use (default: 0)
    --token-pos     Token position to inspect in the per-position bar chart (default: 0)
    --device        cpu / cuda (default: cpu)
    --seed          RNG seed for corruption (default: 42)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from _shared.bootstrap import bootstrap_repo_paths

bootstrap_repo_paths(Path(__file__))

from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.data.corruption import corrupt_token_ids  # noqa: E402
from aitchinson_flow.data.llm_embedding_datamodule import (  # noqa: E402
    _batch_encode,
    _char_ids_to_text,
)
from aitchinson_flow.data.text8_datamodule import _load_text8_splits_cfg  # noqa: E402
from aitchinson_flow.data.transforms.discrete import (  # noqa: E402
    project_log_to_simplex_features,
)
from aitchinson_flow.llms.registry import build_lm  # noqa: E402


# ---------------------------------------------------------------------------
# Core: extract top-K probs from frozen LLM
# ---------------------------------------------------------------------------


def _get_topk_probs(
    lm,
    input_ids: torch.Tensor,
    K: int,
    *,
    renormalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (top_probs, top_indices), both shape (L, K), for a single sequence.

    top_probs are renormalized within the K slots when renormalize=True.
    """
    assert input_ids.dim() == 2 and input_ids.shape[0] == 1, "expects (1, L)"
    with torch.no_grad():
        logits = lm.forward_logits(input_ids=input_ids)  # (1, L, V)
    probs_full = torch.softmax(logits[0].float(), dim=-1)  # (L, V)
    top_probs, top_idx = torch.topk(probs_full, K, dim=-1, sorted=True)  # (L, K)
    if renormalize:
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)
    return top_probs.cpu(), top_idx.cpu()


def _to_ilr_features(top_probs: torch.Tensor, *, eps: float = 1e-12) -> np.ndarray:
    """Convert top-K simplex probabilities ``(L, K)`` to ILR features ``(L, K-1)``.

    ILR requires valid simplex rows; we explicitly re-normalize per row to keep
    this transform stable regardless of the plotting-mode ``--no-renormalize`` flag.
    """
    probs = top_probs.float()
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    log_probs = probs.clamp_min(eps).log()
    ilr = project_log_to_simplex_features(log_probs, mode="ilr")
    return ilr.numpy()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _heatmap(
    ax: plt.Axes,
    probs: np.ndarray,
    title: str,
    token_strs: list[str],
) -> None:
    """probs: (L, K) ndarray — rows=token positions, cols=top-K rank."""
    im = ax.imshow(probs, aspect="auto", origin="upper", cmap="plasma", vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Top-K rank", fontsize=8)
    ax.set_ylabel("Token position", fontsize=8)
    ax.set_yticks(range(len(token_strs)))
    ax.set_yticklabels(
        [f"{i}: {t}" for i, t in enumerate(token_strs)],
        fontsize=6,
    )
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=8))
    ax.tick_params(axis="x", labelsize=7)
    return im


def _feature_heatmap(
    ax: plt.Axes,
    feat: np.ndarray,
    title: str,
    token_strs: list[str],
) -> None:
    """Feature heatmap for ILR coordinates with symmetric color scaling."""
    vmax = float(np.max(np.abs(feat)))
    vmax = max(vmax, 1e-6)
    im = ax.imshow(
        feat,
        aspect="auto",
        origin="upper",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("ILR dimension", fontsize=8)
    ax.set_ylabel("Token position", fontsize=8)
    ax.set_yticks(range(len(token_strs)))
    ax.set_yticklabels([f"{i}: {t}" for i, t in enumerate(token_strs)], fontsize=6)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=8))
    ax.tick_params(axis="x", labelsize=7)
    return im


def _bar_chart(
    ax: plt.Axes,
    valid_probs: np.ndarray,
    invalid_probs: np.ndarray,
    pos: int,
    token_str: str,
    *,
    valid_vocab_idx: np.ndarray | None = None,
    invalid_vocab_idx: np.ndarray | None = None,
    tokenizer=None,
) -> None:
    """Side-by-side bar chart of sorted top-K probs at position ``pos``.

    When ``tokenizer`` and ``*_vocab_idx`` are supplied, x-tick labels show
    ``rank\ntoken_text`` so you can see which vocabulary entry each bar is.
    Valid and invalid may have different top-K tokens, so the label shows
    ``v:<valid_tok> / i:<invalid_tok>`` when they differ.
    """
    K = valid_probs.shape[0]
    x = np.arange(K)
    width = 0.4
    ax.bar(x - width / 2, valid_probs, width, label="valid", color="#1f77b4", alpha=0.85)
    ax.bar(x + width / 2, invalid_probs, width, label="invalid", color="#d62728", alpha=0.85)
    ax.set_title(
        f"Top-K distribution at position {pos} (input token: {token_str!r})",
        fontsize=9,
        fontweight="bold",
    )
    ax.set_ylabel("Probability", fontsize=8)
    ax.set_xlim(-0.8, K - 0.2)
    ax.legend(fontsize=8)

    # Build per-rank x-tick labels with decoded token text when available.
    if tokenizer is not None and valid_vocab_idx is not None and invalid_vocab_idx is not None:

        def _decode(idx: int) -> str:
            try:
                t = tokenizer.decode([idx], skip_special_tokens=False)
                # Collapse whitespace and truncate for readability.
                t = t.replace("\n", "↵").replace("\r", "").strip() or f"<{idx}>"
                return t[:8]
            except Exception:
                return str(idx)

        labels = []
        for rank in range(K):
            vi = int(valid_vocab_idx[rank])
            ii = int(invalid_vocab_idx[rank])
            vt = _decode(vi)
            it = _decode(ii)
            if vi == ii:
                labels.append(f"{rank}:{vt}")
            else:
                labels.append(f"{rank}:{vt}/{it}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_xlabel("Top-K rank  (valid_tok / invalid_tok when different)", fontsize=7)
    else:
        ax.set_xlabel("Top-K rank", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)


def _entropy_plot(
    ax: plt.Axes,
    valid_probs: np.ndarray,
    invalid_probs: np.ndarray,
    token_strs: list[str],
) -> None:
    """Per-position Shannon entropy of the top-K distribution."""

    def _ent(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-12, 1.0)
        return -(p * np.log(p)).sum(axis=-1)

    L = valid_probs.shape[0]
    positions = np.arange(L)
    ax.plot(positions, _ent(valid_probs), "o-", color="#1f77b4", label="valid", lw=1.2, ms=4)
    ax.plot(positions, _ent(invalid_probs), "s--", color="#d62728", label="invalid", lw=1.2, ms=4)
    ax.set_title("Per-token entropy of top-K distribution", fontsize=9, fontweight="bold")
    ax.set_xlabel("Token position", fontsize=8)
    ax.set_ylabel("Entropy (nats)", fontsize=8)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{i}:{t}" for i, t in enumerate(token_strs)],
        rotation=60,
        ha="right",
        fontsize=6,
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=str, default="plots/topk_probs_inspect")
    p.add_argument("--lm-key", type=str, default="hf_causal")
    p.add_argument("--top-k", type=int, default=64)
    p.add_argument("--seq-length", type=int, default=30)
    p.add_argument("--char-window", type=int, default=128)
    p.add_argument("--corrupt-rate", type=float, default=0.15)
    p.add_argument("--window-idx", type=int, default=0)
    p.add_argument("--token-pos", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-renormalize", action="store_true")
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    K = args.top_k
    L = args.seq_length
    device = torch.device(args.device)
    renorm = not args.no_renormalize

    # --- build config & LLM ---
    cfg = Config()
    cfg.llm_topk_probs.lm_key = args.lm_key
    cfg.llm_topk_probs.char_window_length = args.char_window
    cfg.dataset.K = K
    cfg.dataset.L = L

    print(f"[info] loading LLM ({args.lm_key}) ...")
    lm = build_lm(args.lm_key, cfg.teacher)
    lm_vocab_size = lm.vocab_size
    print(f"[info] LLM vocab size = {lm_vocab_size}")

    # --- load one text8 window ---
    print("[info] loading text8 train split ...")
    train_windows, _, _ = _load_text8_splits_cfg(cfg, cfg.text8_dataset.cache_dir, args.char_window)
    total_windows = train_windows.shape[0]
    idx = args.window_idx % total_windows
    char_ids = train_windows[idx]  # (W,)
    raw_text = _char_ids_to_text(char_ids)
    print(f"[info] window #{idx}/{total_windows}: {raw_text[:60]!r} ...")

    # --- tokenize ---
    clean_ids, _ = _batch_encode(lm, [raw_text], max_length=L)  # (1, L)
    clean_ids = clean_ids.to(device)

    # decode token strings for axis labels (best-effort)
    tokenizer = getattr(lm, "tokenizer", None) or getattr(lm, "_tokenizer", None)
    if tokenizer is not None:
        token_strs = [
            tokenizer.decode([int(t)], skip_special_tokens=True) or f"<{int(t)}>"
            for t in clean_ids[0]
        ]
    else:
        token_strs = [str(int(t)) for t in clean_ids[0]]
    print(f"[info] tokens: {token_strs}")

    # --- corrupt ---
    corrupt_ids = corrupt_token_ids(
        clean_ids,
        vocab_size=lm_vocab_size,
        corrupt_rate=args.corrupt_rate,
        seed=args.seed,
    )
    changed = (corrupt_ids != clean_ids).squeeze(0)
    changed_positions = changed.nonzero(as_tuple=False).squeeze(-1).tolist()
    print(f"[info] corrupted positions ({len(changed_positions)}): {changed_positions}")

    # --- extract top-K probs ---
    print("[info] running LLM forward pass (valid) ...")
    valid_probs, valid_idx = _get_topk_probs(lm, clean_ids, K, renormalize=renorm)  # (L, K)
    print("[info] running LLM forward pass (invalid) ...")
    invalid_probs, invalid_idx = _get_topk_probs(lm, corrupt_ids, K, renormalize=renorm)

    vp = valid_probs.numpy()  # (L, K)
    ip = invalid_probs.numpy()  # (L, K)

    # clamp token_pos to valid range
    token_pos = max(0, min(args.token_pos, L - 1))

    # --- Figure 1: raw heatmaps + ILR heatmaps + bar chart ---
    # Give the bar chart subplot extra horizontal room when K is large so
    # rotated tick labels don't overlap.  Each rank needs ~0.25 inches minimum.
    bar_width = max(6, K * 0.25)
    total_width = 24 + bar_width  # four heatmaps + one bar chart
    fig1, axes1 = plt.subplots(
        1,
        5,
        figsize=(total_width, max(6, L * 0.35)),
        gridspec_kw={"width_ratios": [6, 6, 6, 6, bar_width]},
    )
    fig1.suptitle(
        f"Top-{K} LLM probabilities | window #{idx} | corrupt_rate={args.corrupt_rate}",
        fontsize=11,
        fontweight="bold",
    )

    im = _heatmap(axes1[0], vp, "Valid sequence", token_strs)
    _heatmap(axes1[1], ip, "Invalid sequence (corrupted)", token_strs)

    ilr_valid = _to_ilr_features(valid_probs)
    ilr_invalid = _to_ilr_features(invalid_probs)
    im_ilr = _feature_heatmap(
        axes1[2],
        ilr_valid,
        "Valid sequence (renorm + ILR)",
        token_strs,
    )
    _feature_heatmap(
        axes1[3],
        ilr_invalid,
        "Invalid sequence (renorm + ILR)",
        token_strs,
    )

    # mark corrupted rows on both heatmaps
    for ax in axes1[:2]:
        for pos in changed_positions:
            ax.axhline(pos - 0.5, color="cyan", lw=0.8, alpha=0.6)
            ax.axhline(pos + 0.5, color="cyan", lw=0.8, alpha=0.6)

    _bar_chart(
        axes1[4],
        vp[token_pos],
        ip[token_pos],
        token_pos,
        token_strs[token_pos],
        valid_vocab_idx=valid_idx[token_pos].numpy(),
        invalid_vocab_idx=invalid_idx[token_pos].numpy(),
        tokenizer=tokenizer,
    )

    # Manual layout keeps margins stable and avoids colorbar overlap.
    fig1.subplots_adjust(left=0.03, right=0.94, top=0.90, bottom=0.20, wspace=0.30)

    # Place colorbars in fixed axes next to each heatmap group.
    prob_box = axes1[1].get_position()
    ilr_box = axes1[3].get_position()
    cax_prob = fig1.add_axes([prob_box.x1 + 0.004, prob_box.y0, 0.006, prob_box.height])
    cax_ilr = fig1.add_axes([ilr_box.x1 + 0.004, ilr_box.y0, 0.006, ilr_box.height])

    cbar = fig1.colorbar(im, cax=cax_prob, orientation="vertical")
    cbar.set_label("Probability (renorm within top-K)", fontsize=7)
    cbar.ax.tick_params(labelsize=7)

    cbar_ilr = fig1.colorbar(im_ilr, cax=cax_ilr, orientation="vertical")
    cbar_ilr.set_label("ILR coordinate value", fontsize=7)
    cbar_ilr.ax.tick_params(labelsize=7)

    out1 = out_dir / "heatmap_and_bar.png"
    fig1.savefig(out1, dpi=140)
    plt.close(fig1)
    print(f"[info] wrote {out1}")

    # --- Figure 2: entropy per token ---
    fig2, ax2 = plt.subplots(figsize=(max(10, L * 0.5), 4))
    _entropy_plot(ax2, vp, ip, token_strs)
    fig2.tight_layout()
    out2 = out_dir / "entropy_per_token.png"
    fig2.savefig(out2, dpi=140, bbox_inches="tight")
    plt.close(fig2)
    print(f"[info] wrote {out2}")

    # --- Figure 3: difference heatmap (|valid - invalid| per rank) ---
    diff = np.abs(vp - ip)  # (L, K)
    fig3, ax3 = plt.subplots(figsize=(10, max(4, L * 0.3)))
    im3 = ax3.imshow(diff, aspect="auto", origin="upper", cmap="viridis")
    ax3.set_title("|valid probs − invalid probs| per position/rank", fontsize=9, fontweight="bold")
    ax3.set_xlabel("Top-K rank", fontsize=8)
    ax3.set_ylabel("Token position", fontsize=8)
    ax3.set_yticks(range(L))
    ax3.set_yticklabels([f"{i}: {t}" for i, t in enumerate(token_strs)], fontsize=6)
    ax3.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=8))
    ax3.tick_params(axis="x", labelsize=7)
    for pos in changed_positions:
        ax3.axhline(pos - 0.5, color="red", lw=0.8, alpha=0.7)
        ax3.axhline(pos + 0.5, color="red", lw=0.8, alpha=0.7)
    fig3.colorbar(im3, ax=ax3, fraction=0.03, pad=0.01)
    fig3.tight_layout()
    out3 = out_dir / "diff_heatmap.png"
    fig3.savefig(out3, dpi=140, bbox_inches="tight")
    plt.close(fig3)
    print(f"[info] wrote {out3}")

    print("\nDone. Outputs:")
    for p in [out1, out2, out3]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
