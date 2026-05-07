"""One-time WikiText-2 feature cache for the EqM auditor (Phase F).

Runs the LM (default GPT-2) over a subset of WikiText-2 train chunks and
saves logit/hidden-state features for both the clean and span-corrupted
variants. The training loop reads the cache directly — the LM itself is
*never* loaded during training.

Usage:
    python scripts/cache_wiki.py --lm gpt2 --n 300 --L 64 --K 64 \\
        --out data/wiki_cache_gpt2.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from aitchinson_flow.data.wiki import (  # noqa: E402
    _spilled_energy_per_pos,
    _topk_clr,
    span_corrupt,
)


@torch.no_grad()
def _stream_chunks(
    text_iter, tokenizer, *, L: int, n: int, min_chunk_tokens: int = 32
) -> torch.Tensor:
    """Concatenate texts and chunk into L-token windows; return (n, L) ids."""
    eot = tokenizer.eos_token_id
    buf: list[int] = []
    chunks: list[list[int]] = []
    for text in text_iter:
        if not text.strip():
            continue
        ids = tokenizer.encode(text)
        if len(ids) < 2:
            continue
        buf.extend(ids)
        buf.append(eot)
        while len(buf) >= L and len(chunks) < n:
            chunks.append(buf[:L])
            buf = buf[L:]
        if len(chunks) >= n:
            break
    if len(chunks) < n:
        # Pad up if WikiText-2 is small (it's not — this is a guard)
        raise RuntimeError(
            f"only got {len(chunks)} chunks of length {L}; "
            f"need {n}. Reduce --n or check the dataset."
        )
    return torch.tensor(chunks[:n], dtype=torch.long)


@torch.no_grad()
def _lm_forward(
    model, ids: torch.Tensor, *, batch_size: int = 8
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the LM in batches and return (logits, last_hidden) on CPU."""
    device = next(model.parameters()).device
    n = ids.shape[0]
    out_logits: list[torch.Tensor] = []
    out_hidden: list[torch.Tensor] = []
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        batch = ids[s:e].to(device)
        out = model(batch, output_hidden_states=True)
        logits = out.logits  # (B, L, V)
        hidden = out.hidden_states[-1]  # (B, L, H)
        out_logits.append(logits.cpu().float())
        out_hidden.append(hidden.cpu().float())
    return torch.cat(out_logits, dim=0), torch.cat(out_hidden, dim=0)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lm", default="gpt2", choices=["gpt2", "gpt2-medium"])
    p.add_argument("--n", type=int, default=300, help="number of L-token chunks")
    p.add_argument("--L", type=int, default=64, help="tokens per chunk")
    p.add_argument("--K", type=int, default=64, help="top-K logits per position")
    p.add_argument("--corrupt-rate", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", required=True)
    p.add_argument("--lm-batch-size", type=int, default=8)
    args = p.parse_args(argv)

    print(f"[cache_wiki] lm={args.lm}, n={args.n}, L={args.L}, K={args.K}")

    print("[cache_wiki] loading wikitext-2 train …")
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    print(f"  rows: {len(ds)}")

    print(f"[cache_wiki] loading {args.lm} …")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.lm)
    model = AutoModelForCausalLM.from_pretrained(args.lm, torch_dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    V = model.config.vocab_size
    H = model.config.hidden_size
    print(f"  V={V}  H={H}")

    print(f"[cache_wiki] tokenising and chunking → ({args.n}, {args.L}) …")
    text_iter = (row["text"] for row in ds)
    clean_ids = _stream_chunks(text_iter, tokenizer, L=args.L, n=args.n)
    print(f"  clean_ids: {tuple(clean_ids.shape)}")

    print("[cache_wiki] span-corrupting …")
    invalid_ids, mask = span_corrupt(
        clean_ids, vocab_size=V, corrupt_rate=args.corrupt_rate, seed=args.seed
    )
    n_corrupt = int(mask.sum().item())
    pct = 100.0 * n_corrupt / mask.numel()
    print(f"  corrupted positions: {n_corrupt} / {mask.numel()} ({pct:.1f}%)")

    print("[cache_wiki] LM forward on clean …")
    clean_logits, clean_h = _lm_forward(model, clean_ids, batch_size=args.lm_batch_size)
    print(f"  clean_logits: {tuple(clean_logits.shape)}  clean_h: {tuple(clean_h.shape)}")

    print("[cache_wiki] LM forward on invalid …")
    invalid_logits, invalid_h = _lm_forward(
        model, invalid_ids, batch_size=args.lm_batch_size
    )

    print("[cache_wiki] top-K + Spilled Energy …")
    clean_clr, clean_topk_idx = _topk_clr(clean_logits, args.K)
    invalid_clr, invalid_topk_idx = _topk_clr(invalid_logits, args.K)
    clean_SE = _spilled_energy_per_pos(clean_logits, clean_ids)
    invalid_SE = _spilled_energy_per_pos(invalid_logits, invalid_ids)

    cache: dict[str, Any] = {
        "lm": args.lm,
        "L": args.L,
        "K": args.K,
        "V": V,
        "H": H,
        "n": args.n,
        "corrupt_rate": args.corrupt_rate,
        "clean_ids": clean_ids,
        "invalid_ids": invalid_ids,
        "mask_corrupt": mask,
        "clean_clr": clean_clr,
        "invalid_clr": invalid_clr,
        "clean_topk_idx": clean_topk_idx,
        "invalid_topk_idx": invalid_topk_idx,
        "clean_h": clean_h,
        "invalid_h": invalid_h,
        "clean_SE_pos": clean_SE,
        "invalid_SE_pos": invalid_SE,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, out_path)
    print(f"[cache_wiki] wrote {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(
        "[cache_wiki] sanity: clean SE mean = "
        f"{clean_SE.mean().item():.3f}, invalid SE mean = {invalid_SE.mean().item():.3f} "
        "(invalid should be ~much higher)"
    )

    # 1-line summary as JSON for the calling script.
    print(json.dumps({
        "lm": args.lm, "n": args.n, "L": args.L, "K": args.K,
        "V": V, "H": H, "out": str(out_path),
        "clean_SE_mean": float(clean_SE.mean()),
        "invalid_SE_mean": float(invalid_SE.mean()),
    }))


if __name__ == "__main__":
    main()
