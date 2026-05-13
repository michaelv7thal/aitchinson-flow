"""Text8 + GPT-2 feature cache, structurally identical to ``cache_wiki.py``.

text8 IS English text (lowercased, space-only). It tokenises cleanly with the
GPT-2 BPE tokenizer; the resulting cache has the same schema as the wiki
cache (BPE-level positions, top-K logits + last-hidden-state, span-corrupt
negatives) and is consumed by the same ``WikiAuditorDataset`` /
``WikiAuditorDataModule``. Only the source corpus changes — keep text8 as
the experimental substrate for cross-cell consistency.

Usage:
    python scripts/cache_text8_gpt2.py --lm gpt2 --n 300 --L 64 --K 64 \\
        --out data/text8_cache_gpt2.pt
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
    text_iter, tokenizer, *, L: int, n: int
) -> torch.Tensor:
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
        raise RuntimeError(
            f"only got {len(chunks)} chunks of length {L}; "
            f"need {n}. Reduce --n or extend the text8 slice read."
        )
    return torch.tensor(chunks[:n], dtype=torch.long)


@torch.no_grad()
def _lm_forward(model, ids: torch.Tensor, *, batch_size: int = 8):
    device = next(model.parameters()).device
    n = ids.shape[0]
    out_logits: list[torch.Tensor] = []
    out_hidden: list[torch.Tensor] = []
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        batch = ids[s:e].to(device)
        out = model(batch, output_hidden_states=True)
        logits = out.logits
        hidden = out.hidden_states[-1]
        out_logits.append(logits.cpu().float())
        out_hidden.append(hidden.cpu().float())
    return torch.cat(out_logits, dim=0), torch.cat(out_hidden, dim=0)


def _text8_iter(max_chars: int = 8_000_000):
    """Stream text8 as one long string, sliced into reasonable chunks for
    the BPE tokeniser to digest (the entire corpus is ~100 MB; we don't
    need the whole thing for a POC cache)."""
    from datasets import load_dataset

    ds = load_dataset("afmck/text8", split="train")
    full = ds[0]["text"][:max_chars]
    # Yield ~64 KB at a time so the tokeniser doesn't blow up.
    step = 64 * 1024
    for i in range(0, len(full), step):
        yield full[i : i + step]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lm", default="gpt2", choices=["gpt2", "gpt2-medium"])
    p.add_argument("--n", type=int, default=300, help="number of L-token chunks")
    p.add_argument("--L", type=int, default=64, help="BPE tokens per chunk")
    p.add_argument("--K", type=int, default=64, help="top-K logits per position")
    p.add_argument("--corrupt-rate", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", required=True)
    p.add_argument("--lm-batch-size", type=int, default=8)
    p.add_argument(
        "--text8-chars",
        type=int,
        default=4_000_000,
        help="number of chars of text8 to read (corpus is ~100M; default 4M)",
    )
    args = p.parse_args(argv)

    print(f"[cache_text8_gpt2] lm={args.lm}, n={args.n}, L={args.L}, K={args.K}")

    print(f"[cache_text8_gpt2] loading {args.lm} …")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.lm)
    model = AutoModelForCausalLM.from_pretrained(args.lm, torch_dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    V = model.config.vocab_size
    H = model.config.hidden_size
    print(f"  V={V}  H={H}")

    print(
        f"[cache_text8_gpt2] tokenising and chunking → ({args.n}, {args.L}) "
        f"from first {args.text8_chars/1e6:.1f}M chars of text8 …"
    )
    clean_ids = _stream_chunks(
        _text8_iter(max_chars=args.text8_chars),
        tokenizer,
        L=args.L,
        n=args.n,
    )
    print(f"  clean_ids: {tuple(clean_ids.shape)}")

    print("[cache_text8_gpt2] span-corrupting …")
    invalid_ids, mask = span_corrupt(
        clean_ids, vocab_size=V, corrupt_rate=args.corrupt_rate, seed=args.seed
    )
    n_corrupt = int(mask.sum().item())
    pct = 100.0 * n_corrupt / mask.numel()
    print(f"  corrupted positions: {n_corrupt} / {mask.numel()} ({pct:.1f}%)")

    print("[cache_text8_gpt2] LM forward on clean …")
    clean_logits, clean_h = _lm_forward(model, clean_ids, batch_size=args.lm_batch_size)
    print(f"  clean_logits: {tuple(clean_logits.shape)}  clean_h: {tuple(clean_h.shape)}")

    print("[cache_text8_gpt2] LM forward on invalid …")
    invalid_logits, invalid_h = _lm_forward(
        model, invalid_ids, batch_size=args.lm_batch_size
    )

    print("[cache_text8_gpt2] top-K + Spilled Energy …")
    clean_clr, clean_topk_idx = _topk_clr(clean_logits, args.K)
    invalid_clr, invalid_topk_idx = _topk_clr(invalid_logits, args.K)
    clean_SE = _spilled_energy_per_pos(clean_logits, clean_ids)
    invalid_SE = _spilled_energy_per_pos(invalid_logits, invalid_ids)

    cache: dict[str, Any] = {
        "lm": args.lm,
        "corpus": "text8",
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
    print(f"[cache_text8_gpt2] wrote {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(
        "[cache_text8_gpt2] sanity: clean SE mean = "
        f"{clean_SE.mean().item():.3f}, invalid SE mean = {invalid_SE.mean().item():.3f} "
        "(invalid should be ~much higher)"
    )

    print(json.dumps({
        "lm": args.lm, "corpus": "text8", "n": args.n, "L": args.L, "K": args.K,
        "V": V, "H": H, "out": str(out_path),
        "clean_SE_mean": float(clean_SE.mean()),
        "invalid_SE_mean": float(invalid_SE.mean()),
    }))


if __name__ == "__main__":
    main()
