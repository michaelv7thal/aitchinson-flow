"""Store the GPT-2 BPE token counts behind the paper's re-tokenization claims.

The OOD results argue that character-level corruption re-tokenizes the text
(fig/tab sec:ood-word): the paper quotes the clean test set at 13,361 byte-pair
tokens against 22,801 at replace rate 0.15 and 33,014 at rate 0.5, plus the
per-token corruption prevalence. The corrupted counts are stored in the bench
sweep JSONs (``n_bpe_tokens`` per row) but the CLEAN count never was; this
script recomputes the whole table from scratch and stores it as a sidecar next
to the sweep, so every count the paper quotes has a stored source.

Uses the exact bench conventions: same test-split slicing (fit 512 / eval 256),
same corruption functions and per-rate seeds (``seed + int(1000*rate)``), and
the same tokenizer call as ``_gpt2_bpe_scores`` (GPT-2 fast tokenizer, no
special tokens, spans with ``end > start``). No GPT-2 forward pass is needed.

Usage:
    uv run python scripts/dump_bpe_counts.py \
        --ckpt runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM_converge/epoch_final.pt \
        --out bench_ood_final/gpt2_nll/bpe_token_counts.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace as _replace
from pathlib import Path

import torch


def _bootstrap() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))


_bootstrap()

from scripts.eval_full import _config_from_payload  # noqa: E402
from aitchinson_flow.training import build_training_datamodule  # noqa: E402
from aitchinson_flow.data.corruption import (  # noqa: E402
    corrupt_token_ids,
    partially_shuffle_token_ids,
    corrupt_false_info,
    build_vocab_by_len,
)

_ALPH = "abcdefghijklmnopqrstuvwxyz "


def _ids_to_text(char_ids: torch.Tensor) -> list[str]:
    return ["".join(_ALPH[int(i)] if int(i) < len(_ALPH) else "?" for i in row)
            for row in char_ids]


def _corrupt(tok, scheme, rate, K, seed, by_len=None):
    if scheme == "replace":
        return corrupt_token_ids(tok, vocab_size=K, corrupt_rate=rate, seed=seed)
    if scheme == "shuffle":
        return partially_shuffle_token_ids(tok, shuffle_rate=rate, seed=seed)
    if scheme in ("falseinfo", "wordswap"):
        return corrupt_false_info(tok, rate, by_len, seed=seed)
    out = corrupt_token_ids(tok, vocab_size=K, corrupt_rate=rate, seed=seed)
    return partially_shuffle_token_ids(out, shuffle_rate=rate, seed=seed + 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True,
                    help="DirichletFM ckpt — used only for its cfg/datamodule")
    ap.add_argument("--ref-lm", default="gpt2")
    ap.add_argument("--out", default="bench_ood_final/gpt2_nll/bpe_token_counts.json")
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--fit-seqs", type=int, default=512)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--schemes", type=str, default="replace,shuffle,both,falseinfo")
    ap.add_argument("--rates", type=str, default="0.05,0.1,0.15,0.2,0.25,0.3,0.5,0.7,1.0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = _config_from_payload(payload)
    cfg.training = _replace(cfg.training, device="cpu")
    K = cfg.text8_dataset.K
    dm, _ = build_training_datamodule(cfg)
    loaders = {"train": dm.train_dataloader, "val": dm.val_dataloader,
               "test": dm.test_dataloader}
    vl = loaders[args.split]() or dm.train_dataloader()
    seqs = []
    for b in vl:
        seqs.append(b["token_ids"].long())
        if sum(s.shape[0] for s in seqs) >= args.fit_seqs + args.n:
            break
    seqs = torch.cat(seqs)
    fit_tok = seqs[: args.fit_seqs]
    pos_tok = seqs[args.fit_seqs : args.fit_seqs + args.n]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.ref_lm, add_prefix_space=False,
                                        use_fast=True)

    def counts(char_ids: torch.Tensor, changed: torch.Tensor | None):
        """(n_tokens, n_tokens_touching_changed) with the bench's span rule."""
        n_tok = n_hit = 0
        enc = tok(_ids_to_text(char_ids), return_offsets_mapping=True,
                  add_special_tokens=False)
        ch = changed.numpy() if changed is not None else None
        for j, offs in enumerate(enc["offset_mapping"]):
            for (s, e) in offs:
                if e <= s:
                    continue
                n_tok += 1
                if ch is not None and ch[j, s:e].any():
                    n_hit += 1
        return n_tok, n_hit

    by_len = build_vocab_by_len(" ".join(_ids_to_text(fit_tok)))
    n_clean, _ = counts(pos_tok, None)
    print(f"[bpe] clean: {n_clean} tokens over {args.n} windows "
          f"(L={pos_tok.shape[1]}, {int(pos_tok.numel())} chars)")

    rows = []
    schemes = [s for s in args.schemes.split(",") if s.strip()]
    rates = [float(r) for r in args.rates.split(",") if r.strip()]
    for scheme in schemes:
        for r in rates:
            ot = _corrupt(pos_tok.clone(), scheme, r, K,
                          args.seed + int(1000 * r), by_len)
            n_tok, n_hit = counts(ot, ot != pos_tok)
            rows.append({"scheme": scheme, "rate": r, "n_bpe_tokens": n_tok,
                         "vs_clean": round(n_tok / n_clean, 4),
                         "prevalence": round(n_hit / n_tok, 4)})
            print(f"[bpe] {scheme:>9} {r:>5.2f}: {n_tok} tokens "
                  f"({n_tok / n_clean:.4f}x clean), prevalence {n_hit / n_tok:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "ckpt": args.ckpt, "ref_lm": args.ref_lm, "split": args.split,
        "n": args.n, "fit_seqs": args.fit_seqs, "seed": args.seed,
        "n_chars": int(pos_tok.numel()), "n_bpe_tokens_clean": n_clean,
        "rows": rows,
    }, indent=2))
    print(f"[bpe] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
