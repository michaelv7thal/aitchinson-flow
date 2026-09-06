"""Dump the WORDS the word-level corruptions planted, as plain text.

The two word-level schemes of the corruption ladder replace whole words in a
clean ``text8`` window with a different real word of the same length:

  * false information — a random other same-length word from the length-bucketed
    vocabulary (``corrupt_false_info``); lexically valid, contextually off.
  * plausible — the same-length word the MODEL itself scores as most fluent in
    context, the candidate minimising the denoiser NLL over the word's positions
    (``plausible_swap``); lexically valid AND contextually fluent.

Which words each scheme actually plants is the thing the JSON metrics do not
record and the ``*.tokens.pt`` dumps record only as token ids (and ``*.pt`` is
gitignored, so those dumps do not survive a clone). This script turns a dump
into two committed text files, one per scheme, so the substitutions can be read,
diffed and cited without a GPU.

Everything here is CPU-only: it decodes and diffs saved tensors, it never calls
the model.

Usage:
    python scripts/dump_substitutions.py \
        --tokens bench_ood_final/plausible/plausible_swap_var.tokens.pt \
        --out-dir bench_ood_final/plausible
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import torch

_ALPHABET = "abcdefghijklmnopqrstuvwxyz "  # text8 K=27


def _decode(ids) -> str:
    return "".join(_ALPHABET[int(i)] if int(i) < len(_ALPHABET) else "?" for i in ids)


def _word_spans(row: torch.Tensor) -> list[tuple[int, int]]:
    """Fully-contained [a,b) letter-word spans of a window (edge words dropped).

    Same rule as ``corrupt_false_info`` and ``plausible_swap``, so the spans here
    are exactly the slots those two were free to choose from.
    """
    s = _decode(row)
    spans = [(m.start(), m.end()) for m in re.finditer(r"[a-z]+", s)]
    return [(a, b) for (a, b) in spans if a > 0 and b < len(s)]


def substitutions(clean: torch.Tensor, corrupt: torch.Tensor) -> list[dict]:
    """Every word slot whose characters differ between the two windows."""
    subs = []
    for i in range(clean.shape[0]):
        s_clean, s_corrupt = _decode(clean[i]), _decode(corrupt[i])
        for a, b in _word_spans(clean[i]):
            if s_clean[a:b] != s_corrupt[a:b]:
                subs.append({"window": i, "start": a, "end": b,
                             "original": s_clean[a:b], "planted": s_corrupt[a:b],
                             "context": s_clean, "context_corrupt": s_corrupt})
    return subs


def _context(sub: dict, width: int = 28) -> str:
    """The clean sentence around the slot, with ``[original>planted]`` in place."""
    s, a, b = sub["context"], sub["start"], sub["end"]
    left = s[max(0, a - width):a].lstrip()
    right = s[b:b + width].rstrip()
    return f"{left}[{sub['original']}>{sub['planted']}]{right}"


def write_file(path: Path, subs: list[dict], *, scheme: str, blurb: str,
               meta: dict, n_windows: int, sibling: str, shared: int) -> None:
    lines = [
        f"# {scheme} substitutions",
        "#",
    ]
    lines += [f"# {ln}" for ln in blurb.strip().split("\n")]
    lines += [
        "#",
        f"# source      : {meta['source']}",
        f"# checkpoint  : {meta.get('ckpt', 'n/a')}",
        f"# split       : {meta.get('split', 'test')}   windows: {n_windows}"
        f"   rate: {meta.get('rate')}   seed: {meta.get('seed')}",
        f"# words planted: {len(subs)}  over {n_windows} windows"
        f"  ({len(subs) / max(n_windows, 1):.1f} per window)",
        "#",
        f"# The sibling file holds the {sibling} substitutions of the same windows.",
        f"# The two schemes draw their slots from the same eligible words at the same",
        f"# rate, so the budget matches, but the draws are independent: {shared} of the",
        f"# slots below are also corrupted in the sibling file.",
        "#",
        "# window  start  original -> planted   context (clean text, [original>planted])",
        "",
    ]
    for s in subs:
        lines.append(
            f"{s['window']:>6}  {s['start']:>5}  "
            f"{s['original']:<14} -> {s['planted']:<14}  {_context(s)}"
        )
    counts = Counter(s["planted"] for s in subs)
    lines += [
        "",
        f"# distinct planted words: {len(counts)} of {len(subs)} substitutions",
        "# most-planted words:",
    ]
    lines += [f"#   {n:>4}x  {w}" for w, n in counts.most_common(20)]
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}  ({len(subs)} substitutions, {len(counts)} distinct words)")


_FALSEINFO_BLURB = """
Each planted word is a random OTHER word of the same length, drawn from the
length-bucketed text8 vocabulary of build_vocab_by_len (first-seen order, so
roughly frequency-ranked). The text stays lexically valid and the character
statistics barely move, but the word does not fit its context. This is the
cheap scheme: it costs no model calls at all.
"""

_PLAUSIBLE_BLURB = """
Each planted word is the same-length word the MODEL itself finds most fluent in
that slot: of 48 same-length candidates it scores the denoiser NLL over the
word's positions in the clean context and keeps the minimiser, the original
excluded. This is the expensive scheme: one batched forward over 48 rewritten
copies of the whole window per word, so a rate-0.15 run over 64 windows costs
~390 forwards where false information costs none.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", default="bench_ood_final/plausible/plausible_swap_var.tokens.pt",
                    help="a *.tokens.pt dump with eval_tok / rand_tok / plaus_tok")
    ap.add_argument("--out-dir", default="bench_ood_final/plausible")
    ap.add_argument("--ckpt", default="runs/sflm_bench_a100_20g_L256_d1280L14_full/"
                                      "DirichletFM_converge/epoch_final.pt")
    args = ap.parse_args()

    d = torch.load(args.tokens, map_location="cpu")
    clean, rand, plaus = d["eval_tok"], d["rand_tok"], d["plaus_tok"]
    meta = {"source": args.tokens, "ckpt": args.ckpt, "split": "test",
            "rate": d.get("rate"), "seed": d.get("seed")}

    fi = substitutions(clean, rand)
    pl = substitutions(clean, plaus)
    fi_slots = {(s["window"], s["start"]) for s in fi}
    pl_slots = {(s["window"], s["start"]) for s in pl}
    shared = len(fi_slots & pl_slots)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_file(out / "substitutions_falseinfo.txt", fi, scheme="false information",
               blurb=_FALSEINFO_BLURB, meta=meta, n_windows=clean.shape[0],
               sibling="plausible", shared=shared)
    write_file(out / "substitutions_plausible.txt", pl,
               scheme=f"plausible ({d.get('swap_mode', 'min_nll')})",
               blurb=_PLAUSIBLE_BLURB, meta=meta, n_windows=clean.shape[0],
               sibling="false-information", shared=shared)
    print(f"slots corrupted by both schemes: {shared} "
          f"(false info {len(fi)}, plausible {len(pl)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
