"""Throwaway: grep the text8 train/validation/test splits.

Loads the raw character stream of each `afmck/text8` split (the same source
`data/hf_text_loader.py` uses), caches it as a plain .txt, and prints every
match with surrounding context plus the index of the length-L window the match
would land in.

    python scripts/search_text8.py "queen of england"
    python scripts/search_text8.py --regex "b[aeiou]nana" --splits test
    python scripts/search_text8.py "anarchism" --context 120 --max 5 --L 256

Note text8 is lowercase a-z plus space only: no punctuation, no digits
(numbers are spelled out), no newlines. Queries are lowercased by default.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DEFAULT_CACHE = Path.home() / ".cache" / "text8_search"
SPLITS = ("train", "validation", "test")


def load_split(split: str, cache_dir: Path) -> str:
    """Raw text of one split, cached to <cache_dir>/<split>.txt."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{split}.txt"
    if path.exists():
        return path.read_text()

    import datasets  # slow import, only needed on a cache miss

    print(f"[cache miss] downloading/loading text8:{split} ...", file=sys.stderr)
    ds = datasets.load_dataset("afmck/text8", split=split)
    text = " ".join(ds["text"])
    path.write_text(text)
    print(f"[cached] {path} ({len(text):,} chars)", file=sys.stderr)
    return text


def iter_matches(text: str, pattern: re.Pattern[str]):
    for m in pattern.finditer(text):
        yield m.start(), m.end()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("query", help="literal substring, or a regex with --regex")
    p.add_argument("--regex", action="store_true", help="treat query as a regex")
    p.add_argument(
        "--splits",
        default="train,validation,test",
        help="comma-separated subset of train,validation,test (default: all)",
    )
    p.add_argument("--context", type=int, default=60, help="context chars on each side (default 60)")
    p.add_argument("--max", type=int, default=20, help="max matches per split (default 20; 0 = all)")
    p.add_argument("--case-sensitive", action="store_true", help="do not lowercase the query")
    p.add_argument("--L", type=int, default=256, help="window length for the window-index report (default 256)")
    p.add_argument("--count-only", action="store_true", help="print only per-split match counts")
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    args = p.parse_args()

    query = args.query if args.case_sensitive else args.query.lower()
    pattern = re.compile(query if args.regex else re.escape(query))

    total = 0
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        if split not in SPLITS:
            p.error(f"unknown split {split!r}; pick from {SPLITS}")
        text = load_split(split, args.cache_dir)

        hits = list(iter_matches(text, pattern))
        total += len(hits)
        print(f"\n=== {split}  ({len(text):,} chars, {len(text) // args.L:,} windows @ L={args.L}) "
              f"-> {len(hits):,} match(es) ===")
        if args.count_only:
            continue

        shown = hits if args.max == 0 else hits[: args.max]
        for start, end in shown:
            lo = max(0, start - args.context)
            hi = min(len(text), end + args.context)
            left, mid, right = text[lo:start], text[start:end], text[end:hi]
            win = start // args.L
            print(
                f"\n  char {start:,}  window {win:,} (offset {start % args.L})"
                f"  frac {start / len(text):.4f}"
            )
            print(f"    ...{left}[{mid}]{right}...")
        if args.max and len(hits) > args.max:
            print(f"\n  ... {len(hits) - args.max:,} more match(es) suppressed (--max 0 for all)")

    print(f"\ntotal: {total:,} match(es)")
    return 0 if total else 1


if __name__ == "__main__":
    raise SystemExit(main())
