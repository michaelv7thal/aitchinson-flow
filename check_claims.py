#!/usr/bin/env python3
"""Turn the paper's claim-to-artifact table into an executable check.

The traceability matrix in the cleanup plan is prose, and prose rots. This script
reads the same matrix as a TSV and re-reads every number out of the artifact it
claims to come from, so "the paper is still checkable from what remains" becomes
a command with an exit code instead of an assertion.

    python3 check_claims.py claims.tsv [--repo R] [--strict] [--tier T]

TSV columns (tab separated, `#` comments and blank lines ignored):

    id  paper_loc  quantity  expected  tol  tier  artifact  pointer  notes

    id         stable claim id, e.g. AB1
    paper_loc  chapters/results.tex:312, where the number is printed
    quantity   human name, e.g. "KL_uni, fixed-interpolant x deterministic clr"
    expected   the number as the paper prints it
    tol        absolute tolerance; use half a unit of the paper's last digit
    tier       T0/T1/T2/T3 reproduction tier (see the skill)
    artifact   repo-relative path to the JSON the number was read from
    pointer    path into that JSON, or * to search the whole file
    notes      free text

Pointer grammar, `.` separated:

    key                        dict key
    [3]                        list index (parent is a list)
    [scheme=replace,rate=0.15] the single list element matching all predicates
    [0.05]                     literal dict key, for keys containing '.'
                               (parent is a dict; no '=' inside the brackets)

    rows[scheme=replace,rate=0.15].word_max.word.auroc
    rows[detector=NLL,corruption=random].prf_token.prf[0.05].f1

A `*` pointer means "this value appears somewhere in this file". That is a real
check but a weak one, since a file can contain a number by coincidence. Rows that
use it are reported as WEAK and `--strict` makes them failures. Prefer a pointer.

Exit status: 0 if every row passed (WEAK counts as passing unless --strict).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_DEFAULT = Path(__file__).resolve().parent
NUM_RE = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


class PointerError(Exception):
    pass


def split_segments(pointer: str) -> list[str]:
    """Split on '.' but never inside brackets."""
    segs, buf, depth = [], [], 0
    for ch in pointer:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "." and depth == 0:
            segs.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    segs.append("".join(buf))
    return [s for s in segs if s]


def match_pred(item, preds: list[tuple[str, str]]) -> bool:
    for k, v in preds:
        if not isinstance(item, dict) or k not in item:
            return False
        got = item[k]
        try:
            if abs(float(got) - float(v)) > 1e-9:
                return False
        except (TypeError, ValueError):
            if str(got) != v:
                return False
    return True


def resolve(obj, pointer: str):
    """Walk `pointer` into `obj`. Raises PointerError with a locating message."""
    cur = obj
    for seg in split_segments(pointer):
        while seg:
            m = re.match(r"^([^\[\]]+)", seg)
            if m:  # a dict key
                key = m.group(1)
                if not isinstance(cur, dict):
                    raise PointerError(f"'{key}': parent is {type(cur).__name__}, not a dict")
                if key not in cur:
                    keys = list(cur)[:10]
                    raise PointerError(f"key '{key}' absent; available: {keys}")
                cur = cur[key]
                seg = seg[m.end():]
                continue
            m = re.match(r"^\[([^\]]*)\]", seg)
            if not m:
                raise PointerError(f"cannot parse segment '{seg}'")
            inner = m.group(1)
            if isinstance(cur, dict):
                # literal dict-key lookup, for keys that contain '.' and so
                # cannot appear as a plain segment, e.g. prf_token.prf[0.05].f1
                if "=" in inner:
                    raise PointerError(
                        f"'[{inner}]': predicates select list elements, but parent is a dict")
                if inner not in cur:
                    keys = list(cur)[:10]
                    raise PointerError(f"literal key '{inner}' absent; available: {keys}")
                cur = cur[inner]
                seg = seg[m.end():]
                continue
            if not isinstance(cur, list):
                raise PointerError(f"'[{inner}]': parent is {type(cur).__name__}, not a list")
            if re.fullmatch(r"-?\d+", inner):
                idx = int(inner)
                if not -len(cur) <= idx < len(cur):
                    raise PointerError(f"index {idx} out of range (len {len(cur)})")
                cur = cur[idx]
            else:
                preds = []
                for part in inner.split(","):
                    if "=" not in part:
                        raise PointerError(f"predicate '{part}' is not k=v")
                    k, v = part.split("=", 1)
                    preds.append((k.strip(), v.strip()))
                hits = [x for x in cur if match_pred(x, preds)]
                if len(hits) == 0:
                    raise PointerError(f"no list element matches [{inner}]")
                if len(hits) > 1:
                    raise PointerError(f"{len(hits)} elements match [{inner}]; not unique")
                cur = hits[0]
            seg = seg[m.end():]
    return cur


def file_contains(path: Path, value: float, tol: float) -> bool:
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return False
    for m in NUM_RE.finditer(text):
        try:
            if abs(float(m.group(0)) - value) <= tol:
                return True
        except ValueError:
            continue
    return False


def load_rows(tsv: Path) -> list[dict]:
    fields = ["id", "paper_loc", "quantity", "expected", "tol",
              "tier", "artifact", "pointer", "notes"]
    rows = []
    for lineno, raw in enumerate(tsv.read_text().splitlines(), 1):
        line = raw.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t")
        if parts and parts[0].strip() == "id":
            continue  # header
        if len(parts) < 8:
            rows.append({"id": f"line{lineno}", "_error":
                         f"expected >=8 tab-separated columns, got {len(parts)}"})
            continue
        r = dict(zip(fields, [p.strip() for p in parts]))
        r.setdefault("notes", "")
        r["_lineno"] = lineno
        rows.append(r)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tsv", type=Path)
    ap.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    ap.add_argument("--strict", action="store_true",
                    help="treat WEAK (* pointer) rows as failures")
    ap.add_argument("--tier", help="only check rows of this tier, e.g. T1")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()

    if args.no_color or not sys.stdout.isatty():
        g = r = y = d = z = ""
    else:
        g, r, y, d, z = GREEN, RED, YELLOW, DIM, RESET

    if not args.tsv.exists():
        print(f"no such claims file: {args.tsv}", file=sys.stderr)
        return 2

    cache: dict[Path, object] = {}
    results = []

    for row in load_rows(args.tsv):
        rid = row.get("id", "?")
        if "_error" in row:
            results.append((rid, "ERROR", row["_error"]))
            continue
        if args.tier and row.get("tier") != args.tier:
            continue

        try:
            expected = float(row["expected"])
        except ValueError:
            results.append((rid, "ERROR", f"expected '{row['expected']}' is not a number"))
            continue
        try:
            tol = float(row["tol"])
        except ValueError:
            results.append((rid, "ERROR", f"tol '{row['tol']}' is not a number"))
            continue

        path = args.repo / row["artifact"]
        if not path.exists():
            results.append((rid, "ERROR", f"artifact missing: {row['artifact']}"))
            continue

        pointer = row["pointer"]
        if pointer == "*":
            ok = file_contains(path, expected, tol)
            results.append((rid, "WEAK" if ok else "FAIL",
                            f"{expected} {'found' if ok else 'NOT found'} anywhere in "
                            f"{row['artifact']}"))
            continue

        if path not in cache:
            try:
                cache[path] = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as e:
                cache[path] = e
        doc = cache[path]
        if isinstance(doc, Exception):
            results.append((rid, "ERROR", f"{row['artifact']}: {doc}"))
            continue

        try:
            got = resolve(doc, pointer)
        except PointerError as e:
            results.append((rid, "ERROR", f"{row['artifact']} :: {pointer}: {e}"))
            continue
        try:
            gotf = float(got)
        except (TypeError, ValueError):
            results.append((rid, "ERROR", f"{pointer} resolved to {got!r}, not a number"))
            continue

        delta = abs(gotf - expected)
        if delta <= tol:
            results.append((rid, "PASS", f"{expected} == {gotf:.6g} (|d|={delta:.2g})"))
        else:
            results.append((rid, "FAIL",
                            f"paper says {expected}, artifact says {gotf:.6g} "
                            f"(|d|={delta:.3g} > tol {tol})"))

    width = max((len(i) for i, _, _ in results), default=2)
    counts = {"PASS": 0, "WEAK": 0, "FAIL": 0, "ERROR": 0}
    for rid, status, msg in results:
        counts[status] += 1
        color = {"PASS": g, "WEAK": y, "FAIL": r, "ERROR": r}[status]
        print(f"{color}{status:5s}{z} {rid:{width}s}  {msg}")

    print()
    print(f"{counts['PASS']} pass, {counts['WEAK']} weak, "
          f"{counts['FAIL']} fail, {counts['ERROR']} error "
          f"({len(results)} rows)")
    if counts["WEAK"]:
        print(f"{d}WEAK rows used a '*' pointer: the value is in the file but not at a "
              f"named location. Replace with a real pointer to make the check load-bearing.{z}")

    bad = counts["FAIL"] + counts["ERROR"] + (counts["WEAK"] if args.strict else 0)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
