"""Append-only JSONL manifest store for benchmark runs.

The harness writes one JSON object per experiment run to ``results/manifest.jsonl``.
Every record carries the keys defined in the HARNESS INTERFACE CONTRACT::

    {exp_id, status, run_dir, seeds, git_sha, config_hash, gpu, wall_time_s,
     peak_mem_gb, length_fallback, second_order, metrics, artifacts, notes}

This module is intentionally dependency-free (stdlib only) so it can be imported
from training/eval scripts without dragging in torch. It exposes:

    MANIFEST_PATH         — Path("results/manifest.jsonl")
    append(record)        — append one record (creates parent dir)
    lookup(exp_id)        — LAST matching record, or None
    is_done(exp_id)       — True iff a 'done' record exists for exp_id
    load_all()            — list of all records ([] if file missing/empty)

CLI:
    python scripts/manifest.py --list
    python scripts/manifest.py --get EXP_ID
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MANIFEST_PATH = Path("results/manifest.jsonl")


def append(record: dict) -> None:
    """Append ``record`` (a JSON-serialisable dict) to the manifest.

    Creates the parent directory if needed. Append-only: never rewrites or
    truncates existing lines.
    """
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=False)
    with MANIFEST_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load_all() -> list[dict]:
    """Return every record in the manifest, in file order.

    Tolerates a missing or empty file (returns ``[]``) and silently skips blank
    lines.
    """
    if not MANIFEST_PATH.exists():
        return []
    records: list[dict] = []
    with MANIFEST_PATH.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def lookup(exp_id: str) -> dict | None:
    """Return the LAST record whose ``exp_id`` matches, or None."""
    match: dict | None = None
    for record in load_all():
        if record.get("exp_id") == exp_id:
            match = record
    return match


def is_done(exp_id: str) -> bool:
    """True iff at least one record for ``exp_id`` has ``status == "done"``."""
    return any(
        record.get("exp_id") == exp_id and record.get("status") == "done"
        for record in load_all()
    )


def _smoke() -> int:
    """Self-test against a temporary manifest path; prints OK on success."""
    global MANIFEST_PATH
    import tempfile

    original_path = MANIFEST_PATH
    tmp = Path(tempfile.mkdtemp(prefix="manifest_smoke_")) / "manifest.jsonl"
    MANIFEST_PATH = tmp
    try:
        assert load_all() == [], "load_all on missing file should be []"

        done_id = "exp_done"
        failed_id = "exp_failed"
        done_rec = {
            "exp_id": done_id,
            "status": "done",
            "run_dir": "runs/exp_done",
            "seeds": [0],
            "git_sha": "deadbeef",
            "config_hash": "abc123",
            "gpu": "cpu",
            "wall_time_s": 1.0,
            "peak_mem_gb": 0.0,
            "length_fallback": None,
            "second_order": None,
            "metrics": {"KL_uni": 0.01},
            "artifacts": [],
            "notes": "smoke",
        }
        failed_rec = {
            "exp_id": failed_id,
            "status": "failed",
            "run_dir": "runs/exp_failed",
            "seeds": [0],
            "git_sha": "deadbeef",
            "config_hash": "def456",
            "gpu": "cpu",
            "wall_time_s": 0.5,
            "peak_mem_gb": 0.0,
            "length_fallback": None,
            "second_order": None,
            "metrics": {},
            "artifacts": [],
            "notes": "smoke fail",
        }
        # Two records for done_id to verify lookup returns the LAST one.
        done_rec_v2 = dict(done_rec, notes="smoke v2")

        append(done_rec)
        append(failed_rec)
        append(done_rec_v2)

        assert is_done(done_id) is True, "is_done(done_id) must be True"
        assert is_done(failed_id) is False, "is_done(failed_id) must be False"
        assert is_done("missing") is False, "is_done(missing) must be False"

        last = lookup(done_id)
        assert last is not None, "lookup(done_id) must not be None"
        assert last["notes"] == "smoke v2", "lookup must return the LAST record"
        assert lookup("missing") is None, "lookup(missing) must be None"

        all_records = load_all()
        assert len(all_records) == 3, f"expected 3 records, got {len(all_records)}"
    finally:
        if tmp.exists():
            tmp.unlink()
        try:
            tmp.parent.rmdir()
        except OSError:
            pass
        MANIFEST_PATH = original_path

    print("OK manifest smoke: append/lookup/is_done/load_all all pass")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--list", action="store_true", help="print all records")
    group.add_argument("--get", metavar="EXP_ID", help="print LAST record for EXP_ID")
    parser.add_argument("--smoke", action="store_true", help="run self-test and exit")
    args = parser.parse_args()

    if args.smoke:
        return _smoke()

    if args.get is not None:
        record = lookup(args.get)
        if record is None:
            print(f"no record for exp_id={args.get!r}", file=sys.stderr)
            return 1
        print(json.dumps(record, indent=2))
        return 0

    if args.list:
        for record in load_all():
            print(json.dumps(record))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
