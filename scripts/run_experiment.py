"""X2 experiment runner / wrapper.

Thin idempotent driver around the experiment REGISTRY. Each ``exp_id`` maps to a
spec describing a shell command to run (plus its ``run_dir`` and ``seeds``). The
runner consults the append-only manifest (``scripts.manifest``) to skip work that
is already ``done``, executes the command via :func:`subprocess.run`, and writes a
``done`` / ``failed`` record back to the manifest with the timing/provenance keys
required by the HARNESS INTERFACE CONTRACT.

REGISTRY entries are plain data (a ``cmd`` template + metadata) so they can be
inspected and launched later without importing torch or the model package at
module import time.

Usage:
    python scripts/run_experiment.py E1 --list-registry
    python scripts/run_experiment.py E1            # run (skip if already done)
    python scripts/run_experiment.py E1 --force    # re-run even if done
    python scripts/run_experiment.py --smoke       # CPU self-test, <60s, prints OK
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts import manifest  # noqa: E402

# ---------------------------------------------------------------------------
# Experiment registry — pure data. Each value is a spec dict with keys:
#   cmd:        shell command string (run via subprocess with shell=True)
#   run_dir:    where the experiment writes its artifacts (recorded in manifest)
#   seeds:      list[int] of seeds the command sweeps (recorded in manifest)
#   notes:      free-form description (optional)
# These reference real scripts but are intentionally NOT launched here; the
# runner only executes the exp_id passed on the CLI.
# ---------------------------------------------------------------------------
REGISTRY: dict[str, dict] = {
    # E1 — train the DFM arm (the only peer-comparable BPC) at the local scale.
    "E1:DFM": {
        "cmd": (
            "python scripts/train_for_sflm_bench.py "
            "--scale local --only DFM --epochs 50 --seeds 42,43,44"
        ),
        "run_dir": "runs/sflm_bench_local",
        "seeds": [42, 43, 44],
        "notes": "DFM arm; MC-ELBO BPC is the peer-comparable likelihood.",
    },
    # E4a — fit the post-hoc SVGP OOD head on a frozen DirichletFM checkpoint.
    "E4a:SVGP": {
        "cmd": (
            "python scripts/fit_dfm_svgp_hinge.py "
            "--ckpt runs/sflm_bench_local/DirichletFM/epoch_final.pt "
            "--n-epochs 5 --seed 42"
        ),
        "run_dir": "runs/sflm_bench_local/DirichletFM",
        "seeds": [42],
        "notes": "Hinge-trained SVGP OOD detector on frozen DirichletFM features.",
    },
}


def _git_sha() -> str:
    """Return the current commit SHA, or 'unknown' if git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or "unknown"
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _gpu_info() -> tuple[str, float]:
    """Return (gpu_name, peak_mem_gb). ('cpu', 0.0) when CUDA is unavailable.

    torch is imported lazily so the registry / --list-registry path stays cheap.
    """
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name()
            peak = torch.cuda.max_memory_allocated() / 1e9
            return name, float(peak)
    except Exception:  # noqa: BLE001 — torch missing / driver hiccup → fall back to cpu
        pass
    return "cpu", 0.0


def _run_cmd(cmd: str) -> tuple[int, float]:
    """Run ``cmd`` (shell=True), return (returncode, wall_time_s)."""
    start = time.time()
    proc = subprocess.run(cmd, shell=True, cwd=str(ROOT))
    return proc.returncode, time.time() - start


def _record(
    exp_id: str,
    spec: dict,
    *,
    status: str,
    wall_time_s: float,
    notes: str,
) -> dict:
    """Build a manifest record conforming to the interface contract."""
    gpu, peak_mem_gb = _gpu_info()
    return {
        "exp_id": exp_id,
        "status": status,
        "run_dir": spec.get("run_dir", ""),
        "seeds": spec.get("seeds", []),
        "git_sha": _git_sha(),
        "config_hash": "",
        "gpu": gpu,
        "wall_time_s": round(wall_time_s, 3),
        "peak_mem_gb": round(peak_mem_gb, 3),
        "length_fallback": None,
        "second_order": None,
        "metrics": {},
        "artifacts": [],
        "notes": notes,
    }


def run_experiment(exp_id: str, *, force: bool = False) -> int:
    """Run a single registered experiment idempotently.

    Returns a shell exit code (0 = skipped or succeeded, non-zero = failed).
    """
    if exp_id not in REGISTRY:
        known = ", ".join(sorted(REGISTRY))
        print(f"[error] unknown exp_id {exp_id!r}; known: {known}", file=sys.stderr)
        return 2

    if manifest.is_done(exp_id) and not force:
        print(f"[skip] {exp_id} already done (use --force to re-run)")
        return 0

    spec = REGISTRY[exp_id]
    cmd = spec["cmd"]
    print(f"[run] {exp_id}: {cmd}")

    rc, wall = _run_cmd(cmd)
    if rc != 0:
        # Retry exactly once before recording a failure.
        print(f"[retry] {exp_id} exited {rc}; retrying once", file=sys.stderr)
        rc, wall = _run_cmd(cmd)

    if rc == 0:
        manifest.append(
            _record(exp_id, spec, status="done", wall_time_s=wall, notes=spec.get("notes", ""))
        )
        print(f"[done] {exp_id} ({wall:.1f}s)")
        return 0

    manifest.append(
        _record(
            exp_id,
            spec,
            status="failed",
            wall_time_s=wall,
            notes=f"exit code {rc} after retry; {spec.get('notes', '')}".strip(),
        )
    )
    print(f"[failed] {exp_id} exit code {rc} (after retry)", file=sys.stderr)
    return rc


def _smoke() -> int:
    """CPU self-test: register a trivial exp, run against a TEMP manifest, verify.

    Asserts a 'done' record is appended on success and a second run skips. Prints
    OK on success.
    """
    import tempfile

    original_path = manifest.MANIFEST_PATH
    original_registry = dict(REGISTRY)
    tmp = Path(tempfile.mkdtemp(prefix="run_experiment_smoke_")) / "manifest.jsonl"
    manifest.MANIFEST_PATH = tmp

    exp_id = "SMOKE:trivial"
    REGISTRY[exp_id] = {
        "cmd": 'python -c "print(1)"',
        "run_dir": "runs/smoke",
        "seeds": [0],
        "notes": "smoke trivial exp",
    }
    try:
        assert not manifest.is_done(exp_id), "should not be done before first run"

        rc = run_experiment(exp_id)
        assert rc == 0, f"first run should succeed, got rc={rc}"

        rec = manifest.lookup(exp_id)
        assert rec is not None, "a record must have been appended"
        assert rec["status"] == "done", f"expected status 'done', got {rec['status']!r}"
        assert rec["exp_id"] == exp_id
        assert manifest.is_done(exp_id), "is_done must be True after success"
        assert isinstance(rec["wall_time_s"], (int, float)), "wall_time_s must be numeric"
        n_before = len(manifest.load_all())

        # Second run must [skip] and append nothing.
        rc2 = run_experiment(exp_id)
        assert rc2 == 0, f"second run should skip with rc=0, got {rc2}"
        assert len(manifest.load_all()) == n_before, "skip must not append a record"

        # --force must re-run and append another record.
        rc3 = run_experiment(exp_id, force=True)
        assert rc3 == 0, f"forced run should succeed, got {rc3}"
        assert len(manifest.load_all()) == n_before + 1, "force must append a record"
    finally:
        manifest.MANIFEST_PATH = original_path
        REGISTRY.clear()
        REGISTRY.update(original_registry)
        if tmp.exists():
            tmp.unlink()
        try:
            tmp.parent.rmdir()
        except OSError:
            pass

    print("OK run_experiment smoke: run/skip/force + done record all pass")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exp_id", nargs="?", help="experiment id from REGISTRY")
    parser.add_argument("--force", action="store_true", help="re-run even if done")
    parser.add_argument(
        "--list-registry", action="store_true", help="print known exp_ids and exit"
    )
    parser.add_argument("--smoke", action="store_true", help="run CPU self-test and exit")
    args = parser.parse_args()

    if args.smoke:
        return _smoke()

    if args.list_registry:
        for eid in sorted(REGISTRY):
            spec = REGISTRY[eid]
            print(f"{eid}\n    cmd: {spec['cmd']}\n    run_dir: {spec.get('run_dir', '')}")
        return 0

    if not args.exp_id:
        parser.error("exp_id is required (or pass --smoke / --list-registry)")

    return run_experiment(args.exp_id, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
