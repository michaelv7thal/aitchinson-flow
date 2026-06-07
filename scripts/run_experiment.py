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
# L256 capstone registry (scale a100_20g_L256, ~127M). Authored from the
# §4 DAG. The ":eval"/E4 entries below run on the EXISTING seed-42 L256
# checkpoints (no new training); E1 full-split training entries are added
# separately. Paths are relative to repo root (cwd of subprocess.run).
_R = "runs/sflm_bench_a100_20g_L256"        # shared-scale arm checkpoints
_SVGP = "runs/dfm_svgp_L256/epoch_final.pt"  # DirichletFMSvgp Stage-1 base

REGISTRY: dict[str, dict] = {
    # --- E1 generation eval on existing seed-42 L256 checkpoints ---------- #
    "E1:DFM:eval": {
        "cmd": f"python scripts/eval_all.py --ckpt {_R}/DFM/epoch_final.pt --split test --n 64 --bpc-mc 8",
        "run_dir": f"{_R}/DFM", "seeds": [42],
        "notes": "E1 DFM generation + ELBO BPC @ L256 (seed42, 100k-window ckpt). Peer D3PM-uniform 1.61.",
    },
    "E1:SFLMEBM:eval": {
        "cmd": f"python scripts/eval_all.py --ckpt {_R}/SFLMEBM/epoch_final.pt --split test --n 64",
        "run_dir": f"{_R}/SFLMEBM", "seeds": [42],
        "notes": "E1 SFLMEBM generation @ L256 (epoch15, converged); BPC = — (EBM).",
    },
    "E1:SFLMEBM_FM:eval": {
        "cmd": f"python scripts/eval_all.py --ckpt {_R}/SFLMEBM_FM/epoch_final.pt --split test --n 64",
        "run_dir": f"{_R}/SFLMEBM_FM", "seeds": [42],
        "notes": "E1 SFLMEBM_FM generation @ L256.",
    },
    # --- E4 OOD (all on existing checkpoints) ----------------------------- #
    "E4a:SVGP": {
        "cmd": (
            f"python scripts/fit_dfm_svgp_hinge.py --ckpt {_SVGP} "
            f"--out-dir {_R}/DFM_SVGP --n-epochs 5 --lr 1e-3 --eval-n 500 --seed 42 && "
            f"python scripts/sweep_dfm_svgp_corruption.py "
            f"--ckpt {_R}/DFM_SVGP/model_with_svgp_hinge.pt --n 500"
        ),
        "run_dir": f"{_R}/DFM_SVGP", "seeds": [42],
        "notes": "E4a hinge-SVGP @ L256 + corruption-ladder AUROC (shuffle headline).",
    },
    "E4b:SFLMEBM": {
        "cmd": f"python scripts/eval_ood.py --ckpt {_R}/SFLMEBM/epoch_final.pt --n 256",
        "run_dir": f"{_R}/SFLMEBM", "seeds": [42],
        "notes": "E4b native-energy OOD; expect shuffle ~0.5 (chance), sign_inverted possible.",
    },
    "E4c:DFM": {
        "cmd": f"python scripts/eval_ood.py --ckpt {_R}/DFM/epoch_final.pt --n 256",
        "run_dir": f"{_R}/DFM", "seeds": [42],
        "notes": "E4c DFM denoiser-NLL OOD (expect strong seq-level).",
    },
    "E4d:DFM": {
        "cmd": f"python scripts/eval_ood_baselines.py --ckpt {_R}/DFM/epoch_final.pt --n 128",
        "run_dir": f"{_R}/DFM", "seeds": [0],
        "notes": "E4d generative-likelihood OOD baseline + E4e controls.",
    },
    "E4f:hinge_vs_fm": {
        "cmd": f"python scripts/ablate_hinge_vs_fm.py --dfm-ckpt {_SVGP} --out {_R}/ablate_hinge_vs_fm.json",
        "run_dir": _R, "seeds": [42],
        "notes": "E4f 4-arm FM-vs-direct-hinge, replace->shuffle transfer.",
    },
    "E4g:gpt2_bench": {
        "cmd": "python scripts/bench_sflm_ebm.py --scale a100_20g_L256 --n 128 --ref-lm gpt2",
        "run_dir": _R, "seeds": [1234],
        "notes": "E4g GPT-2 spilled-energy external reference + full L256 bench table.",
    },
    # --- DirichletFM: the HIGH-PERFORMANCE generator (Stark et al. 2024) --- #
    # NB: 'DFM' above is Discrete FM (weak baseline). DirichletFM is the
    # headline generator. Its naive BPC (~0.36) is an identity/near-clean
    # artifact demoted to '—' by the <0.5 guard, so it reports generation
    # quality (KL/samples/H_ratio), not a peer-comparable BPC (DFM keeps that).
    # 5 epochs: EqM-family converge by ~2-3 epochs (confirmed on EqM/EqM_OneHot),
    # so 50ep is wasteful. 10k-window scale default, single seed 42.
    "E1:DirichletFM:train": {
        "cmd": "python scripts/train_for_sflm_bench.py --scale a100_20g_L256 --only DirichletFM --epochs 5",
        "run_dir": f"{_R}/DirichletFM", "seeds": [42],
        "notes": "E1 DirichletFM (Dirichlet Flow Matching) — headline high-perf generator; 5ep (converges fast).",
    },
    "E1:EqMLatent:train": {
        "cmd": "python scripts/train_for_sflm_bench.py --scale a100_20g_L256 --only EqMLatent --epochs 5",
        "run_dir": f"{_R}/EqMLatent", "seeds": [42],
        "notes": "E1 EqMLatent @ L256, 5ep.",
    },
    "E1:SFLM:train": {
        "cmd": "python scripts/train_for_sflm_bench.py --scale a100_20g_L256 --only SFLM --epochs 5",
        "run_dir": f"{_R}/SFLM", "seeds": [42],
        "notes": "E1 SFLM @ L256, 5ep.",
    },
    "E1:DirichletFM:eval": {
        "cmd": f"python scripts/eval_all.py --ckpt {_R}/DirichletFM/epoch_final.pt --split test --n 64",
        "run_dir": f"{_R}/DirichletFM", "seeds": [42],
        "notes": "E1 DirichletFM generation @ L256 (KL/samples; BPC = — artifact).",
    },
    "E1:EqM:eval": {
        "cmd": f"python scripts/eval_all.py --ckpt {_R}/EqM/epoch_final.pt --split test --n 64",
        "run_dir": f"{_R}/EqM", "seeds": [42],
        "notes": "E1 EqM (Dirichlet-thickened CLR) generation @ L256; BPC = — (identity path).",
    },
    "E1:EqM_OneHot:eval": {
        "cmd": f"python scripts/eval_all.py --ckpt {_R}/EqM_OneHot/epoch_final.pt --split test --n 64",
        "run_dir": f"{_R}/EqM_OneHot", "seeds": [42],
        "notes": "E1 EqM_OneHot (det. one-hot CLR) generation @ L256 — the Dirichlet-ablation arm.",
    },
    "E1:EqMLatent:eval": {
        "cmd": f"python scripts/eval_all.py --ckpt {_R}/EqMLatent/epoch_final.pt --split test --n 64",
        "run_dir": f"{_R}/EqMLatent", "seeds": [42],
        "notes": "E1 EqMLatent generation @ L256.",
    },
    "E1:SFLM:eval": {
        "cmd": f"python scripts/eval_all.py --ckpt {_R}/SFLM/epoch_final.pt --split test --n 64",
        "run_dir": f"{_R}/SFLM", "seeds": [42],
        "notes": "E1 SFLM generation @ L256.",
    },
    "E4a:sweep": {
        "cmd": f"python scripts/sweep_dfm_svgp_corruption.py --ckpt {_R}/DFM_SVGP/model_with_svgp_hinge.pt --n 500",
        "run_dir": f"{_R}/DFM_SVGP", "seeds": [42],
        "notes": "E4a corruption-ladder AUROC (hinge model + cfg sibling already on disk; fit was done).",
    },
    # Preliminary early read: the DirichletFM generator already trained inside
    # the L256 SVGP base (10k windows) — quick KL_bi check vs Discrete-FM's 1.61.
    "E1:DirichletFM:svgpbase_eval": {
        "cmd": f"python scripts/eval_all.py --ckpt {_SVGP} --split test --n 64",
        "run_dir": "runs/dfm_svgp_L256", "seeds": [42],
        "notes": "Preliminary DirichletFM (SVGP-base, 10k) generation @ L256 — early Dirichlet read.",
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
