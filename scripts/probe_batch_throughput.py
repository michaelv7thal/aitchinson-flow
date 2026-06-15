"""Measure training throughput (windows/s) vs batch size for one arm/scale.

Why: increasing batch size only speeds up wall-clock if the GPU is *compute-
under-utilized* at the smaller batch (per-step time dominated by fixed launch/
dispatch/optimizer overhead). Memory headroom (e.g. 8.8/20 GB) tells you a
bigger batch will FIT, not that it will be faster. This probe settles it
empirically: it times real train steps (forward + CE + backward + opt.step) at
each candidate batch and reports windows/s, so the full-text8 driver can pick
the throughput-optimal batch instead of guessing.

Batches that OOM (without the grad-checkpointing crutch) are skipped — we only
want batches that run at full speed. Prints a table, a parseable
``BEST_BATCH=<n>`` / ``BEST_WIN_PER_S=<x>`` footer, and writes JSON.

Usage:
    python scripts/probe_batch_throughput.py \
        --scale a100_20g_L256_d1280L14_full --arm DirichletFM \
        --batches 16,24,32,40,48 --warmup 6 --iters 25 \
        --out runs/.../batch_probe.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

# scripts/ is sys.path[0] when run as `python scripts/probe_batch_throughput.py`
from train_for_sflm_bench import _model_cfg

import aitchinson_flow.models  # noqa: F401 — populate the model registry
from aitchinson_flow.models.factory import build_model


def _is_oom(err: BaseException) -> bool:
    s = str(err).lower()
    return "out of memory" in s or "nvml" in s or "cuda error" in s


def _time_batch(arm: str, scale: str, B: int, warmup: int, iters: int) -> dict:
    """Build the model at batch B and return timing stats, or an OOM marker."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _model_cfg(arm, epochs=1, scale=scale, B=B)
    L = cfg.training.L
    K = cfg.text8_dataset.K

    model = build_model(cfg).to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    # Synthetic clean token windows — DFM's training_step samples x_t ~ Dir
    # internally, so token_ids is all it needs. Fixed across iters (we measure
    # compute throughput, not data loading).
    gen = torch.Generator(device="cpu").manual_seed(0)
    token_ids = torch.randint(0, K, (B, L), generator=gen).to(device)
    batch = {"token_ids": token_ids}

    def step():
        opt.zero_grad(set_to_none=True)
        out = model.training_step(batch, 0)
        loss = out["loss"]
        loss.backward()
        opt.step()
        return float(loss.detach())

    try:
        for _ in range(warmup):
            step()
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        last_loss = 0.0
        for _ in range(iters):
            last_loss = step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        s_per_it = elapsed / iters
        peak_gb = (
            torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
        )
        result = {
            "batch": B,
            "ok": True,
            "s_per_it": round(s_per_it, 4),
            "windows_per_s": round(B / s_per_it, 2),
            "peak_gb": round(peak_gb, 2),
            "last_loss": round(last_loss, 4),
        }
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:  # type: ignore[attr-defined]
        result = {"batch": B, "ok": False, "error": type(e).__name__,
                  "oom": _is_oom(e)}
        if not _is_oom(e):
            raise
    finally:
        del model, opt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="a100_20g_L256_d1280L14_full")
    ap.add_argument("--arm", default="DirichletFM")
    ap.add_argument("--batches", default="16,24,32,40,48")
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    batches = [int(b) for b in args.batches.split(",") if b.strip()]
    print(f"# probing {args.arm} @ {args.scale}  batches={batches}  "
          f"(warmup={args.warmup}, iters={args.iters})", flush=True)

    results = []
    for B in batches:
        r = _time_batch(args.arm, args.scale, B, args.warmup, args.iters)
        if r["ok"]:
            print(f"  B={B:>3}  {r['s_per_it']:.3f} s/it  "
                  f"{r['windows_per_s']:>7.1f} win/s  peak={r['peak_gb']:.1f} GB",
                  flush=True)
        else:
            tag = "OOM" if r.get("oom") else r.get("error")
            print(f"  B={B:>3}  SKIP ({tag})", flush=True)
        results.append(r)

    ok = [r for r in results if r["ok"]]
    best = max(ok, key=lambda r: r["windows_per_s"]) if ok else None

    if best:
        base = next((r for r in ok if r["batch"] == min(r2["batch"] for r2 in ok)), best)
        speedup = best["windows_per_s"] / base["windows_per_s"]
        print(f"\n# BEST: B={best['batch']}  {best['windows_per_s']:.1f} win/s  "
              f"({speedup:.2f}x vs B={base['batch']})", flush=True)
        print(f"BEST_BATCH={best['batch']}", flush=True)
        print(f"BEST_WIN_PER_S={best['windows_per_s']}", flush=True)
    else:
        print("# BEST: none fit — all candidate batches OOM'd", flush=True)
        print("BEST_BATCH=", flush=True)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"scale": args.scale, "arm": args.arm, "results": results,
             "best": best}, indent=2))
        print(f"# wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
