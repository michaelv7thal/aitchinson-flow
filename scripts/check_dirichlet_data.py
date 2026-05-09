"""Phase 0 + Phase 2 — describe Dirichlet vs deterministic CLR data and pick alpha_peak.

Runs three steps:
  (1) Phase 0.2 — describe deterministic CLR side-by-side with Dirichlet at the
      configured alpha_peak / alpha_base.
  (2) Phase 0.3 — FM-target diagnostic at source_sigma = 0.1 (the EqM default)
      under both encodings, plus per-token x_1 variance (the headline
      "discontinuity smoothing" number).
  (3) Phase 2 — recovery-accuracy sweep over alpha_peak ∈ {10, 50, 200, 1000}
      at fixed alpha_base = 0.1; recommends the smallest peak with ≥99.5%
      argmax recovery on a held-out batch.

Run:
  python scripts/check_dirichlet_data.py
  python scripts/check_dirichlet_data.py --alpha-peaks 10,30,50,100,200
  python scripts/check_dirichlet_data.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
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

from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.data.transforms import (  # noqa: E402
    token_ids_to_features,
    token_ids_to_features_dirichlet,
)
from aitchinson_flow.data.diagnostics import (  # noqa: E402
    describe_clr,
    distance_to_token_basis,
    fm_target_diagnostic,
    per_token_x1_variance,
)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--K", type=int, default=27)
    p.add_argument("--L", type=int, default=40)
    p.add_argument("--B", type=int, default=64)
    p.add_argument("--alpha-base", type=float, default=0.1)
    p.add_argument("--alpha-peak", type=float, default=50.0)
    p.add_argument(
        "--alpha-peaks",
        type=str,
        default="10,50,200,1000",
        help="comma-separated peaks for the Phase-2 sweep",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--source-sigma",
        type=float,
        default=None,
        help="EqM source σ for the FM-target diagnostic; defaults to cfg.eqm.source_sigma",
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=str, default=None)
    return p


def _make_token_batch(B: int, L: int, K: int, *, device: torch.device) -> torch.Tensor:
    return torch.randint(0, K, (B, L), device=device, dtype=torch.long)


def _section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _print_dict(d: dict[str, float]) -> None:
    width = max(len(k) for k in d) if d else 0
    for k in sorted(d):
        print(f"  {k.ljust(width)}  {d[k]:>14.6g}")


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    cfg = Config()
    source_sigma = (
        float(args.source_sigma)
        if args.source_sigma is not None
        else float(cfg.eqm.source_sigma)
    )
    label_smoothing = float(cfg.transformation.label_smoothing)

    K, L, B = args.K, args.L, args.B
    token_ids = _make_token_batch(B, L, K, device=device)

    summary: dict[str, dict[str, float]] = {}

    # -------------- Phase 0.2 -------------- #
    _section(f"Phase 0.2 — describe CLR data (B={B}, L={L}, K={K})")
    x_det = token_ids_to_features(token_ids, K, label_smoothing=label_smoothing)
    x_dir = token_ids_to_features_dirichlet(
        token_ids, K, alpha_peak=args.alpha_peak, alpha_base=args.alpha_base
    )

    print("\n[deterministic CLR]")
    d_det = describe_clr(x_det, name="x_det")
    _print_dict(d_det)

    print(f"\n[Dirichlet CLR  α_peak={args.alpha_peak}, α_base={args.alpha_base}]")
    d_dir = describe_clr(x_dir, name="x_dir")
    _print_dict(d_dir)

    summary["describe.det"] = d_det
    summary["describe.dir"] = d_dir

    # -------------- Phase 0.3 -------------- #
    _section(f"Phase 0.3 — FM-target diagnostic (σ_source={source_sigma})")

    print("\n[deterministic CLR — basis distances]")
    nn_det = distance_to_token_basis(
        x_det, token_ids, K, label_smoothing=label_smoothing
    )
    _print_dict(nn_det)

    print("\n[Dirichlet CLR — basis distances]")
    nn_dir = distance_to_token_basis(
        x_dir, token_ids, K, label_smoothing=label_smoothing
    )
    _print_dict(nn_dir)

    print("\n[deterministic CLR — FM target stats]")
    fm_det = fm_target_diagnostic(x_det, source_sigma=source_sigma)
    _print_dict(fm_det)

    print("\n[Dirichlet CLR — FM target stats]")
    fm_dir = fm_target_diagnostic(x_dir, source_sigma=source_sigma)
    _print_dict(fm_dir)

    print("\n[per-token x_1 variance — deterministic vs Dirichlet]")
    var_det = per_token_x1_variance(x_det, token_ids, K)
    var_dir = per_token_x1_variance(x_dir, token_ids, K)
    _print_dict({**{f"det.{k}": v for k, v in var_det.items()},
                 **{f"dir.{k}": v for k, v in var_dir.items()}})

    summary["nn.det"] = nn_det
    summary["nn.dir"] = nn_dir
    summary["fm.det"] = fm_det
    summary["fm.dir"] = fm_dir
    summary["var.det"] = var_det
    summary["var.dir"] = var_dir

    # -------------- Phase 2 sweep -------------- #
    _section("Phase 2 — α_peak sweep (recovery accuracy + spread)")
    peaks = [float(p) for p in args.alpha_peaks.split(",") if p.strip()]
    sweep_rows: list[dict[str, float]] = []
    print(f"\n{'α_peak':>10} {'recover_acc':>12} {'var_norm_mean':>14} "
          f"{'l2_mean':>10} {'x1_per_tok_var':>16}")
    print("-" * 66)
    for peak in peaks:
        x = token_ids_to_features_dirichlet(
            token_ids, K, alpha_peak=peak, alpha_base=args.alpha_base
        )
        # Recovery via argmax on CLR (equivalent to argmax on probability).
        recovered = x.argmax(dim=-1)
        acc = float((recovered == token_ids).float().mean())
        d = describe_clr(x, name="x")
        v = per_token_x1_variance(x, token_ids, K)
        row = {
            "alpha_peak": peak,
            "recover_acc": acc,
            "var_norm_mean": d["x.var_norm_mean"],
            "l2_mean": d["x.l2_mean"],
            "x1_per_tok_var_mean": v["x1.per_token_l2_var_mean"],
        }
        sweep_rows.append(row)
        print(f"{peak:>10.3f} {acc:>12.5f} {d['x.var_norm_mean']:>14.4f} "
              f"{d['x.l2_mean']:>10.4f} {v['x1.per_token_l2_var_mean']:>16.4f}")

    target = 0.995
    chosen = next(
        (r["alpha_peak"] for r in sweep_rows if r["recover_acc"] >= target),
        None,
    )
    if chosen is None:
        print(f"\n[recommendation] no α_peak in sweep reached recover_acc ≥ {target:.3f}")
        print("  ⇒ widen the upper end of --alpha-peaks; current spread is too high")
    else:
        print(f"\n[recommendation] smallest α_peak with recover_acc ≥ {target:.3f}: "
              f"{chosen:g}")

    summary["phase2_sweep"] = {"rows": sweep_rows, "target_acc": target,
                                "recommended_alpha_peak": chosen}

    if args.out is not None:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"\nWrote summary to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
