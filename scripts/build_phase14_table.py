"""Assemble the Phase 14 headline table from per-run eval.json + ood_eval.json.

Reads:  runs/<name>/eval.json (and eval_<variant>.json variants)
        runs/<name>/ood_eval.json
Writes: a markdown table to stdout (also saved to runs/phase14_table.md).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _row(label: str, eval_d: dict | None, ood_d: dict | None) -> str:
    cells = [label]
    if eval_d is None:
        cells += ["—"] * 4
    else:
        cells += [
            f"{eval_d.get('unigram_kl', float('nan')):.3f}",
            f"{eval_d.get('bigram_kl', float('nan')):.3f}",
            f"{eval_d.get('trigram_kl', float('nan')):.3f}",
            f"{eval_d.get('H_ratio', float('nan')):.3f}",
        ]
    if ood_d is None:
        cells += ["—"] * 3
    else:
        auc = ood_d.get("auc", {}).get("E_seq", {})
        for contrast in ("rand", "subst_0.5", "shuffle_0.5"):
            v = auc.get(contrast)
            cells.append(f"{v['auc']:.3f}" if isinstance(v, dict) else "—")
    return "| " + " | ".join(cells) + " |"


def _header() -> list[str]:
    return [
        "| Run | KL_uni | KL_bi | KL_tri | H_ratio | OOD-AUC clean-vs-rand | OOD-AUC clean-vs-subst_0.5 | OOD-AUC clean-vs-shuffle_0.5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]


def main() -> None:
    rows: list[str] = list(_header())

    for run, label in [
        ("eqm_data50k_ep5_v2", "eqm_data50k_ep5_v2 (NAG)"),
        ("dfm_data50k_ep5_v2", "dfm_data50k_ep5_v2 (Euler-on-tokens)"),
        ("fmclr_data50k_ep5_v2", "fmclr_data50k_ep5_v2 (Euler)"),
    ]:
        run_dir = RUNS / run
        eval_d = _load(run_dir / "eval.json")
        ood_d = _load(run_dir / "ood_eval.json")
        rows.append(_row(label, eval_d, ood_d))

    eqm_dir = RUNS / "eqm_data50k_ep5_v2"
    for variant in sorted(eqm_dir.glob("eval_euler*.json")) if eqm_dir.exists() else []:
        eval_d = _load(variant)
        rows.append(
            _row(f"eqm_data50k_ep5_v2 ({variant.stem.replace('eval_', '')})", eval_d, None)
        )

    for run, label in [
        ("bigram_joint_l03_ep5", "bigram_joint_l03_ep5 (cond.)"),
        ("bigram_joint_l10_ep5", "bigram_joint_l10_ep5 (cond.)"),
        ("longrun_winner_50ep", "longrun_winner_50ep (cond.)"),
    ]:
        run_dir = RUNS / run
        if not run_dir.exists():
            continue
        eval_d = _load(run_dir / "eval.json")
        ood_d = _load(run_dir / "ood_eval.json")
        rows.append(_row(label, eval_d, ood_d))

    table = "\n".join(rows)
    print(table)
    out = RUNS / "phase14_table.md"
    out.write_text(table + "\n")
    print(f"\n[wrote] {out}")


if __name__ == "__main__":
    main()
