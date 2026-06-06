"""E7 — Aggregate ``results/manifest.jsonl`` into ``results/RESULTS.md`` + figs.

Reads the append-only run manifest (one ``done``/``failed``/``running`` record per
experiment; schema in :mod:`scripts.manifest`) and emits the four report-ready
tables plus four figures described in §11 / E7 of
``capstone_experiment_runbook.md``:

  * **Table 1 — Generation @ L=256** — ``KL_uni/bi/tri``, ``H_ratio``, BPC-or-``—``
    overlaid on the published text8 frontier (SEDD 1.32 · D3PM-uniform 1.61 ·
    MDLM ≤1.38 · SFM 1.39 · AR 1.13–1.18).  Identity-path arms get ``—`` for BPC.
  * **Table 2 — Training-signal-class matrix** — one row per (target × data recipe)
    cell: ``KL_uni``, ``KL_bi``, ``Δ@.50``, ``collapsed``.
  * **Table 3 — Recovery Δ@α** — the per-α curve with the headline ``Δ@.50``.
  * **Table 4 — OOD** — detector × corruption-type AUROC; the **shuffle** (order)
    axis is highlighted, the predictive-**variance** AUROC is shown as
    *uninformative*, and the generative-**likelihood** baseline is shown as
    *unreliable*.

Every cell is annotated with provenance — ``exp_id`` / ``run_dir`` / ``seeds`` /
``git_sha`` — and ``collapsed`` / ``sign_inverted`` / ``length_fallback`` surface
as footnotes.  Figures are emitted with matplotlib when available (the import is
guarded; any missing metric is skipped gracefully).

Usage:
    python scripts/aggregate_results.py
    python scripts/aggregate_results.py --manifest results/manifest.jsonl \\
        --out-md results/RESULTS.md --figs results/figs
    python scripts/aggregate_results.py --smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.append(str(_ROOT))

import scripts.manifest as manifest  # noqa: E402

# Published text8 character-level BPC frontier (bits/char) the generation table
# is overlaid on.  Kept in sync with EVAL_ASSESSMENT.md / the runbook E1 row.
FRONTIER: list[tuple[str, str, str]] = [
    ("AR (Transformer)", "1.13–1.18", "autoregressive ceiling"),
    ("Plaid", "1.12", "continuous diffusion"),
    ("SEDD", "1.32", "score-entropy discrete diffusion"),
    ("MDLM", "≤1.38", "masked diffusion LM"),
    ("SFM", "1.39", "simplex/statistical FM"),
    ("D3PM-absorbing", "1.45", "discrete diffusion"),
    ("D3PM-uniform", "1.61", "uniform discrete diffusion (DFM peer)"),
]

# Arms whose BPC is a recovery artifact, never a comparable density (lock-step
# with eval_all._IDENTITY_PATH_BPC / bench_sflm_ebm).
_IDENTITY_PATH_BPC = frozenset({"EqM", "EqM_OneHot", "EqMLatent", "SFLM"})

# Phase → exp_id-prefix routing.  An exp_id like "E2a", "E2a_dsm_clr",
# "E4a.shuffle" all map to their phase by leading-prefix match.
_GEN_PREFIXES = ("E1", "E2c")
_SIGNAL_PREFIXES = ("E2a",)
_RECOVERY_PREFIXES = ("E3",)
_OOD_PREFIXES = ("E4",)

_EM = "—"  # em-dash used for absent / non-comparable cells


# --------------------------------------------------------------------------
# small formatting helpers
# --------------------------------------------------------------------------
def _fmt(value: Any, prec: int = 4) -> str:
    """Format a scalar metric cell; ``None``/NaN/inf → em-dash."""
    if value is None:
        return _EM
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        v = float(value)
        if v != v or v in (float("inf"), float("-inf")):
            return _EM
        return f"{v:.{prec}f}"
    return str(value)


def _md_escape(text: str) -> str:
    return str(text).replace("|", "\\|")


def _seeds_str(record: dict) -> str:
    seeds = record.get("seeds") or []
    if not seeds:
        return _EM
    return ",".join(str(s) for s in seeds)


def _short_sha(record: dict) -> str:
    sha = record.get("git_sha") or ""
    return sha[:8] if sha else _EM


def _provenance(record: dict) -> str:
    """One-line provenance string: exp_id · run_dir · seeds · git_sha."""
    return (
        f"{record.get('exp_id', '?')} · "
        f"`{record.get('run_dir', '?')}` · "
        f"seeds={_seeds_str(record)} · "
        f"sha={_short_sha(record)}"
    )


def _phase_match(exp_id: str, prefixes: tuple[str, ...]) -> bool:
    return any(exp_id == p or exp_id.startswith(p) for p in prefixes)


def _done(records: list[dict], prefixes: tuple[str, ...]) -> list[dict]:
    """``done`` records whose exp_id falls under one of ``prefixes``, in order."""
    return [
        r
        for r in records
        if r.get("status") == "done"
        and _phase_match(str(r.get("exp_id", "")), prefixes)
    ]


def _get(metrics: dict, *keys: str, default: Any = None) -> Any:
    """First present key from ``metrics`` (alias-tolerant)."""
    for k in keys:
        if k in metrics and metrics[k] is not None:
            return metrics[k]
    return default


# --------------------------------------------------------------------------
# Table 1 — Generation @ L=256
# --------------------------------------------------------------------------
def _table_generation(records: list[dict]) -> tuple[str, list[str]]:
    rows = _done(records, _GEN_PREFIXES)
    footnotes: list[str] = []
    lines: list[str] = ["## Table 1 — Generation @ L=256", ""]

    lines += ["**Published text8 BPC frontier (bits/char):**", ""]
    lines += ["| Method | BPC | Note |", "|---|---|---|"]
    for name, bpc, note in FRONTIER:
        lines.append(f"| {name} | {bpc} | {note} |")
    lines.append("")

    if not rows:
        lines += ["_No generation (E1/E2c) records yet._", ""]
        return "\n".join(lines), footnotes

    lines += [
        "| Arm | KL_uni | KL_bi | KL_tri | H_ratio | per-pos H | BPC | "
        "valid | flags | provenance |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    fn_idx = 1
    for r in rows:
        m = r.get("metrics", {}) or {}
        name = _get(m, "model_name", default=r.get("exp_id", "?"))
        valid = bool(_get(m, "generation_metric_valid", default=False))
        bpc = _get(m, "bpc", "BPC")
        # Never show a BPC for identity-path arms or invalid records.
        if name in _IDENTITY_PATH_BPC or not valid:
            bpc_cell = _EM
        else:
            bpc_cell = _fmt(bpc, 2)

        flags: list[str] = []
        if _get(m, "collapsed", default=False):
            flags.append("collapsed")
        lf = r.get("length_fallback")
        if lf is not None:
            flags.append(f"L→{lf}")
        flag_cell = ", ".join(flags) if flags else ""
        if flag_cell:
            flag_cell += f" [^gen{fn_idx}]"
            footnotes.append(
                f"[^gen{fn_idx}]: **{_md_escape(str(name))}** "
                f"({r.get('exp_id', '?')}): {flag_cell.split(' [^')[0]}. "
                + ("collapsed = unigram matches corpus but bigram structure "
                   "absent (KL_uni<0.05 & KL_bi>1.0). " if "collapsed" in flags
                   else "")
                + (f"length_fallback to L={lf} (OOM ladder). "
                   if lf is not None else "")
            )
            fn_idx += 1

        lines.append(
            "| "
            + " | ".join(
                [
                    _md_escape(str(name)),
                    _fmt(_get(m, "KL_uni", "unigram_kl")),
                    _fmt(_get(m, "KL_bi", "bigram_kl")),
                    _fmt(_get(m, "KL_tri", "trigram_kl")),
                    _fmt(_get(m, "H_ratio"), 3),
                    _fmt(_get(m, "per_pos_entropy", "per_pos_H"), 3),
                    bpc_cell,
                    "yes" if valid else "no",
                    flag_cell or _EM,
                    _provenance(r),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines), footnotes


# --------------------------------------------------------------------------
# Table 2 — Training-signal-class matrix
# --------------------------------------------------------------------------
def _table_signal_matrix(records: list[dict]) -> tuple[str, list[str]]:
    rows = _done(records, _SIGNAL_PREFIXES)
    footnotes: list[str] = []
    lines: list[str] = [
        "## Table 2 — Training-signal-class matrix",
        "",
        "_Fixed backbone / data / sampler; vary only the training **target** × "
        "**data recipe**. Predicted: point-prediction targets (FM-velocity, "
        "x1-point) collapse; distributional (CE/KL) and DSM escape._",
        "",
    ]
    if not rows:
        lines += ["_No training-signal-matrix (E2a) records yet._", ""]
        return "\n".join(lines), footnotes

    lines += [
        "| Target | Recipe | KL_uni | KL_bi | Δ@.50 | collapsed | provenance |",
        "|---|---|---|---|---|---|---|",
    ]
    fn_idx = 1
    for r in rows:
        m = r.get("metrics", {}) or {}
        target = _get(m, "target", "signal", "training_target", default=_EM)
        recipe = _get(m, "recipe", "data_recipe", default=_EM)
        collapsed = bool(_get(m, "collapsed", default=False))
        col_cell = "yes" if collapsed else "no"
        if collapsed:
            col_cell += f" [^sig{fn_idx}]"
            footnotes.append(
                f"[^sig{fn_idx}]: cell ({target} × {recipe}, "
                f"{r.get('exp_id', '?')}) collapsed to the unigram mode "
                "(KL_uni<0.05 & KL_bi>1.0)."
            )
            fn_idx += 1
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_escape(str(target)),
                    _md_escape(str(recipe)),
                    _fmt(_get(m, "KL_uni", "unigram_kl")),
                    _fmt(_get(m, "KL_bi", "bigram_kl")),
                    _fmt(_get(m, "delta_50", "delta@0.5", "Δ@.50"), 4),
                    col_cell,
                    _provenance(r),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines), footnotes


# --------------------------------------------------------------------------
# Table 3 — Recovery Δ@α
# --------------------------------------------------------------------------
def _alpha_curve(metrics: dict) -> dict[float, float]:
    """Extract an {alpha: delta} curve from a recovery metrics dict.

    Tolerant of two shapes: a flat ``{"delta@0.5": ...}`` map, or a
    ``{"rows": [{"alpha":..., "delta":...}, ...]}`` list (recovery_check.py).
    """
    curve: dict[float, float] = {}
    rows = metrics.get("rows")
    if isinstance(rows, list):
        for row in rows:
            a = row.get("alpha")
            d = row.get("delta")
            if a is not None and d is not None:
                curve[float(a)] = float(d)
    for k, v in metrics.items():
        if not isinstance(v, (int, float)):
            continue
        key = str(k)
        for tag in ("delta@", "delta_at_", "Δ@", "delta_"):
            if key.startswith(tag):
                suffix = key[len(tag):].replace("p", ".")
                try:
                    curve[float(suffix)] = float(v)
                except ValueError:
                    pass
                break
    return dict(sorted(curve.items()))


def _table_recovery(records: list[dict]) -> tuple[str, list[str], dict]:
    rows = _done(records, _RECOVERY_PREFIXES)
    footnotes: list[str] = []
    healing: dict = {}
    lines: list[str] = [
        "## Table 3 — Recovery Δ@α",
        "",
        "_Headline = **Δ@.50** (token_acc − token_acc_perturbed). Lead with Δ, "
        "not KL — KL rewards the deterministic ref's unigram overfit._",
        "",
    ]
    if not rows:
        lines += ["_No recovery (E3) records yet._", ""]
        return "\n".join(lines), footnotes, healing

    # union of all alphas seen, sorted
    all_alphas: set[float] = set()
    parsed: list[tuple[dict, dict[float, float]]] = []
    for r in rows:
        m = r.get("metrics", {}) or {}
        # Stash any healing-curve metrics (E3b) for the figure.
        hr = m.get("healing") or m.get("healing_curve")
        if isinstance(hr, dict):
            healing.setdefault(str(r.get("exp_id", "?")), hr)
        curve = _alpha_curve(m)
        if curve:
            parsed.append((r, curve))
            all_alphas.update(curve)

    if not parsed:
        lines += ["_Recovery records present but no Δ@α curve found in metrics._", ""]
        return "\n".join(lines), footnotes, healing

    alphas_sorted = sorted(all_alphas)
    header = (
        "| Arm | "
        + " | ".join(f"Δ@{a:.2f}" for a in alphas_sorted)
        + " | headline Δ@.50 | provenance |"
    )
    sep = "|---" * (len(alphas_sorted) + 3) + "|"
    lines += [header, sep]
    for r, curve in parsed:
        m = r.get("metrics", {}) or {}
        name = _get(m, "model_name", "arm", "recipe", default=r.get("exp_id", "?"))
        head = curve.get(0.5)
        if head is None:
            head = _get(m, "delta_50", "delta@0.5", "Δ@.50")
        cells = [_fmt(curve.get(a), 4) for a in alphas_sorted]
        lines.append(
            "| "
            + " | ".join(
                [_md_escape(str(name)), *cells, _fmt(head, 4), _provenance(r)]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines), footnotes, healing


# --------------------------------------------------------------------------
# Table 4 — OOD (detector × corruption)
# --------------------------------------------------------------------------
# AUROC-metric sub-keys, in priority order, with how they should be framed.
_OOD_DETECTORS: list[tuple[str, str, str]] = [
    ("seq_se_auroc", "spilled-energy (seq)", ""),
    ("seq_svgp_auroc", "SVGP hinge prob (seq)", ""),
    ("seq_energy_auroc", "native energy (seq)", ""),
    ("seq_energy_auroc_signfree", "native energy |sign-free|", ""),
    ("pospair_se_auroc", "spilled-energy (pos)", ""),
    ("pospair_pu_auroc", "‖∇E‖ / U_pos (pos)", ""),
    ("svgp_var_auroc", "SVGP predictive variance", "UNINFORMATIVE"),
    ("seq_elbo_auroc", "DFM ELBO likelihood", "UNRELIABLE"),
    ("seq_likelihood_auroc", "generative likelihood", "UNRELIABLE"),
]


def _collect_corruptions(auroc_maps: list[dict]) -> list[str]:
    """Union of corruption-type names, shuffle axis first, then subst, then rand."""
    names: set[str] = set()
    for amap in auroc_maps:
        names.update(k for k, v in amap.items() if isinstance(v, (int, float)))

    def _key(n: str) -> tuple[int, str]:
        if n.startswith("shuffle"):
            return (0, n)
        if n.startswith("subst"):
            return (1, n)
        if n.startswith("rand"):
            return (3, n)
        return (2, n)

    return sorted(names, key=_key)


def _table_ood(records: list[dict]) -> tuple[str, list[str], dict]:
    rows = _done(records, _OOD_PREFIXES)
    footnotes: list[str] = []
    ladder: dict = {}  # detector_label -> {corruption: auroc} for the figure
    lines: list[str] = [
        "## Table 4 — OOD detection (detector × corruption)",
        "",
        "_**Shuffle (order-preserving histogram) is the discriminating axis** — "
        "substitution/random are trivially ~1.0. Predictive **variance** is "
        "shown as *uninformative* (concentration of measure); the generative "
        "**likelihood** baseline is shown as *unreliable* (Nalisnick)._",
        "",
    ]
    if not rows:
        lines += ["_No OOD (E4) records yet._", ""]
        return "\n".join(lines), footnotes, ladder

    # Build (detector-row, exp record, auroc-map) tuples.
    det_rows: list[tuple[str, str, dict, dict]] = []
    for r in rows:
        m = r.get("metrics", {}) or {}
        for key, label, frame in _OOD_DETECTORS:
            amap = m.get(key)
            if isinstance(amap, dict) and any(
                isinstance(v, (int, float)) for v in amap.values()
            ):
                det_rows.append((label, frame, amap, r))

    if not det_rows:
        lines += ["_OOD records present but no AUROC maps found in metrics._", ""]
        return "\n".join(lines), footnotes, ladder

    corruptions = _collect_corruptions([a for _, _, a, _ in det_rows])
    header = (
        "| Detector | "
        + " | ".join(
            (f"**{c}**" if c.startswith("shuffle") else c) for c in corruptions
        )
        + " | note | provenance |"
    )
    sep = "|---" * (len(corruptions) + 3) + "|"
    lines += [header, sep]

    fn_idx = 1
    for label, frame, amap, r in det_rows:
        m = r.get("metrics", {}) or {}
        cells = [_fmt(amap.get(c), 3) for c in corruptions]
        note = frame
        # sign_inverted footnote on native-energy detectors.
        if _get(m, "sign_inverted", default=False) and "native energy" in label:
            note = (note + " " if note else "") + f"sign-inverted [^ood{fn_idx}]"
            footnotes.append(
                f"[^ood{fn_idx}]: **{label}** ({r.get('exp_id', '?')}): "
                "raw energy head is sign-inverted on this arm — the |sign-free| "
                "row recovers the magnitude."
            )
            fn_idx += 1
        lines.append(
            "| "
            + " | ".join(
                [_md_escape(label), *cells, _md_escape(note) or _EM, _provenance(r)]
            )
            + " |"
        )
        ladder.setdefault(label, {}).update(
            {c: amap[c] for c in corruptions if isinstance(amap.get(c), (int, float))}
        )
    lines.append("")
    return "\n".join(lines), footnotes, ladder


# --------------------------------------------------------------------------
# Figures (guarded matplotlib)
# --------------------------------------------------------------------------
def _make_figures(
    records: list[dict],
    ladder: dict,
    healing: dict,
    figs_dir: Path,
) -> list[str]:
    """Emit figures with matplotlib; skip silently if unavailable / no data."""
    written: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[figs] matplotlib unavailable, skipping figures: {exc}")
        return written

    figs_dir.mkdir(parents=True, exist_ok=True)

    # Fig 1 — corruption-ladder AUROC (sorted by corruption rate within type).
    if ladder:
        try:
            corruptions = _collect_corruptions(list(ladder.values()))
            if corruptions:
                fig, ax = plt.subplots(figsize=(8, 5))
                x = list(range(len(corruptions)))
                for label, amap in ladder.items():
                    ys = [amap.get(c, float("nan")) for c in corruptions]
                    ax.plot(x, ys, marker="o", label=label)
                ax.axhline(0.5, color="grey", ls="--", lw=1, label="chance")
                ax.set_xticks(x)
                ax.set_xticklabels(corruptions, rotation=45, ha="right")
                ax.set_ylabel("AUROC")
                ax.set_title("Fig 1 — corruption-ladder AUROC (shuffle = key axis)")
                ax.legend(fontsize=7, loc="lower left")
                fig.tight_layout()
                p = figs_dir / "fig1_corruption_ladder.png"
                fig.savefig(p, dpi=110)
                plt.close(fig)
                written.append(str(p))
        except Exception as exc:  # pragma: no cover
            print(f"[figs] fig1 skipped: {exc}")

    # Fig 2 — healing curve (recovery vs corruption rate).
    if healing:
        try:
            fig, ax = plt.subplots(figsize=(7, 5))
            any_pts = False
            for exp_id, curve in healing.items():
                pts = sorted(
                    (float(k), float(v))
                    for k, v in curve.items()
                    if isinstance(v, (int, float))
                )
                if pts:
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, marker="o", label=exp_id)
                    any_pts = True
            if any_pts:
                ax.set_xlabel("corruption rate r")
                ax.set_ylabel("recovery Δ")
                ax.set_title("Fig 2 — healing curve (recovery vs corruption rate)")
                ax.legend(fontsize=8)
                fig.tight_layout()
                p = figs_dir / "fig2_healing_curve.png"
                fig.savefig(p, dpi=110)
                written.append(str(p))
            plt.close(fig)
        except Exception as exc:  # pragma: no cover
            print(f"[figs] fig2 skipped: {exc}")

    # Fig 3 — field geometry (curl fraction / cos(g,g*) vs γ), if E2d present.
    geo_rows = _done(records, ("E2d",))
    if geo_rows:
        try:
            fig, ax = plt.subplots(figsize=(7, 5))
            any_pts = False
            for r in geo_rows:
                m = r.get("metrics", {}) or {}
                for field in ("curl_fraction", "cos_g_gstar", "curl", "cos"):
                    series = m.get(field)
                    if isinstance(series, dict):
                        pts = sorted(
                            (float(k), float(v))
                            for k, v in series.items()
                            if isinstance(v, (int, float))
                        )
                        if pts:
                            xs, ys = zip(*pts)
                            ax.plot(
                                xs, ys, marker="o",
                                label=f"{r.get('exp_id', '?')}:{field}",
                            )
                            any_pts = True
            if any_pts:
                ax.set_xlabel("γ")
                ax.set_ylabel("curl fraction / cos(g, g*)")
                ax.set_title("Fig 3 — field geometry vs γ")
                ax.legend(fontsize=7)
                fig.tight_layout()
                p = figs_dir / "fig3_field_geometry.png"
                fig.savefig(p, dpi=110)
                written.append(str(p))
            plt.close(fig)
        except Exception as exc:  # pragma: no cover
            print(f"[figs] fig3 skipped: {exc}")

    # Fig 4 — BPC vs frontier.
    try:
        gen_rows = _done(records, _GEN_PREFIXES)
        bpc_pts: list[tuple[str, float]] = []
        for r in gen_rows:
            m = r.get("metrics", {}) or {}
            name = str(_get(m, "model_name", default=r.get("exp_id", "?")))
            bpc = _get(m, "bpc")
            valid = bool(_get(m, "generation_metric_valid", default=False))
            if (
                valid
                and name not in _IDENTITY_PATH_BPC
                and isinstance(bpc, (int, float))
                and bpc == bpc
            ):
                bpc_pts.append((name, float(bpc)))
        # Frontier reference lines (only entries with a single numeric value).
        frontier_pts: list[tuple[str, float]] = []
        for name, bpc, _note in FRONTIER:
            try:
                frontier_pts.append((name, float(bpc.lstrip("≤"))))
            except ValueError:
                continue  # ranges like "1.13–1.18" skipped as point lines
        if bpc_pts or frontier_pts:
            fig, ax = plt.subplots(figsize=(8, 5))
            for name, val in frontier_pts:
                ax.axhline(val, color="grey", ls="--", lw=1)
                ax.text(0.0, val, f" {name} {val:.2f}", fontsize=7, va="bottom")
            if bpc_pts:
                labels, vals = zip(*bpc_pts)
                ax.bar(range(len(vals)), vals, color="#4c72b0")
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_ylabel("BPC (bits/char)")
            ax.set_title("Fig 4 — model BPC vs published text8 frontier")
            fig.tight_layout()
            p = figs_dir / "fig4_bpc_frontier.png"
            fig.savefig(p, dpi=110)
            plt.close(fig)
            written.append(str(p))
    except Exception as exc:  # pragma: no cover
        print(f"[figs] fig4 skipped: {exc}")

    return written


# --------------------------------------------------------------------------
# top-level aggregation
# --------------------------------------------------------------------------
def aggregate(
    *,
    out_md: Path,
    figs_dir: Path,
    make_figs: bool = True,
) -> dict[str, Any]:
    """Read the manifest, build RESULTS.md + figures, return a small summary."""
    records = manifest.load_all()

    n_done = sum(1 for r in records if r.get("status") == "done")
    n_failed = sum(1 for r in records if r.get("status") == "failed")
    n_running = sum(1 for r in records if r.get("status") == "running")

    parts: list[str] = []
    parts.append("# RESULTS — capstone aggregate report")
    parts.append("")
    parts.append(
        f"_Auto-generated by `scripts/aggregate_results.py` from "
        f"`{manifest.MANIFEST_PATH}` ({len(records)} records: "
        f"{n_done} done, {n_failed} failed, {n_running} running)._"
    )
    parts.append("")

    all_footnotes: list[str] = []

    t1, fn1 = _table_generation(records)
    parts.append(t1)
    all_footnotes += fn1

    t2, fn2 = _table_signal_matrix(records)
    parts.append(t2)
    all_footnotes += fn2

    t3, fn3, healing = _table_recovery(records)
    parts.append(t3)
    all_footnotes += fn3

    t4, fn4, ladder = _table_ood(records)
    parts.append(t4)
    all_footnotes += fn4

    figs_written: list[str] = []
    if make_figs:
        figs_written = _make_figures(records, ladder, healing, figs_dir)
        if figs_written:
            parts.append("## Figures")
            parts.append("")
            for p in figs_written:
                rel = Path(p)
                parts.append(f"- `{rel}`")
            parts.append("")

    if all_footnotes:
        parts.append("## Footnotes")
        parts.append("")
        parts.extend(all_footnotes)
        parts.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(parts), encoding="utf-8")

    return {
        "out_md": str(out_md),
        "n_records": len(records),
        "n_done": n_done,
        "figs": figs_written,
    }


# --------------------------------------------------------------------------
# Smoke test
# --------------------------------------------------------------------------
def _smoke() -> int:
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="aggregate_smoke_"))
    tmp_manifest = tmp / "manifest.jsonl"
    out_md = tmp / "RESULTS.md"
    figs_dir = tmp / "figs"

    # Write a temp manifest with synthetic 'done' records spanning the phases
    # E1 / E2a / E3a / E4a (+ a failed record that must be ignored).
    synthetic = [
        {
            "exp_id": "E1_dfm",
            "status": "done",
            "run_dir": "runs/dfm_L256",
            "seeds": [42, 43, 44],
            "git_sha": "deadbeefcafe",
            "config_hash": "h1",
            "gpu": "a100",
            "wall_time_s": 100.0,
            "peak_mem_gb": 18.0,
            "length_fallback": None,
            "second_order": None,
            "metrics": {
                "model_name": "DFM",
                "KL_uni": 0.012,
                "KL_bi": 0.34,
                "KL_tri": 0.88,
                "H_ratio": 0.97,
                "per_pos_entropy": 2.9,
                "bpc": 1.55,
                "generation_metric_valid": True,
                "collapsed": False,
            },
            "artifacts": ["runs/dfm_L256/eval_all.json"],
            "notes": "synthetic",
        },
        {
            "exp_id": "E1_eqm",
            "status": "done",
            "run_dir": "runs/eqm_L256",
            "seeds": [42],
            "git_sha": "deadbeefcafe",
            "config_hash": "h2",
            "gpu": "a100",
            "wall_time_s": 200.0,
            "peak_mem_gb": 19.0,
            "length_fallback": 128,
            "second_order": "checkpointed",
            "metrics": {
                "model_name": "EqM",
                "KL_uni": 0.02,
                "KL_bi": 1.8,
                "KL_tri": 3.0,
                "H_ratio": 0.6,
                "per_pos_entropy": 1.1,
                "bpc": 0.00731,  # identity-path artifact → must render as —
                "generation_metric_valid": False,
                "collapsed": True,
            },
            "artifacts": [],
            "notes": "synthetic collapse",
        },
        {
            "exp_id": "E2a_fmvel_detclr",
            "status": "done",
            "run_dir": "runs/signal_fmvel_detclr",
            "seeds": [42],
            "git_sha": "deadbeefcafe",
            "config_hash": "h3",
            "gpu": "a100",
            "wall_time_s": 50.0,
            "peak_mem_gb": 17.0,
            "length_fallback": None,
            "second_order": None,
            "metrics": {
                "target": "FM-velocity-L2",
                "recipe": "Det-CLR",
                "KL_uni": 0.03,
                "KL_bi": 1.6,
                "delta_50": 0.001,
                "collapsed": True,
            },
            "artifacts": [],
            "notes": "synthetic",
        },
        {
            "exp_id": "E3a_hilbert",
            "status": "done",
            "run_dir": "runs/recovery_hilbert",
            "seeds": [42, 43, 44],
            "git_sha": "deadbeefcafe",
            "config_hash": "h4",
            "gpu": "a100",
            "wall_time_s": 30.0,
            "peak_mem_gb": 16.0,
            "length_fallback": None,
            "second_order": None,
            "metrics": {
                "model_name": "compositional-Hilbert",
                "rows": [
                    {"alpha": 0.1, "delta": 0.02},
                    {"alpha": 0.3, "delta": 0.05},
                    {"alpha": 0.5, "delta": 0.06},
                    {"alpha": 1.0, "delta": 0.03},
                ],
            },
            "artifacts": [],
            "notes": "synthetic",
        },
        {
            "exp_id": "E4a_dfm_svgp",
            "status": "done",
            "run_dir": "runs/dfm_svgp_L256",
            "seeds": [42],
            "git_sha": "deadbeefcafe",
            "config_hash": "h5",
            "gpu": "a100",
            "wall_time_s": 40.0,
            "peak_mem_gb": 15.0,
            "length_fallback": None,
            "second_order": None,
            "metrics": {
                "seq_svgp_auroc": {
                    "shuffle_0.5": 0.93,
                    "subst_0.5": 1.0,
                    "rand": 1.0,
                },
                "svgp_var_auroc": {
                    "shuffle_0.5": 0.50,
                    "subst_0.5": 0.51,
                    "rand": 0.49,
                },
                "seq_elbo_auroc": {
                    "shuffle_0.5": 0.48,
                    "subst_0.5": 0.99,
                    "rand": 0.55,
                },
            },
            "artifacts": [],
            "notes": "synthetic",
        },
        {
            "exp_id": "E1_failed_arm",
            "status": "failed",
            "run_dir": "runs/broken",
            "seeds": [42],
            "git_sha": "deadbeefcafe",
            "config_hash": "h6",
            "gpu": "a100",
            "wall_time_s": 1.0,
            "peak_mem_gb": 0.0,
            "length_fallback": None,
            "second_order": None,
            "metrics": {},
            "artifacts": [],
            "notes": "must be ignored",
        },
    ]

    # Monkeypatch the manifest path so aggregate() reads our temp file.
    original = manifest.MANIFEST_PATH
    manifest.MANIFEST_PATH = tmp_manifest
    try:
        for rec in synthetic:
            manifest.append(rec)
        summary = aggregate(out_md=out_md, figs_dir=figs_dir, make_figs=True)
    finally:
        manifest.MANIFEST_PATH = original

    assert out_md.exists(), "RESULTS.md must be written"
    text = out_md.read_text(encoding="utf-8")
    expected_headers = [
        "## Table 1 — Generation @ L=256",
        "## Table 2 — Training-signal-class matrix",
        "## Table 3 — Recovery Δ@α",
        "## Table 4 — OOD detection",
    ]
    for h in expected_headers:
        assert h in text, f"missing table header: {h!r}"

    # The DFM real BPC must appear; the EqM identity-path BPC must NOT leak a
    # numeric density (rendered as em-dash) and its collapsed flag is footnoted.
    assert "1.55" in text, "DFM BPC 1.55 should appear in Table 1"
    # The EqM row's BPC cell must be an em-dash (identity-path → no density).
    eqm_line = next(
        (ln for ln in text.splitlines() if ln.startswith("| EqM ")), None
    )
    assert eqm_line is not None, "EqM generation row missing"
    eqm_cells = [c.strip() for c in eqm_line.strip("|").split("|")]
    # columns: Arm KL_uni KL_bi KL_tri H_ratio per-pos BPC valid flags prov
    assert eqm_cells[6] == _EM, (
        f"identity-path EqM BPC must render as em-dash, got {eqm_cells[6]!r}"
    )
    assert "collapsed" in text, "collapse flag should surface"
    # Recovery headline column present.
    assert "headline Δ@.50" in text, "recovery headline column missing"
    # Shuffle axis highlighted + variance/likelihood framing present.
    assert "shuffle" in text, "OOD shuffle axis missing"
    assert "UNINFORMATIVE" in text, "variance must be framed uninformative"
    assert "UNRELIABLE" in text, "likelihood baseline must be framed unreliable"
    # Failed record must not have produced a row.
    assert "broken" not in text, "failed records must be ignored"

    print(
        f"OK aggregate_results smoke: wrote {out_md} "
        f"({summary['n_done']} done / {summary['n_records']} records); "
        f"{len(summary['figs'])} fig(s); all 4 table headers present"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--manifest",
        type=str,
        default=str(manifest.MANIFEST_PATH),
        help="path to the JSONL manifest (default: results/manifest.jsonl)",
    )
    ap.add_argument(
        "--out-md",
        type=str,
        default="results/RESULTS.md",
        help="output markdown report path",
    )
    ap.add_argument(
        "--figs",
        type=str,
        default="results/figs",
        help="output directory for figures",
    )
    ap.add_argument(
        "--no-figs",
        action="store_true",
        help="skip figure generation (tables only)",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="self-test against a temp synthetic manifest (<60s); exits 0 on pass.",
    )
    args = ap.parse_args(argv)

    if args.smoke:
        return _smoke()

    # Point the manifest module at the requested file (load_all/append both use
    # the module-level MANIFEST_PATH).
    manifest.MANIFEST_PATH = Path(args.manifest)

    summary = aggregate(
        out_md=Path(args.out_md),
        figs_dir=Path(args.figs),
        make_figs=not args.no_figs,
    )
    print(
        f"wrote {summary['out_md']} "
        f"({summary['n_done']} done / {summary['n_records']} records); "
        f"figs: {summary['figs'] or 'none'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
