"""Consolidate the OOD-detection benchmark into RESULTS.md + headline figures.

Reads every per-detector sweep JSON under a bench_ood/ tree (via
`_bench_common.DETECTOR_KEYS`), the plausible-swap JSON, and the manifest, and
emits:

  * bench_ood/RESULTS.md  — provenance header + comparison tables:
      - Claim 1: per-token AND sequence AUROC on replace/shuffle (NLL vs GPT2-SE +…)
      - Claim 2: false-info sequence AUROC-vs-rate (rises) + per-token (flat)
      - Claim 3: plausible-vs-random per detector
      - Operating point: precision / recall / F1 at a clean-calibrated threshold
        (AUROC only ranks; P/R/F1 say what you actually get when you deploy a
        threshold — and precision is base-rate sensitive, so prevalence is shown)
  * bench_ood/figs/claim1_pertoken.png, claim2_falseinfo.png, claim3_plausible.png,
    prf_f1.png

    uv run python scripts/bench_aggregate.py --bench-dir bench_ood
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from scripts._bench_common import DETECTOR_KEYS  # noqa: E402

_DETS = ["NLL", "BLR", "BLR_ADV", "BLR_FI", "BGMM", "GPT2_SE", "GPT2_NLL"]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _load(bench: Path):
    """Return {det: {(scheme,rate): {'seq':..,'token':..}}} across detectors."""
    out = {}
    for det in _DETS:
        meta = DETECTOR_KEYS[det]
        jp = bench / meta["subdir"] / meta["file"]
        if not jp.exists():
            continue
        data = json.loads(jp.read_text())
        cells = {}
        for r in data.get("rows", []):
            if not r.get("scheme"):
                continue
            cells[(r["scheme"], float(r["rate"]))] = {
                "seq": r.get(meta["seq"]), "token": r.get(meta["token"]),
                "prf_seq": r.get("prf_seq"), "prf_token": r.get("prf_token"),
                # GPT-2 arms only: native BPE granularity, needs no char attribution
                "seq_bpe": r.get("auroc_seq_se_bpe"),
                "token_bpe": r.get("auroc_token_se_bpe"),
                # WORD level: the common unit — every detector has this
                "word_max": r.get("auroc_word_max"),
                "word_mean": r.get("auroc_word_mean"),
                "word_max_full": r.get("word_max"),
                "word_mean_full": r.get("word_mean")}
        out[det] = cells
    return out


def _fmt(x):
    return "—" if x is None or (isinstance(x, float) and x != x) else f"{x:.3f}"


def _table(data, scheme, metric, dets=None):
    """Markdown table: rows = rate, cols = detectors, for one scheme+metric.

    ``dets`` restricts the columns — used to keep native-unit tables honest by never
    putting a per-CHAR AUROC and a per-BPE AUROC in the same table (they count
    different things and must not be read as a ranking).
    """
    rates = sorted({r for det in data for (s, r) in data[det] if s == scheme})
    dets = dets if dets is not None else [d for d in _DETS if d in data]
    if not rates or not dets:
        return "_(no data)_"
    lines = ["| rate | " + " | ".join(dets) + " |",
             "|---|" + "|".join(["---"] * len(dets)) + "|"]
    for rate in rates:
        row = [f"{rate:g}"]
        for d in dets:
            cell = data[d].get((scheme, rate))
            row.append(_fmt(cell.get(metric) if cell else None))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _prf_table(data, scheme, rate, fpr="0.05"):
    """P/R/F1 at a clean-calibrated FPR, per detector, for one scheme+rate.

    AUROC says how well the score RANKS corrupt above clean; these say what you get
    once you must commit to a threshold. `F1_best` is the oracle-threshold ceiling —
    the gap F1@fpr → F1_best is what better calibration could buy you.
    """
    dets = [d for d in _DETS if d in data]
    lines = ["| detector | tok P | tok R | tok F1 | tok F1_best | seq P | seq R | "
             "seq F1 | seq F1_best |",
             "|---|---|---|---|---|---|---|---|---|"]
    any_row = False
    for d in dets:
        cell = data[d].get((scheme, rate))
        if not cell or not cell.get("prf_token"):
            continue
        any_row = True
        t, s = cell.get("prf_token") or {}, cell.get("prf_seq") or {}
        tp, sp = (t.get("prf") or {}).get(fpr, {}), (s.get("prf") or {}).get(fpr, {})
        lines.append(
            f"| {d} | {_fmt(tp.get('precision'))} | {_fmt(tp.get('recall'))} | "
            f"{_fmt(tp.get('f1'))} | {_fmt(t.get('f1_best'))} | "
            f"{_fmt(sp.get('precision'))} | {_fmt(sp.get('recall'))} | "
            f"{_fmt(sp.get('f1'))} | {_fmt(s.get('f1_best'))} |")
    if not any_row:
        return "_(no P/R/F1 recorded — re-run the sweeps)_"
    prev = next((data[d][(scheme, rate)]["prf_token"].get("prevalence")
                 for d in dets if data[d].get((scheme, rate), {}).get("prf_token")), None)
    lines.append(f"\n_token prevalence (corrupt fraction) = {_fmt(prev)} — a random "
                 f"detector's precision. Threshold calibrated at {float(fpr):.0%} FPR "
                 f"on clean tokens._")
    return "\n".join(lines)


def _fig_prf(data, out, fpr="0.05"):
    """Per-token F1 at the calibrated threshold vs corruption rate, per scheme."""
    plt = _plt()
    dets = [d for d in _DETS if d in data]
    schemes = ["replace", "shuffle", "falseinfo"]
    fig, axes = plt.subplots(1, len(schemes), figsize=(15, 4.2), sharey=True)
    for ax, scheme in zip(axes, schemes):
        for d in dets:
            pts = []
            for (s, r), c in data[d].items():
                if s != scheme or not c.get("prf_token"):
                    continue
                v = ((c["prf_token"].get("prf") or {}).get(fpr, {})).get("f1")
                if v is not None and v == v:
                    pts.append((r, v))
            if pts:
                xs, ys = zip(*sorted(pts))
                ax.plot(xs, ys, marker="o", ms=4, label=d)
        ax.set_title(f"per-token F1 — {scheme}", fontsize=10)
        ax.set_xlabel("corruption rate")
        ax.set_ylim(0.0, 1.02)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"F1 @ {float(fpr):.0%} FPR (clean-calibrated)")
    axes[0].legend(fontsize=9)
    fig.suptitle("Operating-point quality — per-token F1 at a clean-calibrated threshold "
                 "(AUROC ranks; F1 is what you deploy)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _fig_claim1(data, out, metric="word_max"):
    """Localization at the COMMON unit (word). Char-level would not be a fair
    head-to-head: GPT-2 cannot score characters without an attribution heuristic."""
    plt = _plt()
    dets = [d for d in _DETS if d in data]
    schemes = ["replace", "shuffle", "falseinfo"]
    fig, axes = plt.subplots(1, len(schemes), figsize=(15, 4.4), sharey=True)
    for ax, scheme in zip(axes, schemes):
        for d in dets:
            pts = sorted((r, c[metric]) for (s, r), c in data[d].items()
                         if s == scheme and c.get(metric) is not None
                         and c[metric] == c[metric])
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="o", ms=4, label=d)
        ax.axhline(0.5, ls="--", lw=0.8, color="grey")
        ax.set_title(f"{scheme}", fontsize=11)
        ax.set_xlabel("corruption rate")
        ax.set_ylim(0.35, 1.02)
        ax.grid(alpha=0.3)
    op = metric.split("_")[-1]
    axes[0].set_ylabel(f"WORD-level AUROC ({op}-pool)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Claim 1 — localization at the common unit (word). GPT2_NLL is the fair "
                 "comparator; GPT2_SE is a sequence-level signal, not a localizer.")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _fig_claim2(data, out):
    plt = _plt()
    dets = [d for d in _DETS if d in data]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    panels = [("seq", "false-info SEQUENCE AUROC (rises with rate)"),
              ("token", "false-info PER-TOKEN AUROC (stays flat/weak)")]
    for ax, (metric, title) in zip(axes, panels):
        for d in dets:
            pts = sorted((r, c[metric]) for (s, r), c in data[d].items()
                         if s == "falseinfo" and c[metric] is not None
                         and c[metric] == c[metric])
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="o", ms=4, label=d)
        ax.axhline(0.5, ls="--", lw=0.8, color="grey")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("false-info swap rate")
        ax.set_ylim(0.4, 1.02)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("AUROC")
    axes[0].legend(fontsize=9)
    fig.suptitle("Claim 2 — false info: sequence-level triage works, per-token does not")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _fig_claim3(plausible, out):
    plt = _plt()
    import numpy as np
    rows = plausible.get("rows", [])
    # NOT filtered through _DETS: the supervised Logistic_* heads only exist here, and
    # they are the ones that actually detect plausible swaps — dropping them would
    # delete the result the figure is meant to show.
    seen, dets = set(), []
    for r in rows:
        if r["detector"] not in seen:
            seen.add(r["detector"])
            dets.append(r["detector"])
    rnd = [next((r["seq_auroc"] for r in rows
                 if r["detector"] == d and r["corruption"] == "random"), float("nan"))
           for d in dets]
    pls = [next((r["seq_auroc"] for r in rows
                 if r["detector"] == d and r["corruption"] == "plausible"), float("nan"))
           for d in dets]
    x = np.arange(len(dets))
    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(dets)), 4.4))
    ax.bar(x - 0.2, rnd, 0.4, label="random swap", color="#1f77b4")
    ax.bar(x + 0.2, pls, 0.4, label="plausible swap", color="#d62728")
    ax.axhline(0.5, ls="--", lw=0.8, color="grey")
    ax.set_xticks(x)
    ax.set_xticklabels(dets, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("sequence AUROC")
    ax.set_ylim(0.0, 1.02)
    ax.legend()
    ax.set_title("Claim 3 — plausible (model-fluent) swaps: char-model detectors → chance")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench-dir", default="bench_ood")
    ap.add_argument("--prf-fpr", default="0.05",
                    help="clean-FPR at which precision/recall/F1 are reported "
                         "(must be one of _bench_common.PRF_FPRS)")
    ap.add_argument("--prf-rate", type=float, default=0.15,
                    help="corruption rate for the P/R/F1 comparison table")
    args = ap.parse_args()
    bench = Path(args.bench_dir)
    data = _load(bench)
    figs = bench / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    manifest = {}
    mp = bench / "manifest.json"
    if mp.exists():
        manifest = json.loads(mp.read_text())

    if data:
        _fig_claim1(data, figs / "claim1_word_max.png", metric="word_max")
        _fig_claim1(data, figs / "claim1_word_mean.png", metric="word_mean")
        _fig_claim2(data, figs / "claim2_falseinfo.png")
        _fig_prf(data, figs / "prf_f1.png", fpr=args.prf_fpr)
    pl_path = bench / "plausible" / "plausible_swap.json"
    plausible = json.loads(pl_path.read_text()) if pl_path.exists() else {}
    if plausible:
        _fig_claim3(plausible, figs / "claim3_plausible.png")

    L = []
    L.append("# OOD-detection benchmark — consolidated results\n")
    c = manifest.get("common", {})
    L.append(f"- **git**: `{manifest.get('git_sha','?')[:12]}`  **gpu**: "
             f"{manifest.get('gpu','?')}  **created**: {manifest.get('created','?')}")
    L.append(f"- **ckpt**: `{manifest.get('ckpt','?')}` (md5 "
             f"`{manifest.get('ckpt_md5','')[:12]}`)")
    L.append(f"- **config**: split={c.get('split')} n={c.get('n')} "
             f"fit_seqs={c.get('fit_seqs')} seed={c.get('seed')} rates={c.get('rates')}")
    L.append(f"- **per-detector t**: {json.dumps(manifest.get('detector_t', {}))}\n")

    L.append("## Claim 1 — localization")
    L.append(
        "> **Evaluation protocol.** Each model is scored in the unit it ACTUALLY operates\n"
        "> in, and the cross-model comparison happens at a common unit:\n"
        "> * flow-matching detectors (NLL / BLR / BGMM) → native unit = **character**\n"
        "> * GPT-2 baselines (spilled energy, NLL) → native unit = **BPE token**\n"
        "> * both are pooled up to **words** → the fair head-to-head\n"
        ">\n"
        "> No score is ever attributed *downward* (BPE→char): GPT-2 has no per-character\n"
        "> opinion, and the old \"spread a token's score uniformly over its chars\" rule\n"
        "> smeared localization. Pooling *up* is exact, so words are the comparison unit —\n"
        "> and words are what the corruptions actually use (`falseinfo`/`plausible` swap\n"
        "> whole words). GPT-2's pre-tokenizer splits on whitespace, so a BPE token never\n"
        "> crosses a word boundary: the word mapping is exact on both sides.\n")
    L.append(
        "> **Two GPT-2 baselines, not interchangeable.** `GPT2_SE` is the *real* cross-step\n"
        "> spilled energy (Minut et al., ICLR 2026): it pairs the logit energy at step i-1\n"
        "> with the marginal energy at step i, so it straddles two decoding steps **by\n"
        "> construction** — a sequence-level signal, not a localizer. `GPT2_NLL` is the same\n"
        "> LM scored with the same-step per-token NLL: that is the fair localization\n"
        "> comparator, and is what this repo previously reported *mislabelled* as spilled\n"
        "> energy. Beating SE at localization would be a straw man — compare to **GPT2_NLL**.\n")

    L.append("### 1a. Native units (each model in its own tokenization)\n")
    L.append("_**Not cross-comparable** — a char AUROC and a BPE AUROC count different "
             "things. These say how well each model localizes in the units it actually "
             "has. Use the WORD table (1c) for the head-to-head._\n")
    fm = [d for d in _DETS if d in data and DETECTOR_KEYS[d].get("unit") == "char"]
    lm = [d for d in _DETS if d in data and DETECTOR_KEYS[d].get("unit") == "bpe"]
    for scheme in ["replace", "shuffle", "falseinfo"]:
        if fm:
            L.append(f"**{scheme} — flow-matching detectors, per-CHARACTER AUROC**\n")
            L.append(_table(data, scheme, "token", dets=fm) + "\n")
        if lm:
            L.append(f"**{scheme} — GPT-2 baselines, per-BPE-TOKEN AUROC**\n")
            L.append(_table(data, scheme, "token", dets=lm) + "\n")

    L.append("### 1b. Sequence level (comparable: per-character normalisation)\n")
    L.append("_Cross-tokenizer sequence scores are made comparable the standard way — "
             "total score divided by the number of CHARACTERS (the bits-per-character "
             "convention). A mean over BPE tokens would NOT be comparable: corrupted text "
             "tokenises into more BPE tokens, which dilutes it._\n")
    for scheme in ["replace", "falseinfo"]:
        L.append(f"**{scheme} — sequence AUROC**\n")
        L.append(_table(data, scheme, "seq") + "\n")

    L.append("### 1c. WORD level — the fair head-to-head (both FM and GPT-2)\n")
    L.append("_`max`-pool asks \"is ANY part of this word surprising?\" (suits a single "
             "replaced character); `mean`-pool asks \"is it surprising on average?\" (suits "
             "weak signal spread over the whole word, which is the false-info regime). "
             "Both are reported rather than picking the flattering one._\n")
    for scheme in ["replace", "shuffle", "falseinfo"]:
        for op in ("max", "mean"):
            L.append(f"**{scheme} — WORD AUROC ({op}-pool)**\n")
            L.append(_table(data, scheme, f"word_{op}") + "\n")

    # attribution-free cross-check for the GPT-2 arms
    gp = [d for d in ("GPT2_SE", "GPT2_NLL") if d in data]
    if gp:
        L.append("**GPT-2 arms at native BPE granularity (no char attribution)**\n")
        L.append("_The per-char numbers above need a BPE->char rule (`boundary`). These do not:_\n"
                 "_they score GPT-2's own tokens, so they show whether a weak per-char result is_\n"
                 "_an attribution artifact or real._\n")
        for scheme in ["replace", "shuffle", "falseinfo"]:
            rts = sorted({r for d in gp for (s, r) in data[d] if s == scheme})
            if not rts:
                continue
            L.append(f"| {scheme} rate | " + " | ".join(
                f"{d} seq | {d} token" for d in gp) + " |")
            L.append("|---|" + "|".join(["---"] * (2 * len(gp))) + "|")
            for rate in rts:
                row = [f"{rate:g}"]
                for d in gp:
                    c = data[d].get((scheme, rate)) or {}
                    row += [_fmt(c.get("seq_bpe")), _fmt(c.get("token_bpe"))]
                L.append("| " + " | ".join(row) + " |")
            L.append("")

    L.append("## Claim 2 — false info: sequence triage vs per-token")
    L.append("_seq AUROC rises with rate (human-in-the-loop); per-token stays flat; "
             "see figs/claim2_falseinfo.png_\n")
    L.append("**falseinfo — sequence AUROC**\n")
    L.append(_table(data, "falseinfo", "seq") + "\n")
    L.append("**falseinfo — per-token AUROC**\n")
    L.append(_table(data, "falseinfo", "token") + "\n")

    L.append("## Claim 3 — plausible (model-fluent) substitutions")
    L.append("_char-model detectors (NLL/BLR/BGMM) → chance on plausible; "
             "see figs/claim3_plausible.png_\n")
    if plausible:
        sp = plausible.get("surprise", {})
        L.append(f"- swapped-char denoiser NLL: random={sp.get('random_swapped_mean_nll'):.3f} "
                 f"plausible={sp.get('plausible_swapped_mean_nll'):.3f} "
                 f"(clean={sp.get('clean_mean_nll'):.3f})\n")
        pdets = sorted({r["detector"] for r in plausible["rows"]},
                       key=lambda d: (d not in _DETS, d))
        L.append("AUROC (ranking) and per-token F1 at the clean-calibrated threshold "
                 "(deployment):\n")
        L.append("| detector | random seq | plausible seq | random tok | plausible tok | "
                 "random tok F1 | plausible tok F1 |")
        L.append("|---|---|---|---|---|---|---|")
        for d in pdets:
            def g(corr, k, _d=d):
                return next((r.get(k) for r in plausible["rows"]
                             if r["detector"] == _d and r["corruption"] == corr), None)

            def gf1(corr, _d=d):
                m = g(corr, "prf_token", _d)
                if not m:
                    return None
                return ((m.get("prf") or {}).get(args.prf_fpr, {})).get("f1")

            if g("random", "seq_auroc") is None:
                continue
            L.append(f"| {d} | {_fmt(g('random','seq_auroc'))} | "
                     f"{_fmt(g('plausible','seq_auroc'))} | "
                     f"{_fmt(g('random','token_auroc'))} | "
                     f"{_fmt(g('plausible','token_auroc'))} | "
                     f"{_fmt(gf1('random'))} | {_fmt(gf1('plausible'))} |")
        L.append("")

    L.append("## Operating point — precision / recall / F1")
    L.append(f"_AUROC is threshold-free (pure ranking). These are what a **deployed** "
             f"detector gives once it must commit to a threshold, calibrated GT-free at "
             f"{float(args.prf_fpr):.0%} FPR on clean data — the same convention the "
             f"healing bench uses for `loc_precision`/`loc_recall`, so the two benches "
             f"are directly comparable. `F1_best` is the oracle-threshold ceiling. "
             f"See figs/prf_f1.png._\n")
    for scheme in ["replace", "shuffle", "falseinfo"]:
        L.append(f"**{scheme} @ rate={args.prf_rate:g}**\n")
        L.append(_prf_table(data, scheme, args.prf_rate, fpr=args.prf_fpr) + "\n")

    L.append("## Figures")
    for f in ["claim1_word_max.png", "claim1_word_mean.png", "claim2_falseinfo.png",
              "claim3_plausible.png", "prf_f1.png"]:
        if (figs / f).exists():
            L.append(f"- `figs/{f}`")

    out_md = bench / "RESULTS.md"
    out_md.write_text("\n".join(L) + "\n")
    print(f"Wrote {out_md}")
    for f in sorted(figs.glob("*.png")):
        print(f"Wrote {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
