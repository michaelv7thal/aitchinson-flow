"""Reproducible HEALING (inpainting) benchmark across localizer methods.

Runs the same heal_dirichlet inpainting pipeline with the detector roster of
tab:ood-prf (NLL / LinE / LinE_all / LinE_fi / LOG_fi / Var / BGMM) on all four
corruption schemes (replace / shuffle / falseinfo / plausible), all at
pinned settings, so fix_rate / damage_rate / net_per_corrupt / loc_precision /
loc_recall are directly comparable. GPT2-SE is excluded (external scorer, no
inpainting path). Records provenance to bench_heal/manifest.json, then consolidates
into bench_heal/RESULTS.md + a figure, and folds in the real-text INSULIN Track A/C
rows (heal_poc_insulin/*.json) as the on-article validation.

    uv run python scripts/run_bench_heal.py --ckpt ...           # full
    uv run python scripts/run_bench_heal.py --ckpt ... --smoke   # tiny
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from scripts._bench_common import (  # noqa: E402
    git_sha, ckpt_md5, gpu_name, now_iso, write_manifest, finalize_manifest,
    code_fingerprint,
)

PY = [sys.executable]
_LOCALIZERS = {  # label -> extra heal_dirichlet args
    "nll": ["--localizer", "nll", "--t-nll", "3.0"],
    "blr": ["--localizer", "linear"],  # linear = BLR hinge energy head (t_eval~4.5)
    "bgmm": ["--localizer", "bgmm", "--t-eval", "7.5", "--bgmm-covariance-type",
             "full", "--bgmm-pca-dim", "64", "--bgmm-max-iter", "1000"],
    # The rest of the detector roster (tab:ood-prf), so repair can be read on the same
    # localizers detection is. Path times mirror the OOD bench arm of the same name:
    # LinE_fi at 4.5, LinE_all and Var at 7.5. LOG_fi takes its negatives at the
    # evaluation rate (heal_dirichlet reads --corrupt-rate for that arm alone), while
    # the two LinE arms keep this bench's --train-rate.
    "blr_adv": ["--localizer", "linear_all", "--t-eval", "7.5"],
    "blr_fi": ["--localizer", "linear_fi", "--t-eval", "4.5"],
    "logreg_fi": ["--localizer", "logistic_fi", "--t-eval", "4.5"],
    "var": ["--localizer", "var", "--t-eval", "7.5", "--var-ridge", "0.1"],
    # LOG_pl: the logistic head on MODEL-GUIDED plausible negatives. It is the only
    # detector that localizes the plausible swap (tab:ood-plausible), so it is the only
    # localizer whose failure to repair one is evidence about repair rather than about
    # detection. Its negatives cost --plausible-n-cands forward passes per swapped
    # word, which is why the fit windows are capped.
    "logreg_pl": ["--localizer", "logistic_pl", "--t-eval", "4.5",
                  "--plaus-fit-seqs", "128"],
}
_SCHEMES = ["replace", "shuffle", "falseinfo", "plausible"]

# Real-text (insulin article) arms. Previously these were run by hand and only READ
# back by the aggregator, which is how they drifted out of sync with the text8 arms
# (they kept the old max-F1 operating point after heal_dirichlet moved to F0.5). Run
# them here so the whole heal bench is one command with one manifest.
_INSULIN = {
    "nll": ["--localizer", "nll"],
    "gmm": ["--localizer", "gmm", "--gmm-n-components", "8",
            "--gmm-covariance-type", "diag", "--gmm-pca-dim", "64"],
    "bgmm": ["--localizer", "bgmm", "--t-eval", "7.5",
             "--bgmm-covariance-type", "full", "--bgmm-pca-dim", "64",
             "--bgmm-max-iter", "1000"],
}
_INSULIN_OUT = {"nll": "heal_insulin_poc.json", "gmm": "heal_insulin_poc_gmm.json",
                "bgmm": "heal_insulin_poc_bgmm.json"}


def _env():
    import os
    e = dict(os.environ)
    e.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return e


def _sel_row(heal_json: dict) -> dict:
    """Report the LEAST-DAMAGING operating point = max net_per_corrupt over the FPR
    sweep (best-case / capability comparison; for false-info this is still negative
    but shows the honest floor rather than the damaging max-F1 point). This is an
    oracle selection (uses demo net); the deployable GT-free choice is the cal-set
    F0.5 recorded per-run in heal_dirichlet.selected_fpr."""
    rows = heal_json.get("sweep", [])
    if not rows:
        return {}
    return max(rows, key=lambda r: (r.get("net_per_corrupt")
                                    if r.get("net_per_corrupt") is not None else -1e9))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt",
                    default="runs/sflm_bench_a100_20g_L256_d1280L14_full/"
                            "DirichletFM_converge/epoch_final.pt")  # the paper's model (md5 9946dce9...)
    ap.add_argument("--out-dir", default="bench_heal_final")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-demo", type=int, default=64)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--corrupt-rate", type=float, default=0.15)
    ap.add_argument("--nfe", type=int, default=100)
    ap.add_argument("--target-fprs", default="0.02,0.05,0.10")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-insulin", action="store_true",
                    help="skip the real-text (insulin article) arms")
    ap.add_argument("--insulin-article",
                    default="heal_poc_insulin/insulin_article_extract.json")
    ap.add_argument("--insulin-out-dir", default="heal_poc_insulin")
    ap.add_argument("--insulin-n-demo", type=int, default=32)
    ap.add_argument("--only", default="",
                    help="comma list of arm names to consider (e.g. "
                         "'var_replace,var_falseinfo'); every other arm keeps its "
                         "record from the existing manifest verbatim and is never "
                         "re-run. Use this to add arms without disturbing arms that "
                         "are already published, whose code fingerprint an edit "
                         "elsewhere in heal_dirichlet.py would otherwise invalidate.")
    ap.add_argument("--resume", action="store_true",
                    help="skip arms already recorded 'done' in the manifest WITH an "
                         "identical cmd (so an interrupted overnight run picks up where "
                         "it stopped, but stale outputs from older code are re-run)")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="skip running arms; re-aggregate existing bench_heal outputs "
                         "(e.g. after fixing the operating-point selection).")
    args = ap.parse_args()
    if args.smoke:
        args.n_demo, args.n_seeds, args.nfe = 8, 2, 30
        args.insulin_n_demo = 4

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        rec = json.loads((out / "manifest.json").read_text())
        _aggregate(out, rec)
        return 0
    common = ["--ckpt", args.ckpt, "--split", args.split,
              "--n-demo", str(args.n_demo), "--n-seeds", str(args.n_seeds),
              "--corrupt-rate", str(args.corrupt_rate), "--nfe", str(args.nfe),
              "--target-fprs", args.target_fprs, "--seed", str(args.seed)]

    arms = []  # dicts: name, kind, localizer, scheme, out, cmd, status
    for loc, loc_args in _LOCALIZERS.items():
        for scheme in _SCHEMES:
            o = str(out / f"{loc}_{scheme}.json")
            arms.append({
                "name": f"{loc}_{scheme}", "kind": "text8", "localizer": loc,
                "scheme": scheme, "out": o, "status": "pending",
                "cmd": PY + ["scripts/heal_dirichlet.py", *common, *loc_args,
                             "--corrupt-scheme", scheme, "--out", o],
            })

    if not args.skip_insulin:
        ins_dir = Path(args.insulin_out_dir)
        ins_dir.mkdir(parents=True, exist_ok=True)
        for loc, loc_args in _INSULIN.items():
            o = str(ins_dir / _INSULIN_OUT[loc])
            arms.append({
                "name": f"insulin_{loc}", "kind": "insulin", "localizer": loc,
                "scheme": "A+C", "out": o, "status": "pending",
                "cmd": PY + ["-u", "scripts/heal_protein_poc.py",
                             "--ckpt", args.ckpt,
                             "--article-json", args.insulin_article,
                             "--name", "insulin",
                             "--corrupt-rate", str(args.corrupt_rate),
                             "--n-demo", str(args.insulin_n_demo),
                             "--n-seeds", str(args.n_seeds),
                             "--nfe", str(args.nfe),
                             "--target-fprs", args.target_fprs,
                             "--seed", str(args.seed), *loc_args, "--out", o],
            })

    # --resume: carry over an arm ONLY if it is recorded `done` with an IDENTICAL cmd
    # AND was produced by the SAME CODE. The cmd alone is not enough — editing a
    # localizer changes what the arm computes while its CLI stays byte-identical, so a
    # cmd-only check would silently keep stale results. `code_fp` invalidates them.
    code_fp = code_fingerprint()
    prev = {}
    mp = out / "manifest.json"
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    if (args.resume or only) and mp.exists():
        try:
            for a in json.loads(mp.read_text()).get("arms", []):
                prev[a.get("name")] = a
        except (OSError, ValueError):
            prev = {}
    to_run, skipped, stale, carried = [], [], [], []
    for arm in arms:
        if only and arm["name"] not in only:
            # Not selected: keep whatever the manifest already records for it, and do
            # not touch the artifact. An arm with no prior record stays 'pending'.
            p = prev.get(arm["name"])
            if p:
                arm.update(p)
                carried.append(arm["name"])
            continue
        arm["code_fp"] = code_fp
        p = prev.get(arm["name"])
        if (args.resume and p and p.get("status") == "done"
                and p.get("cmd") == arm["cmd"] and p.get("code_fp") == code_fp
                and Path(arm["out"]).exists()):
            arm.update(p)  # keep its recorded status/wall_s/log
            skipped.append(arm["name"])
        else:
            if p and p.get("status") == "done":
                stale.append(arm["name"])
            to_run.append(arm)
    if carried:
        print(f"### [only] carrying {len(carried)} unselected arm(s) over untouched: "
              f"{', '.join(carried)}")
    if skipped:
        print(f"### [resume] skipping {len(skipped)} completed arm(s): "
              f"{', '.join(skipped)}")
    if stale:
        print(f"### [resume] RE-RUNNING {len(stale)} arm(s) marked done but produced by "
              f"different code/cmd: {', '.join(stale)}")

    record = {
        "benchmark": "healing_inpainting", "git_sha": git_sha(), "created": now_iso(),
        "gpu": gpu_name(), "ckpt": args.ckpt, "ckpt_md5": ckpt_md5(args.ckpt),
        "common": {"split": args.split, "n_demo": args.n_demo, "n_seeds": args.n_seeds,
                   "corrupt_rate": args.corrupt_rate, "nfe": args.nfe,
                   "target_fprs": args.target_fprs, "seed": args.seed,
                   "smoke": args.smoke, "insulin_n_demo": args.insulin_n_demo,
                   "insulin_article": args.insulin_article},
        "selection": ("deployable operating point = max cal-set F0.5 (precision-weighted; "
                      "healing is damage-averse). Benchmark reports the LEAST-DAMAGING "
                      "point = max net_per_corrupt over the FPR sweep."),
        "localizers": {k: v for k, v in _LOCALIZERS.items()},
        "insulin_localizers": {k: v for k, v in _INSULIN.items()},
        "arms": arms,
    }
    write_manifest(out, record)

    # `record["arms"]` holds ALL arms (skipped ones keep their prior 'done' record, so
    # _aggregate still sees them); only `to_run` is actually executed.
    for arm in to_run:
        log = out / f"{arm['name']}.log"
        print(f"### [{now_iso()}] START {arm['name']}")
        t0 = time.time()
        with open(log, "w") as lf:
            rc = subprocess.run(arm["cmd"], cwd=str(_ROOT), stdout=lf,
                                stderr=subprocess.STDOUT, env=_env()).returncode
        dt = round(time.time() - t0, 1)
        ok = rc == 0 and Path(arm["out"]).exists()
        # `arm` IS the dict inside record["arms"], so this updates the manifest in place
        arm.update(status="done" if ok else "failed", wall_s=dt,
                   returncode=rc, log=str(log))
        write_manifest(out, record)  # checkpoint after each arm -> resumable
        print(f"### [{now_iso()}] {'OK   ' if ok else 'FAIL '} {arm['name']} "
              f"({dt}s, rc={rc})")

    _aggregate(out, record)
    finalize_manifest(out)
    return 0


def _aggregate(out: Path, record: dict):
    """Consolidate heal arms + insulin rows into RESULTS.md + a figure."""
    table = {}    # (localizer, scheme) -> row   [synthetic text8]
    insulin = {}  # localizer -> {"A": row, "C": row}   [real article]
    for arm in record["arms"]:
        if arm["status"] != "done" or not Path(arm["out"]).exists():
            continue
        d = json.loads(Path(arm["out"]).read_text())
        if arm.get("kind") == "insulin":
            # Prefer the FPR sweep (added with the F0.5 fix) so insulin uses the SAME
            # least-damaging selection as the text8 arms. Older JSONs recorded only the
            # single (max-F1) operating point — fall back to it, but it is not
            # comparable, which is exactly how these rows went stale.
            insulin[arm["localizer"]] = {
                "A": (_sel_row({"sweep": d["track_A_sweep"]}) if d.get("track_A_sweep")
                      else d.get("track_A_char_noise", {})),
                "C": (_sel_row({"sweep": d["track_C_sweep"]}) if d.get("track_C_sweep")
                      else d.get("track_C_false_info", {})),
            }
        else:
            table[(arm["localizer"], arm["scheme"])] = _sel_row(d)

    def fmt(x):
        return "—" if x is None else f"{x:+.3f}" if isinstance(x, float) else str(x)

    L = ["# Healing (inpainting) benchmark — consolidated results\n"]
    L.append(f"- **git**: `{record['git_sha'][:12]}`  **gpu**: {record['gpu']}  "
             f"**created**: {record['created']}")
    c = record["common"]
    L.append(f"- **config**: split={c['split']} n_demo={c['n_demo']} "
             f"n_seeds={c['n_seeds']} corrupt_rate={c['corrupt_rate']} nfe={c['nfe']} "
             f"target_fprs={c['target_fprs']}")
    L.append("- localizers: " + " · ".join(
        f"{k} (`{' '.join(v)}`)" for k, v in _LOCALIZERS.items())
        + ". GPT2-SE and GPT2-NLL excluded (external scorers, no inpainting path).\n")

    L.append("Operating point = **least-damaging** (max net/corrupt over the FPR sweep). "
             "`loc_*` are the *localization* metrics — how well the detector finds the "
             "corrupt tokens; `fix`/`damage`/`net` are what the *inpainter* then does "
             "with them.\n")
    L.append("## Synthetic text8 — heal @ selected operating point\n")
    L.append("| localizer | scheme | fpr | loc_P | loc_R | loc_F1 | fix | damage | net/corrupt |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for loc in _LOCALIZERS:
        for sch in _SCHEMES:
            r = table.get((loc, sch))
            if not r:
                continue
            L.append(f"| {loc} | {sch} | {r.get('target_fpr', '—')} | "
                     f"{fmt(r.get('loc_precision'))} | {fmt(r.get('loc_recall'))} | "
                     f"{fmt(r.get('loc_f1'))} | {fmt(r.get('fix_rate'))} | "
                     f"{fmt(r.get('damage_rate'))} | {fmt(r.get('net_per_corrupt'))} |")
    L.append("")

    if insulin:
        L.append("## Real biomedical article (insulin) — Track A (char noise) / "
                 "Track C (false info)\n")
        L.append("| localizer | track | fpr | loc_P | loc_R | loc_F1 | fix | damage | net/corrupt |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for loc, tr in insulin.items():
            for tk in ["A", "C"]:
                r = tr.get(tk, {})
                if not r:
                    continue
                L.append(f"| {loc} | {tk} | {r.get('target_fpr', '—')} | "
                         f"{fmt(r.get('loc_precision'))} | {fmt(r.get('loc_recall'))} | "
                         f"{fmt(r.get('loc_f1'))} | {fmt(r.get('fix_rate'))} | "
                         f"{fmt(r.get('damage_rate'))} | {fmt(r.get('net_per_corrupt'))} |")
        L.append("")

    L.append("**Takeaway:** healing recovers *geometric* corruption (replace/char-noise) "
             "but not *contextual/false-info* — for false info the localizers flag few "
             "words and cannot restore the true token (net ≤ 0). Detection ≠ healing.\n")

    # figure: net_per_corrupt and loc_recall per (localizer, scheme)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        figs = out / "figs"
        figs.mkdir(exist_ok=True)
        locs = list(_LOCALIZERS)
        x = np.arange(len(locs))
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for ax, metric, title in [(axes[0], "net_per_corrupt", "net tokens / corrupt"),
                                  (axes[1], "loc_f1", "localization F1"),
                                  (axes[2], "loc_precision", "localization precision")]:
            for k, sch in enumerate(_SCHEMES):
                ys = [(_sel_get(table, loc, sch, metric)) for loc in locs]
                ax.bar(x + (k - 0.5) * 0.4, ys, 0.4, label=sch)
            ax.axhline(0.0 if metric == "net_per_corrupt" else 0.5, ls="--", lw=0.8,
                       color="grey")
            ax.set_xticks(x)
            ax.set_xticklabels(locs)
            ax.set_title(title)
            ax.grid(alpha=0.3, axis="y")
        axes[0].legend()
        fig.suptitle("Healing benchmark — replace (geometric) vs false-info (contextual)")
        fig.tight_layout()
        fig.savefig(figs / "heal_compare.png", dpi=150)
        plt.close(fig)
        L.append("## Figure\n- `figs/heal_compare.png`")
    except Exception as e:  # noqa: BLE001
        L.append(f"_(figure skipped: {e})_")

    (out / "RESULTS.md").write_text("\n".join(L) + "\n")
    print(f"Wrote {out / 'RESULTS.md'}")


def _sel_get(table, loc, sch, metric):
    r = table.get((loc, sch))
    v = r.get(metric) if r else None
    return float("nan") if v is None else v


if __name__ == "__main__":
    raise SystemExit(main())
