#!/usr/bin/env python3
"""Training history of the full-corpus Dirichlet FM model, and the comparison of
its best-validation checkpoint against the fully annealed one.

The full-corpus model (runs/sflm_bench_a100_20g_L256_d1280L14_full) was trained
in four legs, each warm-started from the previous one.  The trainer wrote no
history file for these runs, so part 1 parses the per-epoch train / validation
loss out of the four tqdm logs under ``_driver/``: the ``epochs:`` bar state
``k/N`` carries, in its postfix, the metrics of epoch ``k+1`` (set_postfix runs
before the bar increments), so epoch ``e`` reads the last bar line at state
``e-1``.  The result is cross-checked against the trainer's own
``[fit] restored best val checkpoint (ep10, val=0.8571)`` line and the driver's
``epoch_best_ep5_val0p8554.pt`` snapshot name.

Part 2 reads the detection and healing benches that were run on the two
checkpoints, ``bench_ood`` / ``bench_heal`` (DirichletFM/epoch_best.pt, epoch 15,
the best-validation checkpoint) and ``bench_ood_final`` / ``bench_heal_final``
(DirichletFM_converge/epoch_final.pt, epoch 23, the fully annealed model of the
paper), verifies the checkpoints against the bench manifests, and tabulates the
headline readouts side by side.

Outputs (all committed):
  runs/sflm_bench_a100_20g_L256_d1280L14_full/anneal_history.json
  bench_ood_final/_compare/checkpoint_comparison.json
  docs/anneal_and_checkpoint_comparison.md

Usage:
  uv run python scripts/anneal_summary.py          # write the three files
  uv run python scripts/anneal_summary.py --tex    # also print LaTeX table bodies
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/sflm_bench_a100_20g_L256_d1280L14_full"
DRV = RUN / "_driver"
HISTORY_OUT = RUN / "anneal_history.json"
COMPARE_OUT = ROOT / "bench_ood_final/_compare/checkpoint_comparison.json"
DOC_OUT = ROOT / "docs/anneal_and_checkpoint_comparison.md"

# One entry per training leg, in order.  ``batch`` is the batch size the leg
# actually ran at (leg 2 was launched at 48, hit out-of-memory at 48 plain and
# at 48 with gradient checkpointing, and ran at 24 with gradient checkpointing).
LEGS = [
    dict(leg=1, arm="DirichletFM", log="train_DirichletFM.log", batch=48, lr_peak=5.196e-4,
         schedule="1-epoch warmup, cosine over 30 planned epochs",
         stop="96 h wall-clock cap, after epoch 10",
         kept="DirichletFM/epoch_final.pt = early-stopping best of the leg (epoch 10)"),
    dict(leg=2, arm="DirichletFM_continue", log="train_DirichletFM_continue.log", batch=24, lr_peak=5.196e-4,
         schedule="warm start from leg 1; 1-epoch warmup, cosine over 10 planned epochs; "
                  "launched at batch 48, out of memory twice, ran at 24 with gradient checkpointing",
         stop="85.53 h wall-clock cap, after epoch 5 (epoch 15 overall)",
         kept="DirichletFM/epoch_best.pt = early-stopping best of the leg (epoch 15) = the best-validation checkpoint"),
    dict(leg=3, arm="DirichletFM_converge", log="train_DirichletFM_converge.log", batch=16, lr_peak=1.5e-4,
         schedule="warm start from leg 2's best; no warmup, cosine to 0 over 8 planned epochs; "
                  "early stopping armed (patience 3) and never triggered",
         stop="stopped after epoch 4 (epoch 19 overall); the last epoch's weights were carried forward",
         kept="none (the epoch-4 snapshot was consumed by leg 4)"),
    dict(leg=4, arm="DirichletFM_converge (tail)", log="train_DirichletFM_converge_tail.log", batch=16, lr_peak=7.5e-5,
         schedule="warm start from leg 3's last epoch; no warmup, cosine to 0 over 4 epochs; no early stopping",
         stop="ran to completion (epoch 23 overall)",
         kept="DirichletFM_converge/epoch_final.pt = last epoch (epoch 23) = the fully annealed model of the paper"),
]

CHECKPOINTS = {
    "best_val": dict(label="best-validation (epoch 15)", epoch=15,
                     ckpt="runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM/epoch_best.pt",
                     ood="bench_ood", heal="bench_heal"),
    "annealed": dict(label="fully annealed (epoch 23)", epoch=23,
                     ckpt="runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM_converge/epoch_final.pt",
                     ood="bench_ood_final", heal="bench_heal_final"),
}
# paper name -> (detector sweep file, per-character AUROC field, healing file stem)
DETECTORS = {
    "NLL": ("nll/denoiser_nll_sweep.json", "auroc_token_nll", "nll"),
    "LinE": ("blr/bayes_linear_sweep.json", "auroc_token_energy", "blr"),
    "BGMM": ("bgmm/bgmm_perpos_sweep.json", "auroc_token_gmm", "bgmm"),
}
RATE = 0.15        # the paper's headline corruption rate (tab:ood-word, tab:heal-synth)
TARGET_FPR = 0.02  # the least-damaging swept operating point (tab:heal-synth)
SCHEMES = ("replace", "shuffle", "falseinfo")

BAR = re.compile(r"epochs:\s+\d+%\|[^|]*\|\s+(\d+)/(\d+)\s+\[([^\]]*)\]")
KV = re.compile(r"([A-Za-z_Δ]+)=([^,\s\]]+)")


def parse_log(path: Path) -> list[dict]:
    """One row per completed epoch of one leg, from the tqdm ``epochs`` bar."""
    text = path.read_text(errors="replace")
    last: dict[int, str] = {}
    for m in BAR.finditer(text):
        k, inner = int(m.group(1)), m.group(3)
        if "train=" in inner:
            last[k] = inner
    if not last:
        raise SystemExit(f"{path}: no epochs bar with a train loss")
    completed = max(last)
    rows = []
    for e in range(1, completed + 1):
        if (e - 1) not in last:
            raise SystemExit(f"{path}: bar state {e - 1} never printed with metrics")
        kv = dict(KV.findall(last[e - 1]))
        rows.append(dict(
            epoch_in_leg=e,
            train_loss=float(kv["train"]),
            val_loss=float(kv["val"]) if "val" in kv else None,
            lr_bar=float(kv["lr"]) if "lr" in kv else None,
            early_stop_counter=kv.get("es"),
        ))
    return rows


def build_history() -> dict:
    rows, legs = [], []
    epoch = 0
    for leg in LEGS:
        lrows = parse_log(DRV / leg["log"])
        first = epoch + 1
        for r in lrows:
            epoch += 1
            rows.append(dict(epoch=epoch, leg=leg["leg"], arm=leg["arm"], batch=leg["batch"], **r))
        legs.append(dict(**{k: v for k, v in leg.items()}, epochs=len(lrows), first_epoch=first, last_epoch=epoch))

    # cross-checks against the trainer's and the driver's own words
    fit = re.search(r"\[fit\] restored best val checkpoint \(ep(\d+), val=([0-9.]+)\)",
                    (DRV / "train_DirichletFM.log").read_text(errors="replace"))
    assert fit and int(fit.group(1)) == 10, fit
    assert abs(rows[9]["val_loss"] - float(fit.group(2))) < 5e-5, (rows[9], fit.group(2))
    snap = re.search(r"epoch_best_ep(\d+)_val0p(\d+)\.pt", (DRV / "converge2.out").read_text(errors="replace"))
    assert snap and int(snap.group(1)) == 5, snap
    leg2 = [r for r in rows if r["leg"] == 2]
    best2 = min(leg2, key=lambda r: r["val_loss"])
    assert best2["epoch_in_leg"] == 5 and abs(best2["val_loss"] - float("0." + snap.group(2))) < 5e-5, (best2, snap.group(0))
    assert rows[-1]["epoch"] == 23, rows[-1]

    by_epoch = {r["epoch"]: r for r in rows}
    checkpoints = {
        10: "DirichletFM/epoch_final.pt (ten-epoch checkpoint; generation eval only)",
        15: "DirichletFM/epoch_best.pt (best-validation checkpoint; bench_ood, bench_heal)",
        23: "DirichletFM_converge/epoch_final.pt (fully annealed; the paper's model)",
    }
    for e, what in checkpoints.items():
        by_epoch[e]["checkpoint"] = what
    anneal = [r for r in rows if r["leg"] >= 3]
    train = [r["train_loss"] for r in anneal]
    summary = dict(
        n_epochs=23,
        val_at_epoch_10=rows[9]["val_loss"],
        val_at_epoch_15=rows[14]["val_loss"],
        val_at_epoch_23=rows[-1]["val_loss"],
        anneal_first_epoch=anneal[0]["epoch"],
        anneal_train_first=train[0], anneal_train_last=train[-1],
        anneal_train_monotone=all(a > b for a, b in zip(train, train[1:])),
        anneal_val_min=min(r["val_loss"] for r in anneal),
        anneal_val_min_epoch=min(anneal, key=lambda r: r["val_loss"])["epoch"],
        anneal_val_max=max(r["val_loss"] for r in anneal),
        anneal_val_max_epoch=max(anneal, key=lambda r: r["val_loss"])["epoch"],
        lowest_val_overall_epoch=min(rows, key=lambda r: r["val_loss"])["epoch"],
        lowest_val_overall=min(r["val_loss"] for r in rows),
    )
    return dict(
        generated_by="scripts/anneal_summary.py",
        source_logs=[str((DRV / l["log"]).relative_to(ROOT)) for l in LEGS],
        note=("train_loss = mean training loss of the epoch; val_loss = mean validation loss on the "
              "held-out validation windows the trainer evaluates once per epoch (each evaluation draws "
              "fresh path times and Dirichlet samples, so it is a noisy readout); lr_bar = the learning "
              "rate printed in the trainer's progress bar for that epoch, as logged."),
        legs=legs, checkpoints=checkpoints, summary=summary, rows=rows,
    )


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def pick(rows: list[dict], **pred) -> dict:
    hits = [r for r in rows if all(
        (abs(float(r.get(k)) - float(v)) < 1e-9) if isinstance(v, float) else r.get(k) == v
        for k, v in pred.items())]
    assert len(hits) == 1, (pred, len(hits))
    return hits[0]


def build_comparison() -> dict:
    ckpts = {}
    for key, c in CHECKPOINTS.items():
        info = dict(c)
        for tree in (c["ood"], c["heal"]):
            man = json.loads((ROOT / tree / "manifest.json").read_text())
            assert man["ckpt"] == c["ckpt"], (tree, man["ckpt"])
            info.setdefault("manifest_md5", man["ckpt_md5"])
            assert man["ckpt_md5"] == info["manifest_md5"], (tree, man["ckpt_md5"])
        p = ROOT / c["ckpt"]
        info["ckpt_on_disk"] = p.is_file()
        if p.is_file():
            info["md5_on_disk"] = md5(p)
            assert info["md5_on_disk"] == info["manifest_md5"], (key, info)
        ckpts[key] = info

    rows = []
    for det, (sweep, char_field, heal_stem) in DETECTORS.items():
        for key, c in CHECKPOINTS.items():
            sw = json.loads((ROOT / c["ood"] / sweep).read_text())["rows"]
            row = dict(localizer=det, checkpoint=key, epoch=c["epoch"])
            for s in SCHEMES:
                r = pick(sw, scheme=s, rate=RATE)
                row[f"char_auroc_{s}"] = r[char_field]
                row[f"word_auroc_max_{s}"] = r["auroc_word_max"]
                row[f"word_auroc_mean_{s}"] = r["auroc_word_mean"]
            for s in ("replace", "falseinfo"):
                h = json.loads((ROOT / c["heal"] / f"{heal_stem}_{s}.json").read_text())
                assert abs(h["corrupt_rate"] - RATE) < 1e-9, h["corrupt_rate"]
                hr = pick(h["sweep"], target_fpr=TARGET_FPR)
                row[f"heal_{s}_loc_f1"] = hr["loc_f1"]
                row[f"heal_{s}_fix"] = hr["fix_rate"]
                row[f"heal_{s}_damage"] = hr["damage_rate"]
                row[f"heal_{s}_net"] = hr["net_per_corrupt"]
            rows.append(row)

    # the GPT-2 arms do not touch the checkpoint: byte-identical rows are the control
    gpt2 = {}
    for f in ("gpt2_nll/gpt2_nll_sweep.json", "gpt2_se/gpt2_spilled_energy_sweep.json"):
        a = json.loads((ROOT / "bench_ood" / f).read_text())["rows"]
        b = json.loads((ROOT / "bench_ood_final" / f).read_text())["rows"]
        gpt2[f] = a == b

    deltas = []
    for det in DETECTORS:
        a = pick(rows, localizer=det, checkpoint="best_val")
        b = pick(rows, localizer=det, checkpoint="annealed")
        deltas.append(dict(localizer=det, **{k: round(b[k] - a[k], 4) for k in a
                                             if isinstance(a[k], float) and k not in ("epoch",)}))
    return dict(
        generated_by="scripts/anneal_summary.py",
        what=("Headline detection and repair readouts of the best-validation checkpoint (epoch 15, "
              "bench_ood / bench_heal) against the fully annealed model (epoch 23, bench_ood_final / "
              "bench_heal_final), both at corruption rate 0.15 on the n=256 test windows; repair at the "
              "least-damaging swept operating point (target FPR 0.02, n_demo=64, 3 seeds)."),
        rate=RATE, target_fpr=TARGET_FPR, checkpoints=ckpts, gpt2_rows_identical=gpt2,
        rows=rows, deltas_annealed_minus_best_val=deltas,
    )


def fmt(x: float, nd: int = 3, sign: bool = False) -> str:
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def write_doc(hist: dict, comp: dict) -> None:
    L = []
    L.append("# The full-corpus Dirichlet FM model: training legs and the checkpoint choice\n")
    L.append("Generated by `scripts/anneal_summary.py` from the four training logs under "
             "`runs/sflm_bench_a100_20g_L256_d1280L14_full/_driver/` and the bench trees named below. "
             "Machine-readable copies: `runs/sflm_bench_a100_20g_L256_d1280L14_full/anneal_history.json` "
             "and `bench_ood_final/_compare/checkpoint_comparison.json`.\n")
    L.append("## Training legs\n")
    L.append("| leg | arm | epochs (overall) | batch | peak lr | schedule | end | checkpoint kept |")
    L.append("|---|---|---|---|---|---|---|---|")
    for g in hist["legs"]:
        L.append(f"| {g['leg']} | {g['arm']} | {g['epochs']} ({g['first_epoch']}–{g['last_epoch']}) | {g['batch']} | "
                 f"{g['lr_peak']:.2e} | {g['schedule']} | {g['stop']} | {g['kept']} |")
    L.append("")
    L.append("## Per-epoch losses\n")
    L.append(hist["note"] + "\n")
    L.append("| epoch | leg | batch | train loss | val loss | lr (bar) | checkpoint |")
    L.append("|---|---|---|---|---|---|---|")
    for r in hist["rows"]:
        L.append(f"| {r['epoch']} | {r['leg']} | {r['batch']} | {r['train_loss']:.4f} | {r['val_loss']:.4f} | "
                 f"{r['lr_bar']:.2e} | {r.get('checkpoint', '')} |")
    s = hist["summary"]
    L.append("")
    L.append(f"Over the anneal (epochs {s['anneal_first_epoch']}–23) the training loss fell "
             f"{'monotonically' if s['anneal_train_monotone'] else 'NON-monotonically'} from "
             f"{s['anneal_train_first']:.4f} to {s['anneal_train_last']:.4f}, while the validation readout moved between "
             f"{s['anneal_val_min']:.4f} (epoch {s['anneal_val_min_epoch']}) and {s['anneal_val_max']:.4f} "
             f"(epoch {s['anneal_val_max_epoch']}). The best-validation checkpoint retained by early stopping is epoch 15 "
             f"(val {s['val_at_epoch_15']:.4f}); the anneal legs kept no per-epoch checkpoint, so the lower single readings "
             f"inside the anneal were never candidate models. The released fully annealed weights (epoch 23) read "
             f"{s['val_at_epoch_23']:.4f}.\n")
    L.append("## Best-validation checkpoint (epoch 15) against the fully annealed model (epoch 23)\n")
    L.append(comp["what"] + "\n")
    for key, c in comp["checkpoints"].items():
        L.append(f"- `{key}`: `{c['ckpt']}` (md5 `{c['manifest_md5']}`; benches `{c['ood']}`, `{c['heal']}`)")
    L.append(f"- GPT-2 arms byte-identical across the two trees (they never load the checkpoint): {comp['gpt2_rows_identical']}\n")
    L.append("| localizer | checkpoint | char AUROC replace | char AUROC shuffle | char AUROC false-info | "
             "word AUROC (max) replace | word AUROC (max) shuffle | word AUROC (max) false-info | "
             "repair replace: loc F1 / fix / damage / net | repair false-info: loc F1 / fix / damage / net |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in comp["rows"]:
        L.append(f"| {r['localizer']} | {CHECKPOINTS[r['checkpoint']]['label']} | "
                 f"{r['char_auroc_replace']:.3f} | {r['char_auroc_shuffle']:.3f} | {r['char_auroc_falseinfo']:.3f} | "
                 f"{r['word_auroc_max_replace']:.3f} | {r['word_auroc_max_shuffle']:.3f} | {r['word_auroc_max_falseinfo']:.3f} | "
                 f"{r['heal_replace_loc_f1']:.3f} / {r['heal_replace_fix']:.3f} / {r['heal_replace_damage']:.3f} / {r['heal_replace_net']:+.3f} | "
                 f"{r['heal_falseinfo_loc_f1']:.3f} / {r['heal_falseinfo_fix']:.3f} / {r['heal_falseinfo_damage']:.3f} / {r['heal_falseinfo_net']:+.3f} |")
    L.append("")
    L.append("Deltas (annealed minus best-validation):\n")
    L.append("| localizer | char replace | char shuffle | char false-info | word-max false-info | net repair replace | net repair false-info |")
    L.append("|---|---|---|---|---|---|---|")
    for d in comp["deltas_annealed_minus_best_val"]:
        L.append(f"| {d['localizer']} | {d['char_auroc_replace']:+.3f} | {d['char_auroc_shuffle']:+.3f} | {d['char_auroc_falseinfo']:+.3f} | "
                 f"{d['word_auroc_max_falseinfo']:+.3f} | {d['heal_replace_net']:+.3f} | {d['heal_falseinfo_net']:+.3f} |")
    L.append("")
    DOC_OUT.write_text("\n".join(L) + "\n")


def print_tex(hist: dict, comp: dict) -> None:
    print("% ---- tab:anneal-history body (epoch & leg & batch & train & val)")
    for r in hist["rows"]:
        mark = {10: r"$^{a}$", 15: r"$^{b}$", 23: r"$^{c}$"}.get(r["epoch"], "")
        print(f"\t\t{r['epoch']}{mark} & {r['leg']} & {r['batch']} & {r['train_loss']:.3f} & {r['val_loss']:.3f} \\\\")
    print("% ---- tab:ckpt-compare body")
    for r in comp["rows"]:
        lab = "best-validation" if r["checkpoint"] == "best_val" else "fully annealed"
        print(f"\t\t{r['localizer']} & {lab} & {r['char_auroc_replace']:.3f} & {r['char_auroc_shuffle']:.3f} & "
              f"{r['word_auroc_max_falseinfo']:.3f} & {r['heal_replace_fix']:.3f} & {r['heal_replace_net']:+.3f} & "
              f"{r['heal_falseinfo_net']:+.3f} \\\\")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tex", action="store_true", help="also print LaTeX table bodies")
    args = ap.parse_args()
    hist = build_history()
    comp = build_comparison()
    HISTORY_OUT.write_text(json.dumps(hist, indent=1, ensure_ascii=False) + "\n")
    COMPARE_OUT.parent.mkdir(parents=True, exist_ok=True)
    COMPARE_OUT.write_text(json.dumps(comp, indent=1, ensure_ascii=False) + "\n")
    write_doc(hist, comp)
    s = hist["summary"]
    print(f"wrote {HISTORY_OUT.relative_to(ROOT)}  ({len(hist['rows'])} epochs; val@10 {s['val_at_epoch_10']:.4f}, "
          f"val@15 {s['val_at_epoch_15']:.4f}, val@23 {s['val_at_epoch_23']:.4f}; anneal train {s['anneal_train_first']:.4f}->"
          f"{s['anneal_train_last']:.4f} monotone={s['anneal_train_monotone']}; anneal val {s['anneal_val_min']:.4f}..{s['anneal_val_max']:.4f})")
    print(f"wrote {COMPARE_OUT.relative_to(ROOT)}  gpt2 identical: {comp['gpt2_rows_identical']}")
    print(f"wrote {DOC_OUT.relative_to(ROOT)}")
    for d in comp["deltas_annealed_minus_best_val"]:
        print(f"  Δ {d['localizer']:5s} char repl {d['char_auroc_replace']:+.3f} shuf {d['char_auroc_shuffle']:+.3f} "
              f"| word-max fi {d['word_auroc_max_falseinfo']:+.3f} | net repair repl {d['heal_replace_net']:+.3f} fi {d['heal_falseinfo_net']:+.3f}")
    if args.tex:
        print_tex(hist, comp)


if __name__ == "__main__":
    main()
