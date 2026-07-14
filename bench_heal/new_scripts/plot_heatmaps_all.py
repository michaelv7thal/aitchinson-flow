"""Unified healing-style per-token heatmaps across ALL detectors.

For each detector sweep JSON under a bench_ood/ tree, render the same
clean / replace30 / falseinfo30 per-token heatmaps: the detector's per-token
score (viridis) with **red box = truly corrupted** and **cyan ▼ = detector-flagged**
(calibrated at the sweep's `flag_thr`). One consistent visual language across
NLL / BLR / BGMM / GPT2-SE, so the reader can eyeball where each detector fires
(and where it misses, e.g. false-info).

Reads the per-detector score key from `_bench_common.DETECTOR_KEYS`.

    uv run python scripts/plot_heatmaps_all.py --bench-dir bench_ood
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


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_heatmap(ex, score_key, label, out_path, max_pos=120):
    plt = _plt()
    from matplotlib.patches import Rectangle

    vals = (ex.get(score_key) or [])[:max_pos]
    if not vals:
        return None
    L = len(vals)
    # One cell = one unit OF THE MODEL'S OWN TOKENIZATION. For the char-level flow
    # models that is a character; for the GPT-2 baselines it is a BPE token, and the
    # red box marks a BPE token that overlaps a corrupted character. Drawing GPT-2 on
    # character cells would mean attributing its score down onto chars — the smearing
    # artifact we removed. `cell_labels` is present iff the example is non-char.
    unit = ex.get("unit", "char")
    cells = ex.get("cell_labels")
    if cells is None:
        cells = list(ex.get("text", "")[:L])
    cells = [str(c) for c in cells[:L]]
    corrupted = (ex.get("corrupted") or [False] * L)[:L]
    flagged = (ex.get("flagged") or [False] * L)[:L]
    corr_idx = [j for j, c in enumerate(corrupted) if c]
    flag_idx = [j for j, f in enumerate(flagged) if f]

    width = 0.12 if unit == "char" else 0.34  # BPE labels are multi-char: give them room
    fig, ax = plt.subplots(1, 1, figsize=(max(8.0, L * width), 2.6))
    im = ax.imshow([vals], aspect="auto", cmap="viridis")
    ax.set_yticks([0])
    ax.set_yticklabels([label])
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    for j in corr_idx:
        ax.add_patch(Rectangle((j - 0.5, -0.5), 1, 1, fill=False,
                               edgecolor="red", lw=1.5))
    if flag_idx:
        ax.scatter(flag_idx, [-0.62] * len(flag_idx), marker="v", s=22,
                   c="#00b3b3", clip_on=False, zorder=5)
    ax.set_xticks(range(L))
    labels = ax.set_xticklabels(
        cells, fontsize=6, family="monospace",
        rotation=(0 if unit == "char" else 90),
        ha=("center" if unit == "char" else "right"))
    for j in corr_idx:
        if j < len(labels):
            labels[j].set_color("red")
            labels[j].set_fontweight("bold")
    unit_name = "char" if unit == "char" else "BPE token"
    ax.set_xlabel(f"position ({unit_name} cells; red box = corrupted; "
                  f"cyan ▼ = flagged)")
    tp = len(set(corr_idx) & set(flag_idx))
    ax.set_title(f"{label} [{unit_name}s] — {ex.get('which', '?')} "
                 f"#{ex.get('idx', 0)}  "
                 f"({len(corr_idx)} corrupted, {len(flag_idx)} flagged, {tp} hit)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench-dir", default="bench_ood")
    ap.add_argument("--max-pos", type=int, default=120)
    ap.add_argument("--idx", type=int, default=0, help="which example passage to plot")
    args = ap.parse_args()
    bench = Path(args.bench_dir)
    written = []
    for det, meta in DETECTOR_KEYS.items():
        jp = bench / meta["subdir"] / meta["file"]
        if not jp.exists():
            print(f"[skip] {det}: {jp} missing")
            continue
        data = json.loads(jp.read_text())
        outdir = bench / meta["subdir"] / "heatmaps"
        outdir.mkdir(parents=True, exist_ok=True)
        for ex in data.get("examples", []):
            if ex.get("idx") != args.idx:
                continue
            name = f"heatmap_{ex.get('which', 'ex')}_{ex.get('idx', 0)}.png"
            p = plot_heatmap(ex, meta["score"], meta["detector_label"],
                             outdir / name, max_pos=args.max_pos)
            if p:
                written.append(p)
                print(f"wrote {p}")
    print(f"\n{len(written)} heatmaps written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
