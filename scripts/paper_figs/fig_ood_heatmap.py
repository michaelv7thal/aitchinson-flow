"""fig:ood-heatmap — per-character denoiser NLL on one test window (paper fig).

Reads BENCH_OOD/nll/heatmap_examples.json (written by scripts/dump_heatmap_examples.py)
and renders one three-panel figure per corruption rate: clean / replace / false
information, the 120-character window wrapped into rows of 40 so the text is
readable. Cell colour is the score on a single-hue light->dark ramp SHARED across
panels and rates (square-root spaced via PowerNorm gamma=0.5, chosen so the
range around the flag threshold stays visible against the largest scores; the
threshold is marked on the colorbar); an orange frame marks ground-truth corrupted characters,
an ink triangle marks characters the detector flags at the 5%-FPR clean threshold.

Usage:
    BENCH_OOD_DIR=bench_ood_final uv run python scripts/paper_figs/fig_ood_heatmap.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import (  # noqa: E402
    BENCH_OOD, ORANGE, INK, INK2, MUTED, BASELINE, TEXTWIDTH_IN,
    apply_style, save,
)

WRAP = 40         # characters per row
IDX = 44          # which example window (Megatokyo; the retired Lewinsky pair was 0/1)
# vertical layout, in x-cell units (aspect is equal, y grows downwards)
TITLE_H = 1.6     # panel title zone
FLAG_H = 0.55     # flag triangles above the strip
STRIP_H = 1.2     # heatmap cell height
TEXT_DY = 0.68    # text baseline below the strip
ROW_PITCH = FLAG_H + STRIP_H + 1.30
PANEL_GAP = 0.8

# single-hue sequential ramp, white -> paper blue -> near-black blue
CMAP = mcolors.LinearSegmentedColormap.from_list(
    "paper_blues", ["#ffffff", "#86b6ef", "#2a78d6", "#104281"])


def _runs(mask):
    """[(start, end)] of consecutive True runs."""
    out, s = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and s is None:
            s = i
        elif not m and s is not None:
            out.append((s, i))
            s = None
    return out


def _panel(ax, ex, title, norm, y0):
    """Draw one panel with its top edge at y0; return its total height."""
    sc = np.array(ex["NLL_t"], dtype=float)
    corr = np.array(ex["corrupted"], dtype=bool)
    flag = np.array(ex["flagged"], dtype=bool)
    text = ex["text"]
    n = len(sc)
    nrows = int(np.ceil(n / WRAP))

    ax.text(0, y0 + 0.55, title, ha="left", va="center",
            fontsize=9, color=INK)
    n_corr, n_flag = int(corr.sum()), int(flag.sum())
    hits = int((corr & flag).sum())
    note = (f"{n_flag} flagged" if not n_corr
            else f"{n_corr} corrupted, {hits} of them flagged ({n_flag} flags)")
    ax.text(WRAP, y0 + 0.55, note, ha="right", va="center",
            fontsize=7.5, color=INK2)

    for r in range(nrows):
        lo, hi = r * WRAP, min((r + 1) * WRAP, n)
        w = hi - lo
        yt = y0 + TITLE_H + r * ROW_PITCH + FLAG_H   # top of this row's strip
        X, Y = np.meshgrid(np.arange(w + 1), np.array([yt, yt + STRIP_H]))
        ax.pcolormesh(X, Y, sc[lo:hi][None, :], cmap=CMAP, norm=norm,
                      edgecolors="white", linewidth=0.4)
        for (s, e) in _runs(corr[lo:hi]):
            ax.add_patch(Rectangle((s, yt), e - s, STRIP_H, fill=False,
                                   edgecolor=ORANGE, linewidth=1.1, zorder=3))
        fx = np.nonzero(flag[lo:hi])[0]
        if len(fx):
            ax.scatter(fx + 0.5, np.full(len(fx), yt - 0.32), marker="v",
                       s=9, color=INK, linewidths=0, zorder=3)
        for i in range(w):
            ax.text(i + 0.5, yt + STRIP_H + TEXT_DY, text[lo + i],
                    ha="center", va="center", fontsize=8,
                    family="monospace", color=INK)

    return TITLE_H + nrows * ROW_PITCH


def _figure(examples, rate_tag, rate_label, norm, thr):
    keys = ["clean", f"replace{rate_tag}", f"falseinfo{rate_tag}"]
    titles = ["clean", f"replace, rate {rate_label}",
              f"false information, rate {rate_label}"]

    total_y = 3 * (TITLE_H + 3 * ROW_PITCH) + 2 * PANEL_GAP
    unit = TEXTWIDTH_IN / (WRAP + 0.4)            # inches per cell
    bottom_in = 0.74                              # legend + colorbar strip
    fig_h = total_y * unit + bottom_in + 0.06
    fig = plt.figure(figsize=(TEXTWIDTH_IN, fig_h))
    ax = fig.add_axes((0.005, bottom_in / fig_h, 0.99,
                       1.0 - bottom_in / fig_h - 0.005))
    ax.set_xlim(-0.2, WRAP + 0.2)
    ax.set_ylim(total_y, 0)
    ax.set_aspect("equal", anchor="N")
    ax.axis("off")

    y0 = 0.0
    for k, ttl in zip(keys, titles):
        y0 += _panel(ax, examples[k], ttl, norm, y0) + PANEL_GAP

    handles = [
        Rectangle((0, 0), 1, 1, fill=False, edgecolor=ORANGE, linewidth=1.1),
        Line2D([], [], marker="v", color=INK, linestyle="none", markersize=4),
    ]
    fig.legend(handles,
               ["corrupted character (ground truth)",
                "flagged by the detector (threshold at 5% FPR on clean text)"],
               loc="lower left", bbox_to_anchor=(0.005, 0.006), ncol=1,
               fontsize=7.5, frameon=False, handlelength=1.2, borderaxespad=0)
    cax = fig.add_axes((0.615, 0.36 / fig_h, 0.375, 0.09 / fig_h))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal",
                      ticks=[0, 2, 5, 10])
    cb.set_label("per-character denoiser NLL\n(shared square-root scale)",
                 fontsize=7.5, color=INK2, labelpad=2)
    cb.ax.tick_params(labelsize=7, color=BASELINE, labelcolor=INK2)
    cb.outline.set_edgecolor(BASELINE)
    cb.outline.set_linewidth(0.7)
    cb.ax.axvline(thr, color=INK, linewidth=1.0)
    cb.ax.text(thr + 0.15, 1.4, "flag threshold",
               transform=cb.ax.get_xaxis_transform(),
               ha="left", fontsize=6.5, color=MUTED)
    return fig


def main():
    apply_style()
    data = json.loads((BENCH_OOD / "nll" / "heatmap_examples.json").read_text())
    examples = {e["which"]: e for e in data["examples"] if e["idx"] == IDX}
    thr = data["flag_thr"]
    vmax = max(max(e["NLL_t"]) for e in data["examples"])
    norm = mcolors.PowerNorm(gamma=0.5, vmin=0.0, vmax=vmax)

    for rate in data["rates"]:
        tag = str(int(round(rate * 100)))
        fig = _figure(examples, tag, f"{rate:.2f}", norm, thr)
        save(fig, f"ood_heatmap_nll_{tag}")
        plt.close(fig)


if __name__ == "__main__":
    main()
