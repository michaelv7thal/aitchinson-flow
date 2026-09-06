"""fig:latent-seq + fig:latent-token — the latent split of the frozen
full-corpus Dirichlet FM backbone (paper sec:latent, tab:latent).

Reads the coordinate dumps written by ``scripts/plot_latent_split.py
--dump-coords`` (test split, rate 0.3). Both figures show one panel per
corruption scheme, and each scheme is taken from the first run below that
carries it, so the figures redraw as the ladder fills in:
  bench_ood/latent_split_all/        replace, shuffle, falseinfo (seq + token)
  bench_ood/latent_split_plausible/  plausible (seq + token)
  bench_ood/latent_split_coords/     legacy: sequence panels + per-token falseinfo
  bench_ood/latent_split_coords_tokrep/  legacy: per-token replace
A scheme no run carries is skipped rather than drawn empty.

The full-dimension probe of tab:latent gets no panel on purpose. Its axis is the
probe's own weight vector, fitted on the very points it would be shown
separating, so the panel would look cleanly split whenever the fit succeeds --
including on permuted labels, which per sequence already reach AUROC 0.93 at 512
points in 1280 dimensions. ``plot_latent_split.py --dump-coords`` does write the
axis (``*_full_xy``) if this is ever wanted for an appendix control.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import _style
from _style import BENCH_OOD, BLUE, ORANGE, INK2

RUNS = [BENCH_OOD / d for d in
        ("latent_split_all", "latent_split_plausible", "latent_split_coords",
         "latent_split_coords_tokrep")]

CLEAN_C, CORR_C = BLUE, ORANGE
SCHEMES = [("replace", "replace"), ("shuffle", "shuffle"),
           ("falseinfo", "false information"), ("plausible", "plausible")]
RNG = np.random.default_rng(0)


def _panels(prefix):
    """[(scheme_key, label, coords, metrics)] for every scheme some run carries."""
    found = []
    for key, label in SCHEMES:
        for run in RUNS:
            if not (run / "latent_split.json").exists():
                continue
            coords, metrics = _load(run)
            if f"{prefix}_{key}_labels" in coords:
                found.append((key, label, coords, metrics))
                break
    return found


def _tok_metrics(metrics, key):
    """per_token is keyed by scheme in current runs, flat in the legacy ones."""
    pt = metrics["per_token"]
    return pt[key] if key in pt else pt


def _load(run):
    coords = np.load(run / "latent_split_coords.npz")
    metrics = json.loads((run / "latent_split.json").read_text())
    return coords, metrics


def _scatter(ax, xy, lab, s=6.0, alpha=0.5):
    """Draw both classes in one interleaved pass so neither sits on top."""
    perm = RNG.permutation(len(lab))
    colors = np.where(lab[perm] == 0, CLEAN_C, CORR_C)
    ax.scatter(xy[perm, 0], xy[perm, 1], s=s, c=colors, alpha=alpha,
               lw=0, rasterized=True)
    ax.locator_params(nbins=4)


def _annot(ax, text):
    ax.text(0.03, 0.965, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=7, color=INK2,
            bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.2))


def _legend(fig):
    handles = [Line2D([], [], marker="o", ls="", ms=4.5, mec="none", mfc=c)
               for c in (CLEAN_C, CORR_C)]
    fig.legend(handles, ["clean", "corrupted"], ncol=2,
               loc="outside upper center")


def fig_seq():
    panels = _panels("seq")
    fig, axes = plt.subplots(2, len(panels),
                             figsize=(_style.TEXTWIDTH_IN, 3.9),
                             layout="constrained", squeeze=False)
    for col, (key, label, coords, metrics) in enumerate(panels):
        m = metrics["per_sequence"][key]
        lab = coords[f"seq_{key}_labels"]
        _scatter(axes[0, col], coords[f"seq_{key}_pca_xy"], lab)
        axes[0, col].set_title(label, pad=4)
        # Name the readout in every annotation, in the words tab:latent gives its
        # columns ("PC 1-2", "Fisher LDA"). The two rows are two different
        # readouts, and a bare "AUROC" leaves the reader to work out which.
        # (author, 2026-08-30: no "probe" on the PC row and no "axis" on the
        # LDA row; the annotation and the x-label both read "Fisher LDA".)
        # Two lines: on one line the label runs into the next panel's y ticks.
        _annot(axes[0, col], f"PC 1-2\nAUROC {m['pca_pc12_probe_auroc']:.3f}")
        _scatter(axes[1, col], coords[f"seq_{key}_lda_xy"], lab)
        _annot(axes[1, col], f"Fisher LDA\nAUROC {m['lda_axis_auroc']:.3f}")
        # Each row labels its own x-axis under each of its panels: a single
        # centred label per row lands between the rows and belongs to neither.
        axes[0, col].set_xlabel("PC1")
        axes[1, col].set_xlabel("Fisher LDA")
        print(f"[seq:{key}] probe={m['pca_pc12_probe_auroc']:.3f} "
              f"lda={m['lda_axis_auroc']:.3f} "
              f"full={m['full_dim_linear_auroc']:.3f}")
    axes[0, 0].set_ylabel("PC2")
    axes[1, 0].set_ylabel(r"leading PC $\perp$ axis")
    _legend(fig)
    _style.save(fig, "latent_seq")


def fig_token():
    panels = _panels("tok")
    fig, axes = plt.subplots(1, len(panels),
                             figsize=(_style.TEXTWIDTH_IN, 2.45),
                             layout="constrained", squeeze=False)
    for col, (key, label, coords, metrics) in enumerate(panels):
        ax = axes[0, col]
        m = _tok_metrics(metrics, key)
        assert m["token_scheme"] == key
        _scatter(ax, coords[f"tok_{key}_energy_xy"], coords[f"tok_{key}_labels"],
                 s=2.5, alpha=0.35)
        ax.set_title(label, pad=4)
        # "hinge axis" is what tab:latent calls this readout in its per-token rows.
        _annot(ax, f"hinge axis\nAUROC {m['energy_axis_auroc']:.3f}")
        print(f"[tok:{key}] energy={m['energy_axis_auroc']:.3f} "
              f"probe={m['pca_pc12_probe_auroc']:.3f} "
              f"full={m['full_dim_linear_auroc']:.3f}")
    axes[0, 0].set_ylabel(r"leading PC $\perp$ axis")
    # One row, so one x-label, but centred on the figure rather than on the
    # middle panel, which at four panels sits visibly off-centre.
    fig.supxlabel(r"energy axis $E = w^{\top} z$", fontsize=9)
    _legend(fig)
    _style.save(fig, "latent_token")


def main():
    _style.apply_style()
    fig_seq()
    fig_token()


if __name__ == "__main__":
    main()
