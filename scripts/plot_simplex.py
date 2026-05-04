"""
Sample text8 windows, apply the log transform, and visualise the
resulting distribution on the simplex in three panels:

  1. ILR coords → UMAP 2D hexbin density (non-linear, respects simplex geometry)
  2. Ternary plot for the three most common characters (space / e / t)
  3. Marginal character frequency bar chart
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch.nn.functional as F
import umap

from aitchinson_flow.data.char_window_dataset import text_to_windows, VOCAB_SIZE
from aitchinson_flow.geometry import ilr

# ── Config ─────────────────────────────────────────────────────────────────────
L = 20
K = VOCAB_SIZE          # 27
N_WINDOWS = 3_000
LABEL_SMOOTHING = 1e-4
ALPHABET = "abcdefghijklmnopqrstuvwxyz "

# ── 1. Load text8 ──────────────────────────────────────────────────────────────
print("Loading text8…")
try:
    from datasets import load_dataset
    ds = load_dataset("afmck/text8", split="train", streaming=True, trust_remote_code=False)
    raw_text = next(iter(ds))["text"]
except Exception as exc:
    sys.exit(
        f"Could not load text8: {exc}\n"
        "Install the `datasets` package and ensure internet access."
    )

# ── 2. Windows → log-space simplex ────────────────────────────────────────────
windows = text_to_windows(raw_text, L)[:N_WINDOWS]           # (N, L)  int64

# Batch log transform equivalent to token_ids_to_features(mode="simplex")
oh = F.one_hot(windows.clamp(0, K - 1), K).float()           # (N, L, K)
log_x = ((1.0 - LABEL_SMOOTHING) * oh + LABEL_SMOOTHING / K).log()  # (N, L, K)
probs = log_x.exp()                                           # (N, L, K) normalised

# ── 3. ILR transform → UMAP 2D ───────────────────────────────────────────────
flat_log = log_x.reshape(-1, K)                              # (N*L, K)
ilr_coords = ilr(flat_log).numpy()                           # (N*L, 26)

print(f"Running UMAP on {len(ilr_coords):,} points…")
reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=42, verbose=False)
xy = reducer.fit_transform(ilr_coords)                       # (N*L, 2)

# ── 4. Ternary coordinates (space / e / t) ────────────────────────────────────
i_spc = ALPHABET.index(" ")   # 26
i_e   = ALPHABET.index("e")   # 4
i_t   = ALPHABET.index("t")   # 19

flat_probs = probs.reshape(-1, K).numpy()                    # (N*L, 27)
flat_ids = windows.reshape(-1).numpy()                       # (N*L,)  true char index

# Only keep positions whose true character is one of the three.
# Other characters have near-zero prob for all three, so renormalising them
# gives ≈ (1/3, 1/3, 1/3) — all at the center.
tri_mask = np.isin(flat_ids, [i_spc, i_e, i_t])
p3 = flat_probs[tri_mask][:, [i_spc, i_e, i_t]]
p3 = p3 / p3.sum(axis=1, keepdims=True)                     # renorm to 3-simplex


def ternary_to_cart(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Equilateral-triangle embedding: left=space, right=e, top=t."""
    x = 0.5 * (2.0 * p[:, 1] + p[:, 2])
    y = (np.sqrt(3.0) / 2.0) * p[:, 2]
    return x, y


tx, ty = ternary_to_cart(p3)

# ── 5. Plot ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    f"Text8 character distributions on the simplex  "
    f"({N_WINDOWS} windows × L={L})",
    fontsize=13,
)

# Panel A — UMAP hexbin density
ax = axes[0]
hb = ax.hexbin(xy[:, 0], xy[:, 1], gridsize=60, cmap="viridis", mincnt=1)
fig.colorbar(hb, ax=ax, label="count")
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.set_title("ILR coords — UMAP 2D density")
ax.set_aspect("equal", "datalim")

# Panel B — ternary plot
ax = axes[1]
ax.set_aspect("equal")
tri_corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])
ax.add_patch(mpatches.Polygon(tri_corners, closed=True, fill=False, edgecolor="black", lw=1.5))
ax.hexbin(tx, ty, gridsize=40, cmap="YlOrRd", mincnt=1, extent=[0, 1, 0, np.sqrt(3) / 2])
d = 0.06
ax.text(-d, -d, "space", ha="right", va="top", fontsize=9)
ax.text(1.0 + d, -d, "e", ha="left", va="top", fontsize=9)
ax.text(0.5, np.sqrt(3.0) / 2.0 + d, "t", ha="center", va="bottom", fontsize=9)
ax.set_xlim(-0.15, 1.15)
ax.set_ylim(-0.1, np.sqrt(3.0) / 2.0 + 0.15)
ax.axis("off")
n_tri = tri_mask.sum()
ax.set_title(f"Ternary density: space / e / t\n(positions with those chars, n={n_tri:,})")

# Panel C — marginal character frequencies
ax = axes[2]
mean_p = flat_probs.mean(axis=0)                             # (27,)
tick_labels = [c if c != " " else "·" for c in ALPHABET]
ax.bar(range(K), mean_p, color="steelblue", width=0.7)
ax.set_xticks(range(K))
ax.set_xticklabels(tick_labels, fontsize=7)
ax.set_xlabel("Character")
ax.set_ylabel("Mean probability")
ax.set_title("Marginal character frequency")

plt.tight_layout()

out = Path(__file__).parent / "simplex_distribution.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.show()
