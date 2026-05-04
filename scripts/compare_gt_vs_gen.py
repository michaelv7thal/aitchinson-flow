"""
scripts/compare_gt_vs_gen.py

Side-by-side comparison of one ground-truth sequence and one flow-matching
generated sequence.  Shows the character-probability heatmaps and decoded text.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import aitchinson_flow.models  # noqa: F401 — populate model registry
from aitchinson_flow.config import Config, TransformerConfig, TransformationConfig
from aitchinson_flow.models import build_model
from aitchinson_flow.geometry import ilr, ilr_inv
from aitchinson_flow.data.char_window_dataset import (
    text_to_windows,
    CHAR2ID,
)

ALPHABET = "".join(sorted(CHAR2ID, key=CHAR2ID.__getitem__))

CKPT = ROOT / "checkpoints/epoch_final.pt"
LABEL_SMOOTHING = 1e-1
ETA = 0.05
MU = 0.9
N_STEPS = 100


def make_log_x(seq: torch.Tensor) -> torch.Tensor:
    """Token IDs (L,) → log-probs (1, L, K)."""
    oh = F.one_hot(seq.clamp(0, K - 1), K).float()
    return ((1.0 - LABEL_SMOOTHING) * oh + LABEL_SMOOTHING / K).log().unsqueeze(0)


@torch.no_grad()
def nag_gd(model, x0, *, n_steps, eta, mu):
    x = x0.clone()
    x_last = x.clone()
    grad = model(x)
    for _ in range(n_steps):
        x_last = x.clone()
        x = x - eta * grad
        grad = model(x + mu * (x - x_last))
    return x


# ── Load model ─────────────────────────────────────────────────────────────────

print("Loading checkpoint…")
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg_d = ckpt["cfg"]
device = torch.device(cfg_d["training"]["device"])

cfg = Config()
cfg.transformer = TransformerConfig(**cfg_d["transformer"])
cfg.transformation = TransformationConfig(**cfg_d["transformation"])
cfg.text8_dataset.L = cfg_d["text8_dataset"]["L"]
cfg.text8_dataset.K = cfg_d["text8_dataset"]["K"]
K = cfg.text8_dataset.K
L = cfg.text8_dataset.L

model = build_model(cfg).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"  epoch {ckpt['epoch']}  |  device: {device}")

# ── Load one test sequence ─────────────────────────────────────────────────────

print("Loading text8 test split…")
try:
    from datasets import load_dataset

    ds = load_dataset(
        "afmck/text8", split="test", streaming=True, trust_remote_code=False
    )
    raw_text = next(iter(ds))["text"]
except Exception as exc:
    sys.exit(f"Could not load text8: {exc}")

seq = text_to_windows(raw_text, L)[0].to(device)  # (L,)
gt_text = "".join(ALPHABET[i] for i in seq.cpu().tolist())
print(f"  Ground truth: '{gt_text}'")

# ── Ground truth: token IDs → ILR ─────────────────────────────────────────────

log_x_gt = make_log_x(seq).to(device)  # (1, L, K)
x_gt_ilr = ilr(log_x_gt)  # (1, L, K-1)
lp_gt = ilr_inv(x_gt_ilr, K)  # (1, L, K) log-probs

# ── Generated: NAG-GD from uniform x0 ─────────────────────────────────────────

print(f"Generating via NAG-GD ({N_STEPS} steps)…")
x0 = torch.zeros(1, L, K - 1, device=device)
x_gen_ilr = nag_gd(model, x0, n_steps=N_STEPS, eta=ETA, mu=MU)  # (1, L, K-1)
lp_gen = ilr_inv(x_gen_ilr, K)  # (1, L, K) log-probs

ids_gen = lp_gen[0].argmax(dim=-1)
gen_text = "".join(ALPHABET[i] for i in ids_gen.cpu().tolist())
print(f"  Generated:    '{gen_text}'")

# ── Tensor-level comparison ────────────────────────────────────────────────────

ilr_mse = (x_gt_ilr - x_gen_ilr).pow(2).mean().item()
prob_kl = F.kl_div(lp_gen, lp_gt.exp(), reduction="batchmean").item()
print(f"\n  ILR MSE:  {ilr_mse:.4f}")
print(f"  KL(gt‖gen): {prob_kl:.4f}")

# ── Plot ───────────────────────────────────────────────────────────────────────

probs_gt = lp_gt[0].exp().cpu()  # (L, K)
probs_gen = lp_gen[0].exp().cpu()  # (L, K)

tick_labels = [c if c != " " else "·" for c in ALPHABET]
vmax = max(probs_gt.max().item(), probs_gen.max().item())

fig, axes = plt.subplots(
    3, 1, figsize=(18, 10), gridspec_kw={"height_ratios": [1, 1, 0.4]}
)
fig.suptitle(
    f"Ground truth vs. flow-matching generated  (epoch {ckpt['epoch']}, "
    f"{N_STEPS} NAG-GD steps)\n"
    f"ILR MSE = {ilr_mse:.4f}   KL(gt‖gen) = {prob_kl:.4f}",
    fontsize=11,
)

for ax, probs, title in zip(
    axes[:2], [probs_gt, probs_gen], ["Ground truth", "Generated (NAG-GD)"]
):
    im = ax.imshow(
        probs.T,
        aspect="auto",
        origin="lower",
        vmin=0,
        vmax=vmax,
        cmap="Blues",
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("Character")
    ax.set_yticks(range(K))
    ax.set_yticklabels(tick_labels, fontsize=6)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    fig.colorbar(im, ax=ax, shrink=0.8, label="P(char)")

# Difference heatmap
diff = (probs_gt - probs_gen).T  # (K, L)
lim = diff.abs().max().item()
ax = axes[2]
im = ax.imshow(
    diff,
    aspect="auto",
    origin="lower",
    vmin=-lim,
    vmax=lim,
    cmap="RdBu_r",
    interpolation="nearest",
)
ax.set_title("Difference  (gt − generated)", fontsize=10)
ax.set_xlabel("Position")
ax.set_ylabel("Char")
ax.set_yticks(range(K))
ax.set_yticklabels(tick_labels, fontsize=6)
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
fig.colorbar(im, ax=ax, shrink=0.8, label="Δ probability")

plt.tight_layout()
out = Path(__file__).parent / "compare_gt_vs_gen.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out}")
plt.show()
