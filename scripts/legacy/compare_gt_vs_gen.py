"""scripts/compare_gt_vs_gen.py

Side-by-side comparison of one ground-truth sequence and one flow-matching
generated sequence in CLR space — character-probability heatmaps and decoded
text. ILR is used only for distance reporting (Aitchison MSE).
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import aitchinson_flow.models  # noqa: F401 — populate model registry
from aitchinson_flow.config import (
    Config,
    EqM,
    Text8DataConfig,
    TransformationConfig,
    TransformerConfig,
)
from aitchinson_flow.data import token_ids_to_features
from aitchinson_flow.data.char_window_dataset import (
    CHAR2ID,
    text_to_windows,
)
from aitchinson_flow.geometry import ilr
from aitchinson_flow.models import build_model

ALPHABET = "".join(sorted(CHAR2ID, key=CHAR2ID.__getitem__))

CKPT = ROOT / "checkpoints/epoch_final.pt"
ETA = 0.05
MU = 0.9
N_STEPS = 100


def _safe_init(cls, d: dict):
    known = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in known})


@torch.no_grad()
def _grad_field(model, x: torch.Tensor) -> torch.Tensor:
    with torch.enable_grad():
        x_req = x.detach().requires_grad_(True)
        energy = (x_req * model(x_req)).sum()
        return torch.autograd.grad(energy, x_req)[0].detach()


def _nag_gd(model, x0, *, n_steps, eta, mu):
    x = x0.clone()
    x_last = x.clone()
    grad = _grad_field(model, x)
    for _ in range(n_steps):
        x_last_prev = x.clone()
        x = x - eta * grad
        grad = _grad_field(model, x + mu * (x - x_last))
        x_last = x_last_prev
    return x


def _decode_logprobs(x_clr: torch.Tensor) -> torch.Tensor:
    return x_clr - torch.logsumexp(x_clr, dim=-1, keepdim=True)


# ── Load model ─────────────────────────────────────────────────────────────────

print("Loading checkpoint…")
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg_d = ckpt["cfg"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = Config()
if "transformer" in cfg_d:
    cfg.transformer = _safe_init(TransformerConfig, cfg_d["transformer"])
if "transformation" in cfg_d:
    cfg.transformation = _safe_init(TransformationConfig, cfg_d["transformation"])
if "text8_dataset" in cfg_d:
    cfg.text8_dataset = _safe_init(Text8DataConfig, cfg_d["text8_dataset"])
if "eqm" in cfg_d:
    cfg.eqm = _safe_init(EqM, cfg_d["eqm"])

K = cfg.text8_dataset.K
L = cfg.text8_dataset.L
LS = cfg.transformation.label_smoothing
SIGMA = cfg.eqm.source_sigma

model = build_model(cfg).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"  epoch {ckpt['epoch']}  device={device}  σ={SIGMA}")


# ── One test sequence ──────────────────────────────────────────────────────────

print("Loading text8 test split…")
from datasets import load_dataset

ds = load_dataset("afmck/text8", split="test", streaming=True, trust_remote_code=False)
raw_text = next(iter(ds))["text"]
seq = text_to_windows(raw_text, L)[0].to(device)
gt_text = "".join(ALPHABET[i] for i in seq.cpu().tolist())
print(f"  Ground truth: '{gt_text}'")


# ── Ground-truth CLR + log-probs ───────────────────────────────────────────────

x_gt_clr = token_ids_to_features(seq.unsqueeze(0).cpu(), K, label_smoothing=LS).to(device)
lp_gt = _decode_logprobs(x_gt_clr)


# ── Generate from noise ────────────────────────────────────────────────────────

print(f"Generating via NAG-GD ({N_STEPS} steps from σ={SIGMA} noise)…")
x0 = SIGMA * torch.randn(1, L, K, device=device)
x0 = x0 - x0.mean(-1, keepdim=True)
x_gen_clr = _nag_gd(model, x0, n_steps=N_STEPS, eta=ETA, mu=MU)
lp_gen = _decode_logprobs(x_gen_clr)
ids_gen = lp_gen[0].argmax(-1)
gen_text = "".join(ALPHABET[i] for i in ids_gen.cpu().tolist())
print(f"  Generated   : '{gen_text}'")


# ── Distances ──────────────────────────────────────────────────────────────────

# Aitchison MSE = MSE in CLR (since CLR is an isometric embedding of the simplex
# under the Aitchison metric). Equivalent to MSE in ILR up to orthogonal rotation.
clr_mse = (x_gt_clr - x_gen_clr).pow(2).mean().item()
ilr_mse = (ilr(x_gt_clr) - ilr(x_gen_clr)).pow(2).mean().item()
prob_kl = F.kl_div(lp_gen, lp_gt.exp(), reduction="batchmean").item()
print(f"\n  CLR MSE : {clr_mse:.4f}   ILR MSE : {ilr_mse:.4f}   KL(gt‖gen): {prob_kl:.4f}")


# ── Plot ───────────────────────────────────────────────────────────────────────

probs_gt = lp_gt[0].exp().cpu()
probs_gen = lp_gen[0].exp().cpu()

fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
fig.suptitle(
    f"GT vs generated  (epoch {ckpt['epoch']}, NAG-GD {N_STEPS} steps, σ={SIGMA})",
    fontsize=12,
)

for ax, probs, title in (
    (axes[0], probs_gt, f'Ground truth: "{gt_text}"'),
    (axes[1], probs_gen, f'Generated:    "{gen_text}"'),
):
    im = ax.imshow(probs.T, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_yticks(range(K))
    ax.set_yticklabels([c if c != " " else "·" for c in ALPHABET], fontsize=6)
    ax.set_ylabel("character")
    ax.set_title(title, fontsize=10, loc="left")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

axes[1].set_xlabel("position")
fig.colorbar(im, ax=axes, shrink=0.85, label="prob")

out = Path(__file__).parent / "compare_gt_vs_gen.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out}")
