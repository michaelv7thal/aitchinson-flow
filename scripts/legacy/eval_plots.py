"""scripts/eval_plots.py

Compact 4-panel evaluation of a trained EqM model in CLR space (the space the
model was trained in).

  A — Reconstruction BPD vs NAG-GD steps  (perturb GT → run sampler)
  B — Generated text samples              (NAG-GD from x0 ~ source_sigma · randn)
  C — GD trajectories in PCA(2)            (CLR → PCA, no ILR re-projection)
  D — Per-character NLL                   (mean −log p[gt_char] across positions)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import dataclasses
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

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
from aitchinson_flow.models import build_model

ALPHABET = "".join(sorted(CHAR2ID, key=CHAR2ID.__getitem__))

# ── Eval config ────────────────────────────────────────────────────────────────
CKPT = ROOT / "checkpoints/epoch_final.pt"
B = 64
N_EVAL = 500
N_SAMPLES = 20
N_TRAJ = 32
ETA = 0.05
MU_NAG = 0.9
N_STEPS_SWEEP = [5, 10, 20, 50, 100, 200]
N_STEPS_MAIN = 100
N_SNAPS = 10


def _safe_init(cls, d: dict):
    known = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in known})


def _decode_logprobs(x_clr: torch.Tensor) -> torch.Tensor:
    return x_clr - torch.logsumexp(x_clr, dim=-1, keepdim=True)


def _tokens_to_text(ids: torch.Tensor) -> list[str]:
    return ["".join(ALPHABET[i] for i in row.tolist()) for row in ids.cpu()]


@torch.no_grad()
def _grad_field(model, x: torch.Tensor) -> torch.Tensor:
    """Conservative gradient ∇⟨x, f(x)⟩ — what the sampler descends."""
    with torch.enable_grad():
        x_req = x.detach().requires_grad_(True)
        energy = (x_req * model(x_req)).sum()
        return torch.autograd.grad(energy, x_req)[0].detach()


def _nag_gd(model, x0: torch.Tensor, *, n_steps: int, eta: float, mu: float):
    x = x0.clone()
    x_last = x.clone()
    grad = _grad_field(model, x)
    for _ in range(n_steps):
        x_last_prev = x.clone()
        x = x - eta * grad
        grad = _grad_field(model, x + mu * (x - x_last))
        x_last = x_last_prev
    return x


def _nag_gd_snapshots(model, x0, *, n_steps, eta, mu, n_snaps):
    x = x0.clone()
    x_last = x.clone()
    grad = _grad_field(model, x)
    record_at = {int(round(i * n_steps / n_snaps)) for i in range(n_snaps + 1)}
    snaps = [x.cpu()] if 0 in record_at else []
    for step in range(1, n_steps + 1):
        x_last_prev = x.clone()
        x = x - eta * grad
        grad = _grad_field(model, x + mu * (x - x_last))
        x_last = x_last_prev
        if step in record_at:
            snaps.append(x.cpu())
    return snaps


# ── Load checkpoint ────────────────────────────────────────────────────────────

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
print(f"  epoch {ckpt['epoch']}  K={K}  L={L}  device={device}  σ={SIGMA}")


# ── Test data ──────────────────────────────────────────────────────────────────

print("Loading text8 test split…")
from datasets import load_dataset

ds = load_dataset("afmck/text8", split="test", streaming=True, trust_remote_code=False)
raw_text = next(iter(ds))["text"]
test_windows = text_to_windows(raw_text, L)[: max(N_EVAL, N_TRAJ)].to(device)


# ── A: BPD vs steps ───────────────────────────────────────────────────────────

print("A: Reconstruction BPD vs steps…")


def _bpd(n_steps: int, mu: float) -> float:
    nll, n_tok = 0.0, 0
    seqs = test_windows[:N_EVAL]
    for i in range(0, len(seqs), B):
        batch = seqs[i : i + B]
        x1 = token_ids_to_features(batch.cpu(), K, label_smoothing=LS).to(device)
        x_init = x1 + 0.1 * torch.randn_like(x1)
        x_init = x_init - x_init.mean(-1, keepdim=True)
        x = _nag_gd(model, x_init, n_steps=n_steps, eta=ETA, mu=mu)
        lp = _decode_logprobs(x)
        nll += F.nll_loss(lp.reshape(-1, K), batch.reshape(-1), reduction="sum").item()
        n_tok += batch.numel()
    return nll / (n_tok * math.log(2))


bpd_gd, bpd_nag = [], []
for n in N_STEPS_SWEEP:
    g = _bpd(n, 0.0)
    a = _bpd(n, MU_NAG)
    bpd_gd.append(g)
    bpd_nag.append(a)
    print(f"  n={n:4d}  GD={g:.4f}  NAG={a:.4f}")


# ── B: Generated text samples ─────────────────────────────────────────────────

print("B: Generated text samples…")
x0 = SIGMA * torch.randn(N_SAMPLES, L, K, device=device)
x0 = x0 - x0.mean(-1, keepdim=True)
x_gen = _nag_gd(model, x0, n_steps=N_STEPS_MAIN, eta=ETA, mu=MU_NAG)
lp_gen = _decode_logprobs(x_gen)
ids_gen = lp_gen.argmax(-1)
generated = _tokens_to_text(ids_gen)


# ── C: PCA trajectories ───────────────────────────────────────────────────────

print("C: PCA trajectories…")
x0_traj = SIGMA * torch.randn(N_TRAJ, L, K, device=device)
x0_traj = x0_traj - x0_traj.mean(-1, keepdim=True)
snaps = _nag_gd_snapshots(
    model, x0_traj, n_steps=N_STEPS_MAIN, eta=ETA, mu=MU_NAG, n_snaps=N_SNAPS
)
T = len(snaps)
M = N_TRAJ * L
flat = np.stack([s.reshape(M, K).numpy() for s in snaps])  # (T, M, K)
pca = PCA(n_components=2).fit(flat[-1])
traj_2d = np.stack([pca.transform(flat[t]) for t in range(T)])

gt_clr = token_ids_to_features(test_windows[:N_TRAJ].cpu(), K, label_smoothing=LS)
gt_2d = pca.transform(gt_clr.reshape(M, K).numpy())


# ── D: Per-character NLL ──────────────────────────────────────────────────────

print("D: Per-character NLL…")
char_nll = np.zeros(K)
char_count = np.zeros(K, dtype=np.int64)

seqs = test_windows[:N_EVAL]
for i in range(0, len(seqs), B):
    batch = seqs[i : i + B]
    x1 = token_ids_to_features(batch.cpu(), K, label_smoothing=LS).to(device)
    x_init = x1 + 0.1 * torch.randn_like(x1)
    x_init = x_init - x_init.mean(-1, keepdim=True)
    x = _nag_gd(model, x_init, n_steps=N_STEPS_MAIN, eta=ETA, mu=MU_NAG)
    lp = _decode_logprobs(x).cpu().numpy().reshape(-1, K)
    ids_np = batch.cpu().numpy().reshape(-1)
    nll_pos = -lp[np.arange(len(ids_np)), ids_np]
    np.add.at(char_nll, ids_np, nll_pos)
    np.add.at(char_count, ids_np, 1)

mean_char_nll = np.where(char_count > 0, char_nll / char_count, np.nan)


# ── Plotting ──────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
fig.suptitle(
    f"EqM evaluation — epoch {ckpt['epoch']}  (η={ETA}, μ={MU_NAG}, σ={SIGMA})",
    fontsize=14,
)

# A
ax = fig.add_subplot(gs[0, 0])
ax.plot(N_STEPS_SWEEP, bpd_gd, "o-", lw=2, color="steelblue", label="GD (μ=0)")
ax.plot(N_STEPS_SWEEP, bpd_nag, "s-", lw=2, color="darkorange", label=f"NAG (μ={MU_NAG})")
ax.set_xscale("log")
ax.set_xlabel("GD steps")
ax.set_ylabel("Reconstruction BPD")
ax.set_title("A — Reconstruction BPD vs steps")
ax.grid(True, which="both", alpha=0.3)
ax.legend()

# B
ax = fig.add_subplot(gs[0, 1])
ax.axis("off")
ax.set_title(f"B — Generated samples (NAG-GD, {N_STEPS_MAIN} steps)", pad=8)
n_cols = 2
n_rows = math.ceil(len(generated) / n_cols)
for i, text in enumerate(generated):
    col = i % n_cols
    row = i // n_cols
    ax.text(
        col / n_cols + 0.01,
        1.0 - (row + 0.8) / n_rows,
        f'{i + 1:2d}. "{text}"',
        transform=ax.transAxes,
        fontsize=9,
        fontfamily="monospace",
        va="center",
    )

# C
ax = fig.add_subplot(gs[1, 0])
for m in range(M):
    ax.plot(traj_2d[:, m, 0], traj_2d[:, m, 1], color="grey", lw=0.3, alpha=0.2)
ax.scatter(traj_2d[-1, :, 0], traj_2d[-1, :, 1], s=10, c="darkorange", label="generated")
ax.scatter(gt_2d[:, 0], gt_2d[:, 1], s=20, c="steelblue", marker="*", label="ground truth")
ax.scatter(traj_2d[0, 0:1, 0], traj_2d[0, 0:1, 1], s=120, c="white", edgecolors="k", lw=1.5, label="x₀")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
ax.set_title("C — Sampling trajectories in CLR/PCA")
ax.legend(fontsize=8)

# D
ax = fig.add_subplot(gs[1, 1])
xs = np.arange(K)
ax.bar(xs, mean_char_nll, color="steelblue", alpha=0.85)
ax.set_xticks(xs)
ax.set_xticklabels([c if c != " " else "·" for c in ALPHABET], fontsize=7)
ax.set_xlabel("Character")
ax.set_ylabel("Mean −log p (nats)")
ax.set_title("D — Per-character NLL on reconstruction")
ax.grid(axis="y", alpha=0.3)

out = Path(__file__).parent / "eval_plots.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out}")
