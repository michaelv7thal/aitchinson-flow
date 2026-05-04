"""
scripts/eval_plots.py

Evaluation panels for a trained EqM model.

EqM trains f(x) ≈ c(γ)·(x₀ − x₁), a gradient field whose zeros are the data
manifold.  Sampling is gradient descent  x ← x − η·f(x),  which subtracts the
field and therefore moves toward x₁ (data), not away from it.

  A — BPD vs GD steps: vanilla GD (μ=0) vs NAG-GD (μ=0.9)
  B — Generated text samples  (NAG-GD, fixed steps)
  C — GD trajectories in ILR space  (PCA 2D)
  D — Per-character NLL
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

import aitchinson_flow.models  # noqa: F401 — populate model registry
from aitchinson_flow.config import (
    Config,
    TransformerConfig,
    TransformationConfig,
    Text8DataConfig,
)
from aitchinson_flow.models import build_model
from aitchinson_flow.geometry import ilr, ilr_inv
from aitchinson_flow.data.char_window_dataset import (
    text_to_windows,
    VOCAB_SIZE,
    CHAR2ID,
)

ALPHABET = "".join(sorted(CHAR2ID, key=CHAR2ID.__getitem__))

# ── Eval config ────────────────────────────────────────────────────────────────
CKPT = ROOT / "checkpoints/epoch_final.pt"
K = VOCAB_SIZE  # 27  (overridden below from checkpoint config)
L: int  # set below from checkpoint config
B = 64  # batch size for BPD / per-char loops
N_EVAL = 500  # test sequences for BPD and per-char NLL
N_SAMPLES = 20  # generated text samples
N_TRAJ = 32  # sequences for trajectory panel

# GD hyperparameters
ETA = 0.05  # step size η
MU_NAG = 0.9  # NAG look-ahead factor μ
N_STEPS_SWEEP = [5, 10, 20, 50, 100, 200]  # for BPD sweep (Panel A)
N_STEPS_MAIN = 100  # steps for generation, trajectory, per-char NLL

LABEL_SMOOTHING = 1e-4


# ── Helpers ────────────────────────────────────────────────────────────────────


def make_log_x(seqs: torch.Tensor) -> torch.Tensor:
    """Token IDs (B, L) → log-probabilities (B, L, K)."""
    oh = F.one_hot(seqs.clamp(0, K - 1), K).float()
    return ((1.0 - LABEL_SMOOTHING) * oh + LABEL_SMOOTHING / K).log()


def tokens_to_text(token_ids: torch.Tensor) -> list[str]:
    return ["".join(ALPHABET[i] for i in row.tolist()) for row in token_ids.cpu()]


def _mean_grad_norm(grad: torch.Tensor) -> float:
    return float(grad.norm(dim=-1).mean())


# ── Samplers ───────────────────────────────────────────────────────────────────


@torch.no_grad()
def nag_gd(
    model: nn.Module,
    x0_ilr: torch.Tensor,
    *,
    n_steps: int,
    eta: float,
    mu: float,
) -> torch.Tensor:
    """Fixed-step NAG-GD sampler.  μ=0 reduces to vanilla GD.

    Update rule (from pseudocode):
        x_last  = x
        x       = x  − η · grad
        grad    = f(x + μ·(x − x_last))
    """
    x = x0_ilr.clone()
    x_last = x.clone()
    grad = ilr(model(x))
    for _ in range(n_steps):
        x_last = x.clone()
        x = x - eta * grad
        grad = ilr(model(x + mu * (x - x_last)))
    return x


@torch.no_grad()
def nag_gd_adaptive(
    model: nn.Module,
    x0_ilr: torch.Tensor,
    *,
    eta: float,
    mu: float,
    g_min: float,
    max_steps: int = 500,
) -> tuple[torch.Tensor, int]:
    """Adaptive NAG-GD: stops when mean gradient norm drops below g_min."""
    x = x0_ilr.clone()
    x_last = x.clone()
    grad = ilr(model(x))
    steps = 0
    while _mean_grad_norm(grad) > g_min and steps < max_steps:
        x_last = x.clone()
        x = x - eta * grad
        grad = ilr(model(x + mu * (x - x_last)))
        steps += 1
    return x, steps


@torch.no_grad()
def nag_gd_snapshots(
    model: nn.Module,
    x0_ilr: torch.Tensor,
    *,
    n_steps: int,
    eta: float,
    mu: float,
    n_snaps: int = 10,
) -> list[torch.Tensor]:
    """Run NAG-GD and return n_snaps+1 ILR tensors at evenly-spaced steps."""
    x = x0_ilr.clone()
    x_last = x.clone()
    grad = ilr(model(x))
    record_at = {int(round(i * n_steps / n_snaps)) for i in range(n_snaps + 1)}
    snaps = [x.cpu()] if 0 in record_at else []
    for step in range(1, n_steps + 1):
        x_last = x.clone()
        x = x - eta * grad
        grad = ilr(model(x + mu * (x - x_last)))
        if step in record_at:
            snaps.append(x.cpu())
    return snaps


# ── Load checkpoint ────────────────────────────────────────────────────────────

print("Loading checkpoint…")
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg_d = ckpt["cfg"]
device = torch.device(cfg_d["training"]["device"])

cfg = Config()
cfg.transformer = TransformerConfig(**cfg_d["transformer"])
cfg.transformation = TransformationConfig(**cfg_d["transformation"])
if "text8_dataset" in cfg_d:
    cfg.text8_dataset = Text8DataConfig(**cfg_d["text8_dataset"])

# Derive L and K from the loaded config so they match pos_emb size
L = cfg.text8_dataset.L
K = cfg.text8_dataset.K

model = build_model(cfg).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"  epoch {ckpt['epoch']}  |  device: {device}")


# ── Load test data ─────────────────────────────────────────────────────────────

print("Loading text8 test split…")
try:
    from datasets import load_dataset

    ds = load_dataset(
        "afmck/text8", split="test", streaming=True, trust_remote_code=False
    )
    raw_text = next(iter(ds))["text"]
except Exception as exc:
    sys.exit(f"Could not load text8: {exc}")

test_windows = text_to_windows(raw_text, L)[:N_EVAL].to(device)
print(f"  {len(test_windows)} test windows")


# ── A: BPD vs GD steps — GD (μ=0) vs NAG-GD (μ=MU_NAG) ───────────────────────

print("A: BPD vs GD steps…")


def _compute_bpd(n_steps: int, mu: float) -> float:
    total_nll, n_tokens = 0.0, 0
    for i in range(0, len(test_windows), B):
        seqs = test_windows[i : i + B]
        x0 = torch.zeros(seqs.shape[0], L, K - 1, device=device)
        x_end = nag_gd(model, x0, n_steps=n_steps, eta=ETA, mu=mu)
        lp = F.log_softmax(ilr_inv(x_end, K), dim=-1)
        total_nll += F.nll_loss(
            lp.reshape(-1, K), seqs.reshape(-1), reduction="sum"
        ).item()
        n_tokens += seqs.numel()
    return total_nll / (n_tokens * math.log(2))


bpd_gd, bpd_nag = [], []
for n in N_STEPS_SWEEP:
    b_gd = _compute_bpd(n, mu=0.0)
    b_nag = _compute_bpd(n, mu=MU_NAG)
    bpd_gd.append(b_gd)
    bpd_nag.append(b_nag)
    print(f"  n={n:4d}  GD={b_gd:.4f}  NAG={b_nag:.4f}")


# ── B: Generate text (NAG-GD) ──────────────────────────────────────────────────

print("B: Generating text samples…")
x0 = torch.zeros(N_SAMPLES, L, K - 1, device=device)
lp = ilr_inv(nag_gd(model, x0, n_steps=N_STEPS_MAIN, eta=ETA, mu=MU_NAG), K)
ids = torch.distributions.Categorical(logits=lp).sample()
generated = tokens_to_text(ids)


# ── C: GD trajectories (PCA) ──────────────────────────────────────────────────

print("C: GD trajectories…")
x0_traj = torch.zeros(N_TRAJ, L, K - 1, device=device)
snaps = nag_gd_snapshots(model, x0_traj, n_steps=50, eta=ETA, mu=MU_NAG, n_snaps=10)

T, M = len(snaps), N_TRAJ * L
snap_flat = np.stack([s.reshape(M, K - 1).numpy() for s in snaps])  # (T, M, K-1)

pca = PCA(n_components=2)
pca.fit(snap_flat[-1])  # fit on final states
traj_2d = np.stack([pca.transform(snap_flat[t]) for t in range(T)])  # (T, M, 2)

true_ilr = ilr(make_log_x(test_windows[:N_TRAJ].cpu()).reshape(M, K)).numpy()
true_2d = pca.transform(true_ilr)


# ── D: Per-character NLL ───────────────────────────────────────────────────────

print("D: Per-character NLL…")
char_nll = np.zeros(K)
char_count = np.zeros(K, dtype=np.int64)

for i in range(0, len(test_windows), B):
    seqs = test_windows[i : i + B]
    x0 = torch.zeros(seqs.shape[0], L, K - 1, device=device)
    lp = F.log_softmax(
        ilr_inv(nag_gd(model, x0, n_steps=N_STEPS_MAIN, eta=ETA, mu=MU_NAG), K), dim=-1
    )
    ids_np = seqs.cpu().numpy().reshape(-1)
    lp_np = lp.cpu().reshape(-1, K).numpy()
    nll_pos = -lp_np[np.arange(len(ids_np)), ids_np]
    np.add.at(char_nll, ids_np, nll_pos)
    np.add.at(char_count, ids_np, 1)

mean_char_nll = np.where(char_count > 0, char_nll / char_count, np.nan)


# ── Plot ───────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.3)
fig.suptitle(
    f"EqM evaluation — epoch {ckpt['epoch']}  (η={ETA}, μ={MU_NAG})", fontsize=14
)

# A ─ BPD vs GD steps
ax = fig.add_subplot(gs[0, 0])
ax.plot(N_STEPS_SWEEP, bpd_gd, marker="o", lw=2, label="GD  (μ=0)", color="steelblue")
ax.plot(
    N_STEPS_SWEEP,
    bpd_nag,
    marker="s",
    lw=2,
    label=f"NAG-GD (μ={MU_NAG})",
    color="darkorange",
)
ax.set_xscale("log")
ax.set_xlabel("GD steps")
ax.set_ylabel("BPD")
ax.set_title(f"BPD vs GD steps  (η={ETA})")
ax.legend()
ax.grid(True, which="both", alpha=0.3)

# B ─ Generated text
ax = fig.add_subplot(gs[0, 1])
ax.axis("off")
ax.set_title(f"Generated text  (NAG-GD, {N_STEPS_MAIN} steps)", pad=10)
for i, text in enumerate(generated):
    ax.text(
        0.03,
        1.0 - (i + 0.6) / N_SAMPLES,
        f'{i + 1:2d}. "{text}"',
        transform=ax.transAxes,
        fontsize=9,
        fontfamily="monospace",
        va="center",
    )

# C ─ GD trajectories (PCA)
ax = fig.add_subplot(gs[1, 0])
cmap = plt.get_cmap("plasma")
for m in range(M):
    for t in range(T - 1):
        ax.plot(
            traj_2d[t : t + 2, m, 0],
            traj_2d[t : t + 2, m, 1],
            color=cmap(t / (T - 1)),
            lw=0.5,
            alpha=0.35,
        )
ax.scatter(
    traj_2d[-1, :, 0],
    traj_2d[-1, :, 1],
    s=8,
    c="royalblue",
    alpha=0.7,
    label="predicted x₁",
    zorder=3,
)
ax.scatter(
    true_2d[:, 0], true_2d[:, 1], s=8, c="tomato", alpha=0.7, label="true x₁", zorder=3
)
ax.scatter(
    traj_2d[0, :1, 0],
    traj_2d[0, :1, 1],
    s=60,
    c="white",
    edgecolors="black",
    lw=1.2,
    zorder=4,
    label="x₀ (uniform)",
)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
ax.set_title("NAG-GD trajectories in ILR space (PCA)")
ax.legend(fontsize=8, loc="upper left")
sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 1))
fig.colorbar(sm, ax=ax, label="t  (0 = uniform → 1 = generated)", shrink=0.75)

# D ─ Per-character NLL
ax = fig.add_subplot(gs[1, 1])
tick_labels = [c if c != " " else "·" for c in ALPHABET]
finite = mean_char_nll[np.isfinite(mean_char_nll)]
norm_c = mcolors.Normalize(float(finite.min()), float(finite.max()))
colors = plt.get_cmap("RdYlGn_r")(norm_c(np.nan_to_num(mean_char_nll, nan=0.0)))
ax.bar(range(K), mean_char_nll, color=colors, width=0.7)
ax.axhline(
    float(np.nanmean(mean_char_nll)),
    color="black",
    ls="--",
    lw=1,
    alpha=0.6,
    label=f"mean = {float(np.nanmean(mean_char_nll)):.2f} nats",
)
ax.set_xticks(range(K))
ax.set_xticklabels(tick_labels, fontsize=7)
ax.set_xlabel("Character")
ax.set_ylabel("Mean NLL (nats)")
ax.set_title(f"Per-character NLL  ({N_STEPS_MAIN} GD steps)")
ax.legend(fontsize=8)

out = Path(__file__).parent / "eval_plots.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.show()
