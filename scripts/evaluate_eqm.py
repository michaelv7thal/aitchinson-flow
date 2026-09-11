"""
scripts/evaluate_eqm.py

Comprehensive evaluation of a trained EqM (Equilibrium Flow Matching) model.

All sampling is done in CLR space (K-dimensional, centred log-ratio), which is
the space the model was trained in.  ILR coordinates (K-1-dimensional isometric
log-ratio) are only used for the PCA trajectory visualisation because they form
a proper Euclidean embedding of the simplex.

Panels
------
  A — Reconstruction BPD vs NAG-GD steps: perturb GT CLR → run GD → NLL
  B — Labeled flow trajectories: ILR/PCA 2-D projection coloured by ground-
       truth character group; circles = generated endpoints, ★ = true targets
  C — Unigram character frequency: generated vs ground-truth (+ KL divergence)
  D — Bigram log₂-ratio heatmap: over/under-generated character transitions
  E — Gradient norm during NAG-GD: energy convergence (log scale)
  F — Gradient norm at GT vs generated: measures how close each lies to an
       energy minimum (lower = more "on manifold")
  G — Generated text samples
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
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

import aitchinson_flow.models  # noqa: F401 – populate model registry
from aitchinson_flow.config import (
    Config,
    EqM,
    LoaderSettings,
    LossConfig,
    Text8DataConfig,
    TransformationConfig,
    TransformerConfig,
    TrainingConfigs,
)
from aitchinson_flow.data import token_ids_to_features
from aitchinson_flow.data.char_window_dataset import (
    CHAR2ID,
    VOCAB_SIZE,
    text_to_windows,
)
from aitchinson_flow.geometry import ilr
from aitchinson_flow.models import build_model

ALPHABET = "".join(sorted(CHAR2ID, key=CHAR2ID.__getitem__))  # 'abcde…z '

# ── Eval hyperparameters ───────────────────────────────────────────────────────
CKPT = ROOT / "checkpoints/epoch_final.pt"
ETA = 0.05  # GD step size η
MU_NAG = 0.9  # NAG look-ahead factor μ
B_EVAL = 64  # batch size for sweeps
N_EVAL = 512  # sequences for BPD / gradient-norm panels
N_GEN = 256  # sequences for unigram / bigram panels
N_TRAJ = 16  # sequences for trajectory panel (keep small → readable plot)
N_SAMPLES = 20  # generated text samples in panel G
N_STEPS_SWEEP = [5, 10, 20, 50]  # step counts for panels A & E
N_STEPS_MAIN = 100  # steps for unigram / bigram / trajectory / gradient panels
N_STEPS_GRAD = 200  # steps for the convergence curve (panel E)
SNAP_COUNT = 12  # trajectory snapshots (panel B)

# Decoding for panel G text samples: "argmax" | "sample" | "topk"
DECODE_MODE = "topk"
DECODE_TEMPERATURE = 1.0  # only used for "sample" / "topk"
DECODE_TOPK = 3  # only used for "topk"


def _decode(log_probs: torch.Tensor) -> torch.Tensor:
    """Token IDs from log-probabilities (..., K) per the configured DECODE_MODE."""
    if DECODE_MODE == "argmax":
        return log_probs.argmax(-1)
    logits = log_probs / DECODE_TEMPERATURE
    if DECODE_MODE == "topk":
        topk = min(DECODE_TOPK, logits.size(-1))
        v, idx = logits.topk(topk, dim=-1)
        probs = torch.softmax(v, dim=-1)
        choice = torch.distributions.Categorical(probs=probs).sample()
        return idx.gather(-1, choice.unsqueeze(-1)).squeeze(-1)
    if DECODE_MODE == "sample":
        return torch.distributions.Categorical(logits=logits).sample()
    raise ValueError(f"Unknown DECODE_MODE={DECODE_MODE!r}")

# Character groups for trajectory colouring
_SPACE = CHAR2ID[" "]
_VOWELS = {CHAR2ID[c] for c in "aeiou"}
_CONS = {CHAR2ID[c] for c in "tnrshl"}
GROUP_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#95a5a6"]
GROUP_LABELS = ["space", "vowels (a/e/i/o/u)", "t/n/r/s/h/l", "other"]


def _char_group(idx: int) -> int:
    if idx == _SPACE:
        return 0
    if idx in _VOWELS:
        return 1
    if idx in _CONS:
        return 2
    return 3


# ── CLR helpers ────────────────────────────────────────────────────────────────


def _decode_logprobs(x_clr: torch.Tensor) -> torch.Tensor:
    """CLR (…, K) → log-softmax (log-probabilities)."""
    return x_clr - torch.logsumexp(x_clr, dim=-1, keepdim=True)


def _tokens_to_text(ids: torch.Tensor) -> list[str]:
    return ["".join(ALPHABET[i] for i in row.tolist()) for row in ids.cpu()]


# ── NAG-GD samplers (all in CLR space) ────────────────────────────────────────


def _grad_field(model, x: torch.Tensor) -> torch.Tensor:
    """Conservative ∇⟨x, f(x)⟩ — what the trained sampler descends."""
    with torch.enable_grad():
        x_req = x.detach().requires_grad_(True)
        energy = (x_req * model(x_req)).sum()
        return torch.autograd.grad(energy, x_req)[0].detach()


def _x0(B: int, L: int, K: int, device, sigma: float) -> torch.Tensor:
    x = sigma * torch.randn(B, L, K, device=device)
    return x - x.mean(-1, keepdim=True)


def _sample(
    model,
    B: int,
    L: int,
    K: int,
    *,
    n_steps: int,
    eta: float,
    mu: float,
    device: torch.device,
    x_init: torch.Tensor | None = None,
    sigma: float = 0.1,
) -> torch.Tensor:
    """Fixed-step NAG-GD on the conservative field → final CLR tensor (B, L, K)."""
    x = x_init.to(device).detach() if x_init is not None else _x0(B, L, K, device, sigma)
    x_last = x.clone()
    grad = _grad_field(model, x)
    for _ in range(n_steps):
        x_last_prev = x.clone()
        x = x - eta * grad
        grad = _grad_field(model, x + mu * (x - x_last))
        x_last = x_last_prev
    return x


def _sample_snapshots(
    model,
    B: int,
    L: int,
    K: int,
    *,
    n_steps: int,
    eta: float,
    mu: float,
    n_snaps: int,
    device: torch.device,
    sigma: float = 0.1,
) -> list[torch.Tensor]:
    """NAG-GD with n_snaps+1 evenly-spaced CLR snapshots."""
    x = _x0(B, L, K, device, sigma)
    x_last = x.clone()
    grad = _grad_field(model, x)
    record_at = {int(round(i * n_steps / n_snaps)) for i in range(n_snaps + 1)}
    snaps: list[torch.Tensor] = [x.cpu()] if 0 in record_at else []
    for step in range(1, n_steps + 1):
        x_last_prev = x.clone()
        x = x - eta * grad
        grad = _grad_field(model, x + mu * (x - x_last))
        x_last = x_last_prev
        if step in record_at:
            snaps.append(x.cpu())
    return snaps


def _sample_grad_norms(
    model,
    B: int,
    L: int,
    K: int,
    *,
    n_steps: int,
    eta: float,
    mu: float,
    device: torch.device,
    sigma: float = 0.1,
) -> list[float]:
    """NAG-GD on conservative field, recording mean ‖∇E‖ at every step."""
    x = _x0(B, L, K, device, sigma)
    x_last = x.clone()
    grad = _grad_field(model, x)
    norms = [float(grad.reshape(B, -1).norm(dim=-1).mean())]
    for _ in range(n_steps):
        x_last_prev = x.clone()
        x = x - eta * grad
        grad = _grad_field(model, x + mu * (x - x_last))
        x_last = x_last_prev
        norms.append(float(grad.reshape(B, -1).norm(dim=-1).mean()))
    return norms


def _grad_norms_at(model, x: torch.Tensor) -> np.ndarray:
    """Per-sequence ‖∇E(x)‖ — uses the conservative field, not raw f(x)."""
    g = _grad_field(model, x)
    B = x.shape[0]
    return g.reshape(B, -1).norm(dim=-1).cpu().numpy()


# ── Load checkpoint ────────────────────────────────────────────────────────────

print("Loading checkpoint…")
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg_d: dict = ckpt["cfg"]

try:
    device = torch.device(str(cfg_d["training"]["device"]))
except Exception:
    device = torch.device("cpu")


# Reconstruct config from checkpoint dict, tolerating missing or extra keys.
def _safe_init(cls, d: dict):
    import dataclasses

    known = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in known})


cfg = Config()
if "transformer" in cfg_d:
    cfg.transformer = _safe_init(TransformerConfig, cfg_d["transformer"])
if "transformation" in cfg_d:
    cfg.transformation = _safe_init(TransformationConfig, cfg_d["transformation"])
if "text8_dataset" in cfg_d:
    cfg.text8_dataset = _safe_init(Text8DataConfig, cfg_d["text8_dataset"])
if "eqm" in cfg_d:
    cfg.eqm = _safe_init(EqM, cfg_d["eqm"])

L = cfg.text8_dataset.L
K = cfg.text8_dataset.K
LS = cfg.transformation.label_smoothing
SIGMA = cfg.eqm.source_sigma

model = build_model(cfg).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"  epoch {ckpt['epoch']}  |  K={K}  L={L}  σ={SIGMA}  |  device: {device}")

tick_labels = [c if c != " " else "·" for c in ALPHABET]

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

_needed = max(N_EVAL, N_GEN, N_TRAJ)
test_windows = text_to_windows(raw_text, L)[:_needed].to(device)
print(f"  loaded {len(test_windows)} test windows")


# ── A: BPD vs GD steps ────────────────────────────────────────────────────────

print("A: BPD vs steps…")


def _bpd(n_steps: int, mu: float) -> float:
    """Reconstruction BPD: perturb GT CLR → run GD → measure NLL against GT.

    Starting from the GT encoding (not from noise) means BPD measures how well
    the model can correct a slightly perturbed data point, so it should decrease
    or stabilise as n_steps grows — not diverge.
    """
    nll, n_tok = 0.0, 0
    seqs = test_windows[:N_EVAL]
    for i in range(0, len(seqs), B_EVAL):
        batch = seqs[i : i + B_EVAL]
        x1 = token_ids_to_features(batch.cpu(), K, label_smoothing=LS).to(device)
        x_init = x1 + 0.1 * torch.randn_like(x1)
        x_init = x_init - x_init.mean(-1, keepdim=True)
        x = _sample(
            model,
            len(batch),
            L,
            K,
            n_steps=n_steps,
            eta=ETA,
            mu=mu,
            device=device,
            x_init=x_init,
        )
        lp = _decode_logprobs(x)
        nll += F.nll_loss(lp.reshape(-1, K), batch.reshape(-1), reduction="sum").item()
        n_tok += batch.numel()
    return nll / (n_tok * math.log(2))


bpd_gd, bpd_nag = [], []
for n in N_STEPS_SWEEP:
    b0, b1 = _bpd(n, 0.0), _bpd(n, MU_NAG)
    bpd_gd.append(b0)
    bpd_nag.append(b1)
    print(f"  n={n:4d}  GD={b0:.4f}  NAG={b1:.4f}")


# ── B: Labeled flow trajectories ──────────────────────────────────────────────

print("B: Labeled trajectories…")
traj_seqs = test_windows[:N_TRAJ]
M = N_TRAJ * L

snaps = _sample_snapshots(
    model,
    N_TRAJ,
    L,
    K,
    n_steps=N_STEPS_MAIN,
    eta=ETA,
    mu=MU_NAG,
    n_snaps=SNAP_COUNT,
    device=device,
    sigma=SIGMA,
)
T = len(snaps)

# Project CLR snapshots → ILR → PCA 2-D.
# ILR is the proper Euclidean embedding of the simplex (no redundant dimension).
snap_ilr = [ilr(s.reshape(M, K)) for s in snaps]  # each: (M, K-1)
pca = PCA(n_components=2)
pca.fit(snap_ilr[-1].numpy())
traj_2d = np.stack([pca.transform(s.numpy()) for s in snap_ilr])  # (T, M, 2)

# Ground-truth CLR → ILR → PCA
gt_clr_traj = token_ids_to_features(traj_seqs.cpu(), K, label_smoothing=LS)
gt_2d = pca.transform(ilr(gt_clr_traj.reshape(M, K)).numpy())  # (M, 2)

# Label each token position by its ground-truth character group
true_ids = traj_seqs.cpu().reshape(M).tolist()
groups = [_char_group(i) for i in true_ids]


# ── C: Unigram character frequency ────────────────────────────────────────────

print("C: Unigram frequency…")
gen_ids_list: list[torch.Tensor] = []
for i in range(0, N_GEN, B_EVAL):
    B_cur = min(B_EVAL, N_GEN - i)
    x = _sample(
        model, B_cur, L, K, n_steps=N_STEPS_MAIN, eta=ETA, mu=MU_NAG, device=device
    )
    gen_ids_list.append(_decode_logprobs(x).argmax(-1).cpu())

gen_ids = torch.cat(gen_ids_list)  # (N_GEN, L)

gen_freq = torch.zeros(K)
gen_freq.scatter_add_(0, gen_ids.reshape(-1), torch.ones(N_GEN * L))
gen_freq /= gen_freq.sum()

gt_freq = torch.zeros(K)
gt_freq.scatter_add_(0, test_windows[:N_GEN].cpu().reshape(-1), torch.ones(N_GEN * L))
gt_freq /= gt_freq.sum()

kl_uni = float(
    F.kl_div((gen_freq + 1e-9).log(), gt_freq + 1e-9, reduction="sum").item()
)


# ── D: Bigram log-ratio heatmap ────────────────────────────────────────────────

print("D: Bigram statistics…")


def _bigram_trans(ids_2d: torch.Tensor) -> np.ndarray:
    """Row-normalised K×K bigram transition matrix (row=prev, col=next)."""
    counts = np.zeros((K, K), dtype=np.float64)
    rows = ids_2d[:, :-1].numpy().ravel()
    cols = ids_2d[:, 1:].numpy().ravel()
    np.add.at(counts, (rows, cols), 1)
    return counts / counts.sum(axis=1, keepdims=True).clip(min=1)


bg_gen = _bigram_trans(gen_ids)
bg_gt = _bigram_trans(test_windows[:N_GEN].cpu())
# log₂ ratio: positive ↔ over-generated, negative ↔ under-generated
bg_logratio = np.log2((bg_gen + 1e-6) / (bg_gt + 1e-6))


# ── E: Gradient norm convergence ──────────────────────────────────────────────

print("E: Gradient norm convergence…")
grad_norms = _sample_grad_norms(
    model, B_EVAL, L, K, n_steps=N_STEPS_GRAD, eta=ETA, mu=MU_NAG, device=device
)


# ── F: Energy gradient at GT vs generated ─────────────────────────────────────
# A well-trained EqM model should assign near-zero gradient to both GT data
# points and to its own generated samples (both should be near energy minima).

print("F: Energy at GT vs generated…")
gt_clr_eval = token_ids_to_features(
    test_windows[:N_EVAL].cpu(), K, label_smoothing=LS
).to(device)
gt_gnorms = _grad_norms_at(model, gt_clr_eval)

gen_x_eval = _sample(
    model, N_EVAL, L, K, n_steps=N_STEPS_MAIN, eta=ETA, mu=MU_NAG, device=device
)
gen_gnorms = _grad_norms_at(model, gen_x_eval)


# ── G: Generated text samples ─────────────────────────────────────────────────

print(f"G: Generating text samples (decode={DECODE_MODE}, T={DECODE_TEMPERATURE})…")
x_txt = _sample(
    model, N_SAMPLES, L, K, n_steps=N_STEPS_MAIN, eta=ETA, mu=MU_NAG, device=device
)
generated = _tokens_to_text(_decode(_decode_logprobs(x_txt)))


# ── Plotting ──────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(20, 20))
gs = gridspec.GridSpec(
    3,
    3,
    figure=fig,
    hspace=0.45,
    wspace=0.35,
    height_ratios=[1, 1, 0.75],
)
fig.suptitle(
    f"EqM evaluation — epoch {ckpt['epoch']}  (η={ETA}, μ_NAG={MU_NAG}, L={L}, K={K})",
    fontsize=14,
    y=0.99,
)

# ── A ─────────────────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
ax.plot(N_STEPS_SWEEP, bpd_gd, "o-", lw=2, color="steelblue", label="GD  (μ=0)")
ax.plot(
    N_STEPS_SWEEP, bpd_nag, "s-", lw=2, color="darkorange", label=f"NAG-GD (μ={MU_NAG})"
)
ax.set_xscale("log")
ax.set_xlabel("GD steps")
ax.set_ylabel("BPD")
ax.set_title("A — Reconstruction BPD vs GD steps")
ax.legend(fontsize=9)
ax.grid(True, which="both", alpha=0.3)

# ── B ─────────────────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
# Trajectory lines, coloured by ground-truth character group
for m in range(M):
    ax.plot(
        traj_2d[:, m, 0],
        traj_2d[:, m, 1],
        color=GROUP_COLORS[groups[m]],
        lw=0.4,
        alpha=0.2,
    )

# Endpoints and ground-truth markers, one scatter per group
for g_idx, label, color in zip(range(4), GROUP_LABELS, GROUP_COLORS):
    mask = np.array(groups) == g_idx
    if not mask.any():
        continue
    # Generated endpoints (circles)
    ax.scatter(
        traj_2d[-1, mask, 0],
        traj_2d[-1, mask, 1],
        s=14,
        c=color,
        alpha=0.75,
        zorder=3,
        label=f"gen: {label}",
    )
    # Ground-truth targets (stars)
    ax.scatter(
        gt_2d[mask, 0],
        gt_2d[mask, 1],
        s=35,
        c=color,
        marker="*",
        alpha=0.95,
        zorder=4,
        edgecolors="k",
        linewidths=0.3,
    )

# Mark the common starting region (all x₀ cluster near ILR origin)
ax.scatter(
    traj_2d[0, :1, 0],
    traj_2d[0, :1, 1],
    s=120,
    c="white",
    edgecolors="black",
    lw=1.5,
    zorder=5,
    label="x₀ (noise)",
)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
ax.set_title(
    "B — Labeled flow trajectories (ILR/PCA)\n"
    "circles = generated  ★ = ground-truth  colour = char group"
)
ax.legend(fontsize=6, loc="upper left", ncol=1)

# ── C ─────────────────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
x_pos = np.arange(K)
w = 0.38
ax.bar(
    x_pos - w / 2,
    gt_freq.numpy(),
    w,
    color="steelblue",
    alpha=0.85,
    label="ground truth",
)
ax.bar(
    x_pos + w / 2,
    gen_freq.numpy(),
    w,
    color="darkorange",
    alpha=0.85,
    label=f"generated  KL={kl_uni:.3f} nats",
)
ax.set_xticks(x_pos)
ax.set_xticklabels(tick_labels, fontsize=6)
ax.set_xlabel("Character")
ax.set_ylabel("Frequency")
ax.set_title(
    f"C — Unigram character frequency\n({N_STEPS_MAIN} GD steps, {N_GEN} sequences)"
)
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

# ── D ─────────────────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
lim = max(float(np.abs(bg_logratio).max()), 0.5)
im = ax.imshow(
    bg_logratio,
    aspect="auto",
    cmap="RdBu_r",
    vmin=-lim,
    vmax=lim,
    interpolation="nearest",
)
ax.set_xticks(range(K))
ax.set_yticks(range(K))
ax.set_xticklabels(tick_labels, fontsize=5, rotation=90)
ax.set_yticklabels(tick_labels, fontsize=5)
ax.set_xlabel("Next character (column)")
ax.set_ylabel("Previous character (row)")
ax.set_title(
    "D — Bigram log₂(gen / GT)\n+red = over-generated  −blue = under-generated"
)
fig.colorbar(im, ax=ax, shrink=0.85, label="log₂ ratio")

# ── E ─────────────────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
steps_e = range(len(grad_norms))
ax.plot(steps_e, grad_norms, lw=1.5, color="mediumseagreen")
ax.axhline(
    cfg.eqm.sample_g_min,
    ls="--",
    lw=1.2,
    color="crimson",
    alpha=0.85,
    label=f"g_min = {cfg.eqm.sample_g_min}",
)
ax.set_xlabel("GD step")
ax.set_ylabel("Mean ‖∇E(x)‖")
ax.set_title("E — Gradient norm convergence (NAG-GD)")
ax.set_yscale("log")
ax.legend(fontsize=9)
ax.grid(True, which="both", alpha=0.3)

# ── F ─────────────────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 2])
bins = np.linspace(0, max(float(gt_gnorms.max()), float(gen_gnorms.max())) * 1.05, 50)
ax.hist(
    gt_gnorms,
    bins=bins,
    color="steelblue",
    alpha=0.7,
    density=True,
    label=f"GT  (μ={gt_gnorms.mean():.2f})",
)
ax.hist(
    gen_gnorms,
    bins=bins,
    color="darkorange",
    alpha=0.7,
    density=True,
    label=f"Generated  (μ={gen_gnorms.mean():.2f})",
)
ax.set_xlabel("‖∇E(x)‖  per sequence")
ax.set_ylabel("Density")
ax.set_title(
    "F — Energy gradient at GT vs generated\n"
    "lower = closer to an energy minimum ('on manifold')"
)
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

# ── G: Generated text (spans all 3 columns) ───────────────────────────────────
ax = fig.add_subplot(gs[2, :])
ax.axis("off")
ax.set_title(
    f"G — Generated text samples  (NAG-GD, {N_STEPS_MAIN} steps, argmax decoding)",
    pad=8,
    fontsize=11,
)
n_cols_txt = 2
n_rows_g = math.ceil(len(generated) / n_cols_txt)
for i, text in enumerate(generated):
    col = i % n_cols_txt
    row = i // n_cols_txt
    ax.text(
        col / n_cols_txt + 0.01,
        1.0 - (row + 0.8) / n_rows_g,
        f'{i + 1:2d}. "{text}"',
        transform=ax.transAxes,
        fontsize=9,
        fontfamily="monospace",
        va="center",
    )

out = Path(__file__).parent / "evaluate_eqm.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out}")
plt.show()
