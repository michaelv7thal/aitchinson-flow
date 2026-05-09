"""MNIST EqM sanity — standard equilibrium flow matching on continuous pixel data.

Goal: show that EqM (conservative-gradient FM, NAG-GD sampling) works on
continuous high-dim data. This is the "no simplex tricks needed" baseline
that establishes the framework's generality. Combined with the
Hilbert counter-example (interior compositional) and the latent-EqM text
result (learned embeddings on K=27), it bounds the regime of applicability.

Setup:
  * Data: MNIST 28×28 grayscale, normalised to ``x in [-1, 1]``, flattened
    to 784-dim vectors. Labels are unused (unconditional generation).
  * Model: small MLP velocity field f(x, γ) with sinusoidal γ time
    conditioning. Conservative-gradient parameterisation: model outputs
    f, FM target is c(γ)·(x_0 - x_1), regression on ∇⟨x, f⟩.
  * Loss: plain MSE (no Hilbert, no CLR — pixels aren't compositional).
  * Sampler: NAG-GD on the conservative gradient.

Eval:
  * Forward FID-lite: compare per-pixel mean/std of generated samples to
    the test split.
  * Diagonal-bin pixel-marginal KL across 256 bins.
  * Visual: save a 8×8 sample grid as PNG.

Usage:
    python scripts/eqm_mnist_sanity.py --out runs/eqm_mnist --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _bootstrap() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_bootstrap()


# ─── Data ───────────────────────────────────────────────────────────────────


def load_mnist(split: str = "train", *, max_n: int | None = None) -> torch.Tensor:
    """Load MNIST as a (N, 784) float tensor in [-1, 1].

    Tries torchvision first; if unavailable, falls back to keras-style
    URL fetch via ``urllib`` and the official IDX format. For the sanity
    run we just need either to work.
    """
    try:
        from torchvision import datasets, transforms
        tr = transforms.Compose([transforms.ToTensor()])
        ds = datasets.MNIST(
            root=str(Path("data") / "mnist_torchvision"),
            train=(split == "train"),
            download=True,
            transform=tr,
        )
        n = len(ds) if max_n is None else min(max_n, len(ds))
        x = torch.stack([ds[i][0] for i in range(n)], dim=0)  # (n, 1, 28, 28)
    except ImportError:
        # Plan B: HuggingFace datasets has MNIST too.
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "Either torchvision or huggingface datasets is required for "
                "MNIST loading."
            ) from e
        ds = load_dataset("mnist", split=split)
        if max_n is not None:
            ds = ds.select(range(min(max_n, len(ds))))
        import numpy as np
        arr = np.stack([np.asarray(im, dtype="uint8") for im in ds["image"]], 0)
        x = torch.from_numpy(arr).float().unsqueeze(1) / 255.0  # (n, 1, 28, 28)
    x = x.view(x.shape[0], -1)              # (n, 784)
    return x * 2.0 - 1.0                     # [-1, 1]


# ─── Model ──────────────────────────────────────────────────────────────────


def sinusoidal_embed(t: torch.Tensor, d: int) -> torch.Tensor:
    half = d // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
    args = t[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class MLPVelocity(nn.Module):
    """Velocity field f: R^784 → R^784 with γ time conditioning. Modest
    capacity (~2-3M params) — enough to learn coarse MNIST structure on
    a laptop GPU."""

    def __init__(self, D: int = 784, d: int = 512, time_d: int = 64) -> None:
        super().__init__()
        self.time_d = time_d
        self.time_proj = nn.Linear(time_d, d)
        self.in_proj = nn.Linear(D, d)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d), nn.GELU())
            for _ in range(4)
        ])
        self.out_proj = nn.Linear(d, D)

    def forward(self, x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x) + self.time_proj(sinusoidal_embed(gamma, self.time_d))
        for b in self.blocks:
            h = h + b(h)
        return self.out_proj(h)


# ─── Training step ───────────────────────────────────────────────────────────


def fm_step(
    model: MLPVelocity,
    x1: torch.Tensor,
    *,
    sigma: float,
    gamma_power: float,
) -> torch.Tensor:
    B, D = x1.shape
    device = x1.device
    x0 = sigma * torch.randn(B, D, device=device)
    gamma = torch.rand(B, device=device).pow(gamma_power)
    x_g = (1 - gamma[:, None]) * x0 + gamma[:, None] * x1
    x_g.requires_grad_(True)
    u_tgt = (1.0 - gamma)[:, None] * (x0 - x1)

    v = model(x_g, gamma)
    energy = (x_g * v).sum()
    grad_g = torch.autograd.grad(energy, x_g, create_graph=True)[0]
    return F.mse_loss(grad_g, u_tgt)


# ─── Sampling ────────────────────────────────────────────────────────────────


@torch.no_grad()
def sample_nag(
    model: MLPVelocity,
    n: int,
    D: int,
    *,
    sigma: float,
    eta: float = 0.1,
    mu: float = 0.9,
    max_steps: int = 200,
    grad_clip: float = 1.0,
) -> torch.Tensor:
    device = next(model.parameters()).device
    x = sigma * torch.randn(n, D, device=device)

    def grad_fn(x_in: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            x_req = x_in.detach().requires_grad_(True)
            gamma_one = torch.ones(x_req.shape[0], device=device)
            energy = (x_req * model(x_req, gamma_one)).sum()
            g = torch.autograd.grad(energy, x_req, create_graph=False)[0].detach()
            n_per = g.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            return g * (n_per.clamp(max=grad_clip) / n_per)

    x_last = x.clone()
    g = grad_fn(x)
    for _ in range(max_steps):
        x_last = x
        x = x - eta * g
        g = grad_fn(x + mu * (x - x_last))
    return x


# ─── Eval ───────────────────────────────────────────────────────────────────


def pixel_marginal_kl(gen: torch.Tensor, ref: torch.Tensor, *, bins: int = 64) -> float:
    """KL between aggregate pixel-value histograms (across all pixels and
    samples). Cheap proxy for "model captures the marginal pixel
    distribution"."""
    edges = torch.linspace(-1.0, 1.0, bins + 1)
    g_hist = torch.histc(gen, bins=bins, min=-1.0, max=1.0)
    r_hist = torch.histc(ref, bins=bins, min=-1.0, max=1.0)
    smoothing = 1e-6
    g_p = (g_hist + smoothing) / (g_hist.sum() + bins * smoothing)
    r_p = (r_hist + smoothing) / (r_hist.sum() + bins * smoothing)
    return float((g_p * (g_p.log() - r_p.log())).sum())


def save_grid_png(samples: torch.Tensor, path: Path, n: int = 64) -> None:
    """Save an n=64 grid (8×8) of generated samples to ``path`` as an 8-bit PNG."""
    try:
        import torchvision  # noqa: F401
        from torchvision.utils import save_image
    except ImportError:
        # Fallback: write a flat grayscale grid via PIL.
        try:
            from PIL import Image
        except ImportError:
            return
        n = min(n, samples.shape[0])
        side = int(n ** 0.5)
        n = side * side
        x = samples[:n].clamp(-1, 1).add(1).div(2).cpu().view(n, 28, 28)
        rows = []
        for r in range(side):
            rows.append(torch.cat([x[r * side + c] for c in range(side)], dim=1))
        grid = torch.cat(rows, dim=0)
        Image.fromarray((grid * 255).clamp(0, 255).numpy().astype("uint8"), "L").save(path)
        return
    n = min(n, samples.shape[0])
    side = int(n ** 0.5)
    n = side * side
    x = samples[:n].clamp(-1, 1).add(1).div(2).view(n, 1, 28, 28).cpu()
    save_image(x, str(path), nrow=side)


# ─── Driver ─────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str, default="runs/eqm_mnist")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-eval", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--sigma", type=float, default=0.5,
                    help="source σ — match to data std (~0.5 after normalize).")
    ap.add_argument("--gamma-power", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=200,
                    help="NAG-GD sample steps")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    print(f"Loading MNIST (max_n={args.n_train})…")
    train_x = load_mnist("train", max_n=args.n_train).to(device)
    eval_x = load_mnist("test", max_n=args.n_eval).to(device)
    D = train_x.shape[-1]
    print(f"  train {tuple(train_x.shape)}, eval {tuple(eval_x.shape)}, D={D}")

    model = MLPVelocity(D=D, d=512).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,d}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    n_steps = (train_x.shape[0] // args.batch) * args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_steps)

    history: list[dict] = []
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(train_x.shape[0], device=device)
        losses = []
        for i in range(0, train_x.shape[0], args.batch):
            idx = perm[i : i + args.batch]
            x1 = train_x[idx]
            loss = fm_step(model, x1, sigma=args.sigma, gamma_power=args.gamma_power)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            losses.append(float(loss.detach()))
        avg = sum(losses) / max(len(losses), 1)
        print(f"epoch {ep + 1}/{args.epochs}: flow_loss={avg:.4f}  "
              f"lr={optim.param_groups[0]['lr']:.2e}")
        history.append({"epoch": ep + 1, "flow_loss": avg,
                        "lr": optim.param_groups[0]["lr"]})

    # Sample.
    model.eval()
    print(f"Sampling {args.n_eval} via NAG (max_steps={args.max_steps})…")
    gen = sample_nag(model, args.n_eval, D, sigma=args.sigma, max_steps=args.max_steps)

    pkl = pixel_marginal_kl(gen, eval_x)
    gen_mean = float(gen.mean()); gen_std = float(gen.std())
    ref_mean = float(eval_x.mean()); ref_std = float(eval_x.std())
    print(f"  pixel_marginal_KL={pkl:.4f}  "
          f"gen_mean/std={gen_mean:.3f}/{gen_std:.3f}  "
          f"ref_mean/std={ref_mean:.3f}/{ref_std:.3f}")

    grid_path = out / "samples.png"
    try:
        save_grid_png(gen, grid_path, n=64)
        print(f"  wrote sample grid {grid_path}")
    except Exception as e:
        print(f"  could not save grid: {e}")

    record = {
        "pixel_marginal_kl": pkl,
        "gen_mean": gen_mean,
        "gen_std": gen_std,
        "ref_mean": ref_mean,
        "ref_std": ref_std,
        "history": history,
        "n_params": n_params,
        "args": vars(args),
    }
    (out / "results.json").write_text(json.dumps(record, indent=2))
    print(f"Wrote {out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
