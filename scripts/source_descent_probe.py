"""What the EqM descent does when it starts at the source (gamma = 0).

Why this exists
---------------
``fig:image-text`` in the paper's conclusion used to put mu_1, the unigram
mean, at the centre of the simplex panel, as the point the descent ends at.
That is not what happens.  The sampler starts at the random source x_0 and the
field sharpens whichever vertex that draw already leans on, so the sample is a
one-hot sequence out at the data radius and mu_1 only tilts the choice of
character.  ``scripts/recovery_native_path.py`` shows the endpoint of this
(alpha=1.0: accuracy 0.082, log p(true) = -11.45) but not the mechanism.  This
script measures the mechanism, on the same checkpoint:

* the radius and the mean argmax probability of the returned iterate, against
  the data radius 12.27 and mu_1's 2.43;
* how often the returned argmax is the character the random source already
  pointed at, against the chance rate implied by the two marginals;
* how early in the descent each position's final character is fixed;
* the unigram and uniform KL of the source and of the sample.

Run:
    uv run python scripts/source_descent_probe.py \
        --ckpt runs/compu_mse_det/epoch_final.pt --n 256 --steps 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def _bootstrap() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))


_bootstrap()

import aitchinson_flow.models  # noqa: E402,F401
from aitchinson_flow.data.char_window_dataset import CHAR2ID  # noqa: E402
from aitchinson_flow.models import build_model  # noqa: E402
from aitchinson_flow.training import build_training_datamodule  # noqa: E402
from scripts.eval_full import _config_from_payload  # noqa: E402

ALPHABET = "".join(sorted(CHAR2ID, key=CHAR2ID.__getitem__))
SNAPSHOTS = (1, 2, 3, 5, 10, 20, 50, 100, 200)


def _decode(ids_row: torch.Tensor) -> str:
    return "".join(ALPHABET[i] for i in ids_row.tolist())


def _freq(ids: torch.Tensor, K: int) -> torch.Tensor:
    c = torch.bincount(ids.reshape(-1), minlength=K).float()
    return c / c.sum()


def _kl(p: torch.Tensor, q: torch.Tensor) -> float:
    p, q = p.clamp_min(1e-12), q.clamp_min(1e-12)
    return float((p * (p.log() - q.log())).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = _config_from_payload(payload)
    device = cfg.training.device
    model = build_model(cfg).to(device)
    model.load_state_dict(payload.get("model_state_dict", payload))
    model.eval()

    dm, _ = build_training_datamodule(cfg)
    K, L = cfg.text8_dataset.K, cfg.text8_dataset.L
    s = cfg.eqm
    p_uni = _freq(dm.splits.train.long(), K)
    p_unif = torch.full((K,), 1.0 / K)

    # The sampler of sec:eqm-gen-method: NAG-GD on the conservative gradient,
    # per-position gradient clipping, lowest-gradient iterate returned. Unrolled
    # here rather than calling model.sample() so the trajectory is observable.
    def clip(g: torch.Tensor) -> torch.Tensor:
        n = g.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        return g * (n.clamp(max=s.sample_grad_clip) / n)

    def grad_of(z: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            zr = z.detach().requires_grad_(True)
            energy = (zr * model.forward(zr, None)).sum()
            return torch.autograd.grad(energy, zr)[0].detach()

    torch.manual_seed(args.seed)
    x = s.source_sigma * torch.randn(args.n, L, K, device=device)
    x = x - x.mean(-1, keepdim=True)
    x0 = x.clone()
    ids0 = x0.argmax(-1)

    x_last = x.clone()
    g = clip(grad_of(x))
    best_x, best_g = x.clone(), g.norm(dim=-1).mean().item()
    traj, snaps = [], {}
    for t in range(1, args.steps + 1):
        if g.reshape(args.n, -1).norm(dim=-1).max() < s.sample_g_min:
            break
        x_last = x
        x = x - s.sample_eta * g
        g = clip(grad_of(x + s.sample_mu * (x - x_last)))
        gm = g.norm(dim=-1).mean().item()
        if gm < best_g:
            best_g, best_x = gm, x.clone()
        ids_t = x.argmax(-1)
        traj.append({
            "step": t,
            "grad_norm": gm,
            "radius": float(x.norm(dim=-1).mean()),
            "agree_with_source_argmax": float((ids_t == ids0).float().mean()),
            "kl_unigram": _kl(_freq(ids_t.cpu(), K), p_uni),
        })
        if t in SNAPSHOTS:
            snaps[t] = ids_t.clone()

    ids_out = best_x.argmax(-1)
    p_in, p_out = _freq(ids0.cpu(), K), _freq(ids_out.cpu(), K)
    out = {
        "ckpt": args.ckpt,
        "n": args.n,
        "L": L,
        "steps_run": len(traj),
        "data_radius_per_pos": 12.2723,
        "mu1_norm_per_pos": 2.4284,
        "source": {
            "radius": float(x0.norm(dim=-1).mean()),
            "kl_unigram": _kl(p_in, p_uni),
            "kl_uniform": _kl(p_in, p_unif),
        },
        "sample": {
            "radius": float(best_x.norm(dim=-1).mean()),
            "mean_max_prob": float(best_x.softmax(-1).max(-1).values.mean()),
            "agree_with_source_argmax": float((ids_out == ids0).float().mean()),
            "chance_agreement": float((p_out * p_in).sum()),
            "kl_unigram": _kl(p_out, p_uni),
            "kl_uniform": _kl(p_out, p_unif),
        },
        # fraction of positions already carrying their final character at step t
        "argmax_settled": {
            str(t): float((v == ids_out).float().mean()) for t, v in snaps.items()
        },
        "sample_strings": {
            "source_argmax": _decode(ids0[0].cpu()),
            "sample": _decode(ids_out[0].cpu()),
        },
        "trajectory": traj,
    }
    print(json.dumps({k: v for k, v in out.items() if k != "trajectory"}, indent=2))

    dest = args.out or str(Path(args.ckpt).parent / "source_descent.json")
    Path(dest).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
