"""Recovery on EqM's OWN training path, for the deterministic-CLR arm.

Why this exists
---------------
``scripts/recovery_check.py`` perturbs EqM by adding isotropic Gaussian noise
in CLR feature space::

    sig_perturb = alpha * embed_norm            # embed_norm = 12.2723
    z_init = z_clean + sig_perturb * randn_like(z_clean)

``embed_norm`` is the per-POSITION norm but ``randn_like`` is per-COORDINATE,
so the per-position displacement is ``alpha * 12.2723 * sqrt(27) ~ alpha*63.8``
and the initialisation sits at radius ``12.27*sqrt(1+27*alpha^2)``.  The
training interpolant never exceeds radius 12.27, so every alpha that visibly
corrupts text starts the descent 1.9x-5.3x outside the trained region.

Every transport arm, by contrast, is perturbed by its OWN training corruption
(DirichletFM: a partial-path Dirichlet draw; DFM: uniform token corruption at
keep-prob 1-alpha; SFM: a partial geodesic).  This script gives EqM the same
treatment: initialise at the training interpolant

    x_gamma = (1-gamma)*x0 + gamma*x1,   gamma = 1-alpha,   x0 ~ N(0, source_sigma^2)

so alpha=0 is clean data and alpha=1 is EqM's own source noise, then descend.

What to read off
----------------
``acc_pt`` is the pre-descent accuracy of the initialisation.  If it stays at
1.0 across the sweep, EqM's native corruption never makes a token ambiguous
and there is nothing for a recovery test to repair -- the recovery experiment
is not constructible from EqM's own path, which is itself the result.
``delta`` then measures whether the descent DAMAGES an in-distribution input.

Run:
    uv run python scripts/recovery_native_path.py \
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


def _decode(ids_row: torch.Tensor) -> str:
    return "".join(ALPHABET[i] for i in ids_row.tolist())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--alphas", default="0.05,0.1,0.2,0.3,0.4,0.5,0.9,0.99,1.0")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = _config_from_payload(payload)
    device = cfg.training.device
    model = build_model(cfg).to(device)
    state = payload.get("model_state_dict", payload)
    model.load_state_dict(state)
    model.eval()

    dm, _ = build_training_datamodule(cfg)
    val_ids = dm.splits.val.long()
    K = cfg.text8_dataset.K
    L = cfg.text8_dataset.L
    sigma = cfg.eqm.source_sigma

    from aitchinson_flow.data.transforms import token_ids_to_features

    val_pick = val_ids[: args.n].to(device)
    ls = cfg.transformation.label_smoothing
    x1 = token_ids_to_features(val_pick, K, label_smoothing=ls)
    data_r = x1.norm(dim=-1).mean().item()

    print(f"ckpt            {args.ckpt}")
    print(f"model           {cfg.training.model_name}")
    print(f"dirichlet_samp  {cfg.transformation.dirichlet_sampling}")
    print(f"gamma_power     {cfg.eqm.gamma_power}")
    print(f"source_sigma    {sigma}")
    print(f"data radius/pos {data_r:.4f}   (max radius the training path reaches)")
    print()
    print("Native-path recovery:  x_init = (1-g)*x0 + g*x1,  g = 1-alpha")
    print(f"{'alpha':>7}{'gamma':>7}{'r_init':>9}{'r_out':>9}{'acc_pt':>8}{'acc_rc':>8}"
          f"{'lp_pt':>9}{'lp_rc':>9}")

    rows = []
    for a in [float(v) for v in args.alphas.split(",") if v.strip()]:
        torch.manual_seed(args.seed + int(a * 1000))
        g = 1.0 - a
        x0 = sigma * torch.randn_like(x1)
        x_init = (1.0 - g) * x0 + g * x1
        with torch.no_grad():
            lp_pt = model.decode_to_logprobs(x_init)
            ids_pt = lp_pt.argmax(-1).cpu()
            x = model.sample(args.n, L, x_init=x_init, max_steps=args.steps)
            lp_rc = model.decode_to_logprobs(x)
            ids = lp_rc.argmax(-1).cpu()
        acc_pt = float((ids_pt == val_pick.cpu()).float().mean())
        acc = float((ids == val_pick.cpu()).float().mean())
        # Argmax is scale-invariant along the ray, so it cannot see the state
        # being dragged towards the basin. These two can.
        tgt = val_pick.unsqueeze(-1)
        logp_pt = float(lp_pt.gather(-1, tgt).mean())
        logp_rc = float(lp_rc.gather(-1, tgt).mean())
        r = x_init.norm(dim=-1).mean().item()
        r_rc = float(x.norm(dim=-1).mean())
        print(f"{a:>7}{g:>7.2f}{r:>9.2f}{r_rc:>9.2f}{acc_pt:>8.4f}{acc:>8.4f}"
              f"{logp_pt:>9.3f}{logp_rc:>9.3f}")
        rows.append(
            {
                "alpha": a,
                "gamma": g,
                "radius": r,
                "radius_out": r_rc,
                "logp_true_pre": logp_pt,
                "logp_true_post": logp_rc,
                "token_acc_perturbed": acc_pt,
                "token_acc": acc,
                "delta": acc - acc_pt,
                "gt_sample0": _decode(val_pick[0].cpu()),
                "pt_sample0": _decode(ids_pt[0]),
                "rc_sample0": _decode(ids[0]),
            }
        )

    print("\nsample[0] at each alpha (gt / pre-descent / post-descent):")
    for r in rows:
        print(f"  a={r['alpha']:<5} gt='{r['gt_sample0']}'")
        print(f"{'':11}pt='{r['pt_sample0']}'")
        print(f"{'':11}rc='{r['rc_sample0']}'")

    out = args.out or str(Path(args.ckpt).parent / "recovery_native_path.json")
    json.dump({"data_radius": data_r, "rows": rows}, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
