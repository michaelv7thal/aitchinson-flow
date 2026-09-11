"""Does the descent sharpen to the WRONG vertex, or does it do nothing?

`recovery_check.py` already shows that under the isotropic perturbation the
decoded output equals the decoded perturbed input character for character at
every alpha.  That is consistent with two very different behaviours:

  (a) the field is inert off-manifold and hands the state back untouched, or
  (b) the field applies the same rule it applies on its own path -- sharpen to
      whichever vertex currently dominates -- and simply lands on the wrong
      vertex wherever the noise flipped the ordering.

The radius separates them.  Under (a) the output stays out at the perturbed
radius 12.27*sqrt(1+K*alpha^2).  Under (b) it collapses onto the data shell at
~12.27, a sharp one-hot, exactly as it does on the native path.

Also reports agreement with the perturbed input's argmax (should be ~1.0 under
both) alongside agreement with the ground truth, so the two are never confused.

Run:
    uv run python scripts/recovery_isotropic_probe.py \
        --ckpt runs/compu_mse_det/epoch_final.pt --alphas 0.2,0.3,0.4,0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def _bootstrap() -> None:
    root = Path(__file__).resolve().parent.parent
    for q in (root / "src", root):
        if str(q) not in sys.path:
            sys.path.insert(0, str(q))


_bootstrap()

import aitchinson_flow.models  # noqa: E402,F401
from aitchinson_flow.data.char_window_dataset import CHAR2ID  # noqa: E402
from aitchinson_flow.models import build_model  # noqa: E402
from aitchinson_flow.training import build_training_datamodule  # noqa: E402
from scripts.eval_full import _config_from_payload  # noqa: E402

ALPHABET = "".join(sorted(CHAR2ID, key=CHAR2ID.__getitem__))


def _dec(row):
    return "".join(ALPHABET[i] for i in row.tolist())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--alphas", default="0.2,0.3,0.4,0.5")
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
    val_ids = dm.splits.val.long()
    K, L = cfg.text8_dataset.K, cfg.text8_dataset.L

    from aitchinson_flow.data.transforms import token_ids_to_features

    val_pick = val_ids[: args.n].to(device)
    x1 = token_ids_to_features(val_pick, K, label_smoothing=cfg.transformation.label_smoothing)
    embed_norm = x1.norm(dim=-1).mean().item()

    print(f"data radius/pos {embed_norm:.4f}")
    print("Isotropic perturbation, exactly as scripts/recovery_check.py builds it\n")
    print(f"{'alpha':>6}{'r_in':>8}{'r_out':>8}{'acc_gt':>8}{'acc_pt':>8}"
          f"{'=pt argmax':>12}{'p_true_in':>11}{'p_true_out':>12}")

    rows = []
    for a in [float(v) for v in args.alphas.split(",") if v.strip()]:
        torch.manual_seed(args.seed + int(a * 1000))
        z = x1 + (a * embed_norm) * torch.randn_like(x1)
        with torch.no_grad():
            lp_in = model.decode_to_logprobs(z)
            ids_pt = lp_in.argmax(-1).cpu()
            x = model.sample(args.n, L, x_init=z, max_steps=args.steps)
            lp_out = model.decode_to_logprobs(x)
            ids = lp_out.argmax(-1).cpu()
        tgt = val_pick.unsqueeze(-1)
        row = {
            "alpha": a,
            "radius_in": z.norm(dim=-1).mean().item(),
            "radius_out": float(x.norm(dim=-1).mean()),
            "acc_vs_truth": float((ids == val_pick.cpu()).float().mean()),
            "acc_pt_vs_truth": float((ids_pt == val_pick.cpu()).float().mean()),
            "agrees_with_perturbed_argmax": float((ids == ids_pt).float().mean()),
            "p_true_in": float(lp_in.gather(-1, tgt).exp().mean()),
            "p_true_out": float(lp_out.gather(-1, tgt).exp().mean()),
            "pt_sample0": _dec(ids_pt[0]),
            "rc_sample0": _dec(ids[0]),
        }
        rows.append(row)
        print(f"{a:>6}{row['radius_in']:>8.2f}{row['radius_out']:>8.2f}"
              f"{row['acc_vs_truth']:>8.4f}{row['acc_pt_vs_truth']:>8.4f}"
              f"{row['agrees_with_perturbed_argmax']:>12.4f}"
              f"{row['p_true_in']:>11.4f}{row['p_true_out']:>12.4f}")

    print("\nsample[0]:")
    for r in rows:
        print(f"  a={r['alpha']}  pt='{r['pt_sample0']}'")
        print(f"{'':10}rc='{r['rc_sample0']}'")

    out = args.out or str(Path(args.ckpt).parent / "isotropic_probe.json")
    json.dump({"data_radius": embed_norm, "rows": rows}, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
