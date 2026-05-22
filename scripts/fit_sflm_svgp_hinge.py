"""Stage 2 (hinge-trained SVGP) for SFLMSvgp.

Reads a frozen Stage-1 SFLM checkpoint (typically
``runs/sflm_bench_<scale>/SFLM/epoch_final.pt``), builds the
:class:`SFLMSvgp` wrapper around it, and trains the pooler + SVGP head +
energy head via the contrastive energy hinge using
``batch['token_ids_invalid']`` from :class:`CorruptingCollate` as the
negatives.

Mirrors ``scripts/fit_dfm_svgp_hinge.py`` — the comparison target is one
recipe applied to two generators (DFM vs SFLM) so the cluster numbers
are apples-to-apples.

Outputs:
  <out-dir>/model_with_svgp_hinge.pt   (state_dict of the wrapped SFLMSvgp)
  <out-dir>/sflm_svgp_hinge_eval.json  (per-split AUROC of probability)

Usage:
    python scripts/fit_sflm_svgp_hinge.py \\
        --ckpt runs/sflm_bench_cluster/SFLM/epoch_final.pt \\
        --n-epochs 5 --lr 1e-3 --eval-n 500
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
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
from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.data.corruption import (  # noqa: E402
    corrupt_token_ids, partially_shuffle_token_ids,
)
from aitchinson_flow.models import build_model  # noqa: E402
from aitchinson_flow.models.sflm_svgp import SFLMSvgp  # noqa: E402
from aitchinson_flow.training import build_training_datamodule  # noqa: E402
from scripts.eval_full import _config_from_payload  # noqa: E402


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores).astype(np.float64)
    labels = np.asarray(labels).astype(int)
    if (labels == 0).sum() == 0 or (labels == 1).sum() == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    return float(
        (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True,
                    help="path to the Stage-1 SFLM epoch_final.pt")
    ap.add_argument("--out-dir", default=None,
                    help="default = same directory as --ckpt")
    ap.add_argument("--n-epochs", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--margin", type=float, default=None,
                    help="override cfg.dfm_svgp.margin_energy")
    ap.add_argument("--gamma-eval", type=float, default=None,
                    help="override cfg.sflm.eval_gamma for the hinge "
                         "(SVGP is trained on pooled features at this γ).")
    ap.add_argument("--lengthscale", type=float, default=None,
                    help="initial Matern-5/2 lengthscale (defaults to "
                         "sqrt(d_embed)).")
    ap.add_argument("--eval-n", type=int, default=500,
                    help="held-out batch size for AUROC eval")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # 1) Rebuild Config from the Stage-1 checkpoint and wrap as SFLMSvgp.
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg: Config = _config_from_payload(payload)
    if cfg.training.model_name != "SFLM":
        raise SystemExit(
            f"--ckpt's model_name is {cfg.training.model_name!r}, expected 'SFLM'"
        )
    cfg.training = replace(cfg.training, model_name="SFLMSvgp")
    device = cfg.training.device

    model = build_model(cfg).to(device)
    assert isinstance(model, SFLMSvgp)
    # Load SFLM weights into the wrapped sub-module. (Stage-1 state_dict has
    # no SVGP/pooler keys → strict=False is required.)
    state = payload.get("model_state_dict", payload)
    sflm_keys = {("sflm." + k): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(sflm_keys, strict=False)
    if unexpected:
        print(f"[warn] unexpected keys when porting SFLM weights: "
              f"{unexpected[:4]}{'...' if len(unexpected) > 4 else ''}")

    # 2) Datamodule with corruption ON (the collate emits token_ids_invalid).
    cfg.text8_dataset = replace(
        cfg.text8_dataset, train_corrupt_rate=0.15, eval_corrupt_rate=0.15,
    )
    dm, _ = build_training_datamodule(cfg)
    loader = dm.train_dataloader()

    # 3) Run the Stage-2 hinge fit.
    torch.manual_seed(args.seed)
    info = model.fit_svgp_hinge(
        loader,
        gamma_eval=args.gamma_eval,
        n_epochs=args.n_epochs,
        max_steps=args.max_steps,
        lr=args.lr,
        margin_energy=args.margin,
        train_pooler=True,
        verbose=True,
        lengthscale_init=args.lengthscale,
    )

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.ckpt).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ckpt = out_dir / "model_with_svgp_hinge.pt"
    torch.save({"cfg": _cfg_to_dict(cfg),
                "model_state_dict": model.state_dict(),
                "stage2_info": {"n_steps": info["n_steps"]}}, out_ckpt)
    print(f"\nwrote {out_ckpt}  (n_steps={info['n_steps']})")

    # 4) Quick OOD AUROC on a held-out batch: positive (clean val) vs
    # several corruption families.  Mirrors DFM-SVGP eval format.
    val_ids = dm.splits.val.long()
    g = torch.Generator().manual_seed(args.seed)
    pick = torch.randperm(val_ids.shape[0], generator=g)[: args.eval_n]
    clean = val_ids[pick].to(device)
    K = cfg.text8_dataset.K
    corruptions: dict[str, torch.Tensor] = {
        "scrambled":      partially_shuffle_token_ids(
            clean, shuffle_rate=1.0, seed=args.seed + 7),
        "subst_0.3":      corrupt_token_ids(
            clean, vocab_size=K, corrupt_rate=0.3, seed=args.seed),
        "random_simplex": torch.randint(
            0, K, clean.shape, device=device,
            generator=torch.Generator(device=device).manual_seed(args.seed)),
    }
    out_scores: dict[str, dict[str, float]] = {}
    pos = model.ood_score(clean)
    out_scores["positive"] = {
        "mean_prob": float(pos["prob"].mean()),
        "mean_std":  float(pos["std"].mean()),
    }
    for name, neg_ids in corruptions.items():
        neg = model.ood_score(neg_ids)
        probs = torch.cat([pos["prob"], neg["prob"]]).cpu().numpy()
        stds = torch.cat([pos["std"],  neg["std"]]).cpu().numpy()
        labels = np.concatenate([
            np.zeros(pos["prob"].numel()),
            np.ones(neg["prob"].numel()),
        ])
        out_scores[name] = {
            "auroc_prob": _auroc(probs, labels),
            "auroc_std":  _auroc(stds, labels),
            "mean_prob":  float(neg["prob"].mean()),
            "mean_std":   float(neg["std"].mean()),
        }
        print(f"  {name:18s} AUROC(prob)={out_scores[name]['auroc_prob']:.4f}  "
              f"AUROC(std)={out_scores[name]['auroc_std']:.4f}")

    out_json = out_dir / "sflm_svgp_hinge_eval.json"
    out_json.write_text(json.dumps(out_scores, indent=2))
    print(f"wrote {out_json}")
    return 0


def _cfg_to_dict(cfg: Config) -> dict[str, Any]:
    from dataclasses import asdict
    d: dict[str, Any] = {}
    for f in cfg.__dataclass_fields__:
        section = getattr(cfg, f)
        try:
            d[f] = asdict(section)
        except TypeError:
            d[f] = section
    # device is a torch.device — keep it serialisable.
    if isinstance(d.get("training", {}).get("device"), torch.device):
        d["training"]["device"] = str(d["training"]["device"])
    return d


if __name__ == "__main__":
    raise SystemExit(main())
