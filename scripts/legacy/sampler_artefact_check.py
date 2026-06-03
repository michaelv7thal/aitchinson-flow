"""Quick post-hoc test of the Phase 4 sampler-artefact hypothesis.

Hypothesis: ``cfg.eqm.sample_return_best=True`` returns the noise init when
the trained model has weak gradients at γ=0 (OOD region for Dirichlet-
trained EqM). Two side-by-side evaluations on the same checkpoint:

  (A) ``return_best=True`` — what eval_full.py reports
  (B) ``return_best=False`` — trajectory endpoint after 200 NAG steps

If (A) gives uniform-ish samples and (B) gives non-uniform, that confirms
the sampler is the bottleneck, not training.

Usage:
    python scripts/sampler_artefact_check.py \\
        --ckpt runs/dphase4_lambda_recalib/epoch_final.pt --n 64
"""

from __future__ import annotations

import argparse
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

import aitchinson_flow.models  # noqa: E402,F401 — populate registry
from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.data.char_window_dataset import CHAR2ID  # noqa: E402
from aitchinson_flow.models import build_model  # noqa: E402
from aitchinson_flow.training import build_training_datamodule  # noqa: E402
from scripts.eval_full import _config_from_payload  # type: ignore  # noqa: E402

ALPHABET = "".join(sorted(CHAR2ID, key=CHAR2ID.__getitem__))


def _ngram_counts_flat(ids2d: torch.Tensor, K: int, n: int) -> torch.Tensor:
    L = ids2d.shape[1]
    if L < n:
        return torch.zeros(K**n)
    idx = torch.zeros(ids2d.shape[0], L - n + 1, dtype=torch.long)
    for i in range(n):
        idx = idx + ids2d[:, i : L - n + 1 + i].long() * (K ** (n - 1 - i))
    counts = torch.zeros(K**n)
    counts.scatter_add_(0, idx.reshape(-1), torch.ones(idx.numel()))
    return counts


def _kl_smoothed(gen: torch.Tensor, ref: torch.Tensor) -> float:
    smoothing = 1e-6
    Ksize = gen.numel()
    gp = (gen + smoothing) / (gen.sum() + Ksize * smoothing)
    rp = (ref + smoothing) / (ref.sum() + Ksize * smoothing)
    return float((gp * (gp.log() - rp.log())).sum())


def _entropy(counts: torch.Tensor) -> float:
    p = (counts + 1e-9) / (counts.sum() + counts.numel() * 1e-9)
    return float(-(p * p.log()).sum())


def _sample_with_seed(model, n, L, *, return_best, max_steps, seed=42):
    torch.manual_seed(seed)
    with torch.no_grad():
        x = model.sample(n, L, max_steps=max_steps, return_best=return_best)
        log_probs = model.decode_to_logprobs(x)
        return log_probs.argmax(-1).cpu()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()

    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = _config_from_payload(payload)
    device = cfg.training.device
    model = build_model(cfg).to(device)
    state = payload["model_state_dict"] if "model_state_dict" in payload else payload
    model.load_state_dict(state)
    model.eval()

    dm, _ = build_training_datamodule(cfg)
    train_ids = dm.splits.train.long()
    K = cfg.text8_dataset.K
    L = cfg.text8_dataset.L

    ref_uni = _ngram_counts_flat(train_ids, K, 1)
    ref_bi = _ngram_counts_flat(train_ids, K, 2)
    H_gt = _entropy(ref_uni)

    for label, return_best in (("return_best=True", True), ("return_best=False", False)):
        ids = _sample_with_seed(model, args.n, L, return_best=return_best,
                                max_steps=args.steps, seed=42)
        gen_uni = _ngram_counts_flat(ids, K, 1)
        gen_bi = _ngram_counts_flat(ids, K, 2)
        kl_u = _kl_smoothed(gen_uni, ref_uni)
        kl_b = _kl_smoothed(gen_bi, ref_bi)
        H_gen = _entropy(gen_uni)
        sample_text = "".join(ALPHABET[int(i)] for i in ids[0])
        print(f"[{label}]  KL_uni={kl_u:.4f}  KL_bi={kl_b:.4f}  "
              f"H_gen={H_gen:.4f} (H_gt={H_gt:.4f})")
        print(f"    sample[0]: {sample_text!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
