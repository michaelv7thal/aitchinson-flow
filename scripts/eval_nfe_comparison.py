"""NFE-vs-quality sweep comparing EqM and DFM on text8.

Mirrors the evaluation style of the DFM papers (Gat et al. 2024;
Discrete Flow Maps 2025): plot generation quality against the number
of model forward passes (NFE) used during sampling.

For EqM, NFE = number of NAG-GD gradient evaluations (max_steps).
For DFM, NFE = number of Euler steps.

Metrics (all model-agnostic — comparable across architectures)
-------
* H_unigram  — unigram entropy of generated character distribution (bits)
               text8 true ≈ 1.6 bits; uniform noise ≈ 4.75 bits
* bigram_ll  — average bigram log-likelihood under empirical text8 model (nats)
               higher is better; tells you if common character transitions appear

BPD is intentionally omitted: EqM has no tractable likelihood, so its
"reconstruction BPD" measures something different from DFM's ELBO BPD,
making direct comparison misleading.

Usage
-----
    python scripts/eval_nfe_comparison.py \\
        --eqm-checkpoint checkpoints/eqm/best.pt \\
        --dfm-checkpoint checkpoints/dfm/best.pt \\
        --nfe 1 2 4 8 16 32 64 128 256 \\
        --n-samples 512 \\
        --output plots/nfe_comparison.png
"""

from __future__ import annotations

import argparse
import sys
import math
from pathlib import Path

import torch
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Bootstrap repo paths (same pattern as main.py)
# ---------------------------------------------------------------------------

def _bootstrap(anchor: Path) -> None:
    repo_root = anchor.resolve().parent.parent
    src = repo_root / "src"
    for p in (str(src), str(repo_root)):
        if p not in sys.path:
            sys.path.insert(0, p)


_bootstrap(Path(__file__))

import aitchinson_flow.models  # noqa: E402,F401  — populate model REGISTRY
from aitchinson_flow.config import Config
from aitchinson_flow.training.checkpoint import load_checkpoint
from aitchinson_flow.models import build_model, EquilibriumFlowMatching, DiscreteFlowMatching

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def unigram_entropy(token_ids: torch.Tensor, K: int = 27) -> float:
    """Shannon entropy (bits) of the empirical unigram character distribution."""
    counts = torch.bincount(token_ids.reshape(-1), minlength=K).float()
    probs = counts / counts.sum()
    probs = probs.clamp(min=1e-12)
    return -(probs * probs.log2()).sum().item()


def build_bigram_model(batches: list[torch.Tensor], K: int = 27) -> torch.Tensor:
    """Fit an empirical bigram log-probability table from reference token sequences.

    Returns (K, K) tensor of log P(next | current), smoothed with add-1.
    """
    counts = torch.ones(K, K)  # add-1 smoothing
    for ids in batches:
        ids = ids.reshape(-1)
        counts[ids[:-1], ids[1:]] += 1
    log_probs = counts.log() - counts.sum(dim=1, keepdim=True).log()
    return log_probs  # (K, K)


def bigram_loglikelihood(token_ids: torch.Tensor, bigram_log_probs: torch.Tensor) -> float:
    """Mean bigram log-likelihood per transition (nats)."""
    ids = token_ids.reshape(-1)
    ll = bigram_log_probs[ids[:-1], ids[1:]]
    return ll.mean().item()


# ---------------------------------------------------------------------------
# Per-model evaluation at a fixed NFE
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_eqm_at_nfe(
    model: EquilibriumFlowMatching,
    nfe: int,
    n_samples: int,
    bigram_log_probs: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    K = model.cfg.text8_dataset.K
    L = model.cfg.text8_dataset.L

    x_gen = model.sample(n_samples, L, max_steps=nfe)  # (B, L, K) CLR
    gen_ids = x_gen.argmax(dim=-1)                      # (B, L) token IDs

    return {
        "entropy": unigram_entropy(gen_ids, K),
        "bigram_ll": bigram_loglikelihood(gen_ids, bigram_log_probs.to(device)),
    }


@torch.no_grad()
def eval_dfm_at_nfe(
    model: DiscreteFlowMatching,
    nfe: int,
    n_samples: int,
    bigram_log_probs: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    K = model.cfg.text8_dataset.K
    L = model.cfg.text8_dataset.L

    gen_ids = model.sample(n_samples, L, nfe=nfe)  # (B, L) token IDs

    return {
        "entropy": unigram_entropy(gen_ids, K),
        "bigram_ll": bigram_loglikelihood(gen_ids, bigram_log_probs.to(device)),
    }


# ---------------------------------------------------------------------------
# Reference statistics from real text8 validation data
# ---------------------------------------------------------------------------

def _load_reference_batches(cfg: Config, n_samples: int, device: torch.device):
    from aitchinson_flow.training import build_training_datamodule
    dm, _ = build_training_datamodule(cfg)
    loader = dm.val_dataloader()
    if loader is None:
        raise RuntimeError("No validation dataloader available.")
    batches, collected = [], 0
    for batch in loader:
        batches.append(batch["token_ids"].to(device))
        collected += batches[-1].shape[0]
        if collected >= n_samples:
            break
    return batches


def _reference_stats(batches: list[torch.Tensor], K: int) -> dict[str, float]:
    all_ids = torch.cat([b.reshape(-1) for b in batches])
    counts = torch.bincount(all_ids, minlength=K).float()
    probs = counts / counts.sum()
    probs = probs.clamp(min=1e-12)
    entropy = -(probs * probs.log2()).sum().item()

    bigram = build_bigram_model(batches, K)
    bll = bigram_loglikelihood(torch.cat(batches, dim=0), bigram)
    return {"entropy": entropy, "bigram_ll": bll}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eqm-checkpoint", type=str, default=None)
    p.add_argument("--dfm-checkpoint", type=str, default=None)
    p.add_argument(
        "--nfe", type=int, nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256],
        metavar="N",
    )
    p.add_argument("--n-samples", type=int, default=512)
    p.add_argument("--output", type=str, default="plots/nfe_comparison.png")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def _load_model(checkpoint_path: str, device: torch.device):
    cfg = Config()
    model = build_model(cfg)
    load_checkpoint(checkpoint_path, model=model, map_location=device)
    model.to(device).eval()
    return model


def main() -> None:
    args = parse_args()
    if args.eqm_checkpoint is None and args.dfm_checkpoint is None:
        print("Provide at least one of --eqm-checkpoint or --dfm-checkpoint.")
        raise SystemExit(1)

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    cfg = Config()
    K = cfg.text8_dataset.K

    print("Loading reference data...")
    ref_batches = _load_reference_batches(cfg, args.n_samples, device)
    bigram_log_probs = build_bigram_model(ref_batches, K)
    ref_stats = _reference_stats(ref_batches, K)
    print(f"  Reference  H={ref_stats['entropy']:.3f} bits  bigram_ll={ref_stats['bigram_ll']:.3f}")

    results: dict[str, dict[int, dict[str, float]]] = {}

    if args.eqm_checkpoint:
        print(f"\nLoading EqM from {args.eqm_checkpoint}")
        eqm = _load_model(args.eqm_checkpoint, device)
        assert isinstance(eqm, EquilibriumFlowMatching)
        results["EqM"] = {}
        for nfe in sorted(args.nfe):
            m = eval_eqm_at_nfe(eqm, nfe, args.n_samples, bigram_log_probs, device)
            results["EqM"][nfe] = m
            print(f"  EqM  NFE={nfe:4d}  H={m['entropy']:.3f}  bigram_ll={m['bigram_ll']:.3f}")

    if args.dfm_checkpoint:
        print(f"\nLoading DFM from {args.dfm_checkpoint}")
        dfm = _load_model(args.dfm_checkpoint, device)
        assert isinstance(dfm, DiscreteFlowMatching)
        results["DFM"] = {}
        for nfe in sorted(args.nfe):
            m = eval_dfm_at_nfe(dfm, nfe, args.n_samples, bigram_log_probs, device)
            results["DFM"][nfe] = m
            print(f"  DFM  NFE={nfe:4d}  H={m['entropy']:.3f}  bigram_ll={m['bigram_ll']:.3f}")

    # ---- Plot ----
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_ent, ax_bll) = plt.subplots(1, 2, figsize=(11, 4))
    colors = {"EqM": "steelblue", "DFM": "darkorange"}
    markers = {"EqM": "o", "DFM": "s"}

    for model_name, nfe_data in results.items():
        nfes = sorted(nfe_data)
        entropies  = [nfe_data[n]["entropy"]   for n in nfes]
        bigram_lls = [nfe_data[n]["bigram_ll"] for n in nfes]
        kw = dict(label=model_name, color=colors[model_name], marker=markers[model_name])
        ax_ent.plot(nfes, entropies,  **kw)
        ax_bll.plot(nfes, bigram_lls, **kw)

    # Reference lines (true text8 statistics)
    for ax, key, label in (
        (ax_ent, "entropy",   f"text8 truth ({ref_stats['entropy']:.2f} bits)"),
        (ax_bll, "bigram_ll", f"text8 truth ({ref_stats['bigram_ll']:.2f})"),
    ):
        ax.axhline(ref_stats[key], color="black", linestyle="--", linewidth=1, label=label)

    ax_ent.set_xscale("log")
    ax_ent.set_xlabel("NFE (log scale)")
    ax_ent.set_ylabel("Unigram Entropy (bits) →  text8 truth")
    ax_ent.set_title("Unigram Entropy vs NFE")
    ax_ent.legend()
    ax_ent.grid(True, alpha=0.3)

    ax_bll.set_xscale("log")
    ax_bll.set_xlabel("NFE (log scale)")
    ax_bll.set_ylabel("Bigram Log-Likelihood (nats ↑)")
    ax_bll.set_title("Bigram LL vs NFE")
    ax_bll.legend()
    ax_bll.grid(True, alpha=0.3)

    # Uniform-noise baseline for entropy
    ax_ent.axhline(math.log2(27), color="gray", linestyle=":", linewidth=1,
                   label=f"Uniform noise ({math.log2(27):.2f} bits)")
    ax_ent.legend()

    fig.suptitle("EqM vs DFM — generation quality on text8", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")


if __name__ == "__main__":
    main()
