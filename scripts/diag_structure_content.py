"""Quantify the K-isolation confound: how much *sequential structure* each
corpus actually contains, and whether EqM_OneHot beat the unigram-only baseline.

The DNA-vs-text8 comparison isolates vocab size K only up to a data-structure
confound (DNA's adjacent bases are far closer to independent than English
letters). This script measures that confound directly, training-free, from the
reference corpora:

  * "structure present"  = KL(ref_bigram || indep_bigram), the divergence of the
    true adjacent-pair law from the product of its own unigram marginals
    (≈ adjacent mutual information). Large ⇒ lots of bigram structure to capture.
  * "unigram-only KL"     = KL(indep || ref), the score a *perfect-unigram,
    no-structure* generator gets on the eval's KL_bi/KL_tri metric. This is the
    bar EqM_OneHot must beat to have captured any structure.

Compared against each model's measured KL_bi/KL_tri it answers: did the model
capture structure, or merely match marginals (or worse, inject wrong structure)?
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.append(str(_ROOT))

import torch  # noqa: E402

import aitchinson_flow.models  # noqa: E402,F401  populate REGISTRY / resolve import cycle
from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.training import build_training_datamodule  # noqa: E402
from scripts.eval_full import _ngram_counts  # noqa: E402

# Measured EqM_OneHot model KL (vs each corpus's own reference).
MODEL_KL = {
    "text8 (K=27)": {"KL_bi": 1.5694, "KL_tri": 6.9696},
    "DNA (K=4)": {"KL_bi": 0.0307, "KL_tri": 0.0685},
}


def _kl_pq(p: torch.Tensor, q: torch.Tensor) -> float:
    """KL(p || q) over flat prob vectors, matching eval_full._kl smoothing."""
    sm = 1e-6
    n = p.numel()
    pp = (p + sm) / (p.sum() + n * sm)
    qq = (q + sm) / (q.sum() + n * sm)
    return float((pp * (pp.log() - qq.log())).sum())


def _indep_prob(p_uni: torch.Tensor, n: int) -> torch.Tensor:
    """Product-of-marginals law over n positions, flattened to (K**n,)."""
    prod = p_uni
    for _ in range(n - 1):
        prod = torch.outer(prod.reshape(-1), p_uni).reshape(-1)
    return prod


def _kl_counts(gc: torch.Tensor, rc: torch.Tensor, K: int, n: int) -> float:
    """KL(gen || ref) on n-gram *counts* — identical estimator to
    eval_full.ngram_kl, so finite-sample bias matches the model's reported KL."""
    sm = 1e-6
    ksz = K ** n
    gp = (gc + sm) / (gc.sum() + ksz * sm)
    rp = (rc + sm) / (rc.sum() + ksz * sm)
    return float((gp * (gp.log() - rp.log())).sum())


def _finite_indep_kl(p_uni, ref_bi, ref_tri, K, *, n=256, L=40, seed=0):
    """KL_bi/KL_tri of a *256-sample* i.i.d.-from-marginal generator — the
    apples-to-apples 'no-structure' baseline under the model's own estimator."""
    g = torch.Generator().manual_seed(seed)
    samp = torch.multinomial(p_uni, n * L, replacement=True, generator=g).view(n, L)
    gb = _ngram_counts(samp, 2, K=K)
    gt = _ngram_counts(samp, 3, K=K)
    return _kl_counts(gb, ref_bi, K, 2), _kl_counts(gt, ref_tri, K, 3)


def analyze(name: str, windows: torch.Tensor, K: int) -> None:
    uni = _ngram_counts(windows, 1, K=K)
    bi = _ngram_counts(windows, 2, K=K)
    tri = _ngram_counts(windows, 3, K=K)
    p_uni = uni / uni.sum()
    p_bi = bi / bi.sum()
    p_tri = tri / tri.sum()

    H_uni = float(-(p_uni.clamp(min=1e-12) * p_uni.clamp(min=1e-12).log()).sum())
    H_max = float(torch.log(torch.tensor(float(K))))

    indep_bi = _indep_prob(p_uni, 2)
    indep_tri = _indep_prob(p_uni, 3)

    struct_bi = _kl_pq(p_bi, indep_bi)       # structure present (bigram)
    struct_tri = _kl_pq(p_tri, indep_tri)    # structure present (trigram)

    # Finite-sample (256-sample) no-structure baseline — same estimator as the
    # model's reported KL, so finite-sample bias cancels in the comparison.
    fs_bi, fs_tri = _finite_indep_kl(p_uni, bi, tri, K)

    m = MODEL_KL.get(name, {})
    print(f"\n### {name}   (N={windows.shape[0]} windows, L={windows.shape[1]})")
    print(f"  unigram entropy H_uni        : {H_uni:.4f} nats  "
          f"({H_uni / H_max * 100:.1f}% of max ln{K}={H_max:.3f})")
    print(f"  bigram  structure present    : {struct_bi:.4f} nats  "
          f"= KL(ref || independent)")
    print(f"  trigram structure present    : {struct_tri:.4f} nats")
    print(f"  no-structure baseline KL_bi  : {fs_bi:.4f}   "
          f"(256 i.i.d.-from-marginal samples, model's estimator)")
    print(f"  no-structure baseline KL_tri : {fs_tri:.4f}")
    if m:
        print(f"  --> EqM_OneHot model KL_bi   : {m['KL_bi']:.4f}   "
              f"({'BEAT baseline' if m['KL_bi'] < fs_bi else 'NO BETTER than baseline'})")
        print(f"  --> EqM_OneHot model KL_tri  : {m['KL_tri']:.4f}   "
              f"({'BEAT baseline' if m['KL_tri'] < fs_tri else 'NO BETTER than baseline'})")


def main() -> None:
    # text8 (K=27), 10k train windows — matches the EqM_OneHot baseline split.
    cfg = Config()
    cfg.text8_dataset = replace(cfg.text8_dataset, max_train_windows=10_000, L=40)
    dm, _ = build_training_datamodule(cfg)
    analyze("text8 (K=27)", dm.splits.train.long(), 27)

    # DNA (K=4), the cached promoter windows.
    saved = torch.load("data_cache/dna/promoters_L40.pt", weights_only=False)
    analyze("DNA (K=4)", saved["train"].long(), 4)

    print("\n" + "-" * 64)
    print("Reading: a model that captured structure has KL_bi << unigram-only")
    print("baseline. If model KL_bi > baseline, it injected WRONG structure.")


if __name__ == "__main__":
    main()
