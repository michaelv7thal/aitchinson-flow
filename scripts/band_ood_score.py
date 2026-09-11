"""Band-scaled OOD scorecard for an EqM checkpoint.

The question
------------
EqM was in the paper because its learned energy promised a per-token OOD score
for free. On the whole-path fields that promise died: the trained energy has a
minimum at every vertex, the wrong ones included, so it scores a position by how
sharp its state is and not by whether its token fits the context.

But a band-trained field (eqm.decay_strategy=band, γ ∈ [γ_lo, γ_hi]) has never
seen a state at the data radius, which is exactly where that readout was taken.
The recovery result (paper: tab:eqm-band-recovery) showed the same field repairs
text once its input is rescaled into the band. This script asks the matching
question for detection: rescale the candidate window into the band, THEN read the
energy and the per-position gradient norm, and see whether they separate clean
text from corrupted text.

What it computes
----------------
Per (γ, scheme, rate) cell, for the field's two native readouts

    U_pos = ‖∇_x E(x)‖ per position   (model.position_uncertainty)
    E_seq = ⟨x, f(x)⟩ per sequence    (model.energy)

on the mapped state

    x = γ · clr(ids)                         (--noise-draws 0, pure rescale)
    x = γ · clr(ids) + (1-γ) · σ · ε         (default, the band's own training
                                              distribution, averaged over draws)

it reports

    * per-CHARACTER AUROC of U_pos: corrupted vs untouched positions
    * per-WORD AUROC of U_pos, max- and mean-pooled over each word's characters
    * per-SEQUENCE AUROC of E_seq and of |E_seq|, clean windows vs corrupted ones

through scripts/_bench_common.py, the same helpers the published detectors use,
so the numbers are directly comparable to the paper's detector tables. γ=1.0 is
the data-scale control: that is the reading the paper already has.

The comparison targets (paper, sec:ood, frozen full-corpus Dirichlet FM at a 15%
replace rate): word-level AUROC 0.981 for the training-free denoiser NLL, at most
0.75 for the GPT-2 baselines. Chance is 0.5. A band field that lands anywhere
above ~0.7 on words would be a genuine per-token score, which is the thing the
equilibrium route was supposed to hand over for free.

Cost: forward+backward passes only, no sampling. Minutes, not hours.

Usage
-----
    # the band cell trained by sweeps/band_L256.yaml
    uv run python scripts/band_ood_score.py \
        --ckpt runs/band_L256_ep10_d10k/epoch_final.pt \
        --out  runs/band_L256_ep10_d10k/band_ood.json

    # the published whole-path benchmark arm, as the control
    uv run python scripts/band_ood_score.py \
        --ckpt runs/sflm_bench_a100_20g_L256/EqM_OneHot_gp1p0/epoch_final.pt \
        --out  runs/sflm_bench_a100_20g_L256/EqM_OneHot_gp1p0/band_ood.json

    # dev-scale cells (L=40), where the band result was established
    uv run python scripts/band_ood_score.py --ckpt runs/band_strict_d50k/epoch_final.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import aitchinson_flow.models  # noqa: E402,F401 — populate the model registry
from aitchinson_flow.data.corruption import (  # noqa: E402
    corrupt_token_ids,
    partially_shuffle_token_ids,
)
from aitchinson_flow.data.transforms import token_ids_to_features  # noqa: E402
from aitchinson_flow.models import build_model  # noqa: E402
from aitchinson_flow.training import build_training_datamodule  # noqa: E402

from scripts._bench_common import det_metrics, word_metrics  # noqa: E402
from scripts.eval_full import _config_from_payload  # noqa: E402

# Paper reference points, printed with the results for orientation.
REF = {
    "dirfm_nll_word_auroc_at_15": 0.981,
    "gpt2_best_word_auroc_at_15": 0.75,
}


def _load(ckpt: str, device: str):
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = _config_from_payload(payload)
    model = build_model(cfg).to(device)
    state = payload["model_state_dict"] if "model_state_dict" in payload else payload
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def _windows(cfg, n: int, split: str) -> torch.Tensor:
    dm, _ = build_training_datamodule(cfg)
    loaders = {"train": dm.train_dataloader, "val": dm.val_dataloader,
               "test": dm.test_dataloader}
    dl = loaders[split]() or dm.train_dataloader()
    got: list[torch.Tensor] = []
    for b in dl:
        got.append(b["token_ids"].long())
        if sum(t.shape[0] for t in got) >= n:
            break
    seqs = torch.cat(got)[:n]
    if seqs.shape[0] < n:
        raise SystemExit(f"split {split!r} yielded {seqs.shape[0]} windows, need {n}")
    return seqs


def _grad_norm_per_pos(model, x: torch.Tensor) -> torch.Tensor:
    """Fallback for models without .position_uncertainty: per-position ‖∇E‖."""
    with torch.enable_grad():
        xr = x.detach().requires_grad_(True)
        energy = (xr * model.forward(xr)).sum()
        grad = torch.autograd.grad(energy, xr)[0]
    return grad.norm(dim=-1)


def _map_into_band(feats: torch.Tensor, gamma: float, sigma: float,
                   generator: torch.Generator | None) -> torch.Tensor:
    """x = γ·feats (+ (1-γ)·σ·ε), projected to the zero-sum hyperplane."""
    x = gamma * feats
    if generator is not None:
        eps = torch.randn(feats.shape, generator=generator,
                          device=feats.device, dtype=feats.dtype)
        eps = eps - eps.mean(dim=-1, keepdim=True)
        x = x + (1.0 - gamma) * sigma * eps
    return x - x.mean(dim=-1, keepdim=True)


@torch.no_grad()
def _score(model, ids: torch.Tensor, *, gamma: float, K: int,
           label_smoothing: float, sigma: float, draws: int, chunk: int,
           device: str, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(U_pos (B,L) on CPU, E_seq (B,) on CPU), averaged over noise draws.

    The generator is seeded from ``seed`` and consumed in the same chunk order
    every call, so a clean batch and a corrupted batch of the same size see the
    SAME noise draws. The comparison is therefore paired: none of the AUROC can
    come from one side happening to get a luckier ε.
    """
    has_native = hasattr(model, "position_uncertainty")
    u_draws: list[torch.Tensor] = []
    e_draws: list[torch.Tensor] = []
    n_draws = max(1, draws)
    for d in range(n_draws):
        gen = None
        if draws > 0:
            gen = torch.Generator(device=device)
            gen.manual_seed(seed + 1000 * d)
        u_parts, e_parts = [], []
        for i in range(0, ids.shape[0], chunk):
            tb = ids[i:i + chunk].to(device)
            feats = token_ids_to_features(tb, K, label_smoothing=label_smoothing)
            x = _map_into_band(feats, gamma, sigma, gen)
            u = (model.position_uncertainty(x) if has_native
                 else _grad_norm_per_pos(model, x))
            u_parts.append(u.detach().cpu())
            e_parts.append(model.energy(x).detach().cpu())
        u_draws.append(torch.cat(u_parts))
        e_draws.append(torch.cat(e_parts))
    return (torch.stack(u_draws).mean(dim=0), torch.stack(e_draws).mean(dim=0))


def _corrupt(ids: torch.Tensor, scheme: str, rate: float, *, K: int, seed: int):
    if scheme == "replace":
        return corrupt_token_ids(ids, vocab_size=K, corrupt_rate=rate, seed=seed)
    if scheme == "shuffle":
        return partially_shuffle_token_ids(ids, shuffle_rate=rate, seed=seed)
    raise SystemExit(f"unknown scheme {scheme!r} (use replace or shuffle)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None,
                    help="default: <ckpt dir>/band_ood.json")
    ap.add_argument("--n", type=int, default=256, help="# eval windows")
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--gammas", default="0.005,0.01,0.016,0.02,0.03,1.0",
                    help="band positions to read the energy at; 1.0 is the "
                         "data-scale control, i.e. the reading the paper has")
    ap.add_argument("--schemes", default="replace,shuffle")
    ap.add_argument("--rates", default="0.05,0.1,0.15,0.2,0.3",
                    help="the paper's detector ladder; 0.15 is its headline rate")
    ap.add_argument("--noise-draws", type=int, default=4,
                    help="average the score over this many draws of the band's "
                         "own source noise; 0 = deterministic rescale only")
    ap.add_argument("--chunk", type=int, default=8,
                    help="windows per forward+backward; lower it if L=256 OOMs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model, cfg = _load(args.ckpt, device)
    K = cfg.text8_dataset.K
    label_smoothing = cfg.transformation.label_smoothing
    sigma = float(cfg.eqm.source_sigma)
    eqm = cfg.eqm
    band = (getattr(eqm, "decay_strategy", "linear") == "band")
    print(f"[band-ood] {args.ckpt}")
    print(f"[band-ood] L={cfg.training.L} K={K} sigma={sigma} "
          f"decay={eqm.decay_strategy} gamma_lo={getattr(eqm, 'gamma_lo', None)} "
          f"gamma_hi={getattr(eqm, 'gamma_hi', None)} "
          f"gamma_star={getattr(eqm, 'gamma_star', None)} "
          f"lambda={eqm.gradient_lambda}")
    if not band:
        print("[band-ood] NOTE: this checkpoint was trained on the whole path, "
              "so it is the control, not a band cell.")

    clean = _windows(cfg, args.n, args.split)
    gammas = [float(g) for g in args.gammas.split(",") if g.strip()]
    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    rates = [float(r) for r in args.rates.split(",") if r.strip()]

    rows: list[dict] = []
    for gamma in gammas:
        u_clean, e_clean = _score(
            model, clean, gamma=gamma, K=K, label_smoothing=label_smoothing,
            sigma=sigma, draws=args.noise_draws, chunk=args.chunk,
            device=device, seed=args.seed)
        u_clean_flat = u_clean.numpy().reshape(-1)
        e_clean_np = e_clean.numpy()
        tag = "data scale (control)" if gamma == 1.0 else "in band"
        print(f"\n=== gamma={gamma:g}  [{tag}]  "
              f"clean |E| mean={float(e_clean.abs().mean()):.4g} "
              f"U_pos mean={float(u_clean.mean()):.4g}")
        print(f"{'scheme':>8} {'rate':>5} {'AU_char':>8} {'AU_word_max':>12} "
              f"{'AU_word_mean':>13} {'AU_seq_E':>9} {'AU_seq_|E|':>11}")
        for scheme in schemes:
            for rate in rates:
                corr = _corrupt(clean.clone(), scheme, rate, K=K,
                                seed=args.seed + int(1000 * rate))
                changed = corr != clean
                u_corr, e_corr = _score(
                    model, corr, gamma=gamma, K=K,
                    label_smoothing=label_smoothing, sigma=sigma,
                    draws=args.noise_draws, chunk=args.chunk,
                    device=device, seed=args.seed)
                cm = changed.numpy().reshape(-1)
                u_corr_flat = u_corr.numpy().reshape(-1)
                nan = float("nan")
                if cm.any() and (~cm).any():
                    m_char = det_metrics(u_clean_flat, u_corr_flat[~cm],
                                         u_corr_flat[cm])
                    au_char = m_char["auroc"]
                else:
                    m_char, au_char = {}, nan
                w_max = word_metrics(u_clean, clean, u_corr, corr, changed,
                                     op="max")
                w_mean = word_metrics(u_clean, clean, u_corr, corr, changed,
                                      op="mean")
                e_corr_np = e_corr.numpy()
                au_seq_e = det_metrics(e_clean_np, e_clean_np, e_corr_np)["auroc"]
                au_seq_abs = det_metrics(np.abs(e_clean_np), np.abs(e_clean_np),
                                         np.abs(e_corr_np))["auroc"]
                print(f"{scheme:>8} {rate:>5.2f} {au_char:>8.4f} "
                      f"{w_max['word'].get('auroc', nan):>12.4f} "
                      f"{w_mean['word'].get('auroc', nan):>13.4f} "
                      f"{au_seq_e:>9.4f} {au_seq_abs:>11.4f}")
                rows.append({
                    "gamma": gamma, "scheme": scheme, "rate": rate,
                    "n": int(clean.shape[0]),
                    "auroc_char_upos": au_char,
                    "auroc_word_max_upos": w_max["word"].get("auroc"),
                    "auroc_word_mean_upos": w_mean["word"].get("auroc"),
                    "auroc_seq_energy": au_seq_e,
                    "auroc_seq_energy_abs": au_seq_abs,
                    "prf_char_upos": m_char,
                    "word_max": w_max, "word_mean": w_mean,
                    "upos_clean_mean": float(u_clean.mean()),
                    "upos_corr_mean": float(u_corr.mean()),
                    "energy_clean_mean": float(e_clean.mean()),
                    "energy_corr_mean": float(e_corr.mean()),
                })

    out = Path(args.out or (Path(args.ckpt).parent / "band_ood.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "ckpt": args.ckpt,
        "detector": "BandScaledEnergy",
        "detector_long": "per-position ‖∇E‖ and sequence energy read after "
                         "rescaling the candidate window into the trained band",
        "L": cfg.training.L, "K": K, "source_sigma": sigma,
        "decay_strategy": eqm.decay_strategy,
        "gamma_lo": getattr(eqm, "gamma_lo", None),
        "gamma_hi": getattr(eqm, "gamma_hi", None),
        "gamma_star": getattr(eqm, "gamma_star", None),
        "gradient_lambda": eqm.gradient_lambda,
        "band_trained": band,
        "n": int(clean.shape[0]), "split": args.split,
        "noise_draws": args.noise_draws, "seed": args.seed,
        "reference": REF, "rows": rows,
    }, indent=2))

    best = max((r for r in rows
                if r["auroc_word_max_upos"] == r["auroc_word_max_upos"]),
               key=lambda r: r["auroc_word_max_upos"], default=None)
    print(f"\nWrote {out}")
    if best is not None:
        print(f"best word-max AUROC {best['auroc_word_max_upos']:.4f} at "
              f"gamma={best['gamma']:g}, {best['scheme']} {best['rate']:.2f}")
    print(f"paper reference at rate 0.15: Dirichlet FM denoiser NLL "
          f"{REF['dirfm_nll_word_auroc_at_15']:.3f} word-level, GPT-2 baselines "
          f"at most {REF['gpt2_best_word_auroc_at_15']:.2f}, chance 0.5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
