"""E4d/E4e — generative-likelihood OOD baseline + controls.

The missing baseline for Obj 3 (OOD detection): can a *density model* tell
clean text8 windows apart from corrupted ones using its own likelihood?  We use
the DFM MC-ELBO (``model.elbo_bpc``) as a per-sequence density/OOD score and ask
whether ranking by likelihood separates clean from corrupted at a sweep of
corruption rates, for three corruption schemes:

  * **substitution** — replace a fraction of positions with uniform random
    tokens (changes the token histogram; ``data/corruption.corrupt_token_ids``).
  * **shuffle** — permute a fraction of positions per sequence, *preserving the
    token multiset* (``data/corruption.partially_shuffle_token_ids``).  This is
    the order-only axis: the unigram histogram is untouched, only n-gram order
    is destroyed.  This is the axis density models are predicted to FAIL on.
  * **random** — replace *every* position with a uniform random token (rate is
    ignored; the whole sequence is noise).

Score convention: ``elbo_bpc`` is a bits/char NLL upper bound, so **higher bpc
⇒ lower likelihood ⇒ more OOD**.  We therefore feed the corrupted bpc as the
positive class to the AUROC helper (the class that *should* score higher).  We
also report ``sign_inverted`` (raw AUROC < 0.45): a density model that is
*confidently wrong* (assigns higher likelihood to corrupted text — the
Nalisnick et al. 2019 "deep generative models don't know what they don't know"
phenomenon) shows up as an inverted AUROC, esp. on the histogram-preserving
shuffle axis where the unigram likelihood is unchanged.

Per-sequence ELBO: ``model.elbo_bpc`` returns a batch-mean.  The simplest
correct way to get a *per-sequence* score is to call it on single-row batches
(one sequence at a time); the MC estimator and the per-step ×N reweighting are
identical for B=1, so the returned scalar is exactly that sequence's bound.  We
use a shared RNG seed per (clean, corrupted) pair so the MC noise is matched.

Controls (E4e): degenerate inputs whose likelihood ranking is diagnostic of
*what the detector actually keys on*:

  * **const_char** — ``"aaaa..."`` (a single repeated non-space token).
  * **all_spaces** — ``"     ..."`` (the most common single token in text8).
  * **valid_perm** — a clean sequence with a full histogram-preserving
    permutation (a "valid" rearrangement; tests whether the detector flags pure
    order changes that keep the unigram intact).

For each control we report its bpc and whether each detector "flags" it as OOD
(bpc above the clean median — i.e. lower likelihood than typical clean text).

Usage:
    python scripts/eval_ood_baselines.py --ckpt runs/<dfm>/epoch_final.pt \\
        --n 128 --rates 0.1,0.3,0.5,0.7,1.0 --out runs/<dfm>/ood_baselines.json
    python scripts/eval_ood_baselines.py --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.append(str(_ROOT))

import torch  # noqa: E402

import aitchinson_flow.models  # noqa: E402,F401  populate REGISTRY
from aitchinson_flow.data.char_window_dataset import CHAR2ID  # noqa: E402
from aitchinson_flow.data.corruption import (  # noqa: E402
    corrupt_token_ids,
    partially_shuffle_token_ids,
)
from aitchinson_flow.models import build_model  # noqa: E402
from aitchinson_flow.training import build_training_datamodule  # noqa: E402
from scripts.eval_full import _config_from_payload  # noqa: E402

# Arms that expose a genuine variational bits/char bound usable as a density.
_DENSITY_ARMS = frozenset({"DFM", "DirichletFM"})

# Default corruption-rate ladder (E4d).
_DEFAULT_RATES = (0.1, 0.3, 0.5, 0.7, 1.0)

# Raw AUROC below this ⇒ the detector ranks corrupted as *more* likely than
# clean (confidently wrong); we surface it as ``sign_inverted`` (Nalisnick).
_SIGN_INVERT_THRESH = 0.45

_SPACE_ID = CHAR2ID[" "]


def _auroc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """AUROC with ``pos`` the class that should score higher.

    Same rank-sum estimator as ``scripts/bench_sflm_ebm.py:_auroc`` (kept
    consistent so the numbers compose with the rest of the OOD bench).
    """
    pos, neg = pos.flatten().float().cpu(), neg.flatten().float().cpu()
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    comb = torch.cat([pos, neg])
    order = comb.argsort()
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, comb.numel() + 1, dtype=torch.float)
    return float(
        (ranks[: pos.numel()].sum() - pos.numel() * (pos.numel() + 1) / 2)
        / (pos.numel() * neg.numel())
    )


def _full_shuffle(token_ids: torch.Tensor, *, seed: int | None = None) -> torch.Tensor:
    """Full histogram-preserving permutation of every position per sequence.

    Used for the ``valid_perm`` control (rate=1.0 shuffle that keeps the token
    multiset identical to the source)."""
    gen: torch.Generator | None = None
    if seed is not None:
        gen = torch.Generator(device=token_ids.device)
        gen.manual_seed(seed)
    out = token_ids.clone()
    bsz, seq_len = out.shape
    for b in range(bsz):
        perm = torch.randperm(seq_len, device=out.device, generator=gen)
        out[b] = out[b, perm]
    return out


def _random_ids(
    token_ids: torch.Tensor, *, vocab_size: int, seed: int | None = None
) -> torch.Tensor:
    """Replace every position with a uniform random token (pure noise)."""
    gen: torch.Generator | None = None
    if seed is not None:
        gen = torch.Generator(device=token_ids.device)
        gen.manual_seed(seed)
    return torch.randint(
        0, vocab_size, token_ids.shape, device=token_ids.device, generator=gen
    )


def _corrupt(
    scheme: str,
    token_ids: torch.Tensor,
    rate: float,
    *,
    vocab_size: int,
    seed: int | None = None,
) -> torch.Tensor:
    """Dispatch a corruption scheme at a given rate. Reuses the repo's
    ``data/corruption`` utilities for substitution / shuffle."""
    if scheme == "substitution":
        return corrupt_token_ids(
            token_ids, vocab_size=vocab_size, corrupt_rate=rate, seed=seed
        )
    if scheme == "shuffle":
        return partially_shuffle_token_ids(token_ids, shuffle_rate=rate, seed=seed)
    if scheme == "random":
        # Whole-sequence noise; rate is ignored (documented in the module docstring).
        return _random_ids(token_ids, vocab_size=vocab_size, seed=seed)
    raise ValueError(f"unknown corruption scheme: {scheme!r}")


@torch.no_grad()
def _per_sequence_bpc(
    model, token_ids: torch.Tensor, *, n_mc: int, n_steps: int, seed: int | None = None
) -> torch.Tensor:
    """Per-sequence MC-ELBO bits/char, by calling ``elbo_bpc`` on single-row
    batches (B=1).  Returns a (B,) tensor on CPU.

    ``elbo_bpc`` reduces with ``.mean()`` over the batch; for a 1-row batch that
    mean is exactly the single sequence's bound.  We reseed per row (with the
    same ``seed`` across clean/corrupted pairs upstream) so the Monte-Carlo
    noise is matched between the two members of a pair, reducing AUROC variance.
    """
    scores = []
    for i in range(token_ids.shape[0]):
        if seed is not None:
            torch.manual_seed(seed + i)
        row = token_ids[i : i + 1]
        bpc_i = float(model.elbo_bpc(row, n_mc=n_mc, n_steps=n_steps))
        scores.append(bpc_i)
    return torch.tensor(scores, dtype=torch.float)


def evaluate_ood_baselines(
    ckpt: str | Path,
    *,
    n: int = 128,
    rates: tuple[float, ...] = _DEFAULT_RATES,
    n_mc: int = 8,
    n_steps: int = 1000,
    schemes: tuple[str, ...] = ("substitution", "shuffle", "random"),
    seed: int = 0,
) -> dict[str, Any]:
    """Run the likelihood-OOD baseline + controls for a density-arm checkpoint.

    Returns a dict with per-scheme AUROC tables and the control verdicts.
    """
    ckpt = Path(ckpt)
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = _config_from_payload(payload)
    model_name = cfg.training.model_name
    if model_name not in _DENSITY_ARMS:
        raise ValueError(
            f"eval_ood_baselines needs a density arm with elbo_bpc "
            f"({sorted(_DENSITY_ARMS)}); got model_name={model_name!r}"
        )

    device = cfg.training.device
    model = build_model(cfg).to(device)
    state = payload.get("model_state_dict", payload)
    model.load_state_dict(state)
    model.eval()
    if not hasattr(model, "elbo_bpc"):
        raise ValueError(f"{model_name} does not expose elbo_bpc")

    K = int(cfg.text8_dataset.K)
    L = int(cfg.text8_dataset.L)

    dm, _ = build_training_datamodule(cfg)
    test_ids = getattr(dm.splits, "test", None)
    if test_ids is None or test_ids.numel() == 0:
        test_ids = dm.splits.val
    test_ids = test_ids.long()
    n = max(1, min(n, test_ids.shape[0]))
    clean = test_ids[:n].to(device)

    # Clean per-sequence density (shared across all schemes / controls).
    clean_bpc = _per_sequence_bpc(
        model, clean, n_mc=n_mc, n_steps=n_steps, seed=seed
    )
    clean_median = float(clean_bpc.median())

    # ── (1) Corruption ladder ────────────────────────────────────────────────
    schemes_out: dict[str, Any] = {}
    for scheme in schemes:
        eff_rates = (1.0,) if scheme == "random" else rates
        rate_rows: list[dict[str, Any]] = []
        for rate in eff_rates:
            corrupted = _corrupt(
                scheme, clean, float(rate), vocab_size=K, seed=seed + 1
            )
            corr_bpc = _per_sequence_bpc(
                model, corrupted, n_mc=n_mc, n_steps=n_steps, seed=seed
            )
            # Higher bpc ⇒ lower likelihood ⇒ more OOD ⇒ corrupted is the
            # positive class (should score higher).
            auroc = _auroc(corr_bpc, clean_bpc)
            sign_inverted = bool(auroc == auroc and auroc < _SIGN_INVERT_THRESH)
            rate_rows.append(
                {
                    "rate": float(rate),
                    "auroc": auroc,
                    "sign_inverted": sign_inverted,
                    "corrupted_bpc_mean": float(corr_bpc.mean()),
                    "clean_bpc_mean": float(clean_bpc.mean()),
                }
            )
        # Headline AUROC for the scheme = the strongest corruption rate.
        headline = rate_rows[-1]
        schemes_out[scheme] = {
            "rates": rate_rows,
            "auroc": headline["auroc"],
            "sign_inverted": headline["sign_inverted"],
            "note": (
                "histogram-preserving (order-only); density predicted near-chance"
                if scheme == "shuffle"
                else (
                    "whole-sequence uniform noise"
                    if scheme == "random"
                    else "token-substitution (changes histogram)"
                )
            ),
        }

    # ── (2) Controls ─────────────────────────────────────────────────────────
    # const_char: all 'a' (id 0). all_spaces: all space (id _SPACE_ID).
    # valid_perm: a full histogram-preserving permutation of the clean ids.
    const_char = torch.zeros((n, L), dtype=torch.long, device=device)  # 'a'*L
    all_spaces = torch.full((n, L), _SPACE_ID, dtype=torch.long, device=device)
    valid_perm = _full_shuffle(clean, seed=seed + 2)

    controls_out: dict[str, Any] = {}
    for cname, cids in (
        ("const_char", const_char),
        ("all_spaces", all_spaces),
        ("valid_perm", valid_perm),
    ):
        cbpc = _per_sequence_bpc(model, cids, n_mc=n_mc, n_steps=n_steps, seed=seed)
        bpc_mean = float(cbpc.mean())
        # AUROC vs clean (does the detector rank this control as OOD?).
        auroc = _auroc(cbpc, clean_bpc)
        # "Flagged" ⇒ majority of control sequences have lower likelihood
        # (higher bpc) than the clean median.
        flagged = bool((cbpc > clean_median).float().mean() > 0.5)
        controls_out[cname] = {
            "bpc_mean": bpc_mean,
            "auroc_vs_clean": auroc,
            "flagged_ood": flagged,
        }

    return {
        "model_name": model_name,
        "ckpt": str(ckpt),
        "n": int(n),
        "L": L,
        "K": K,
        "n_mc": int(n_mc),
        "n_steps": int(n_steps),
        "score": "per_sequence_elbo_bpc (higher=more OOD; B=1 batches)",
        "clean_bpc_mean": float(clean_bpc.mean()),
        "clean_bpc_median": clean_median,
        "schemes": schemes_out,
        "controls": controls_out,
        "interpretation": (
            "Nalisnick et al. 2019: generative likelihood is unreliable for OOD. "
            "Expect near-chance / sign-inverted AUROC on the order-only shuffle "
            "axis (histogram preserved), and constant-char / all-spaces controls "
            "scoring as in-distribution-or-better despite being degenerate."
        ),
    }


# --------------------------------------------------------------------------
# Smoke test — tiny DFM checkpoint, CPU, synthetic; <60s.
# --------------------------------------------------------------------------
def _smoke() -> int:
    import dataclasses
    import tempfile
    from dataclasses import replace as _replace

    from aitchinson_flow.config import Config

    torch.manual_seed(0)
    cfg = Config()
    cfg.training = _replace(cfg.training, model_name="DFM", device="cpu", L=12, K=27)
    cfg.transformer = _replace(
        cfg.transformer, d_model=32, nhead=4, num_layers=1, d_latent=32
    )
    cfg.text8_dataset = _replace(
        cfg.text8_dataset, L=12, K=27, max_train_windows=32, max_eval_windows=32
    )

    model = build_model(cfg)
    model.eval()

    def _to_dict(obj: Any) -> Any:
        if dataclasses.is_dataclass(obj):
            return {
                f.name: _to_dict(getattr(obj, f.name))
                for f in dataclasses.fields(obj)
            }
        return obj

    payload = {
        "cfg": _to_dict(cfg),
        "model_state_dict": model.state_dict(),
        "epoch": 0,
        "global_step": 0,
    }
    tmp = Path(tempfile.mkdtemp())
    ckpt = tmp / "tiny_dfm.pt"
    torch.save(payload, ckpt)

    res = evaluate_ood_baselines(
        ckpt,
        n=8,
        rates=(0.3, 1.0),
        n_mc=2,
        n_steps=8,
        seed=0,
    )

    out = tmp / "ood_baselines.json"
    out.write_text(json.dumps(res, indent=2))

    # Contract assertions.
    assert res["model_name"] == "DFM", res["model_name"]
    assert set(res["schemes"]) == {"substitution", "shuffle", "random"}, res["schemes"]
    for scheme, sd in res["schemes"].items():
        assert "auroc" in sd and "sign_inverted" in sd, (scheme, sd)
        assert isinstance(sd["sign_inverted"], bool), (scheme, sd)
        a = sd["auroc"]
        assert a == a, f"{scheme} auroc is NaN"
        assert 0.0 <= a <= 1.0, f"{scheme} auroc out of range: {a}"
        assert sd["rates"], f"{scheme} has no rate rows"
    assert set(res["controls"]) == {"const_char", "all_spaces", "valid_perm"}, (
        res["controls"]
    )
    for cname, cd in res["controls"].items():
        assert "bpc_mean" in cd and "flagged_ood" in cd, (cname, cd)
        assert isinstance(cd["flagged_ood"], bool), (cname, cd)

    print(
        "OK eval_ood_baselines smoke: "
        f"sub_auroc={res['schemes']['substitution']['auroc']:.3f} "
        f"shuf_auroc={res['schemes']['shuffle']['auroc']:.3f} "
        f"rand_auroc={res['schemes']['random']['auroc']:.3f} | "
        f"controls flagged="
        + ",".join(
            f"{k}:{int(v['flagged_ood'])}" for k, v in res["controls"].items()
        )
        + f" | wrote {out}"
    )
    return 0


def _parse_rates(s: str) -> tuple[float, ...]:
    return tuple(float(x) for x in s.split(",") if x.strip())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ckpt", type=str, default=None, help="DFM / DirichletFM checkpoint."
    )
    ap.add_argument("--n", type=int, default=128, help="# test sequences.")
    ap.add_argument(
        "--rates",
        type=str,
        default=",".join(str(r) for r in _DEFAULT_RATES),
        help="Comma-separated corruption-rate ladder.",
    )
    ap.add_argument("--n-mc", type=int, default=8, help="MC samples for elbo_bpc.")
    ap.add_argument("--n-steps", type=int, default=1000, help="ELBO grid resolution.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output JSON path (default: <ckpt_dir>/ood_baselines.json).",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="CPU synthetic-data self-test (<60s); exits 0 on pass.",
    )
    args = ap.parse_args(argv)

    if args.smoke:
        return _smoke()

    if args.ckpt is None:
        ap.error("--ckpt is required unless --smoke is given")

    res = evaluate_ood_baselines(
        args.ckpt,
        n=args.n,
        rates=_parse_rates(args.rates),
        n_mc=args.n_mc,
        n_steps=args.n_steps,
        seed=args.seed,
    )

    print(f"{res['model_name']}  clean_bpc={res['clean_bpc_mean']:.3f}")
    for scheme, sd in res["schemes"].items():
        inv = " [SIGN-INVERTED]" if sd["sign_inverted"] else ""
        print(f"  {scheme:>13}: AUROC={sd['auroc']:.3f}{inv}  ({sd['note']})")
    print("  controls (flagged_ood / bpc_mean):")
    for cname, cd in res["controls"].items():
        print(
            f"    {cname:>11}: flagged={cd['flagged_ood']!s:>5}  "
            f"bpc={cd['bpc_mean']:.3f}  auroc_vs_clean={cd['auroc_vs_clean']:.3f}"
        )

    out_path = (
        Path(args.out)
        if args.out
        else Path(args.ckpt).parent / "ood_baselines.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
