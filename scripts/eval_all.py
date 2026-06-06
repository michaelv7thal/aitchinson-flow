"""Unified generation-eval module (harness component X1).

One entry point — :func:`evaluate` — produces the full generation scorecard for
*any* arm in the registry from a single checkpoint, with the validity / collapse
guards the rest of the harness relies on.  It reuses
:func:`scripts.eval_full.evaluate_checkpoint` for the n-gram KL / entropy /
sample fields (so the KL_uni/bi/tri and H numbers are byte-for-byte the same as
the canonical scorecard) and adds:

  * ``per_pos_entropy`` — mean per-position character entropy of the generated
    sequences.  For each of the L positions we estimate the char distribution
    across the ``n_samples`` generated sequences, take its entropy (nats), and
    average over positions.  A collapsed field (every sample identical) gives
    ≈0; a maximally diffuse field approaches log(K) (≈3.296 for K=27).
  * ``bpc`` — a peer-comparable bits/char density, computed via
    ``model.elbo_bpc`` / ``model.bpd`` (the variational ELBO) on held-out
    ``split`` ids — but ONLY for DFM / DirichletFM.  Every other arm gets
    ``bpc=None`` (the identity-path arms have no comparable likelihood; see
    ``EVAL_ASSESSMENT.md`` Obj 1).
  * ``generation_metric_valid`` — False when the arm is in the identity-path
    set OR ``bpc`` is None / non-finite / below the text8 sanity floor (0.5).
  * ``collapsed`` — the canonical mode-collapse flag: ``KL_uni < 0.05`` (the
    unigram already matches the corpus peak) AND ``KL_bi > 1.0`` (but the
    bigram structure is absent) ⇒ collapse to the unigram mode.

Usage:
    python scripts/eval_all.py --ckpt PATH --split test --n 64
    python scripts/eval_all.py --smoke
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
from aitchinson_flow.models import build_model  # noqa: E402
from aitchinson_flow.training import build_training_datamodule  # noqa: E402
from scripts.eval_full import (  # noqa: E402
    _config_from_payload,
    evaluate_checkpoint,
    ids_to_text,
)

# Arms whose ``bpd()`` reads the input through an (essentially identity)
# encode→decode_to_logprobs path, so PPL≈1.0 / BPC≈0 is a *recovery* artifact,
# NOT a data NLL/ELBO comparable to published text8 BPC.  Kept in lock-step
# with bench_sflm_ebm.py / eval_generation.py:_IDENTITY_PATH_BPC.
_IDENTITY_PATH_BPC = frozenset({"EqM", "EqM_OneHot", "EqMLatent", "SFLM"})

# A finite text8 char-NLL bound is ≳ the corpus entropy floor; anything below
# this (or non-finite) is a degenerate identity-recovery value, not a bound.
_BPC_SANITY_FLOOR = 0.5

# Arms that expose a genuine variational bits/char bound via elbo_bpc / bpd.
_BPC_DENSITY_ARMS = frozenset({"DFM", "DirichletFM"})


def _bpc_is_valid(model_name: str, bpc: float | None) -> bool:
    """Whether ``bpc`` for ``model_name`` is a peer-comparable density."""
    if model_name in _IDENTITY_PATH_BPC:
        return False
    if bpc is None:
        return False
    if bpc != bpc or bpc in (float("inf"), float("-inf")):  # NaN / ±inf
        return False
    return bpc >= _BPC_SANITY_FLOOR


def _per_position_entropy(gen_ids: torch.Tensor, K: int) -> float:
    """Mean per-position char entropy (nats) of generated sequences.

    ``gen_ids`` is (n_samples, L) integer ids.  For each position l we build the
    empirical char distribution over the n_samples sequences, compute its
    entropy, and average across positions.  Returns NaN if there are no samples.
    """
    if gen_ids.numel() == 0 or gen_ids.ndim != 2:
        return float("nan")
    n, _L = gen_ids.shape
    # one-hot counts per position → (L, K)
    counts = torch.zeros(gen_ids.shape[1], K)
    onehot = torch.zeros(n, gen_ids.shape[1], K)
    onehot.scatter_(2, gen_ids.long().unsqueeze(-1).clamp_(0, K - 1), 1.0)
    counts = onehot.sum(dim=0)  # (L, K)
    p = counts / counts.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    ent = -(p * (p.clamp(min=1e-12)).log()).sum(dim=-1)  # (L,)
    return float(ent.mean())


@torch.no_grad()
def _sample_gen_ids(model, cfg, n: int, L: int) -> torch.Tensor:
    """Unconditional sample → (n, L) cpu long ids, normalising the two repo
    sampling conventions (EqM-family decode_to_logprobs vs categorical
    denoisers whose ``sample`` already returns ids)."""
    if hasattr(model, "decode_to_logprobs"):
        x = model.sample(n, L)
        log_probs = model.decode_to_logprobs(x)
        return log_probs.argmax(-1).cpu().long()
    return model.sample(n, L).cpu().long()


def evaluate(
    ckpt: str | Path,
    *,
    model_kind: str | None = None,
    split: str = "test",
    n_samples: int = 64,
    bpc_mc: int = 8,
    n_steps: int = 200,
) -> dict[str, Any]:
    """Unified generation scorecard for one checkpoint.

    Returns a dict with exactly the harness-contract keys:
        model_name, KL_uni, KL_bi, KL_tri, H_gen, H_gt, H_ratio,
        per_pos_entropy, bpc, generation_metric_valid, collapsed, samples.
    """
    ckpt = Path(ckpt)
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = _config_from_payload(payload)

    # ``model_kind`` is a *label / dispatch* override only — it decides which
    # arm the result is reported as (and therefore the bpc dispatch + the
    # identity-path validity guard).  The model is still built from the
    # checkpoint's own cfg architecture so the saved state_dict always loads.
    model_name = model_kind or cfg.training.model_name

    # --- KL / entropy / samples via the canonical scorecard (reused) ----------
    base = evaluate_checkpoint(
        ckpt, n_samples=n_samples, n_steps=n_steps,
    )
    KL_uni = float(base["unigram_kl"])
    KL_bi = float(base["bigram_kl"])
    KL_tri = float(base["trigram_kl"])
    H_gen = float(base["H_gen"])
    H_gt = float(base["H_gt"])
    H_ratio = float(base["H_ratio"])
    samples = list(base.get("samples", []))

    # --- build the model for sampling (per_pos_entropy) + bpc -----------------
    device = cfg.training.device
    model = build_model(cfg).to(device)
    state = payload.get("model_state_dict", payload)
    model.load_state_dict(state)
    model.eval()
    K = int(cfg.text8_dataset.K)
    L = int(cfg.text8_dataset.L)

    # per-position entropy from a fresh unconditional sample.
    gen_ids = _sample_gen_ids(model, cfg, n_samples, L)
    per_pos_entropy = _per_position_entropy(gen_ids, K)
    if not samples:
        samples = ids_to_text(gen_ids[:8], K=K)

    # --- bpc: only DFM / DirichletFM produce a comparable ELBO bound ----------
    bpc: float | None = None
    if model_name in _BPC_DENSITY_ARMS:
        dm, _ = build_training_datamodule(cfg)
        held = getattr(dm.splits, split, None)
        if held is None or held.numel() == 0:
            held = dm.splits.val
        held = held.long().to(device)[: max(1, min(len(held), 4 * n_samples))]
        try:
            if hasattr(model, "elbo_bpc"):
                bpc = float(model.elbo_bpc(held, n_mc=bpc_mc))
            elif hasattr(model, "bpd"):
                bpc = float(model.bpd(held, n_mc=bpc_mc))
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:  # pragma: no cover
            print(f"[warn] bpc computation failed for {model_name}: {e}")
            bpc = None

    generation_metric_valid = _bpc_is_valid(model_name, bpc)
    if not generation_metric_valid:
        # Don't propagate a degenerate / non-comparable BPC downstream.
        bpc = None

    collapsed = bool(KL_uni < 0.05 and KL_bi > 1.0)

    return {
        "model_name": model_name,
        "KL_uni": KL_uni,
        "KL_bi": KL_bi,
        "KL_tri": KL_tri,
        "H_gen": H_gen,
        "H_gt": H_gt,
        "H_ratio": H_ratio,
        "per_pos_entropy": per_pos_entropy,
        "bpc": bpc,
        "generation_metric_valid": generation_metric_valid,
        "collapsed": collapsed,
        "samples": samples,
    }


# --------------------------------------------------------------------------
# Smoke test — tiny DFM checkpoint, CPU, synthetic; <60s.
# --------------------------------------------------------------------------
def _smoke() -> int:
    import tempfile
    from dataclasses import replace as _replace

    from aitchinson_flow.config import Config

    torch.manual_seed(0)
    cfg = Config()
    # Tiny CPU DFM: small d_model / few layers / short L / few windows.
    cfg.training = _replace(cfg.training, model_name="DFM", device="cpu", L=12, K=27)
    cfg.transformer = _replace(
        cfg.transformer, d_model=32, nhead=4, num_layers=1, d_latent=32,
    )
    cfg.text8_dataset = _replace(
        cfg.text8_dataset, L=12, K=27,
        max_train_windows=32, max_eval_windows=32,
    )

    model = build_model(cfg)
    model.eval()

    # Save a tiny checkpoint with the cfg-as-dict + model_state_dict layout the
    # rest of the harness expects.
    import dataclasses

    def _to_dict(obj: Any) -> Any:
        if dataclasses.is_dataclass(obj):
            return {f.name: _to_dict(getattr(obj, f.name))
                    for f in dataclasses.fields(obj)}
        return obj

    cfg_dict = _to_dict(cfg)
    # device is a string here (we set "cpu"); _config_from_payload pops it.
    payload = {
        "cfg": cfg_dict,
        "model_state_dict": model.state_dict(),
        "epoch": 0,
        "global_step": 0,
    }

    tmp = Path(tempfile.mkdtemp())
    ckpt = tmp / "tiny_dfm.pt"
    torch.save(payload, ckpt)

    res = evaluate(ckpt, split="test", n_samples=8, bpc_mc=2, n_steps=4)

    required = {
        "model_name", "KL_uni", "KL_bi", "KL_tri", "H_gen", "H_gt",
        "H_ratio", "per_pos_entropy", "bpc", "generation_metric_valid",
        "collapsed", "samples",
    }
    missing = required - set(res)
    assert not missing, f"missing contract keys: {missing}"
    assert res["model_name"] == "DFM", res["model_name"]
    assert isinstance(res["bpc"], float) and res["bpc"] == res["bpc"], (
        f"DFM bpc must be a finite float, got {res['bpc']!r}"
    )
    assert res["bpc"] not in (float("inf"), float("-inf"))
    assert res["generation_metric_valid"] is True, (
        f"DFM generation_metric_valid should be True, got {res}"
    )
    assert isinstance(res["per_pos_entropy"], float) and (
        res["per_pos_entropy"] == res["per_pos_entropy"]
    ), res["per_pos_entropy"]
    assert isinstance(res["collapsed"], bool)
    assert isinstance(res["samples"], list) and res["samples"], res["samples"]

    # Identity-path label must force generation_metric_valid=False (and bpc=None)
    # regardless of any computed value.
    res_eqm = evaluate(
        ckpt, model_kind="EqM", split="test", n_samples=8, bpc_mc=2, n_steps=4,
    )
    assert res_eqm["model_name"] == "EqM"
    assert res_eqm["generation_metric_valid"] is False, res_eqm
    assert res_eqm["bpc"] is None, res_eqm["bpc"]

    print(
        f"OK eval_all smoke: DFM bpc={res['bpc']:.3f} "
        f"per_pos_H={res['per_pos_entropy']:.3f} "
        f"gen_valid={res['generation_metric_valid']} "
        f"collapsed={res['collapsed']} | "
        f"EqM-label gen_valid={res_eqm['generation_metric_valid']} "
        f"bpc={res_eqm['bpc']}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--split", type=str, default="test",
                    choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=64, dest="n_samples")
    ap.add_argument("--model-kind", type=str, default=None,
                    help="Override the arm label / bpc dispatch (default: read "
                         "from the checkpoint cfg).")
    ap.add_argument("--bpc-mc", type=int, default=8)
    ap.add_argument("--steps", type=int, default=200, dest="n_steps")
    ap.add_argument("--out", type=str, default=None,
                    help="Output JSON path (default: <ckpt_dir>/eval_all.json).")
    ap.add_argument("--smoke", action="store_true",
                    help="CPU synthetic-data self-test (<60s); exits 0 on pass.")
    args = ap.parse_args(argv)

    if args.smoke:
        return _smoke()

    if args.ckpt is None:
        ap.error("--ckpt is required unless --smoke is given")

    res = evaluate(
        args.ckpt,
        model_kind=args.model_kind,
        split=args.split,
        n_samples=args.n_samples,
        bpc_mc=args.bpc_mc,
        n_steps=args.n_steps,
    )

    bpc_s = "—" if res["bpc"] is None else f"{res['bpc']:.3f}"
    print(
        f"{res['model_name']}  KL_uni={res['KL_uni']:.4f}  "
        f"KL_bi={res['KL_bi']:.4f}  KL_tri={res['KL_tri']:.4f}  "
        f"H_ratio={res['H_ratio']:.3f}  per_pos_H={res['per_pos_entropy']:.3f}  "
        f"bpc={bpc_s}  valid={res['generation_metric_valid']}  "
        f"collapsed={res['collapsed']}"
    )
    for s in res["samples"][:4]:
        print(f"  {s!r}")

    out_path = (Path(args.out) if args.out
                else Path(args.ckpt).parent / "eval_all.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
