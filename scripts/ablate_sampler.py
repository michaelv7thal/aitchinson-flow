"""E2b — sampler-only ablation on ONE fixed EqM checkpoint.

Loads a single trained EqM checkpoint and varies **only the sampler**, holding
the field (weights) fixed. The point is to separate "what the energy field
learned" from "how the sampler descends it" — every cell decodes the same
trained ⟨x, f(x)⟩ field, just with a different integrator.

Sampler cells (all EqM.sample on the same checkpoint):
  * nag             — NAG-GD on the conservative gradient (EqM.sample default).
  * euler_grad      — FM-style Euler integrator using ∇⟨x,f⟩ (use_grad=True).
  * euler_raw       — same Euler integrator on the raw velocity f (use_grad=False).
  * sde             — Langevin SDE sampler (method="sde").

Annealed Langevin is deliberately **dropped** for EqM: plain EqM exposes no
``score(x, sigma)`` (the σ-conditioned score needed by
``sampling/annealed_langevin.py``), so there is no ``EqM.sample(method="annealed")``.
Annealed Langevin is only meaningful for the DSM arm (EqMDSM / ScoreDSM), which
do expose ``score``. We record a clear note to that effect instead of calling a
nonexistent sampler.

For each cell we decode the samples to token IDs and report n-gram generation
metrics (KL_uni / KL_bi against the training corpus, via eval_full.ngram_kl) plus
a couple of argmax decodes.

Usage:
    python scripts/ablate_sampler.py --ckpt runs/<eqm>/epoch_final.pt \\
        --n 256 --steps 200 --out runs/<eqm>/ablate_sampler.json
    # local sanity (no HF / no ckpt — builds & saves a tiny EqM):
    python scripts/ablate_sampler.py --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import replace as _replace
from pathlib import Path
from typing import Any


def _bootstrap() -> None:
    root = Path(__file__).resolve().parent.parent
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    if str(root) not in sys.path:
        sys.path.append(str(root))


_bootstrap()

import torch  # noqa: E402

import aitchinson_flow.models  # noqa: E402,F401 — populate registry
from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.data.char_window_dataset import CHAR2ID, VOCAB_SIZE  # noqa: E402
from aitchinson_flow.models import build_model  # noqa: E402
from aitchinson_flow.training import build_training_datamodule  # noqa: E402
from scripts.eval_full import _config_from_payload, ngram_kl, unigram_kl  # noqa: E402

ALPHABET = "".join(sorted(CHAR2ID, key=CHAR2ID.__getitem__))

# Annealed Langevin needs a σ-conditioned score(x, sigma); only EqMDSM/ScoreDSM
# expose it. Plain EqM does not, so the cell is dropped (not silently faked).
_ANNEALED_NOTE = (
    "annealed Langevin dropped for EqM: plain EqM exposes no score(x, sigma) "
    "(EqM.sample has no method='annealed'). Annealed Langevin is reported only "
    "for the DSM arm (EqMDSM / ScoreDSM), which expose the σ-conditioned score "
    "required by sampling/annealed_langevin.py."
)

# The four sampler cells, each a kwargs dict for EqM.sample(n, L, max_steps=steps).
_SAMPLER_CELLS: list[tuple[str, dict[str, Any]]] = [
    ("nag", {}),  # cfg default sampler="nag"
    ("euler_grad", {"method": "euler", "use_grad": True}),
    ("euler_raw", {"method": "euler", "use_grad": False}),
    ("sde", {"method": "sde"}),
]


def _decode(ids: torch.Tensor, *, K: int) -> list[str]:
    return ["".join(ALPHABET[int(i)] for i in row) for row in ids.cpu()]


def _gen_metrics(
    gen_ids: torch.Tensor, ref_ids: torch.Tensor, *, K: int
) -> dict[str, Any]:
    kl_u, H_gen, H_ref = unigram_kl(gen_ids, ref_ids, K=K)
    kl_b = ngram_kl(gen_ids, ref_ids, 2, K=K)
    return {
        "KL_uni": kl_u,
        "KL_bi": kl_b,
        "H_gen": H_gen,
        "H_gt": H_ref,
        "H_ratio": (H_gen / H_ref) if H_ref > 0 else float("nan"),
        "samples": _decode(gen_ids[: min(4, gen_ids.shape[0])], K=K),
    }


def _run_cell(
    model,
    *,
    n: int,
    L: int,
    steps: int,
    K: int,
    ref_ids: torch.Tensor,
    sample_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Sample with one sampler configuration and compute generation metrics."""
    with torch.no_grad():
        x = model.sample(n, L, max_steps=steps, **sample_kwargs)
        log_probs = model.decode_to_logprobs(x)
        gen_ids = log_probs.argmax(-1).cpu()
    metrics = _gen_metrics(gen_ids, ref_ids, K=K)
    metrics["sample_kwargs"] = dict(sample_kwargs)
    return metrics


def ablate(
    ckpt_path: str | Path,
    *,
    n: int,
    steps: int,
    ref_ids: torch.Tensor | None = None,
) -> dict[str, Any]:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = _config_from_payload(payload)
    model_name = cfg.training.model_name
    if model_name != "EqM":
        # Only the plain-EqM conservative field is the intended subject; the
        # cell mechanics (method="euler"/"sde", use_grad) are EqM.sample-specific.
        print(
            f"[warn] checkpoint model_name={model_name!r} is not 'EqM'; "
            "the sampler cells assume the EqM.sample API."
        )

    device = cfg.training.device
    model = build_model(cfg).to(device)
    state = payload["model_state_dict"] if "model_state_dict" in payload else payload
    model.load_state_dict(state)
    model.eval()

    L = cfg.text8_dataset.L
    K = int(cfg.text8_dataset.K)
    # Reference n-gram corpus is the training split. The smoke path injects a
    # synthetic ``ref_ids`` so it never touches HF / the on-disk window cache.
    if ref_ids is None:
        dm, _ = build_training_datamodule(cfg)
        ref_ids = dm.splits.train.long()

    cells: dict[str, Any] = {}
    for name, sk in _SAMPLER_CELLS:
        m = _run_cell(
            model, n=n, L=L, steps=steps, K=K, ref_ids=ref_ids, sample_kwargs=sk
        )
        cells[name] = m
        print(
            f"[{name:11s}] KL_uni={m['KL_uni']:.4f}  KL_bi={m['KL_bi']:.4f}  "
            f"H_ratio={m['H_ratio']:.3f}  sample[0]={m['samples'][0]!r}"
        )

    print(f"[annealed   ] (skipped) {_ANNEALED_NOTE}")

    return {
        "ckpt": str(ckpt_path),
        "model_name": model_name,
        "n_samples": n,
        "sample_steps": steps,
        "annealed_langevin_note": _ANNEALED_NOTE,
        "cells": cells,
    }


def _build_tiny_eqm_ckpt(path: Path) -> None:
    """Construct a tiny EqM, save a repo-compatible checkpoint at ``path``."""
    from aitchinson_flow.training.checkpoint import save_checkpoint

    cfg = Config()
    K = cfg.text8_dataset.K
    cfg.training = _replace(cfg.training, model_name="EqM", L=16, K=K)
    cfg.text8_dataset = _replace(cfg.text8_dataset, L=16, K=K)
    cfg.transformer = _replace(cfg.transformer, d_model=32, num_layers=1, nhead=2)

    model = build_model(cfg).to("cpu")
    save_checkpoint(
        path,
        model=model,
        cfg=cfg,
        optimizer=None,
        epoch=0,
        global_step=0,
    )


def _smoke() -> int:
    """Build a tiny EqM, save a tiny ckpt, run every sampler cell on CPU."""
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "tiny_eqm.pt"
        _build_tiny_eqm_ckpt(ckpt)

        # Synthetic reference corpus (matches the tiny ckpt's L/K) so smoke
        # never triggers an HF download or touches the window cache.
        ref_ids = torch.randint(0, VOCAB_SIZE, (16, 16))

        out_path = Path(td) / "ablate_sampler.json"
        result = ablate(ckpt, n=8, steps=3, ref_ids=ref_ids)
        out_path.write_text(json.dumps(result, indent=2))

        # Round-trip the json the way a consumer would.
        loaded = json.loads(out_path.read_text())
        cells = loaded["cells"]
        expected = {"nag", "euler_grad", "euler_raw", "sde"}
        assert set(cells) == expected, f"cells={set(cells)} != {expected}"
        for name, m in cells.items():
            assert "KL_uni" in m and "KL_bi" in m, f"missing KL in cell {name}"
            assert "samples" in m and m["samples"], f"no samples in cell {name}"
        assert loaded.get("annealed_langevin_note"), "missing annealed note"
        assert "annealed" not in cells, "annealed must NOT be an EqM cell"
        assert loaded["model_name"] == "EqM"

    print(
        "OK ablate_sampler smoke: 4 EqM sampler cells "
        f"({', '.join(sorted(expected))}) + annealed-Langevin note present; "
        "json validated."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=None, help="EqM checkpoint to ablate.")
    ap.add_argument("--n", type=int, default=256, help="number of samples per cell")
    ap.add_argument(
        "--steps", type=int, default=200, help="sampler steps (max_steps/NFE)"
    )
    ap.add_argument("--out", type=str, default=None, help="json output path")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="CPU self-test: build a tiny EqM, save a tiny ckpt, run all cells.",
    )
    args = ap.parse_args(argv)

    if args.smoke:
        return _smoke()

    if not args.ckpt:
        ap.error("--ckpt is required unless --smoke")

    result = ablate(args.ckpt, n=args.n, steps=args.steps)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
