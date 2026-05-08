"""Phase 17 — quantitative healing/recovery test for EqM-style checkpoints.

Probes the EqM design promise: random init → equilibrium descent → valid text.
By initialising the sampler at *partially-scrambled* clean text (instead of
random noise) and measuring how many corrupted positions snap back to the
original token, we directly test whether the trained energy field has
basins at clean text.

For each scramble rate r, sample N initialisations from val text, replace
fraction r of tokens with uniform random ones, run NAG-GD (or whatever
sampler ``model.sample`` is configured for) from there, and report:

  * recovery_at_corrupt   — fraction of *corrupted* positions whose
                            argmax-decoded token is the original clean
                            token. The headline metric for fidelity.
  * recovery_at_uncorrupt — fraction of *uncorrupted* positions that
                            stay correct. Should be ~1.0 for a well-
                            behaved sampler; lower indicates the sampler
                            is destroying structure rather than fixing it.
  * recovered_kl_bi       — bigram KL of the decoded outputs vs. training
                            distribution. Tells us whether outputs are
                            valid text *regardless* of which sentence
                            they correspond to.

Outputs:
    runs/<name>/healing.json   per-rate metrics + 8 example decodes
    runs/<name>/healing.png    recovery curves (corrupt + uncorrupt vs r)

Usage:
    python scripts/eval_healing.py --ckpt runs/eqm_data50k_ep5_mse_v3/epoch_final.pt \\
        --out runs/eqm_data50k_ep5_mse_v3/healing.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import aitchinson_flow.models  # noqa: E402,F401
from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.data.char_window_dataset import CHAR2ID, VOCAB_SIZE  # noqa: E402
from aitchinson_flow.data.corruption import corrupt_token_ids  # noqa: E402
from aitchinson_flow.data.transforms import token_ids_to_features  # noqa: E402
from aitchinson_flow.models import build_model  # noqa: E402
from aitchinson_flow.training import build_training_datamodule  # noqa: E402

from scripts.eval_full import _config_from_payload, ngram_kl  # noqa: E402


ALPHABET = "".join(sorted(CHAR2ID, key=CHAR2ID.__getitem__))
K = VOCAB_SIZE


def _ids_to_text(ids: torch.Tensor) -> list[str]:
    return ["".join(ALPHABET[int(i)] for i in row) for row in ids.cpu()]


def _heal_one_rate(
    model: Any,
    clean_ids: torch.Tensor,
    *,
    rate: float,
    K: int,
    label_smoothing: float,
    max_steps: int,
    seed: int,
) -> dict[str, Any]:
    """Run the sampler from a partially-scrambled clean init at one rate.

    Returns per-rate metrics + 8 sample decodes for inspection.
    """
    device = next(model.parameters()).device
    B, L = clean_ids.shape
    clean_ids = clean_ids.to(device)

    if rate <= 0.0:
        scrambled_ids = clean_ids.clone()
    else:
        scrambled_ids = corrupt_token_ids(
            clean_ids, vocab_size=K, corrupt_rate=rate, seed=seed
        )

    # Convert to CLR features for the sampler's x_init.
    x_init = token_ids_to_features(scrambled_ids, K, label_smoothing=label_smoothing)
    x_init = x_init.to(device)

    # Healing: run the sampler from the scrambled CLR as initial state.
    with torch.no_grad():
        x_out = model.sample(B, L, max_steps=max_steps, x_init=x_init)
        log_probs = model.decode_to_logprobs(x_out)
        out_ids = log_probs.argmax(-1)

    corrupted_mask = (scrambled_ids != clean_ids)
    uncorrupted_mask = ~corrupted_mask
    n_corrupted = int(corrupted_mask.sum().item())
    n_uncorrupted = int(uncorrupted_mask.sum().item())

    if n_corrupted > 0:
        recovery_at_corrupt = float(
            (out_ids[corrupted_mask] == clean_ids[corrupted_mask]).float().mean()
        )
    else:
        recovery_at_corrupt = float("nan")  # rate=0; nothing to recover

    if n_uncorrupted > 0:
        recovery_at_uncorrupt = float(
            (out_ids[uncorrupted_mask] == clean_ids[uncorrupted_mask]).float().mean()
        )
    else:
        recovery_at_uncorrupt = float("nan")

    # Bigram-KL of outputs against training distribution: independent of
    # which clean sentence the init came from. Captures "is the output
    # valid text" even if the sampler walked away from the source.
    return {
        "rate": rate,
        "n_corrupted": n_corrupted,
        "n_uncorrupted": n_uncorrupted,
        "recovery_at_corrupt": recovery_at_corrupt,
        "recovery_at_uncorrupt": recovery_at_uncorrupt,
        "out_ids": out_ids.cpu(),
        "scrambled_ids": scrambled_ids.cpu(),
        "clean_ids": clean_ids.cpu(),
    }


def evaluate_healing(
    ckpt_path: str | Path,
    *,
    n_samples: int = 256,
    max_steps: int = 200,
    rates: tuple[float, ...] = (0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0),
    seed: int = 1234,
) -> dict[str, Any]:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = _config_from_payload(payload)
    device = cfg.training.device
    K_ = cfg.text8_dataset.K
    label_smoothing = cfg.transformation.label_smoothing

    model = build_model(cfg).to(device)
    state = payload["model_state_dict"] if "model_state_dict" in payload else payload
    model.load_state_dict(state)
    model.eval()

    if not hasattr(model, "decode_to_logprobs"):
        raise RuntimeError(
            f"healing test requires CLR-feature output (decode_to_logprobs); "
            f"{type(model).__name__} doesn't expose one. Skipping."
        )
    if not hasattr(model, "sample"):
        raise RuntimeError(f"{type(model).__name__} has no sample()")

    dm, _ = build_training_datamodule(cfg)
    val_ids = dm.splits.val.long()
    train_ids = dm.splits.train.long()

    rng = torch.Generator().manual_seed(seed)
    pick = torch.randperm(val_ids.shape[0], generator=rng)[:n_samples]
    clean_ids = val_ids[pick].to(device)

    rows: list[dict[str, Any]] = []
    full_outputs: dict[str, dict[str, Any]] = {}

    for r in rates:
        cell = _heal_one_rate(
            model,
            clean_ids,
            rate=float(r),
            K=K_,
            label_smoothing=label_smoothing,
            max_steps=max_steps,
            seed=seed + int(round(r * 1000)),
        )
        kl_bi = ngram_kl(cell["out_ids"], train_ids, 2)
        kl_uni = ngram_kl(cell["out_ids"], train_ids, 1)
        # Three example decodes per rate: original / scrambled / out.
        idx = list(range(min(3, n_samples)))
        examples = [
            {
                "clean":     _ids_to_text(cell["clean_ids"][[i]])[0],
                "scrambled": _ids_to_text(cell["scrambled_ids"][[i]])[0],
                "recovered": _ids_to_text(cell["out_ids"][[i]])[0],
            }
            for i in idx
        ]
        rows.append({
            "rate": r,
            "n_corrupted": cell["n_corrupted"],
            "n_uncorrupted": cell["n_uncorrupted"],
            "recovery_at_corrupt": cell["recovery_at_corrupt"],
            "recovery_at_uncorrupt": cell["recovery_at_uncorrupt"],
            "out_kl_bi": float(kl_bi),
            "out_kl_uni": float(kl_uni),
            "examples": examples,
        })
        print(
            f"  rate={r:>4.2f}  "
            f"recover@corrupt={cell['recovery_at_corrupt']:.3f}  "
            f"recover@uncorrupt={cell['recovery_at_uncorrupt']:.3f}  "
            f"KL_bi={kl_bi:.3f}"
        )

    return {
        "ckpt": str(ckpt_path),
        "model_name": cfg.training.model_name,
        "n_samples": n_samples,
        "max_steps": max_steps,
        "epoch": int(payload.get("epoch", -1)) if not isinstance(payload.get("epoch"), str) else payload.get("epoch"),
        "rows": rows,
    }


def _make_figure(summary: dict[str, Any], out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[healing] matplotlib unavailable; skipping figure")
        return

    rates = [r["rate"] for r in summary["rows"]]
    rec_c = [r["recovery_at_corrupt"] for r in summary["rows"]]
    rec_u = [r["recovery_at_uncorrupt"] for r in summary["rows"]]
    kls = [r["out_kl_bi"] for r in summary["rows"]]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(rates, rec_c, "o-", color="#3a7ca5", label="recover @ corrupt position")
    axes[0].plot(rates, rec_u, "s-", color="#a55a3a", label="recover @ uncorrupt position")
    axes[0].axhline(1.0 / 27, color="gray", linestyle=":", alpha=0.6, label="chance (1/K)")
    axes[0].set_xlabel("scramble rate r")
    axes[0].set_ylabel("token recovery rate")
    axes[0].set_title(f"Recovery vs scramble — {summary['model_name']}")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.3)

    axes[1].plot(rates, kls, "o-", color="#3a7c5a")
    axes[1].set_xlabel("scramble rate r")
    axes[1].set_ylabel("output bigram KL vs training")
    axes[1].set_title("Output validity (lower = more text-like)")
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        f"Healing test — {summary['model_name']} ({Path(summary['ckpt']).parent.name})"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n", type=int, default=256, dest="n_samples")
    p.add_argument("--steps", type=int, default=200, dest="max_steps")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--no-figure", action="store_true")
    args = p.parse_args(argv)

    print(f"[healing] ckpt={args.ckpt}  n={args.n_samples}  steps={args.max_steps}")
    summary = evaluate_healing(
        args.ckpt,
        n_samples=args.n_samples,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"[healing] wrote {out_path}")
        if not args.no_figure:
            fig_path = out_path.with_suffix(".png")
            _make_figure(summary, fig_path)
            print(f"[healing] wrote {fig_path}")


if __name__ == "__main__":
    main()
