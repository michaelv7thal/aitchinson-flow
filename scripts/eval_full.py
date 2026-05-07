"""Canonical scorecard for an EqM checkpoint.

Loads the checkpoint, samples B sequences with the patched NAG-GD sampler, and
emits a single-line JSON record with unigram/bigram/trigram KL, entropy, energy
gradient norms, and 8 argmax decodes.

Output: writes the JSON to --out, also pretty-prints a one-line summary.

Usage:
    python scripts/eval_full.py --ckpt PATH --n 256 --steps 200 --out runs/<name>/eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

import aitchinson_flow.models  # noqa: E402,F401 — populate registry
from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.data.char_window_dataset import CHAR2ID, VOCAB_SIZE  # noqa: E402
from aitchinson_flow.models import build_model  # noqa: E402
from aitchinson_flow.training import build_training_datamodule  # noqa: E402

ALPHABET = "".join(sorted(CHAR2ID, key=CHAR2ID.__getitem__))
K = VOCAB_SIZE


def ids_to_text(ids: torch.Tensor) -> list[str]:
    return ["".join(ALPHABET[int(i)] for i in row) for row in ids.cpu()]


def unigram_kl(gen_ids: torch.Tensor, ref_ids: torch.Tensor) -> tuple[float, float, float]:
    gen = torch.zeros(K).scatter_add_(
        0, gen_ids.reshape(-1), torch.ones_like(gen_ids.reshape(-1), dtype=torch.float)
    )
    ref = torch.zeros(K).scatter_add_(
        0, ref_ids.reshape(-1), torch.ones_like(ref_ids.reshape(-1), dtype=torch.float)
    )
    gen = (gen + 1e-9) / (gen.sum() + K * 1e-9)
    ref = (ref + 1e-9) / (ref.sum() + K * 1e-9)
    kl = float((gen * (gen.log() - ref.log())).sum())
    H_gen = float(-(gen * gen.log()).sum())
    H_ref = float(-(ref * ref.log()).sum())
    return kl, H_gen, H_ref


def _ngram_counts(ids: torch.Tensor, n: int) -> torch.Tensor:
    """Return a flat (K**n,) count tensor over all length-n contiguous windows."""
    flat = ids.reshape(-1, ids.shape[-1])
    L = flat.shape[1]
    if L < n:
        return torch.zeros(K**n)
    idx = torch.zeros(flat.shape[0], L - n + 1, dtype=torch.long)
    for i in range(n):
        idx = idx + flat[:, i : L - n + 1 + i].long() * (K ** (n - 1 - i))
    counts = torch.zeros(K**n)
    counts.scatter_add_(0, idx.reshape(-1), torch.ones(idx.numel()))
    return counts


def ngram_kl(gen_ids: torch.Tensor, ref_ids: torch.Tensor, n: int) -> float:
    gc = _ngram_counts(gen_ids, n)
    rc = _ngram_counts(ref_ids, n)
    smoothing = 1e-6
    gp = (gc + smoothing) / (gc.sum() + (K**n) * smoothing)
    rp = (rc + smoothing) / (rc.sum() + (K**n) * smoothing)
    return float((gp * (gp.log() - rp.log())).sum())


def _config_from_payload(payload: dict[str, Any]) -> Config:
    """Rebuild a Config dataclass from the dict saved with a checkpoint.

    Falls back to defaults for any missing keys (older checkpoints).
    """
    cfg = Config()
    saved = payload.get("cfg") or {}
    for section_name in ("training", "text8_dataset", "transformation", "transformer", "eqm", "dfm", "loss"):
        section_dict = saved.get(section_name) or {}
        if not section_dict:
            continue
        section = getattr(cfg, section_name)
        if not is_dataclass(section):
            continue
        valid = {k: v for k, v in section_dict.items() if k in {f.name for f in section.__dataclass_fields__.values()}}
        # Skip 'device' — string in payload, torch.device in config.
        valid.pop("device", None)
        if section_name == "training":
            valid.pop("device", None)
        try:
            setattr(cfg, section_name, replace(section, **valid))
        except TypeError:
            # Some sections are frozen and need from-scratch construction
            kls = type(section)
            setattr(cfg, section_name, kls(**valid))
    return cfg


def _apply_overrides(cfg: Config, overrides: dict[str, Any]) -> Config:
    """Apply dotted overrides post-load. Lets W1 re-evaluate the same EqM
    checkpoint under different sampler configs without re-training."""
    from dataclasses import replace, is_dataclass

    for key, value in overrides.items():
        parts = key.split(".")
        *parents, leaf = parts
        obj: Any = cfg
        for p in parents:
            obj = getattr(obj, p)
        if not is_dataclass(obj):
            raise TypeError(f"override path '{key}' does not end at a dataclass")
        new = replace(obj, **{leaf: value})
        target: Any = cfg
        for p in parents[:-1]:
            target = getattr(target, p)
        if parents:
            setattr(target, parents[-1], new)
        else:
            setattr(cfg, leaf, value)
    return cfg


def evaluate_checkpoint(
    ckpt_path: str | Path,
    *,
    n_samples: int,
    n_steps: int,
    grad_at_n: int = 64,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = Config()
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = _config_from_payload(payload)
    if overrides:
        cfg = _apply_overrides(cfg, overrides)
    device = cfg.training.device

    model = build_model(cfg).to(device)
    state = payload["model_state_dict"] if "model_state_dict" in payload else payload
    model.load_state_dict(state)
    model.eval()

    dm, _ = build_training_datamodule(cfg)
    train_ids = dm.splits.train.long()

    L = cfg.text8_dataset.L
    model_name = cfg.training.model_name

    # Sample. Two API conventions in the repo:
    # - EqM: model.sample(B, L, max_steps=...) returns (B, L, K) CLR features.
    # - DFM: model.sample(B, L, nfe=...)        returns (B, L) long token IDs.
    with torch.no_grad():
        if hasattr(model, "decode_to_logprobs"):
            x = model.sample(n_samples, L, max_steps=n_steps)
            log_probs = model.decode_to_logprobs(x)
            gen_ids = log_probs.argmax(-1).cpu()
        else:
            x = model.sample(n_samples, L, nfe=n_steps)
            gen_ids = x.cpu().long()

    kl_u, H_gen, H_ref = unigram_kl(gen_ids, train_ids)
    kl_b = ngram_kl(gen_ids, train_ids, 2)
    kl_t = ngram_kl(gen_ids, train_ids, 3)

    # Gradient norms only meaningful for energy-based models (EqM).
    if hasattr(model, "position_uncertainty"):
        val_features = dm._val_ds._features[: min(grad_at_n, len(dm._val_ds))].to(device)
        grad_gt = float(model.position_uncertainty(val_features).mean().item())
        grad_gen = float(
            model.position_uncertainty(x[: min(grad_at_n, x.shape[0])]).mean().item()
        )
    else:
        grad_gt = float("nan")
        grad_gen = float("nan")

    samples = ids_to_text(gen_ids[:8])

    return {
        "ckpt": str(ckpt_path),
        "model_name": model_name,
        "epoch": int(payload.get("epoch", -1)) if not isinstance(payload.get("epoch"), str) else payload.get("epoch"),
        "global_step": int(payload.get("global_step", -1)) if not isinstance(payload.get("global_step"), str) else payload.get("global_step"),
        "n_samples": n_samples,
        "sample_steps": n_steps,
        "unigram_kl": kl_u,
        "bigram_kl": kl_b,
        "trigram_kl": kl_t,
        "H_gen": H_gen,
        "H_gt": H_ref,
        "H_ratio": H_gen / H_ref if H_ref > 0 else float("nan"),
        "grad_at_gen": grad_gen,
        "grad_at_gt": grad_gt,
        "samples": samples,
    }


def _parse_override(s: str) -> tuple[str, Any]:
    """Parse `key=value` with simple type inference (bool / int / float / str)."""
    if "=" not in s:
        raise ValueError(f"--override expects key=value, got: {s!r}")
    key, raw = s.split("=", 1)
    if raw.lower() in ("true", "false"):
        return key, raw.lower() == "true"
    if raw.lower() in ("none", "null"):
        return key, None
    try:
        return key, int(raw)
    except ValueError:
        pass
    try:
        return key, float(raw)
    except ValueError:
        pass
    return key, raw


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n", type=int, default=256, dest="n_samples")
    p.add_argument("--steps", type=int, default=200, dest="n_steps")
    p.add_argument("--out", type=str, default=None)
    p.add_argument(
        "--override",
        action="append",
        default=[],
        help="Post-load cfg override, e.g. --override eqm.sampler=euler "
             "(repeatable). Useful for evaluating one checkpoint under "
             "multiple sampler configs.",
    )
    args = p.parse_args(argv)

    overrides = dict(_parse_override(s) for s in args.override) if args.override else None

    result = evaluate_checkpoint(
        args.ckpt,
        n_samples=args.n_samples,
        n_steps=args.n_steps,
        overrides=overrides,
    )
    if overrides:
        result["overrides"] = overrides

    print(
        f"ckpt={Path(args.ckpt).parent.name}/{Path(args.ckpt).name}  "
        f"epoch={result['epoch']}  step={result['global_step']}  "
        f"KL_uni={result['unigram_kl']:.4f}  KL_bi={result['bigram_kl']:.4f}  "
        f"KL_tri={result['trigram_kl']:.4f}  H_ratio={result['H_ratio']:.3f}  "
        f"|∇E|gen/gt={result['grad_at_gen']:.3f}/{result['grad_at_gt']:.3f}"
    )
    print("samples:")
    for s in result["samples"]:
        print(f"  '{s}'")

    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
