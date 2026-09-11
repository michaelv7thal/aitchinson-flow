"""Practical demonstration of an EqMLatent checkpoint's two working modes.

(1) Sequence healing — corrupt a clean text snippet at varying noise
    levels, run the sampler from the corrupted point, decode, compare
    to the original.

(2) Per-position uncertainty quantification — encode several types of
    inputs (clean text, randomly-permuted text, scrambled text, foreign-
    looking strings), report per-position ‖∇⟨z, f(z)⟩‖, demonstrate that
    the score is higher for off-manifold inputs.

Usage:
    python scripts/use_trained_model.py \\
        --ckpt runs/latent_d128_trainable_tied_ce0_ep20/epoch_final.pt
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

import aitchinson_flow.models  # noqa: E402,F401
from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.data.char_window_dataset import CHAR2ID  # noqa: E402
from aitchinson_flow.models import build_model  # noqa: E402
from aitchinson_flow.training import build_training_datamodule  # noqa: E402
from scripts.eval_full import _config_from_payload  # noqa: E402

ALPHABET = "".join(sorted(CHAR2ID, key=CHAR2ID.__getitem__))


def text_to_ids(text: str, L: int, pad_char: str = " ") -> torch.Tensor:
    text = text.lower()
    text = "".join(c if c in CHAR2ID else " " for c in text)
    text = (text + pad_char * L)[:L]
    return torch.tensor([CHAR2ID[c] for c in text], dtype=torch.long).unsqueeze(0)


def ids_to_text(ids: torch.Tensor) -> list[str]:
    return ["".join(ALPHABET[int(i)] for i in row) for row in ids.cpu()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
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
    K = cfg.text8_dataset.K
    L = cfg.text8_dataset.L
    embed = model.embed.weight.detach().to(device)

    # ─── (1) Sequence healing ─────────────────────────────────────────
    print("=" * 70)
    print("MODE 1 — Sequence healing")
    print("=" * 70)

    clean = "the capital of one government after another fell to"
    ids_clean = text_to_ids(clean, L=L).to(device)

    print(f"\nClean text ({L} chars):")
    print(f"  {ids_to_text(ids_clean)[0]!r}")

    z_clean = model.encode(ids_clean)
    embed_norm = embed.norm(dim=-1).mean().item()
    for alpha in (0.05, 0.10, 0.20, 0.40, 0.60):
        torch.manual_seed(42)
        z_corr = z_clean + alpha * embed_norm * torch.randn_like(z_clean)
        # decode the corrupted point directly (without healing)
        argmax_corr = model.decode_to_token_ids(z_corr).cpu()
        with torch.no_grad():
            z_healed = model.sample(1, L, max_steps=args.steps, x_init=z_corr)
        argmax_healed = model.decode_to_token_ids(z_healed).cpu()
        n_match = int((argmax_healed == ids_clean.cpu()).sum())
        print(f"\nα = {alpha:.2f}  σ_perturb = {alpha * embed_norm:.3f}")
        print(f"  corrupted (no healing): {ids_to_text(argmax_corr)[0]!r}")
        print(f"  healed   (after sampler): {ids_to_text(argmax_healed)[0]!r}")
        print(f"  recovery: {n_match}/{L} ({100*n_match/L:.1f}%)")

    # ─── (2) Per-position uncertainty ─────────────────────────────────
    print()
    print("=" * 70)
    print("MODE 2 — Per-position uncertainty quantification")
    print("=" * 70)

    in_dist = "the capital of one government after another fell to"
    permuted = "wllf trvno aetan rooenehrtmnga laupcei pteflo lt"  # same chars, scrambled
    foreign  = "qzxbmdpvhqjzkwfnxqcuvyzjglkpqxbnvfwsmhqzykrxnglq"  # rare-letter-heavy
    repeat   = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

    inputs = [
        ("in-distribution", in_dist),
        ("permuted (same chars, no order)", permuted),
        ("foreign (rare-letter-heavy)", foreign),
        ("repetitive (single char)", repeat),
    ]

    print(f"\n{'description':<40} {'mean':>8} {'max':>8} {'argmax_pos':>10}")
    print("-" * 75)
    for label, text in inputs:
        ids = text_to_ids(text, L=L).to(device)
        z = model.encode(ids)
        unc = model.position_uncertainty(z)[0].detach().cpu()
        print(f"{label:<40} {unc.mean().item():>8.3f} {unc.max().item():>8.3f} "
              f"{int(unc.argmax()):>10d}")
        # Show per-position scores aligned with text.
        text_pad = (text.lower() + " " * L)[:L]
        scores_str = " ".join(f"{c}{int(u*10):x}" for c, u in zip(text_pad, unc.tolist()))
        # too long to print full; just truncate
        # print("    pos→ " + " ".join(text_pad))
        # print("    unc→ " + " ".join(f"{int(u*10):x}" for u in unc.tolist()))

    print()
    print("Interpretation: 'mean' uncertainty is the sequence-level OOD score —")
    print("higher = more out-of-distribution. 'argmax_pos' is the position the")
    print("model finds most surprising; useful for OCR/typo localisation.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
