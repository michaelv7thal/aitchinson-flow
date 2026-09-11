"""Qualitative evaluation for DirichletFMSvgp — produces three sections:

  1) Ground-truth training-corpus samples (10 real sequences from val).
  2) Unconditional model samples (10 sequences from the trained DFM).
  3) OOD diagnostic: 10 valid (real) + 10 invalid (scrambled) sequences,
     each annotated with the SVGP posterior std (epistemic UQ), latent
     mean, and in-distribution probability.

Both terminal output and a JSON file are written. Run *after*
``scripts/fit_dfm_svgp_hinge.py`` has populated ``model_with_svgp_hinge.pt``.

Usage:
    python scripts/eval_dfm_svgp_qualitative.py \\
        --ckpt runs/dfm_svgp_poc/model_with_svgp_hinge.pt \\
        --n 10 --out runs/dfm_svgp_poc/svgp_qualitative.json
"""

from __future__ import annotations

import argparse
import json
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
from aitchinson_flow.data.char_window_dataset import CHAR2ID  # noqa: E402
from aitchinson_flow.models import build_model  # noqa: E402
from aitchinson_flow.training import build_training_datamodule  # noqa: E402
from scripts.eval_full import _config_from_payload  # noqa: E402

ALPHABET = "".join(sorted(CHAR2ID, key=CHAR2ID.__getitem__))


def _decode_ids(ids: torch.Tensor) -> list[str]:
    """(N, L) long tensor → list of N strings."""
    out = []
    for row in ids.cpu():
        out.append("".join(ALPHABET[int(i)] for i in row))
    return out


def _hr(title: str, width: int = 80) -> str:
    bar = "=" * width
    return f"\n{bar}\n{title}\n{bar}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True,
                    help="Path to model_with_svgp_hinge.pt (from fit_dfm_svgp_hinge.py)")
    ap.add_argument("--n", type=int, default=10,
                    help="number of sequences per section")
    ap.add_argument("--t-eval", type=float, default=None,
                    help="override cfg.dfm_svgp.t_eval for OOD scoring")
    ap.add_argument("--out", type=str, default=None,
                    help="JSON output path (default: <ckpt-dir>/svgp_qualitative.json)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-nfe", type=int, default=None,
                    help="override cfg.dirichlet_fm.sample_nfe for unconditional sampling")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    # Prefer cfg embedded in this payload; fall back to the sibling
    # epoch_final.pt which always has the proper cfg (run_sweep saves it
    # there). The model_with_svgp_hinge.pt produced by fit_dfm_svgp_hinge.py is a
    # state_dict-only blob and needs the sibling cfg.
    if isinstance(payload, dict) and "cfg" in payload:
        cfg = _config_from_payload(payload)
    else:
        sibling = Path(args.ckpt).parent / "epoch_final.pt"
        if not sibling.exists():
            raise SystemExit(
                f"No cfg in {args.ckpt} and no sibling epoch_final.pt to recover from"
            )
        sibling_payload = torch.load(sibling, map_location="cpu", weights_only=False)
        cfg = _config_from_payload(sibling_payload)

    device = cfg.training.device
    model = build_model(cfg).to(device)
    state = payload["model_state_dict"] if "model_state_dict" in payload else payload
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        # Almost certainly the SVGP buffers were not in the ckpt yet.
        svgp_missing = [m for m in missing if "svgp" in m]
        if svgp_missing:
            print("[warn] SVGP weights missing from ckpt — was fit_dfm_svgp_hinge.py run?")
            print("       Proceeding with unfitted SVGP (uniform posterior std).")
    model.eval()

    t_eval = float(args.t_eval if args.t_eval is not None else cfg.dfm_svgp.t_eval)
    L = cfg.text8_dataset.L

    dm, _ = build_training_datamodule(cfg)
    val_loader = dm.val_dataloader() or dm.train_dataloader()

    # Pull a batch of ground-truth sequences
    gt_batch = next(iter(val_loader))
    gt_tok_all = gt_batch["token_ids"].to(device).long()
    gt_tok = gt_tok_all[: args.n]
    gt_text = _decode_ids(gt_tok)

    # ----- Section 1: Ground-truth training-corpus samples -------------
    section1: list[dict] = []
    print(_hr("[1/3] Ground-truth training-corpus samples (val split)"))
    for i, t in enumerate(gt_text):
        print(f"  GT [{i:>2}]  {t!r}")
        section1.append({"idx": i, "text": t})

    # ----- Section 2: Unconditional model samples ----------------------
    print(_hr("[2/3] Unconditional model samples"))
    nfe = args.sample_nfe if args.sample_nfe is not None else cfg.dirichlet_fm.sample_nfe
    with torch.no_grad():
        gen_tok = model.sample(args.n, L, nfe=nfe)
    gen_text = _decode_ids(gen_tok)
    section2: list[dict] = []
    for i, t in enumerate(gen_text):
        print(f"  GEN[{i:>2}]  {t!r}")
        section2.append({"idx": i, "text": t})

    # ----- Section 3: OOD diagnostic — valid vs invalid + std ----------
    print(_hr(f"[3/3] OOD diagnostic — SVGP posterior at t_eval={t_eval:.2f}"))

    valid_scores = model.ood_score(gt_tok, t_eval=t_eval)

    # Build invalids: per-row shuffle (preserves unigram, breaks bigram).
    perms = torch.argsort(torch.rand(args.n, L, device=device), dim=-1)
    invalid_tok = gt_tok.gather(1, perms)
    invalid_text = _decode_ids(invalid_tok)
    invalid_scores = model.ood_score(invalid_tok, t_eval=t_eval)

    print(f"  {'idx':>3}  {'std':>8}  {'mean':>8}  {'prob':>6}  text")
    print(f"  {'---':>3}  {'---':>8}  {'---':>8}  {'---':>6}  ----")
    section3_valid: list[dict] = []
    for i in range(args.n):
        s = float(valid_scores["std"][i].item())
        m = float(valid_scores["mean"][i].item())
        p = float(valid_scores["prob"][i].item())
        print(f"  V[{i:>2}]  {s:>8.4f}  {m:>+8.4f}  {p:>6.3f}  {gt_text[i]!r}")
        section3_valid.append({"idx": i, "text": gt_text[i], "std": s, "mean": m, "prob": p})

    print()
    section3_invalid: list[dict] = []
    for i in range(args.n):
        s = float(invalid_scores["std"][i].item())
        m = float(invalid_scores["mean"][i].item())
        p = float(invalid_scores["prob"][i].item())
        print(f"  I[{i:>2}]  {s:>8.4f}  {m:>+8.4f}  {p:>6.3f}  {invalid_text[i]!r}")
        section3_invalid.append({"idx": i, "text": invalid_text[i], "std": s, "mean": m, "prob": p})

    # Aggregate stats for the OOD section
    v_std = valid_scores["std"].cpu()
    i_std = invalid_scores["std"].cpu()
    v_prob = valid_scores["prob"].cpu()
    i_prob = invalid_scores["prob"].cpu()
    print()
    print(f"  summary  valid:    mean(std)={v_std.mean():.4f}  mean(prob)={v_prob.mean():.3f}")
    print(f"  summary  invalid:  mean(std)={i_std.mean():.4f}  mean(prob)={i_prob.mean():.3f}")
    print(f"  summary  Δmean(std) = {i_std.mean() - v_std.mean():+.4f}  "
          f"(positive ⇒ invalid more uncertain, as expected)")
    print(f"  summary  Δmean(prob)= {v_prob.mean() - i_prob.mean():+.3f}  "
          f"(positive ⇒ valid scored higher as in-dist, as expected)")

    out_path = Path(args.out) if args.out else Path(args.ckpt).parent / "svgp_qualitative.json"
    payload_json = {
        "ckpt": str(args.ckpt),
        "t_eval": t_eval,
        "n": int(args.n),
        "sample_nfe": int(nfe),
        "ground_truth": section1,
        "unconditional": section2,
        "ood_valid": section3_valid,
        "ood_invalid": section3_invalid,
        "summary": {
            "valid_mean_std": float(v_std.mean()),
            "valid_mean_prob": float(v_prob.mean()),
            "invalid_mean_std": float(i_std.mean()),
            "invalid_mean_prob": float(i_prob.mean()),
            "delta_mean_std": float(i_std.mean() - v_std.mean()),
            "delta_mean_prob": float(v_prob.mean() - i_prob.mean()),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload_json, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
