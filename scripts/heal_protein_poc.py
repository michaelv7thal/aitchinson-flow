"""Heal-from-context proof of concept on a real protein Wikipedia article.

Motivation
----------
"Does the DirichletFM healer recover corrupted tokens from surrounding context?"
BRCA1 is a *bad* probe: 'brca' occurs 0x in the text8 train split and text8
spells digits out ("brca1" -> "brca one"), so the model never saw the token and
*cannot* heal it. We use an in-distribution protein (default: insulin, 347 train
occurrences, pure-letter name) whose statistics the model actually learned.

Same machinery as scripts/heal_dirichlet.py (NLL localizer -> threshold
calibrated on generic text8 -> inpaint -> score); the demo *set* is windows of
the protein article. Three tracks:

  (A) char noise  -- random --corrupt-rate character substitution (misspellings /
      non-words -> high surprise -> the easy, denoise-typos case).
  (B) targeted    -- corrupt every letter of the protein name in a window and
      show it healed back purely from context.
  (C) false info  -- replace --corrupt-rate of whole WORDS with a *different real
      same-length English word* (lexically valid, semantically wrong -> locally
      fluent -> the hard case: does context-healing correct plausible falsehoods
      or only noise?).

A and C are scored identically (fix_rate / net-per-corrupt / loc P,R) so they are
directly comparable at the same GT-free operating point.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch


def _bootstrap() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    for p in (repo_root / "src", repo_root):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


_bootstrap()

from datasets import load_dataset  # noqa: E402

from scripts.heal_dirichlet import (  # noqa: E402
    _decode,
    _load_dirichletfm,
    aggregate,
    calibrate_threshold,
    fbeta,
    inpaint,
    make_bgmm_localizer,
    make_gmm_localizer,
    make_nll_localizer,
    score_healing,
)
from aitchinson_flow.data.char_window_dataset import text_to_windows  # noqa: E402
from aitchinson_flow.data.corruption import (  # noqa: E402
    build_vocab_by_len,
    corrupt_false_info,
    corrupt_token_ids,
)

_DIGITS = {"0": " zero ", "1": " one ", "2": " two ", "3": " three ", "4": " four ",
           "5": " five ", "6": " six ", "7": " seven ", "8": " eight ", "9": " nine "}


def wikifil(text: str) -> str:
    """Matt Mahoney's text8 preprocessing: lowercase, spell each digit, map every
    remaining non-[a-z] char to space, collapse runs."""
    text = text.lower()
    for d, w in _DIGITS.items():
        text = text.replace(d, w)
    text = re.sub(r"[^a-z]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_article(path: str) -> str:
    pages = json.load(open(path))["query"]["pages"]
    return next(iter(pages.values()))["extract"]


def name_positions(window_ids: torch.Tensor, name: str) -> list[tuple[int, int]]:
    s = _decode(window_ids)
    return [(m.start() + 1, m.end() - 1)
            for m in re.finditer(rf"(?<![a-z]){re.escape(name)}(?![a-z])", s)]


def run_track(corrupt_batches, demo_clean, score, thr, model, *, nfe, device):
    """Localize (NLL>thr) -> inpaint -> score, averaged over the corruption seeds."""
    per_seed, first = [], None
    for s, dc in enumerate(corrupt_batches):
        mask = score(dc) > thr
        healed = inpaint(model, dc, mask, nfe=nfe, device=device)
        per_seed.append(score_healing(demo_clean, dc, healed, mask, dc != demo_clean))
        if s == 0:
            first = (dc, mask, healed)
    return aggregate(per_seed), first


def _render(clean_row, dc_row, mask_row, healed_row) -> dict:
    tc = (dc_row != clean_row).tolist()
    return {
        "clean": _decode(clean_row), "corrupted": _decode(dc_row), "healed": _decode(healed_row),
        "truec": "".join("^" if x else " " for x in tc),
        "flag": "".join("*" if x else " " for x in mask_row.tolist()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--article-json", required=True)
    ap.add_argument("--name", default="insulin")
    ap.add_argument("--out", default="scratchpad/heal_protein_poc.json")
    ap.add_argument("--corrupt-rate", type=float, default=0.15)
    ap.add_argument("--n-demo", type=int, default=32)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--n-cal", type=int, default=128)
    ap.add_argument("--vocab-chars", type=int, default=8_000_000,
                    help="# train chars scanned to build the false-info word list")
    ap.add_argument("--target-fprs", type=str, default="0.02,0.05,0.10")
    ap.add_argument("--t-nll", type=float, default=3.0)
    ap.add_argument("--nfe", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    # --- localizer: training-free NLL (default) / GMM / Bayesian-DP-GMM density ---
    ap.add_argument("--localizer", choices=["nll", "gmm", "bgmm"], default="nll",
                    help="'nll' = per-token denoiser surprise (local, misses false info); "
                         "'gmm' = Gaussian-mixture density on frozen contextual features; "
                         "'bgmm' = Bayesian DP infinite-mixture density (infers K)")
    ap.add_argument("--t-eval", type=float, default=None,
                    help="[gmm/bgmm] path-time for the density features (default: "
                         "cfg.dfm_svgp.t_eval for gmm, 7.5 for bgmm — falseinfo-best)")
    ap.add_argument("--fit-seqs", type=int, default=256,
                    help="[gmm/bgmm] # generic text8 train windows to fit the mixture")
    ap.add_argument("--gmm-n-components", type=int, default=8)
    ap.add_argument("--gmm-covariance-type",
                    choices=["diag", "full", "tied", "spherical"], default="diag")
    ap.add_argument("--gmm-pca-dim", type=int, default=64)
    ap.add_argument("--gmm-reg-covar", type=float, default=1e-4)
    # --- bgmm knobs (Bayesian DP infinite mixture) ---
    ap.add_argument("--bgmm-max-components", type=int, default=20)
    ap.add_argument("--bgmm-covariance-type",
                    choices=["diag", "full", "tied", "spherical"], default="full")
    ap.add_argument("--bgmm-pca-dim", type=int, default=64)
    ap.add_argument("--bgmm-max-iter", type=int, default=1000)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = _load_dirichletfm(args.ckpt, device)
    K, L = cfg.text8_dataset.K, cfg.text8_dataset.L
    print(f"[poc] ckpt={args.ckpt}  K={K} L={L} t_max={model.dfm.t_max} device={device}", flush=True)

    # --- article -> text8 windows (demo set); prefer windows containing the name ---
    clean_txt = wikifil(load_article(args.article_json))
    all_win = text_to_windows(clean_txt, L)
    n_name = len(re.findall(rf"(?<![a-z]){re.escape(args.name)}(?![a-z])", clean_txt))
    has_name = [i for i in range(all_win.shape[0]) if name_positions(all_win[i], args.name)]
    order = has_name + [i for i in range(all_win.shape[0]) if i not in set(has_name)]
    demo_clean = all_win[order][: args.n_demo]
    print(f"[poc] '{args.name}': {len(clean_txt)} text8 chars -> {all_win.shape[0]} windows; "
          f"name appears ~{n_name}x; demo={demo_clean.shape[0]} ({len(has_name)} contain name)",
          flush=True)

    # --- generic text8 (test split): calibration windows + train words for false-info ---
    ds = load_dataset("afmck/text8")
    cal_tok = text_to_windows(ds["test"][0]["text"][: (args.n_cal + 4) * L], L)[: args.n_cal]
    by_len = build_vocab_by_len(ds["train"][0]["text"][: args.vocab_chars])
    print(f"[poc] calibration={cal_tok.shape[0]} generic windows; "
          f"false-info vocab lengths {min(by_len)}-{max(by_len)} "
          f"(e.g. len-7: {by_len[7][:6]})", flush=True)

    # --- localizer + GT-free operating point (max cal F0.5 on generic text8) ---
    print(f"[poc] localizer={args.localizer}", flush=True)
    bgmm_n_eff = None
    t_eval = None
    if args.localizer in ("gmm", "bgmm"):
        default_t = 7.5 if args.localizer == "bgmm" else float(cfg.dfm_svgp.t_eval)
        t_eval = float(args.t_eval) if args.t_eval is not None else default_t
        fit_tok = text_to_windows(
            ds["train"][0]["text"][(args.vocab_chars): (args.vocab_chars) + (args.fit_seqs + 4) * L],
            L,
        )[: args.fit_seqs]
        if args.localizer == "bgmm":
            score, _, bgmm_n_eff = make_bgmm_localizer(
                model, fit_tok, t_eval, device,
                max_components=args.bgmm_max_components,
                covariance_type=args.bgmm_covariance_type,
                pca_dim=args.bgmm_pca_dim, max_iter=args.bgmm_max_iter, seed=args.seed)
        else:
            score, _ = make_gmm_localizer(
                model, fit_tok, t_eval, device, n_components=args.gmm_n_components,
                covariance_type=args.gmm_covariance_type, pca_dim=args.gmm_pca_dim,
                reg_covar=args.gmm_reg_covar, seed=args.seed)
    else:
        score, _ = make_nll_localizer(model, args.t_nll, device)
    cal_corr = corrupt_token_ids(cal_tok.clone(), vocab_size=K,
                                 corrupt_rate=args.corrupt_rate, seed=args.seed + 1)
    fprs = [float(x) for x in args.target_fprs.split(",") if x.strip()]
    thr_by_fpr = {f: calibrate_threshold(score, cal_tok, cal_corr, target_fpr=f) for f in fprs}
    # Same damage-averse rule as heal_dirichlet: F0.5 (precision-weighted), NOT F1.
    # Max-F1 over-weights recall and picks the DAMAGING end of the sweep.
    sel_fpr = max(fprs, key=lambda f: fbeta(thr_by_fpr[f][1]))
    thr = thr_by_fpr[sel_fpr][0]
    print(f"[poc] operating point: fpr={sel_fpr} thr={thr:.3f} (GT-free, max cal-F0.5)",
          flush=True)

    # =================== Track A: random character noise ===================
    print("\n=== Track A: random character corruption (denoise typos) ===", flush=True)
    A_batches = [corrupt_token_ids(demo_clean.clone(), vocab_size=K,
                                   corrupt_rate=args.corrupt_rate, seed=args.seed + 100 + s)
                 for s in range(args.n_seeds)]

    # =================== Track C: sparse FALSE INFORMATION ===================
    C_batches = [corrupt_false_info(demo_clean, args.corrupt_rate, by_len, seed=args.seed + 200 + s)
                 for s in range(args.n_seeds)]

    # Sweep EVERY operating point (not just the selected one) so the benchmark can
    # report the least-damaging point, exactly as the text8 arms do. The selected
    # (GT-free, F0.5) row stays the headline.
    def _sweep(batches, tag):
        print(f"\n=== Track {tag}: FPR sweep ===", flush=True)
        print(f"{'fpr':>5} {'thr':>9} {'loc_P':>6} {'loc_R':>6} {'loc_F1':>7} {'fix':>6} "
              f"{'dmg':>6} {'net/corrupt':>12}", flush=True)
        rows, sel_agg, sel_first = [], None, None
        for f in fprs:
            t = thr_by_fpr[f][0]
            agg, first = run_track(batches, demo_clean, score, t, model,
                                   nfe=args.nfe, device=device)
            mark = "  <- selected" if f == sel_fpr else ""
            print(f"{f:>5.2f} {t:>9.3f} {agg['loc_precision']:>6.3f} "
                  f"{agg['loc_recall']:>6.3f} {agg['loc_f1']:>7.3f} {agg['fix_rate']:>6.3f} "
                  f"{agg['damage_rate']:>6.3f} {agg['net_per_corrupt']:>+12.3f}{mark}",
                  flush=True)
            rows.append({"target_fpr": f, "threshold": t,
                         "calibration": thr_by_fpr[f][1], **agg})
            if f == sel_fpr:
                sel_agg, sel_first = agg, first
        assert sel_agg is not None and sel_first is not None, \
            f"sel_fpr={sel_fpr} not in target_fprs={fprs}"
        return rows, sel_agg, sel_first

    A_sweep, A_agg, A_first = _sweep(A_batches, "A (char noise)")
    C_sweep, C_agg, C_first = _sweep(C_batches, "C (false information)")

    def nchars(batches):
        return float(np.mean([(b != demo_clean).float().mean().item() for b in batches]))

    print(f"\n{'track':<26}{'corrupt%chars':>13}{'loc_P':>7}{'loc_R':>7}"
          f"{'fix':>7}{'dmg':>7}{'net/corrupt':>14}", flush=True)
    for lbl, agg, bs in [("A: char noise", A_agg, A_batches),
                         ("C: false information", C_agg, C_batches)]:
        print(f"{lbl:<26}{100*nchars(bs):>12.1f}%{agg['loc_precision']:>7.3f}"
              f"{agg['loc_recall']:>7.3f}{agg['fix_rate']:>7.3f}{agg['damage_rate']:>7.3f}"
              f"{agg['net_per_corrupt']:>+9.3f}±{agg['net_per_corrupt_std']:.3f}", flush=True)

    # example renders (seed 0)
    clean = demo_clean.cpu()
    exA = _render(clean[0], A_first[0].cpu()[0], A_first[1].cpu()[0], A_first[2].cpu()[0])
    exC = _render(clean[0], C_first[0].cpu()[0], C_first[1].cpu()[0], C_first[2].cpu()[0])
    for tag, ex in [("A", exA), ("C", exC)]:
        print(f"\n--- Track {tag} example window (seed 0) ---", flush=True)
        for k in ("clean", "corrupted", "healed"):
            print(f"  {k:<9}: {ex[k]!r}", flush=True)

    # =================== Track B: targeted name showcase ===================
    print(f"\n=== Track B: corrupt every letter of '{args.name}', heal from context ===", flush=True)
    g = torch.Generator().manual_seed(args.seed + 7)
    showcase = []
    for wi in has_name[:3]:
        win = all_win[wi].clone()
        corr = win.clone()
        pos = []
        for (a, b) in name_positions(win, args.name):
            for p in range(a, b):
                cur = int(corr[p])
                new = int(torch.randint(0, K - 1, (1,), generator=g).item())  # a-z only
                corr[p] = new + 1 if new >= cur else new
                pos.append(p)
        mask_row = (score(corr.unsqueeze(0)) > thr)[0]
        healed = inpaint(model, corr.unsqueeze(0), mask_row.unsqueeze(0), nfe=args.nfe, device=device)[0]
        restored = all(int(healed[p]) == int(win[p]) for p in pos)
        print(f"\n  [window {wi}] name letters corrupted={len(pos)} "
              f"flagged={sum(int(mask_row[p]) for p in pos)} -> fully restored: {restored}", flush=True)
        print(f"    corr  : {_decode(corr)!r}", flush=True)
        print(f"    healed: {_decode(healed)!r}", flush=True)
        showcase.append({"window": int(wi), "name_chars_corrupted": len(pos),
                         "name_restored": bool(restored), "clean": _decode(win),
                         "corrupted": _decode(corr), "healed": _decode(healed)})

    out = {
        "ckpt": args.ckpt, "protein": args.name, "K": K, "L": L,
        "localizer": args.localizer, "t_nll": args.t_nll, "nfe": args.nfe,
        "density_t_eval": (t_eval if args.localizer in ("gmm", "bgmm") else None),
        "bgmm_n_effective": bgmm_n_eff,
        "corrupt_rate": args.corrupt_rate, "n_demo": int(demo_clean.shape[0]),
        "n_seeds": args.n_seeds, "name_occurrences_in_article": n_name,
        "operating_point": {"fpr": sel_fpr, "threshold": thr},
        "selection": "max cal-set F0.5 (precision-weighted, GT-free; healing damage-averse)",
        "target_fprs": fprs,
        "track_A_char_noise": {**A_agg, "corrupt_frac_chars": nchars(A_batches), "example": exA},
        "track_C_false_info": {**C_agg, "corrupt_frac_chars": nchars(C_batches), "example": exC},
        "track_A_sweep": A_sweep,
        "track_C_sweep": C_sweep,
        "track_B_targeted_name": showcase,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n[poc] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
