"""GPT-2 Spilled-Energy OOD baseline on text8 — MATCHED to the DirichletFM-NLL sweep.

The project's "forced baseline": a frozen pretrained GPT-2 read as an EBM, scored
with the **spilled energy** of Minut, Dewidar & Masi, "Spilled Energy in Large
Language Models" (ICLR 2026, arXiv:2602.18671). Zero-train, single forward pass.

    SE_i = logsumexp(logits_i) - logits_{i-1}[x_i]      (Definition 4.1 / Eq. 8)

This is a CROSS-STEP quantity: the logit energy of token x_i is measured at step
i-1, the marginal (free) energy at step i. The chain rule says the two should
cancel; the residual is the signal.

**Correction (2026-07):** this script previously computed the SAME-STEP quantity
``logsumexp(logits_{i-1}) - logits_{i-1}[x_i] = -log p(x_i | x_<i)`` — a plain
per-token NLL — and called it spilled energy. It is not (they differ by the
cross-step log-partition drift). The real ΔE is now the default; the old baseline
is still available, honestly named, via ``--score nll``. Sign and indices are
validated against the authors' reference implementation by
``scripts/validate_spilled_energy.py``.

**Attribution (Bug 2).** GPT-2 scores BPE tokens; our detector scores characters.
The old code spread each token's score UNIFORMLY over its characters, which smears
localization (one corrupt char inflates every char of its BPE token). Default is now
``--char-attrib boundary`` (score lands on the token's last character). The sweep
ALSO reports metrics at native **BPE-token granularity** (``auroc_*_bpe``), which
needs no attribution rule at all and is the honest cross-check of the per-char
number.

Run on the SAME test sequences (same --fit-seqs offset, --n, --split) and the SAME
corruption ladder + seeds as ``scripts/ood_denoiser_nll.py`` so the sequence- and
per-token AUROC columns are a controlled apples-to-apples baseline for the
DirichletFM-NLL detector. (SE needs no fit set; --fit-seqs only aligns the eval
slice with the NLL sweep.)

Usage:
    uv run python scripts/ood_gpt2_spilled_energy.py \
        --ckpt runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM/epoch_final.pt \
        --split test --rates 0.05,0.1,0.15,0.2,0.25,0.3,0.5,0.7,1.0 \
        --out ood_out/gpt2_se/gpt2_spilled_energy_sweep.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace as _replace
from pathlib import Path

import numpy as np
import torch


def _bootstrap() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))


_bootstrap()

from scripts.bench_sflm_ebm import (  # noqa: E402
    _load_gpt2, _gpt2_bpe_scores, _bpe_to_char,
)
from scripts.ood_variance_perpos import _auroc  # noqa: E402
from scripts.eval_full import _config_from_payload  # noqa: E402
from aitchinson_flow.training import build_training_datamodule  # noqa: E402
from aitchinson_flow.data.corruption import (  # noqa: E402
    corrupt_token_ids,
    partially_shuffle_token_ids,
    corrupt_false_info,
    build_vocab_by_len,
)
from scripts._bench_common import (  # noqa: E402
    heal_style_examples_bpe, det_metrics, word_metrics_from_segments,
)

_ALPH = "abcdefghijklmnopqrstuvwxyz "  # text8 K=27 decode (for falseinfo vocab)


def _corrupt(tok, scheme, rate, K, seed, by_len=None):
    if scheme == "replace":
        return corrupt_token_ids(tok, vocab_size=K, corrupt_rate=rate, seed=seed)
    if scheme == "shuffle":
        return partially_shuffle_token_ids(tok, shuffle_rate=rate, seed=seed)
    if scheme in ("falseinfo", "wordswap"):
        return corrupt_false_info(tok, rate, by_len, seed=seed)
    out = corrupt_token_ids(tok, vocab_size=K, corrupt_rate=rate, seed=seed)
    return partially_shuffle_token_ids(out, shuffle_rate=rate, seed=seed + 1)


def _bpe_labels(spans, changed_row) -> np.ndarray:
    """1 for each BPE token whose character span overlaps a corrupted char."""
    return np.array([int(bool(changed_row[s:e].any())) for (s, e) in spans], dtype=int)


def _bpe_pool(scores, spans, changed):
    """Flatten per-BPE scores + labels across a batch.

    Returns (seq_scores (B,), tok_scores (n_tokens,), tok_labels (n_tokens,)).
    This is the ATTRIBUTION-FREE view: GPT-2 natively scores BPE tokens, so
    evaluating at that granularity needs no BPE->char heuristic at all. A BPE
    token counts as corrupt if ANY character it spans was corrupted.
    """
    seq, tk, lb = [], [], []
    for j in range(len(scores)):
        s = scores[j].numpy()
        seq.append(float(s.mean()) if len(s) else float("nan"))
        tk.append(s)
        lb.append(_bpe_labels(spans[j], changed[j].numpy()))
    return (np.array(seq), np.concatenate(tk) if tk else np.array([]),
            np.concatenate(lb) if lb else np.array([]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True,
                    help="DirichletFM ckpt — used ONLY for its cfg/datamodule so the "
                         "test sequences match the NLL sweep (GPT-2 does the scoring)")
    ap.add_argument("--ref-lm", default="gpt2", help="HF causal LM (gpt2, distilgpt2, ...)")
    ap.add_argument("--out", default="bench_ood_final/gpt2_se/gpt2_spilled_energy_sweep.json")
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--fit-seqs", type=int, default=512,
                    help="eval-slice offset; must match the NLL sweep (SE uses no fit)")
    ap.add_argument("--n", type=int, default=256, help="# eval sequences per split")
    ap.add_argument("--schemes", type=str, default="replace,shuffle,both,falseinfo")
    ap.add_argument("--rates", type=str, default="0.05,0.1,0.15,0.2,0.25,0.3,0.5,0.7,1.0")  # the bench ladder; the paper reads 0.15/0.30
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--score", choices=["spilled", "nll"], default="spilled",
                    help="'spilled' = the paper's CROSS-STEP energy discrepancy "
                         "(Minut et al. 2026, Def 4.1); 'nll' = the same-step "
                         "-log p(x_i|x_<i) this repo previously mislabelled as "
                         "spilled energy (kept as an honest, separate baseline)")
    ap.add_argument("--char-attrib", choices=["boundary", "uniform"], default="boundary",
                    help="BPE->char attribution: 'boundary' puts the token's score on "
                         "its last char (no smearing); 'uniform' is the old behaviour "
                         "that spread it over every char and blurred localization")
    ap.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = _config_from_payload(payload)
    cfg.training = _replace(cfg.training, device=device)
    K = cfg.text8_dataset.K
    dm, _ = build_training_datamodule(cfg)
    loaders = {"train": dm.train_dataloader, "val": dm.val_dataloader,
               "test": dm.test_dataloader}
    vl = loaders[args.split]() or dm.train_dataloader()
    seqs = []
    for b in vl:
        seqs.append(b["token_ids"].long())
        if sum(s.shape[0] for s in seqs) >= args.fit_seqs + args.n:
            break
    seqs = torch.cat(seqs)
    fit_tok = seqs[: args.fit_seqs]
    pos_tok = seqs[args.fit_seqs : args.fit_seqs + args.n]
    print(f"[gpt2-se] ref_lm={args.ref_lm} split={args.split} K={K} "
          f"eval={pos_tok.shape[0]} (slice [{args.fit_seqs}:{args.fit_seqs+args.n}])")

    schemes = [s for s in args.schemes.split(",") if s.strip()]
    # false-info replacement vocab (built once from the pre-eval fit windows), if needed
    by_len = None
    if any(s in ("falseinfo", "wordswap") for s in schemes):
        fit_txt = " ".join(
            "".join(_ALPH[int(i)] if int(i) < len(_ALPH) else "?" for i in row)
            for row in fit_tok)
        by_len = build_vocab_by_len(fit_txt)
        print(f"[gpt2-se] false-info vocab: lengths {min(by_len)}-{max(by_len)}")

    model, tok = _load_gpt2(args.ref_lm, device)
    L = int(pos_tok.shape[1])
    print(f"[gpt2-se] score={args.score} char_attrib={args.char_attrib}")

    def bpe(t):  # -> (per-BPE scores, char spans)
        return _gpt2_bpe_scores(model, tok, t, chunk=args.chunk, score=args.score)

    def score(t):  # (B,L) per-char, for the char-granularity metrics
        sc, sp = bpe(t)
        return _bpe_to_char(sc, sp, L, attrib=args.char_attrib)

    Sp_bpe, Sp_spans = bpe(pos_tok)
    Sp = _bpe_to_char(Sp_bpe, Sp_spans, L, attrib=args.char_attrib)
    Sp_seq = Sp.mean(1).cpu().numpy()
    clean_mean = float(Sp.mean())
    # The paper's own diagnostic: the cross-step spilled energy should sit near ZERO
    # on clean, well-modelled text (it is a consistency gap that vanishes for a correct
    # model), whereas an NLL is substantially positive. A clean mean far from 0 under
    # --score spilled means we are (still) computing an NLL somewhere.
    clean_bpe_mean = float(torch.cat(Sp_bpe).mean())
    print(f"[gpt2-se] clean per-char {args.score} mean = {clean_mean:.4f}  "
          f"| clean per-BPE mean = {clean_bpe_mean:.4f}"
          f"{'   (SE: expect ~0)' if args.score == 'spilled' else '   (NLL: expect >0)'}")
    # clean-sequence + clean-token BPE scores, for the attribution-free metrics
    Sp_bpe_seq, Sp_bpe_tok, _ = _bpe_pool(Sp_bpe, Sp_spans,
                                          torch.zeros_like(pos_tok, dtype=torch.bool))
    rows = [{"scheme": None, "rate": 0.0, "n": int(pos_tok.shape[0]),
             "se_seq_mean": clean_mean, "se_bpe_mean": clean_bpe_mean,
             "n_bpe_tokens": int(len(Sp_bpe_tok))}]
    print(f"\n{'scheme':>9} {'rate':>5} {'AUROC_seq_SE':>13} {'AUROC_tok_SE':>13} "
          f"{'AUtok_BPE':>10} {'AUwrd_max':>9} {'AUwrd_men':>9} "
          f"{'tokP@5':>7} {'tokR@5':>7} {'tokF1@5':>8}")
    rates = [float(r) for r in args.rates.split(",") if r.strip()]
    Sp_tok = Sp.cpu().numpy().reshape(-1)  # clean per-token SE: threshold calibration
    for scheme in schemes:
        for r in rates:
            ot = _corrupt(pos_tok.clone(), scheme, r, K, args.seed + int(1000 * r),
                          by_len=by_len)
            So_bpe, So_spans = bpe(ot)
            So = _bpe_to_char(So_bpe, So_spans, L, attrib=args.char_attrib)
            So_seq = So.mean(1).cpu().numpy()
            lab = np.r_[np.zeros(len(Sp_seq)), np.ones(So.shape[0])]
            au_seq = _auroc(np.r_[Sp_seq, So_seq], lab)
            changed = (ot != pos_tok)
            cm = changed.cpu().numpy().reshape(-1).astype(int)
            au_tok = float("nan")
            if changed.any() and (~changed).any():
                au_tok = _auroc(So.cpu().numpy().reshape(-1), cm)
            # --- operating-point metrics (P/R/F1 at clean-calibrated thresholds) ---
            m_seq = det_metrics(Sp_seq, Sp_seq, So_seq)
            so_tok = So.cpu().numpy().reshape(-1)
            m_tok = det_metrics(Sp_tok, so_tok[cm == 0], so_tok[cm == 1])
            p5 = m_tok["prf"].get("0.05", {})

            # --- BPE-granularity (attribution-free): GPT-2's native units ----------
            # No BPE->char heuristic is involved, so this is the honest check on
            # whether the per-char number is limited by the detector or by attribution.
            ob_seq, ob_tok, ob_lab = _bpe_pool(So_bpe, So_spans, changed.cpu())
            au_seq_bpe = _auroc(np.r_[Sp_bpe_seq, ob_seq],
                                np.r_[np.zeros(len(Sp_bpe_seq)), np.ones(len(ob_seq))])
            au_tok_bpe = float("nan")
            m_seq_bpe = det_metrics(Sp_bpe_seq, Sp_bpe_seq, ob_seq)
            m_tok_bpe = {}
            if ob_lab.any() and (~ob_lab.astype(bool)).any():
                au_tok_bpe = _auroc(ob_tok, ob_lab)
                m_tok_bpe = det_metrics(Sp_bpe_tok, ob_tok[ob_lab == 0],
                                        ob_tok[ob_lab == 1])

            # --- WORD level: the COMMON unit with the char-level detectors ---------
            # Pool BPE->word directly (never via the char attribution): GPT-2 stays in
            # its native units all the way up, and words are exact for both sides since
            # a GPT-2 BPE token never crosses a whitespace boundary.
            w_max = word_metrics_from_segments(Sp_bpe, Sp_spans, pos_tok,
                                               So_bpe, So_spans, ot, changed, op="max")
            w_mean = word_metrics_from_segments(Sp_bpe, Sp_spans, pos_tok,
                                                So_bpe, So_spans, ot, changed, op="mean")

            print(f"{scheme:>9} {r:>5.2f} {au_seq:>13.4f} {au_tok:>13.4f} "
                  f"{au_tok_bpe:>10.4f} "
                  f"{w_max['word'].get('auroc', float('nan')):>9.4f} "
                  f"{w_mean['word'].get('auroc', float('nan')):>9.4f} "
                  f"{p5.get('precision', float('nan')):>7.3f} "
                  f"{p5.get('recall', float('nan')):>7.3f} "
                  f"{p5.get('f1', float('nan')):>8.3f}")
            rows.append({"scheme": scheme, "rate": r, "n": int(ot.shape[0]),
                         "auroc_seq_se": au_seq, "auroc_token_se": au_tok,
                         "prf_seq": m_seq, "prf_token": m_tok,
                         # attribution-free, native-granularity companions
                         "auroc_seq_se_bpe": au_seq_bpe,
                         "auroc_token_se_bpe": au_tok_bpe,
                         "prf_seq_bpe": m_seq_bpe, "prf_token_bpe": m_tok_bpe,
                         "n_bpe_tokens": int(len(ob_tok)),
                         # WORD level — comparable head-to-head with the char models
                         "word_max": w_max, "word_mean": w_mean,
                         "auroc_word_max": w_max["word"].get("auroc"),
                         "auroc_word_mean": w_mean["word"].get("auroc"),
                         "se_seq_mean": float(So.mean())})

    # ---- healing-style examples, in GPT-2's OWN units (BPE tokens) -------------
    # One heatmap cell = one BPE token; a red box marks a BPE token overlapping a
    # corrupted character. Rendering GPT-2 on character cells would require pushing
    # its score down onto chars, which is exactly the attribution artifact we removed.
    print("\n=== healing-style examples (BPE-token cells) ===")
    examples, flag_thr = heal_style_examples_bpe(
        bpe, pos_tok, pos_tok[:2], K=K, by_len=by_len,
        score_key="SE_t", example_fpr=0.05, max_cells=64)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "ref_lm": args.ref_lm, "split": args.split, "K": K,
        "n": int(pos_tok.shape[0]), "fit_seqs_offset": args.fit_seqs,
        "detector": ("GPT2_SpilledEnergy" if args.score == "spilled"
                     else "GPT2_NLL"),
        "detector_long": (
            "cross-step spilled energy SE_i = logsumexp(logits_i) - logits_{i-1}[x_i] "
            "under a frozen pretrained GPT-2 read as an EBM (Minut, Dewidar & Masi, "
            "ICLR 2026, arXiv:2602.18671, Def 4.1/Eq.8); zero-train; matched to the "
            "DirichletFM-NLL sweep"
            if args.score == "spilled" else
            "per-char NLL -log p_GPT2(x_i | x_<i) under a frozen pretrained GPT-2 "
            "(same-step; this is what the repo previously mislabelled as 'spilled "
            "energy'); zero-train; matched to the DirichletFM-NLL sweep"),
        "score": args.score,
        "char_attrib": args.char_attrib,
        "clean_char_mean": clean_mean,
        "clean_bpe_mean": clean_bpe_mean,
        "spilled_energy_ref": {
            "paper": "Minut, Dewidar & Masi, Spilled Energy in LLMs, ICLR 2026",
            "arxiv": "2602.18671",
            "code": "github.com/OmnAI-Lab/spilled-energy (src/spilled_energy/energy.py)",
            "formula": "SE_i = logsumexp(logits_i) - logits_{i-1}[x_i]  (= -E_marg + E_logit)",
            "validated_by": "scripts/validate_spilled_energy.py",
        },
        "flag_thr": flag_thr, "rows": rows, "examples": examples,
    }, indent=2))
    print(f"\nWrote {out_path}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
            rws = [r for r in rows if r.get("scheme")]
            schs = []
            for r in rws:
                if r["scheme"] not in schs:
                    schs.append(r["scheme"])
            for ax, key, title in [(axes[0], "auroc_seq_se", "Sequence SE AUROC"),
                                   (axes[1], "auroc_token_se", "Per-token SE AUROC")]:
                for sc in schs:
                    pts = [(r["rate"], r.get(key)) for r in rws if r["scheme"] == sc]
                    pts = [(x, y) for x, y in pts if y is not None and y == y]
                    if pts:
                        xs, ys = zip(*sorted(pts))
                        ax.plot(xs, ys, marker="o", label=sc)
                ax.axhline(0.5, ls="--", lw=0.8, color="grey")
                ax.set_ylim(0.0, 1.02)
                ax.grid(alpha=0.3)
                ax.set_title(title)
                ax.set_xlabel("corruption rate")
                ax.set_ylabel("AUROC")
            axes[0].legend(title="scheme", fontsize=9)
            fig.suptitle(f"GPT-2 Spilled Energy — OOD AUROC vs corruption ladder "
                         f"({args.ref_lm}, split={args.split}, n={pos_tok.shape[0]})")
            fig.tight_layout()
            p = out_path.with_name(out_path.stem + "_auroc.png")
            fig.savefig(p, dpi=150)
            plt.close(fig)
            print(f"Wrote {p}")
        except Exception as e:
            print(f"[plot] skipped: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
