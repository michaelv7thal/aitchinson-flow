"""Detector transfer test: does the text8-trained false-info detector survive on
REAL out-of-domain text (the insulin article)?

The false-info per-token localizer (full-dim LOGISTIC head trained on text8
false-info negatives) reaches ~0.72 held-out ON text8. But text8 test is
in-distribution; a biomedical article is out-of-domain, so its *clean* features
are already partly OOD to the model — the "both would be OOD" regime. This script
fits the detector on text8 and evaluates per-token + sequence AUROC on:
  * text8 held-out  (clean vs false-info)   — the in-distribution reference
  * insulin article (clean vs false-info)   — the out-of-domain transfer
so the drop (if any) is measured, not assumed.

    python scripts/eval_detector_transfer.py \
        --ckpt runs/.../DirichletFM/epoch_best.pt \
        --article-json heal_poc_insulin/insulin_article_extract.json \
        --out bench_ood/transfer/insulin_transfer.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def _bootstrap() -> None:
    root = Path(__file__).resolve().parent.parent
    for p in (root / "src", root):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


_bootstrap()

from scripts.ood_variance_perpos import _load_dirichletfm, _auroc  # noqa: E402
from scripts.ood_plausible_swap import train_logistic_head  # noqa: E402
from scripts.bench_sflm_ebm import _load_gpt2, _gpt2_bpe_scores  # noqa: E402
from scripts._bench_common import word_metrics, word_metrics_from_segments  # noqa: E402
from scripts.heal_protein_poc import wikifil, load_article  # noqa: E402
from aitchinson_flow.data.char_window_dataset import text_to_windows  # noqa: E402
from aitchinson_flow.data.corruption import (  # noqa: E402
    corrupt_false_info, build_vocab_by_len,
)
from datasets import load_dataset  # noqa: E402

_ALPH = "abcdefghijklmnopqrstuvwxyz "


def _decode_rows(tok):
    return " ".join("".join(_ALPH[int(i)] if int(i) < len(_ALPH) else "?"
                            for i in r) for r in tok)


def _score_auroc(score_fn, clean, corr, K, device):
    ch = (corr != clean)
    sc_c, sc_o = score_fn(clean), score_fn(corr)
    lab = np.r_[np.zeros(sc_c.shape[0]), np.ones(sc_o.shape[0])]
    seq = _auroc(np.r_[sc_c.mean(1).numpy(), sc_o.mean(1).numpy()], lab)
    tok = float("nan")
    if ch.any() and (~ch).any():
        tok = _auroc(sc_o.numpy().reshape(-1), ch.numpy().reshape(-1).astype(int))
    return {"seq_auroc": seq, "token_auroc": tok, "n_changed": int(ch.sum())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--article-json", default="heal_poc_insulin/insulin_article_extract.json")
    ap.add_argument("--out", default="bench_ood/transfer/insulin_transfer.json")
    ap.add_argument("--t-eval", type=float, default=4.5)
    ap.add_argument("--fit-seqs", type=int, default=512)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--rate", type=float, default=0.15)
    ap.add_argument("--vocab-chars", type=int, default=8_000_000)
    ap.add_argument("--ref-lm", default="gpt2",
                    help="HF causal LM run as the external baseline on the same "
                         "transfer track (GPT2_NLL, the fair localization comparator); "
                         "'' disables the baseline")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = _load_dirichletfm(args.ckpt, device)
    K, L = cfg.text8_dataset.K, cfg.text8_dataset.L

    # text8: fit windows (train the detector) + held-out eval windows (test split)
    ds = load_dataset("afmck/text8")
    fit_tok = text_to_windows(
        ds["train"][0]["text"][args.vocab_chars: args.vocab_chars + (args.fit_seqs + 4) * L],
        L)[: args.fit_seqs]
    eval_t8 = text_to_windows(ds["test"][0]["text"][: (args.n + 4) * L], L)[: args.n]
    by_len = build_vocab_by_len(ds["train"][0]["text"][: args.vocab_chars])

    # insulin article windows (out-of-domain)
    art = wikifil(load_article(args.article_json))
    ins = text_to_windows(art, L)[: args.n]
    print(f"[transfer] fit={fit_tok.shape[0]} text8_eval={eval_t8.shape[0]} "
          f"insulin_eval={ins.shape[0]} t_eval={args.t_eval}")

    # train the full-dim LOGISTIC false-info detector on text8
    fi_fit = corrupt_false_info(fit_tok.clone(), args.rate, by_len, seed=args.seed + 5)
    det = train_logistic_head(model, fit_tok, fi_fit, args.t_eval, device, seed=args.seed)

    def score(tok):
        return det(tok)

    # external baseline: the same GPT-2 NLL arm as the main bench (fair localization
    # comparator), evaluated on the identical clean/corrupt pairs of both domains
    gpt2, gpt2_tok = (None, None)
    if args.ref_lm:
        gpt2, gpt2_tok = _load_gpt2(args.ref_lm, device)

    def _gpt2_eval(clean, corr, changed):
        cv, cs = _gpt2_bpe_scores(gpt2, gpt2_tok, clean, chunk=32, score="nll")
        ov, os_ = _gpt2_bpe_scores(gpt2, gpt2_tok, corr, chunk=32, score="nll")
        wm = word_metrics_from_segments(cv, cs, clean, ov, os_, corr, changed, op="max")
        # native BPE token AUROC (attribution-free): corrupt = span overlaps a changed char
        ch = changed.numpy()
        tok_sc = np.concatenate([v.numpy() for v in ov])
        tok_lb = np.concatenate([
            np.array([int(bool(ch[j, s:e].any())) for (s, e) in os_[j]], dtype=int)
            for j in range(len(ov))])
        bpe_tok = _auroc(tok_sc, tok_lb)
        # per-char-normalised sequence score (the bench 1b convention)
        L = clean.shape[1]
        seq_sc = np.r_[[float(v.sum()) / L for v in cv], [float(v.sum()) / L for v in ov]]
        seq_lb = np.r_[np.zeros(len(cv)), np.ones(len(ov))]
        return {"word_auroc_max": wm["word"].get("auroc"),
                "word_seq_auroc": wm["seq"].get("auroc"),
                "bpe_token_auroc": bpe_tok,
                "seq_auroc_charnorm": _auroc(seq_sc, seq_lb)}

    # evaluate on both domains (clean vs false-info)
    t8_corr = corrupt_false_info(eval_t8.clone(), args.rate, by_len, seed=args.seed + 9)
    ins_corr = corrupt_false_info(ins.clone(), args.rate, by_len, seed=args.seed + 9)
    res = {}
    for name, clean, corr in [("text8_indist", eval_t8, t8_corr),
                              ("insulin_ood", ins, ins_corr)]:
        changed = corr != clean
        r = _score_auroc(score, clean, corr, K, device)
        wm = word_metrics(score(clean).numpy(), clean, score(corr).numpy(), corr,
                          changed, op="max")
        r["word_auroc_max"] = wm["word"].get("auroc")
        r["word_seq_auroc"] = wm["seq"].get("auroc")
        if gpt2 is not None:
            r["gpt2_nll"] = _gpt2_eval(clean, corr, changed)
        res[name] = r

    print(f"\n{'domain':>16} {'seq_AUROC':>10} {'token_AUROC':>12} {'word_AUROC':>11}"
          + (f" {'gpt2_word':>10} {'gpt2_bpe_tok':>13} {'gpt2_seq':>9}" if gpt2 is not None else ""))
    for name, r in res.items():
        line = (f"{name:>16} {r['seq_auroc']:>10.3f} {r['token_auroc']:>12.3f} "
                f"{r['word_auroc_max']:>11.3f}")
        if gpt2 is not None:
            g = r["gpt2_nll"]
            line += (f" {g['word_auroc_max']:>10.3f} {g['bpe_token_auroc']:>13.3f} "
                     f"{g['seq_auroc_charnorm']:>9.3f}")
        print(line)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "ckpt": args.ckpt, "detector": "logistic_falseinfo_fulldim",
        "baseline": f"{args.ref_lm}_nll" if args.ref_lm else None,
        "t_eval": args.t_eval, "rate": args.rate, "n": args.n,
        "results": res,
        "note": "text8=in-distribution reference; insulin=out-of-domain transfer; "
                "word AUROC = max-pool, the bench's fair common unit",
    }, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
