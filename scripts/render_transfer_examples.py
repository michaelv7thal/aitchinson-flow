"""Render the tab:ood-transfer windows, clean against their false-information draw.

The transfer table reports AUROC on two domains but never shows the reader what a
corrupted window looks like on either. This reproduces the EXACT windows and the
EXACT corruption draw behind that table (same split offsets, same rate, same seed as
scripts/eval_detector_transfer_all.py) and dumps, per domain:

  * the clean and corrupted character strings of the quoted windows,
  * every swap as (start, end, original, replacement),
  * for each replacement, its whole-word frequency in the text8 TRAIN split.

That last column is the reason this script writes an artifact rather than just
printing. The corruption is described as replacing a word by "a different real word
of the same length", and the replacements are indeed drawn from the corpus
vocabulary, but the corpus vocabulary of a 90M-character character-level dump is not
a dictionary: it contains acronyms, transliterations and one-off strings. Recording
the frequency of every replacement lets the paper say how word-like the draws
actually are instead of asserting it.

    python scripts/render_transfer_examples.py \
        --out bench_ood_final/transfer/transfer_examples.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def _bootstrap() -> None:
    root = Path(__file__).resolve().parent.parent
    for p in (root / "src", root):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


_bootstrap()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts.heal_protein_poc import wikifil, load_article  # noqa: E402
from scripts._bench_common import (  # noqa: E402
    git_sha, now_iso, word_spans, pool_to_words, word_labels,
)
from scripts.ood_variance_perpos import _load_dirichletfm  # noqa: E402
from scripts.ood_plausible_swap import train_logistic_head  # noqa: E402
from aitchinson_flow.data.char_window_dataset import text_to_windows  # noqa: E402
from aitchinson_flow.data.corruption import (  # noqa: E402
    corrupt_false_info, build_vocab_by_len,
)
from datasets import load_dataset  # noqa: E402

_ALPH = "abcdefghijklmnopqrstuvwxyz "


def _dec(ids) -> str:
    return "".join(_ALPH[int(i)] for i in ids)


def _swaps(clean: str, corr: str) -> list[dict]:
    """Maximal runs of differing characters, with the word each run sits in.

    The per-character label is `corrupt != clean`, so a swapped word shows up as one
    or more runs, split wherever the replacement happens to share a character with
    the original. The enclosing word is recovered by walking out to the spaces.
    """
    out, i, L = [], 0, len(clean)
    while i < L:
        if clean[i] != corr[i]:
            j = i
            while j < L and clean[j] != corr[j]:
                j += 1
            a = clean.rfind(" ", 0, i) + 1
            b = clean.find(" ", j)
            b = L if b < 0 else b
            out.append({"start": i, "end": j,
                        "orig_run": clean[i:j], "repl_run": corr[i:j],
                        "orig_word": clean[a:b], "repl_word": corr[a:b]})
            i = j
        else:
            i += 1
    return out


def _render(clean: str, corr: str, width: int) -> list[str]:
    lines = []
    for s in range(0, len(clean), width):
        seg = slice(s, min(s + width, len(clean)))
        lines += [f"clean   |{clean[seg]}|",
                  f"corrupt |{corr[seg]}|",
                  "changed |" + "".join("^" if a != b else " "
                                        for a, b in zip(clean[seg], corr[seg])) + "|"]
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--article-json",
                    default="bench_ood_final/transfer/semaglutide_article_extract.json")
    ap.add_argument("--out", default="bench_ood_final/transfer/transfer_examples.json")
    ap.add_argument("--windows", default="0,1", help="window indices to render")
    ap.add_argument("--excerpt", type=int, default=126,
                    help="characters of each window quoted in the paper excerpt")
    ap.add_argument("--width", type=int, default=126)
    # these five must match eval_detector_transfer_all.py exactly
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--rate", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vocab-chars", type=int, default=8_000_000)
    ap.add_argument("--L", type=int, default=256)
    ap.add_argument("--ckpt",
                    default="runs/sflm_bench_a100_20g_L256_d1280L14_full/"
                            "DirichletFM_converge/epoch_final.pt")
    ap.add_argument("--fit-seqs", type=int, default=512)
    ap.add_argument("--t-eval", type=float, default=4.5)
    ap.add_argument("--example-fpr", type=float, default=0.05,
                    help="clean-calibrated false-positive rate for the flag threshold, "
                         "the deployed operating point of the detector tables")
    ap.add_argument("--no-detect", action="store_true",
                    help="skip the detector pass and only render the corruption")
    args = ap.parse_args()

    L, N = args.L, args.n
    ds = load_dataset("afmck/text8")
    train_txt = ds["train"][0]["text"]
    by_len = build_vocab_by_len(train_txt[: args.vocab_chars])
    train_counts = Counter(train_txt.split())

    eval_t8 = text_to_windows(ds["test"][0]["text"][: (N + 4) * L], L)[:N]
    art = wikifil(load_article(args.article_json))
    ins = text_to_windows(art, L)[:N]
    corr = {"text8_indist": corrupt_false_info(eval_t8.clone(), args.rate, by_len,
                                               seed=args.seed + 9),
            "semaglutide_ood": corrupt_false_info(ins.clone(), args.rate, by_len,
                                                  seed=args.seed + 9)}
    clean = {"text8_indist": eval_t8, "semaglutide_ood": ins}

    # ---- the detector: LOG_fi, refit exactly as the transfer track fits it ----
    flags = {}
    detector_meta = None
    if not args.no_detect:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(args.seed)
        model, cfg = _load_dirichletfm(args.ckpt, device)
        fit_tok = text_to_windows(
            train_txt[args.vocab_chars: args.vocab_chars + (args.fit_seqs + 4) * L],
            L)[: args.fit_seqs]
        fi_fit = corrupt_false_info(fit_tok.clone(), args.rate, by_len, seed=args.seed + 5)
        score = train_logistic_head(model, fit_tok, fi_fit, args.t_eval, device,
                                    seed=args.seed)
        # One threshold for BOTH domains, calibrated on clean text8 words alone: this is
        # the deployed setting, a detector fitted once on text8 and carried across.
        cw_t8 = pool_to_words(score(clean["text8_indist"]).numpy(),
                              word_spans(clean["text8_indist"]), "max")
        flat = np.concatenate(cw_t8)
        thr = float(np.quantile(flat, 1.0 - args.example_fpr))
        detector_meta = {"detector": "LOG_fi", "t_eval": args.t_eval,
                         "fit_seqs": args.fit_seqs, "example_fpr": args.example_fpr,
                         "threshold": thr, "pooling": "word max",
                         "calibrated_on": "clean text8 test words, both domains share it"}
        print(f"\n[detector] LOG_fi @ t={args.t_eval}, word-max threshold {thr:.4f} "
              f"at {args.example_fpr:.0%} FPR on clean text8 words")
        for dom in clean:
            sp_c = word_spans(clean[dom])
            sp_o = word_spans(corr[dom])
            wc = pool_to_words(score(clean[dom]).numpy(), sp_c, "max")
            wo = pool_to_words(score(corr[dom]).numpy(), sp_o, "max")
            lab = word_labels((corr[dom] != clean[dom]), sp_o)
            flags[dom] = {"spans_clean": sp_c, "spans_corr": sp_o,
                          "word_clean": wc, "word_corr": wo, "word_label": lab,
                          "thr": thr}
            tp = sum(int(((w > thr) & (l > 0)).sum()) for w, l in zip(wo, lab))
            fp = sum(int(((w > thr) & (l == 0)).sum()) for w, l in zip(wo, lab))
            fn = sum(int(((w <= thr) & (l > 0)).sum()) for w, l in zip(wo, lab))
            print(f"  {dom:>16}  flagged-correct {tp}  false-alarm {fp}  missed {fn}"
                  f"  (precision {tp/max(1,tp+fp):.3f}, recall {tp/max(1,tp+fn):.3f})")

    idxs = [int(x) for x in args.windows.split(",") if x.strip()]
    domains: dict = {}
    for dom in clean:
        wins = []
        for i in idxs:
            c, o = _dec(clean[dom][i]), _dec(corr[dom][i])
            sw = _swaps(c, o)
            for s in sw:
                s["repl_word_train_count"] = int(train_counts.get(s["repl_word"], 0))
                s["orig_word_train_count"] = int(train_counts.get(s["orig_word"], 0))
            words = None
            if dom in flags:
                f = flags[dom]
                sp, sc, lb = f["spans_corr"][i], f["word_corr"][i], f["word_label"][i]
                # the SAME word scored in the clean window: a false alarm next to a
                # swap can be the swap's doing rather than the word's own, and only
                # the clean-window score separates the two
                scl = f["word_clean"][i]
                words = [{"word": o[a:b], "clean_word": c[a:b], "start": int(a),
                          "end": int(b), "score": float(v),
                          "clean_score": (float(scl[k]) if k < len(scl) else None),
                          "flagged": bool(v > f["thr"]), "corrupt": bool(l > 0),
                          "flagged_when_clean": (bool(scl[k] > f["thr"])
                                                 if k < len(scl) else None)}
                         for k, ((a, b), v, l) in enumerate(zip(sp, sc, lb))]
            wins.append({"index": i, "clean": c, "corrupt": o, "swaps": sw,
                         "words": words,
                         "excerpt_clean": c[: args.excerpt],
                         "excerpt_corrupt": o[: args.excerpt],
                         "render": _render(c, o, args.width)})
            print(f"\n=== {dom} window {i} ===")
            print("\n".join("  " + ln for ln in wins[-1]["render"]))
            if words:
                mark = [" "] * L
                for w in words:
                    if w["flagged"]:
                        for k in range(w["start"], w["end"]):
                            mark[k] = "#" if w["corrupt"] else "!"
                for s0 in range(0, L, args.width):
                    seg = slice(s0, min(s0 + args.width, L))
                    print("  flagged |" + "".join(mark[seg]) + "|")
                print("            (# = flagged and corrupt, ! = false alarm)")
            print("  swaps: " + ", ".join(
                f"{s['orig_word']}->{s['repl_word']} ({s['repl_word_train_count']}x)"
                for s in sw))
        domains[dom] = wins

    # how word-like are the replacements, over ALL windows of both domains?
    stats = {}
    for dom in clean:
        reps = []
        for i in range(N):
            c, o = _dec(clean[dom][i]), _dec(corr[dom][i])
            reps += [s["repl_word"] for s in _swaps(c, o)]
        reps = [r for r in reps if r]
        cnts = [train_counts.get(r, 0) for r in reps]
        n = max(1, len(cnts))
        stats[dom] = {
            "n_replacements": len(cnts),
            "median_train_count": sorted(cnts)[len(cnts) // 2] if cnts else None,
            "frac_lt_10": sum(c < 10 for c in cnts) / n,
            "frac_lt_100": sum(c < 100 for c in cnts) / n,
            "frac_ge_1000": sum(c >= 1000 for c in cnts) / n,
            "rarest_examples": [r for r, c in sorted(zip(reps, cnts),
                                                     key=lambda t: t[1])[:25]],
        }
        s = stats[dom]
        print(f"\n[{dom}] {s['n_replacements']} replacements  "
              f"median train count {s['median_train_count']}  "
              f"<10x {s['frac_lt_10']:.1%}  <100x {s['frac_lt_100']:.1%}  "
              f">=1000x {s['frac_ge_1000']:.1%}")
        print("   rarest:", ", ".join(s["rarest_examples"][:15]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "git_sha": git_sha(), "written": now_iso(), "article": args.article_json,
        "provenance": ("windows and corruption draw identical to "
                       "scripts/eval_detector_transfer_all.py: text8 test split "
                       f"first {N} windows of L={L}, article windows likewise, "
                       f"corrupt_false_info(rate={args.rate}, seed={args.seed + 9}), "
                       f"replacement vocabulary from the first {args.vocab_chars} "
                       "characters of the train split"),
        "rate": args.rate, "seed": args.seed, "n": N, "L": L,
        "detector": detector_meta,
        "windows": domains,
        "replacement_word_likeness": stats,
        "note": "changed characters are corrupt != clean, so a swapped word can show "
                "as several runs where replacement and original share a character; "
                "orig_word/repl_word give the enclosing whitespace-delimited word.",
    }, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
