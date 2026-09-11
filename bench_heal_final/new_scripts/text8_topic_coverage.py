"""How much of the transfer article's TOPIC is already in text8?

tab:ood-transfer calls the semaglutide article "out of domain by construction", and
the supporting fact in the paper is narrow and verifiable: the word *semaglutide*
never occurs in the corpus. That says the ENTITY is absent. It does not say the
topic is, and the two are different claims: text8 is itself Wikipedia, so
encyclopedic prose about drugs, peptide hormones and diabetes may well be in there.

This script measures both, so the paper can say which one it is:

  1. whole-word counts in each text8 split for the drug itself, its class, the
     underlying biology, and the conditions it treats;
  2. the article's own vocabulary against the text8 TRAIN vocabulary: what share of
     its word TYPES and word TOKENS the corpus has never seen, with a text8 test
     sample scored the same way as the in-distribution baseline;
  3. the article's most frequent content words, each with its text8 train count, so
     the unseen part of the vocabulary can be read rather than assumed.

    python scripts/text8_topic_coverage.py \
        --out bench_ood_final/transfer/text8_topic_coverage.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


def _bootstrap() -> None:
    root = Path(__file__).resolve().parent.parent
    for p in (root / "src", root):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


_bootstrap()

from scripts.heal_protein_poc import wikifil, load_article  # noqa: E402
from scripts._bench_common import git_sha, now_iso  # noqa: E402
from datasets import load_dataset  # noqa: E402

# text8 is lowercase a-z plus space: digits are spelled out by wikifil, so "GLP-1"
# enters the corpus as "glp one" and "glp" is a searchable whole word.
TERMS = {
    "the drug and its brands": [
        "semaglutide", "ozempic", "wegovy", "rybelsus",
    ],
    "the same drug class": [
        "liraglutide", "exenatide", "byetta", "victoza", "dulaglutide",
        "lixisenatide", "albiglutide", "tirzepatide", "incretin", "glp",
        "agonist", "agonists",
    ],
    "underlying biology": [
        "glucagon", "insulin", "peptide", "peptides", "hormone", "hormones",
        "pancreas", "pancreatic", "islet", "islets", "receptor", "receptors",
        "glucose", "metabolism", "metabolic", "endocrine", "subcutaneous",
    ],
    "the conditions it treats": [
        "diabetes", "diabetic", "obesity", "obese", "overweight",
        "hypoglycemia", "hyperglycaemia", "hyperglycemia",
    ],
    "pharmacology and trials": [
        "pharmacokinetics", "placebo", "clinical", "trials", "dose", "dosage",
        "nausea", "efficacy", "approval", "nordisk",
    ],
}

# multi-word forms, counted with the same whole-word boundaries. The target of the
# drug is only ever written as a phrase, so a single-token search would miss it and a
# bare "glp" would find the wrong thing (in text8 it is the Guyana Labor Party and
# Good Laboratory Practice, never the receptor).
PHRASES = {
    "the molecular target": ["glucagon like", "glp one", "glucagon like peptide"],
    "the manufacturer": ["novo nordisk"],
    "the disease": ["type two diabetes"],
}

# words that carry no topic (counting them would drown the signal)
_STOP = set("""a an the and or but if of to in on at by for with from as is are was
were be been being it its this that these those he she they we you i his her their
our your not no which who whom what when where how than then so such can may also
one two three four five six seven eight nine ten first second new other some any
all more most much many into over under between during about after before while
have has had do does did will would should could there here own same other""".split())


def _counts(words: Counter, terms: list[str]) -> dict:
    return {t: int(words.get(t, 0)) for t in terms}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--article-json",
                    default="bench_ood_final/transfer/semaglutide_article_extract.json")
    ap.add_argument("--out",
                    default="bench_ood_final/transfer/text8_topic_coverage.json")
    ap.add_argument("--top-content", type=int, default=40,
                    help="# most frequent article content words to report")
    ap.add_argument("--context", type=int, default=3,
                    help="# corpus snippets to quote per probe term")
    ap.add_argument("--context-terms", default="glucagon,incretin,glp,liraglutide",
                    help="terms to quote train-split context for")
    ap.add_argument("--context-chars", type=int, default=160)
    args = ap.parse_args()

    ds = load_dataset("afmck/text8")
    texts = {s: ds[s][0]["text"] for s in ("train", "validation", "test")}
    words = {s: Counter(t.split()) for s, t in texts.items()}
    for s in words:
        print(f"[text8] {s:<11} {len(texts[s]):>10,} chars  "
              f"{sum(words[s].values()):>9,} words  {len(words[s]):>7,} types")

    # ---- 1. topic terms per split -------------------------------------------
    by_group = {}
    for group, terms in TERMS.items():
        by_group[group] = {s: _counts(words[s], terms) for s in words}
        tr = by_group[group]["train"]
        print(f"\n[{group}]  (train counts)")
        for t in terms:
            print(f"   {t:<18} {tr[t]:>7,}")

    # ---- 1b. multi-word forms, same boundaries, all splits -------------------
    phrase_counts = {}
    for group, phrases in PHRASES.items():
        phrase_counts[group] = {
            p: {s: len(re.findall(rf"(?<![a-z]){re.escape(p)}(?![a-z])", texts[s]))
                for s in texts}
            for p in phrases
        }
        print(f"\n[{group}]  (train / validation / test)")
        for p, c in phrase_counts[group].items():
            print(f"   {p:<24} {c['train']:>6} {c['validation']:>4} {c['test']:>4}")

    # ---- 2. the article's vocabulary against text8 train ---------------------
    art = wikifil(load_article(args.article_json))
    art_words = art.split()
    art_types = Counter(art_words)
    train_vocab = words["train"]
    unseen_types = sorted(w for w in art_types if w not in train_vocab)
    unseen_tokens = sum(art_types[w] for w in unseen_types)
    art_stats = {
        "chars": len(art), "tokens": len(art_words), "types": len(art_types),
        "unseen_types": len(unseen_types),
        "unseen_type_rate": len(unseen_types) / max(1, len(art_types)),
        "unseen_tokens": unseen_tokens,
        "unseen_token_rate": unseen_tokens / max(1, len(art_words)),
        "unseen_type_list": unseen_types[:200],
    }

    # in-distribution baseline: an equal-length slice of the text8 TEST split, scored
    # the same way, so the article's rate has something to be large or small against
    base_text = texts["test"][: len(art)]
    base_words = base_text.split()
    base_types = Counter(base_words)
    b_unseen_types = [w for w in base_types if w not in train_vocab]
    b_unseen_tokens = sum(base_types[w] for w in b_unseen_types)
    base_stats = {
        "chars": len(base_text), "tokens": len(base_words), "types": len(base_types),
        "unseen_types": len(b_unseen_types),
        "unseen_type_rate": len(b_unseen_types) / max(1, len(base_types)),
        "unseen_tokens": b_unseen_tokens,
        "unseen_token_rate": b_unseen_tokens / max(1, len(base_words)),
    }
    print(f"\n[vocabulary vs text8 train]"
          f"\n   article    {art_stats['types']:>6,} types  "
          f"{art_stats['unseen_types']:>5,} unseen ({art_stats['unseen_type_rate']:.1%})  "
          f"tokens unseen {art_stats['unseen_token_rate']:.2%}"
          f"\n   text8 test {base_stats['types']:>6,} types  "
          f"{base_stats['unseen_types']:>5,} unseen ({base_stats['unseen_type_rate']:.1%})  "
          f"tokens unseen {base_stats['unseen_token_rate']:.2%}")

    # ---- 3. the article's top content words, with their train counts ---------
    content = [(w, c) for w, c in art_types.most_common() if w not in _STOP and len(w) > 2]
    top = [{"word": w, "article": c, "text8_train": int(train_vocab.get(w, 0))}
           for w, c in content[: args.top_content]]
    print(f"\n[article's top {args.top_content} content words: text8 train count]")
    for r in top:
        mark = "  UNSEEN" if r["text8_train"] == 0 else ""
        print(f"   {r['word']:<18} article {r['article']:>4}   train {r['text8_train']:>8,}{mark}")

    # ---- 4. context: what text8 actually says where the topic does appear ----
    ctx = {}
    for term in [t for t in args.context_terms.split(",") if t.strip()]:
        hits = []
        for m in re.finditer(rf"(?<![a-z]){re.escape(term)}(?![a-z])", texts["train"]):
            a = max(0, m.start() - args.context_chars // 2)
            hits.append(texts["train"][a: a + args.context_chars].strip())
            if len(hits) >= args.context:
                break
        ctx[term] = hits
        print(f"\n[context] {term} ({train_vocab.get(term, 0):,} in train)")
        for h in hits:
            print(f"   ...{h}...")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "source": "HF afmck/text8", "git_sha": git_sha(), "written": now_iso(),
        "article": args.article_json,
        "split_chars": {s: len(t) for s, t in texts.items()},
        "split_words": {s: int(sum(c.values())) for s, c in words.items()},
        "split_types": {s: len(c) for s, c in words.items()},
        "term_counts": by_group,
        "phrase_counts": phrase_counts,
        "article_vs_train": art_stats,
        "text8_test_baseline_vs_train": base_stats,
        "article_top_content_words": top,
        "train_context": ctx,
        "note": "whole-word counts over the space-separated text8 splits (wikifil "
                "spells digits out, so 'GLP-1' enters as 'glp one'). unseen_* are "
                "measured against the TRAIN split vocabulary, the corpus the "
                "backbone was fitted on.",
    }, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
