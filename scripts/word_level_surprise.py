"""Denoiser surprise at a swapped WORD, not just at the changed characters.

``ood_plausible_swap.py`` reports one "how plausible is plausible" statistic, the
mean denoiser NLL over the characters that actually changed
(``surprise.{clean,random_swapped,plausible_swapped}_mean_nll`` in
``plausible_swap*.json``). The paper's comparison unit, however, is the WORD:
each word's score is the max (default) or the mean of the character scores inside
it (methods, "Comparison unit"). This script reads the same saved windows and
reports the same statistic at that unit, so the prose can quote a word-level
number without changing the protocol.

Everything is deterministic: the windows come from the ``*.tokens.pt`` dump of a
finished run and the NLL localizer is a single no-grad forward pass at
``--t-nll`` on the Dirichlet mean, no sampling. The script first re-derives the
three published character-level means and refuses to write its output unless
they reproduce, which is what makes the word-level numbers provably the same
run's.

Word spans are the fully-contained ``[a-z]+`` spans of the clean window (edge
words dropped), the same rule ``corrupt_false_info`` and ``plausible_swap`` use
to pick their slots. A swapped word is a span holding at least one changed
character.

Usage:
    python scripts/word_level_surprise.py \
        --tokens bench_ood_final/plausible/plausible_swap_var.tokens.pt \
        --ckpt runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM_converge/epoch_final.pt \
        --out bench_ood_final/plausible/word_level_surprise.json
"""

from __future__ import annotations

import argparse
import json
import re
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

from scripts.heal_dirichlet import make_nll_localizer  # noqa: E402
from scripts.ood_variance_perpos import _load_dirichletfm  # noqa: E402

_ALPH = "abcdefghijklmnopqrstuvwxyz "  # text8 K=27

# The three character-level means published in plausible_swap*.json, quoted in
# chapters/results.tex sec:ood-plausible. Guard rails, not inputs.
_PUBLISHED = {
    "clean_mean_nll": 0.12417630851268768,
    "random_swapped_mean_nll": 0.5349077582359314,
    "plausible_swapped_mean_nll": 0.0032652695663273335,
}


def _decode(ids) -> str:
    return "".join(_ALPH[int(i)] if int(i) < len(_ALPH) else "?" for i in ids)


def _word_spans(row: torch.Tensor) -> list[tuple[int, int]]:
    """Fully-contained [a,b) letter-word spans of a window (edge words dropped)."""
    s = _decode(row)
    spans = [(m.start(), m.end()) for m in re.finditer(r"[a-z]+", s)]
    return [(a, b) for (a, b) in spans if a > 0 and b < len(s)]


def _pooled(score: torch.Tensor, spans: list[tuple[int, int]]) -> dict[str, list[float]]:
    """Per-word max and mean of a character-level score over the given spans."""
    return {
        "max": [float(score[a:b].max()) for a, b in spans],
        "mean": [float(score[a:b].mean()) for a, b in spans],
    }


def _stats(vals: list[float]) -> dict[str, float]:
    t = torch.tensor(vals, dtype=torch.float64)
    return {"n": int(t.numel()), "mean": float(t.mean()), "median": float(t.median())}


def _char_stats(score: torch.Tensor, keep: torch.Tensor) -> dict[str, float]:
    """Mean/median of a character-level score over the characters ``keep`` marks."""
    v = score[keep].double()
    if not v.numel():
        return {"n": 0, "mean": float("nan"), "median": float("nan")}
    return {"n": int(v.numel()), "mean": float(v.mean()), "median": float(v.median())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="bench_ood_final/plausible/plausible_swap_var.tokens.pt")
    ap.add_argument("--ckpt", default="runs/sflm_bench_a100_20g_L256_d1280L14_full/"
                                      "DirichletFM_converge/epoch_final.pt")
    ap.add_argument("--t-nll", type=float, default=3.0, help="must match the run's t_nll")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="max allowed drift against the published character-level means")
    ap.add_argument("--out", default="bench_ood_final/plausible/word_level_surprise.json")
    args = ap.parse_args()

    dump = torch.load(args.tokens, map_location="cpu", weights_only=False)
    clean, rand, plaus = dump["eval_tok"], dump["rand_tok"], dump["plaus_tok"]
    print(f"[wls] {args.tokens}: n={clean.shape[0]} L={clean.shape[1]} "
          f"rate={dump['rate']} seed={dump['seed']} swap_mode={dump['swap_mode']}")

    model, _ = _load_dirichletfm(args.ckpt, args.device)
    model.eval()
    nll_score, _ = make_nll_localizer(model, args.t_nll, args.device, chunk=args.chunk)
    nll = {"clean": nll_score(clean), "random": nll_score(rand), "plausible": nll_score(plaus)}

    # ---- character level: reproduce the published statistic ------------------
    ch = {"random": rand != clean, "plausible": plaus != clean}
    char_level = {
        "clean_mean_nll": float(nll["clean"].mean()),
        "random_swapped_mean_nll": float(nll["random"][ch["random"]].mean()),
        "plausible_swapped_mean_nll": float(nll["plausible"][ch["plausible"]].mean()),
    }
    drift = {k: abs(v - _PUBLISHED[k]) for k, v in char_level.items()}
    print("[wls] character level (published in parentheses):")
    for k, v in char_level.items():
        print(f"        {k:34s} {v:.6f}  ({_PUBLISHED[k]:.6f})  drift {drift[k]:.2e}")
    if max(drift.values()) > args.tol:
        raise SystemExit(
            f"[wls] REFUSING to write: character-level means drifted by "
            f"{max(drift.values()):.2e} > tol {args.tol:.0e}. The word-level numbers "
            f"would not be this run's. Check --ckpt, --t-nll and --tokens."
        )

    # ---- word level ----------------------------------------------------------
    # Spans come from the clean window; the corruptions are same-length words, so
    # the corrupted windows must carry the identical span structure. Assert it.
    all_spans: list[list[tuple[int, int]]] = []
    for i in range(clean.shape[0]):
        spans = _word_spans(clean[i])
        for arm, tok in (("random", rand), ("plausible", plaus)):
            if _word_spans(tok[i]) != spans:
                raise SystemExit(f"[wls] window {i}: {arm} changed the word spans")
        all_spans.append(spans)

    out: dict[str, dict] = {"clean_all_words": {}, "random": {}, "plausible": {}}

    # Clean baseline over every eligible word: the word-level analogue of the
    # published "mean over clean text".
    clean_all = {"max": [], "mean": []}
    for i, spans in enumerate(all_spans):
        p = _pooled(nll["clean"][i], spans)
        clean_all["max"] += p["max"]
        clean_all["mean"] += p["mean"]
    out["clean_all_words"] = {k: _stats(v) for k, v in clean_all.items()}

    # Per arm: the swapped words, and the words they replaced (same spans, clean
    # window). The two arms do NOT share slots, so each carries its own baseline.
    for arm, tok in (("random", rand), ("plausible", plaus)):
        planted = {"max": [], "mean": []}
        replaced = {"max": [], "mean": []}
        words_planted: list[str] = []
        for i, spans in enumerate(all_spans):
            hit = [(a, b) for a, b in spans if bool(ch[arm][i, a:b].any())]
            pp, pr = _pooled(nll[arm][i], hit), _pooled(nll["clean"][i], hit)
            for k in ("max", "mean"):
                planted[k] += pp[k]
                replaced[k] += pr[k]
            words_planted += [_decode(tok[i][a:b]) for a, b in hit]
        out[arm] = {
            "swapped_word": {k: _stats(v) for k, v in planted.items()},
            "word_it_replaced": {k: _stats(v) for k, v in replaced.items()},
            "n_words": len(words_planted),
            "chars_changed": int(ch[arm].sum()),
        }

    # ---- character level, row by row ----------------------------------------
    # The same five populations at the character unit, so the two units can be
    # printed side by side. "all" is every character inside those words, "changed"
    # only the characters that differ from clean (the published statistic for the
    # two planted rows, and what those characters scored before the swap for the
    # two rows beneath them).
    eligible = torch.zeros_like(clean, dtype=torch.bool)
    hit_words = {arm: torch.zeros_like(clean, dtype=torch.bool) for arm in ("random", "plausible")}
    for i, spans in enumerate(all_spans):
        for a, b in spans:
            eligible[i, a:b] = True
            for arm in ("random", "plausible"):
                if bool(ch[arm][i, a:b].any()):
                    hit_words[arm][i, a:b] = True

    chars: dict[str, dict] = {
        "clean_all_chars": _char_stats(nll["clean"], torch.ones_like(clean, dtype=torch.bool)),
        "clean_in_eligible_words": _char_stats(nll["clean"], eligible),
    }
    for arm in ("random", "plausible"):
        chars[arm] = {
            "swapped_word": {"all": _char_stats(nll[arm], hit_words[arm]),
                             "changed": _char_stats(nll[arm], ch[arm])},
            "word_it_replaced": {"all": _char_stats(nll["clean"], hit_words[arm]),
                                 "changed": _char_stats(nll["clean"], ch[arm])},
        }

    payload = {
        "source_tokens": args.tokens,
        "ckpt": args.ckpt,
        "t_nll": args.t_nll,
        "rate": dump["rate"],
        "seed": dump["seed"],
        "swap_mode": dump["swap_mode"],
        "n": int(clean.shape[0]),
        "L": int(clean.shape[1]),
        "pooling": "each word's score is the max or the mean of the character NLLs "
                   "inside it (methods, Comparison unit)",
        "char_level": char_level,
        "char_level_rows": chars,
        "char_level_published": _PUBLISHED,
        "char_level_max_drift": max(drift.values()),
        "word_level": out,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")

    print("\n[wls] character level, mean over characters (all / changed):")
    print(f"        clean, every character          "
          f"{chars['clean_all_chars']['mean']:.3f}   (n={chars['clean_all_chars']['n']})")
    print(f"        clean, inside eligible words    "
          f"{chars['clean_in_eligible_words']['mean']:.3f}   "
          f"(n={chars['clean_in_eligible_words']['n']})")
    for arm in ("random", "plausible"):
        for key in ("swapped_word", "word_it_replaced"):
            c = chars[arm][key]
            print(f"        {arm:9s} {key:17s} {c['all']['mean']:.3f} / {c['changed']['mean']:.3f}"
                  f"   (n={c['all']['n']} / {c['changed']['n']})")

    print("\n[wls] word level, mean over words (max-pool / mean-pool):")
    print(f"        clean, every eligible word      "
          f"{out['clean_all_words']['max']['mean']:.3f} / "
          f"{out['clean_all_words']['mean']['mean']:.3f}   "
          f"(n={out['clean_all_words']['max']['n']})")
    for arm in ("random", "plausible"):
        a = out[arm]
        print(f"        {arm:8s} planted word           "
              f"{a['swapped_word']['max']['mean']:.3f} / "
              f"{a['swapped_word']['mean']['mean']:.3f}   (n={a['n_words']})")
        print(f"        {arm:8s} word it replaced       "
              f"{a['word_it_replaced']['max']['mean']:.3f} / "
              f"{a['word_it_replaced']['mean']['mean']:.3f}")
    print(f"\n[wls] wrote {args.out}")


if __name__ == "__main__":
    main()
