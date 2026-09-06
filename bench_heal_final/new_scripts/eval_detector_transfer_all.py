"""Full-roster detector transfer: every detector of tab:ood-prf on the out-of-domain
track that scripts/eval_detector_transfer.py runs for two arms only.

The published transfer table (tab:ood-transfer) carries the refit full-dimension
logistic head and the GPT-2 likelihood. Those two alone cannot say whether the
domain shift costs the FITTED heads something the fit-free readouts avoid, so this
script runs the whole tab:ood-prf roster on the identical windows, the identical
corruption draw and the identical metrics.

PROVENANCE is inherited verbatim from eval_detector_transfer.py, so the two
published rows reproduce bit for bit (``--check`` asserts it against the published
JSON):

  * fit windows   text8 TRAIN characters [8e6, 8e6 + (fit_seqs+4)*L), first fit_seqs
  * text8 eval    the first ``n`` windows of the text8 TEST split
  * transfer eval the first ``n`` windows of the article extract
  * corruption    corrupt_false_info at ``--rate`` on both eval domains, seed+9,
                  and at the per-arm training rate on the fit windows
  * by_len        built from the first ``--vocab-chars`` characters of TRAIN

PER-ARM RECIPES mirror bench_ood_final/manifest.json (the arms behind tab:ood-prf):

    NLL        denoiser NLL @ t_nll 3.0                              fit-free
    LinE       hinge head @ t 4.5, negatives replace @ train-rate 0.5
    LinE_all   hinge head @ t 7.5, negatives replace+shuffle+falseinfo+both @ 0.5
    LinE_fi    hinge head @ t 4.5, negatives falseinfo @ 0.5
    LOG_fi     logistic head @ t 4.5, negatives falseinfo @ the EVALUATION rate
    Var        Laplace predictive variance @ t 7.5, ridge 0.1     closed form, clean only
    BGMM       BayesianGaussianMixture @ t 7.5, PCA-64, 20 comps, full cov
    GPT-2 SE   cross-step spilled energy, native BPE tokens
    GPT-2 NLL  same-step per-token NLL, native BPE tokens

ONE DELIBERATE DEVIATION from the bench. ood_bayes_linear.py builds the
false-information training vocabulary from the fit windows' own text; here every arm
uses the same corpus-level ``by_len`` that corrupts the evaluation domains, which is
what the published LOG_fi arm already did. One convention inside one table beats
matching two different ones.

    python scripts/eval_detector_transfer_all.py \
        --ckpt runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM_converge/epoch_final.pt \
        --out bench_ood_final/transfer/semaglutide_transfer_all.json --check
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
from scripts.ood_plausible_swap import (  # noqa: E402
    train_logistic_head, make_variance_head,
)
from scripts.heal_dirichlet import make_nll_localizer, make_bgmm_localizer  # noqa: E402
from scripts.bench_sflm_ebm import _load_gpt2, _gpt2_bpe_scores  # noqa: E402
from scripts._bench_common import (  # noqa: E402
    word_metrics, word_metrics_from_segments, make_adversarial_negatives,
    git_sha, ckpt_md5, gpu_name, now_iso,
)
from scripts.heal_protein_poc import wikifil, load_article  # noqa: E402
from aitchinson_flow.data.char_window_dataset import text_to_windows  # noqa: E402
from aitchinson_flow.data.corruption import (  # noqa: E402
    corrupt_false_info, build_vocab_by_len,
)
from datasets import load_dataset  # noqa: E402

_ALPH = "abcdefghijklmnopqrstuvwxyz "


# --------------------------------------------------------------------------- #
# metrics: one scoring pass per (arm, domain), three granularities out of it
# --------------------------------------------------------------------------- #
def _char_metrics(score_fn, clean, corr, changed):
    """Per-character detector -> {token, word(max-pool), sequence} AUROC.

    Mirrors eval_detector_transfer.py exactly: the sequence score is the mean of the
    per-position scores, the token AUROC ranks changed characters against unchanged
    ones inside the corrupted windows, and the word AUROC max-pools both sides.
    """
    sc_c, sc_o = score_fn(clean), score_fn(corr)
    lab = np.r_[np.zeros(sc_c.shape[0]), np.ones(sc_o.shape[0])]
    seq = _auroc(np.r_[sc_c.mean(1).numpy(), sc_o.mean(1).numpy()], lab)
    ch = changed
    tok = float("nan")
    if ch.any() and (~ch).any():
        tok = _auroc(sc_o.numpy().reshape(-1), ch.numpy().reshape(-1).astype(int))
    wm = word_metrics(sc_c.numpy(), clean, sc_o.numpy(), corr, ch, op="max")
    return {"token_auroc": tok,
            "word_auroc_max": wm["word"].get("auroc"),
            "seq_auroc": seq,
            "word_seq_auroc": wm["seq"].get("auroc"),
            "n_changed": int(ch.sum())}


def _gpt2_metrics(gpt2, gpt2_tok, clean, corr, changed, score_kind):
    """BPE-native baseline -> the same three granularities.

    The token column is the LM's own unit (a BPE token counts as corrupt when its
    span overlaps a changed character), the word column pools BPE segments straight
    up to words, and the sequence score is normalised per character so a sequence
    that tokenises into more BPE tokens is not diluted.
    """
    cv, cs = _gpt2_bpe_scores(gpt2, gpt2_tok, clean, chunk=32, score=score_kind)
    ov, os_ = _gpt2_bpe_scores(gpt2, gpt2_tok, corr, chunk=32, score=score_kind)
    wm = word_metrics_from_segments(cv, cs, clean, ov, os_, corr, changed, op="max")
    ch = changed.numpy()
    tok_sc = np.concatenate([v.numpy() for v in ov])
    tok_lb = np.concatenate([
        np.array([int(bool(ch[j, s:e].any())) for (s, e) in os_[j]], dtype=int)
        for j in range(len(ov))])
    L = clean.shape[1]
    seq_sc = np.r_[[float(v.sum()) / L for v in cv], [float(v.sum()) / L for v in ov]]
    seq_lb = np.r_[np.zeros(len(cv)), np.ones(len(ov))]
    return {"token_auroc": _auroc(tok_sc, tok_lb),
            "word_auroc_max": wm["word"].get("auroc"),
            "seq_auroc": _auroc(seq_sc, seq_lb),
            "word_seq_auroc": wm["seq"].get("auroc"),
            "n_changed": int(changed.sum())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt",
                    default="runs/sflm_bench_a100_20g_L256_d1280L14_full/"
                            "DirichletFM_converge/epoch_final.pt")
    ap.add_argument("--article-json",
                    default="bench_ood_final/transfer/semaglutide_article_extract.json")
    ap.add_argument("--out", default="bench_ood_final/transfer/semaglutide_transfer_all.json")
    ap.add_argument("--domain-name", default="semaglutide_ood")
    ap.add_argument("--fit-seqs", type=int, default=512)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--rate", type=float, default=0.15,
                    help="evaluation corruption rate (both domains)")
    ap.add_argument("--train-rate", type=float, default=0.5,
                    help="training-negative rate for the three LinE arms (the bench "
                         "default; LOG_fi keeps --rate, as it does in the paper)")
    ap.add_argument("--vocab-chars", type=int, default=8_000_000)
    ap.add_argument("--ref-lm", default="gpt2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arms", default="all",
                    help="comma list of arm keys to run, or 'all'")
    ap.add_argument("--bgmm-max-iter", type=int, default=1000)
    ap.add_argument("--check", action="store_true",
                    help="assert LOG_fi and GPT-2 NLL reproduce the published "
                         "bench_ood_final/transfer/semaglutide_transfer_gpt2.json")
    args = ap.parse_args()

    # ---- prologue: byte-identical to eval_detector_transfer.py so LOG_fi lands on
    # ---- the published draw (the corruption helpers all take explicit seeds, so the
    # ---- later arms cannot perturb it either).
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = _load_dirichletfm(args.ckpt, device)
    K, L = cfg.text8_dataset.K, cfg.text8_dataset.L

    ds = load_dataset("afmck/text8")
    fit_tok = text_to_windows(
        ds["train"][0]["text"][args.vocab_chars: args.vocab_chars + (args.fit_seqs + 4) * L],
        L)[: args.fit_seqs]
    eval_t8 = text_to_windows(ds["test"][0]["text"][: (args.n + 4) * L], L)[: args.n]
    by_len = build_vocab_by_len(ds["train"][0]["text"][: args.vocab_chars])

    art = wikifil(load_article(args.article_json))
    ins = text_to_windows(art, L)[: args.n]
    print(f"[transfer-all] fit={fit_tok.shape[0]} text8_eval={eval_t8.shape[0]} "
          f"{args.domain_name}_eval={ins.shape[0]} device={device}")

    # the evaluation corruption: one draw, shared by every arm and both domains
    t8_corr = corrupt_false_info(eval_t8.clone(), args.rate, by_len, seed=args.seed + 9)
    ins_corr = corrupt_false_info(ins.clone(), args.rate, by_len, seed=args.seed + 9)
    domains = [("text8_indist", eval_t8, t8_corr),
               (args.domain_name, ins, ins_corr)]

    def _negatives(schemes, rate):
        """(clean, corrupt) training pairs, one per scheme, on the fit windows."""
        adv = make_adversarial_negatives(fit_tok, K=K, by_len=by_len, schemes=schemes,
                                         rate=rate, seed=args.seed)
        print(f"  [negatives] {[(s, int(m.sum())) for s, _, m in adv]}")
        return [(fit_tok, ct) for _s, ct, _m in adv]

    # ---- arm builders: each returns a per-character score(tok) -> (B,L) closure ----
    def _build_log_fi():
        fi_fit = corrupt_false_info(fit_tok.clone(), args.rate, by_len, seed=args.seed + 5)
        return train_logistic_head(model, fit_tok, fi_fit, 4.5, device, seed=args.seed)

    def _build_hinge(schemes, t_eval):
        pairs = _negatives(schemes, args.train_rate)
        (cl0, ct0), extra = pairs[0], pairs[1:]
        return train_logistic_head(model, cl0, ct0, t_eval, device, seed=args.seed,
                                   kind="hinge", extra_pairs=extra)

    ARMS = {
        "LOG_fi":   ("logistic head @ t4.5, falseinfo negatives @ eval rate", _build_log_fi),
        "NLL":      ("denoiser NLL @ t3.0, fit-free",
                     lambda: make_nll_localizer(model, 3.0, device)[0]),
        "LinE":     ("hinge head @ t4.5, replace negatives",
                     lambda: _build_hinge(["replace"], 4.5)),
        "LinE_all": ("hinge head @ t7.5, replace+shuffle+falseinfo+both negatives",
                     lambda: _build_hinge(["replace", "shuffle", "falseinfo", "both"], 7.5)),
        "LinE_fi":  ("hinge head @ t4.5, falseinfo negatives",
                     lambda: _build_hinge(["falseinfo"], 4.5)),
        "Var":      ("Laplace predictive variance @ t7.5, ridge 0.1",
                     lambda: make_variance_head(model, fit_tok, 7.5, device, ridge=0.1)),
        "BGMM":     ("BayesianGaussianMixture @ t7.5, PCA-64, 20 comps, full cov",
                     lambda: make_bgmm_localizer(model, fit_tok, 7.5, device,
                                                 max_components=20, covariance_type="full",
                                                 pca_dim=64, reg_covar=1e-4,
                                                 max_fit_pos=80000, n_init=1,
                                                 max_iter=args.bgmm_max_iter,
                                                 init_params="k-means++",
                                                 seed=args.seed)[0]),
    }
    order = ["LOG_fi", "NLL", "LinE", "LinE_all", "LinE_fi", "Var", "BGMM"]
    want = order if args.arms == "all" else [a for a in order if a in args.arms.split(",")]

    results: dict = {name: {} for name, _c, _o in domains}
    meta: dict = {}
    for key in want:
        desc, build = ARMS[key]
        t0 = time.time()
        print(f"\n=== {key}: {desc} ===")
        score_fn = build()
        for dname, clean, corr in domains:
            m = _char_metrics(score_fn, clean, corr, corr != clean)
            results[dname][key] = m
            print(f"  {dname:>16} token={m['token_auroc']:.4f} "
                  f"word={m['word_auroc_max']:.4f} seq={m['seq_auroc']:.4f}")
        meta[key] = {"recipe": desc, "unit": "character", "wall_s": round(time.time() - t0, 1)}
        del score_fn

    # ---- the two external baselines, in their native BPE unit ----
    if args.ref_lm and (args.arms == "all" or "GPT2" in args.arms):
        gpt2, gpt2_tok = _load_gpt2(args.ref_lm, device)
        for key, kind in (("GPT-2 SE", "spilled"), ("GPT-2 NLL", "nll")):
            t0 = time.time()
            print(f"\n=== {key}: {args.ref_lm} {kind}, native BPE tokens ===")
            for dname, clean, corr in domains:
                m = _gpt2_metrics(gpt2, gpt2_tok, clean, corr, corr != clean, kind)
                results[dname][key] = m
                print(f"  {dname:>16} token={m['token_auroc']:.4f} "
                      f"word={m['word_auroc_max']:.4f} seq={m['seq_auroc']:.4f}")
            meta[key] = {"recipe": f"{args.ref_lm} {kind}", "unit": "bpe_token",
                         "wall_s": round(time.time() - t0, 1)}

    # ---- printed table, paper column order ----
    rows = [k for k in order + ["GPT-2 SE", "GPT-2 NLL"] if k in results["text8_indist"]]
    print(f"\n{'Detector':<10} | {'text8 (held-out)':^24} | {args.domain_name+' (transfer)':^24}")
    print(f"{'':<10} | {'token':>7} {'word':>7} {'seq':>7} | {'token':>7} {'word':>7} {'seq':>7}")
    print("-" * 74)
    for k in rows:
        a, b = results["text8_indist"][k], results[args.domain_name][k]
        print(f"{k:<10} | {a['token_auroc']:>7.3f} {a['word_auroc_max']:>7.3f} "
              f"{a['seq_auroc']:>7.3f} | {b['token_auroc']:>7.3f} "
              f"{b['word_auroc_max']:>7.3f} {b['seq_auroc']:>7.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "ckpt": args.ckpt, "ckpt_md5": ckpt_md5(args.ckpt), "git_sha": git_sha(),
        "gpu": gpu_name(), "written": now_iso(),
        "article": args.article_json,
        "rate": args.rate, "train_rate": args.train_rate, "n": args.n,
        "fit_seqs": args.fit_seqs, "seed": args.seed, "vocab_chars": args.vocab_chars,
        "arms": meta,
        "results": results,
        "note": "text8_indist = held-out text8 test windows (the head is fitted on TRAIN "
                "windows offset by vocab_chars); the transfer domain is the article. "
                "word AUROC = max-pool, the bench's common unit; token AUROC is each "
                "detector's native unit (characters for ours, BPE for GPT-2). "
                "Extends bench_ood_final/transfer/semaglutide_transfer_gpt2.json from "
                "two arms to the full tab:ood-prf roster on the identical draw.",
    }, indent=2))
    print(f"\nWrote {out}")

    # ---- reproduction check against the published two-arm artifact ----
    if args.check:
        ref = json.loads(Path(
            "bench_ood_final/transfer/semaglutide_transfer_gpt2.json").read_text())
        bad = []
        for dname in ("text8_indist", args.domain_name):
            r = ref["results"].get(dname, {})
            checks = [
                ("LOG_fi", "token_auroc", r.get("token_auroc")),
                ("LOG_fi", "word_auroc_max", r.get("word_auroc_max")),
                ("LOG_fi", "seq_auroc", r.get("seq_auroc")),
                ("GPT-2 NLL", "token_auroc", r.get("gpt2_nll", {}).get("bpe_token_auroc")),
                ("GPT-2 NLL", "word_auroc_max", r.get("gpt2_nll", {}).get("word_auroc_max")),
                ("GPT-2 NLL", "seq_auroc", r.get("gpt2_nll", {}).get("seq_auroc_charnorm")),
            ]
            for arm, field, expect in checks:
                if expect is None or arm not in results[dname]:
                    continue
                got = results[dname][arm][field]
                d = abs(got - expect)
                flag = "ok " if d < 1e-6 else "DIFF"
                if d >= 1e-6:
                    bad.append((dname, arm, field, expect, got, d))
                print(f"[check] {flag} {dname:>16} {arm:<10} {field:<15} "
                      f"published={expect:.6f} rerun={got:.6f} d={d:.2e}")
        print("[check] " + ("PASS: published arms reproduce exactly"
                            if not bad else f"FAIL: {len(bad)} mismatched fields"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
