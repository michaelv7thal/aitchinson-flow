"""Plausible-vs-random word-swap OOD experiment on DirichletFM.

Backs the paper claim: *the detector senses DISTRIBUTIONAL implausibility, not
factual truth.* We contrast two word-level corruptions applied to the SAME word
slots of the same clean passages:

  * random   — replace a word with a RANDOM different same-length real word
               (``corrupt_false_info``): lexically valid, contextually OFF.
  * plausible — replace a word with the same-length real word the MODEL ITSELF
               finds most fluent in context, i.e. the candidate MINIMISING the
               denoiser NLL over the word's positions (excluding the original).
               Lexically valid AND contextually fluent, but factually wrong.

Both change the same number of tokens at the same positions; only the *choice* of
replacement differs. We then score clean / random / plausible with two detectors:

  * BGMM  — the Bayesian DP-mixture feature density (heal_dirichlet.make_bgmm_
            localizer), the paper's OOD head — reads manifold density, NOT local
            surprise.
  * NLL   — the training-free denoiser surprise (reference; by construction the
            plausible swap was chosen to minimise this, so NLL should be near
            chance on plausible — the interesting question is whether BGMM is too).

PREDICTION (claim 3): AUROC(random) > AUROC(plausible) ~= 0.5 for BOTH detectors.
If BGMM detects plausible swaps where NLL cannot, that REFINES the story (density
catches context violations surprise misses); if it too is at chance, the passage
is genuinely invisible — factual falsehood without distributional departure is
undetectable by a char-level model with no world knowledge.

Usage:
    python scripts/ood_plausible_swap.py \
        --ckpt runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM/epoch_best.pt \
        --split test --n 64 --rate 0.15 --t-eval 6.0 --t-nll 3.0 --n-cands 48 \
        --out ood_out_best/plausible_swap/plausible_swap.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _bootstrap() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))


_bootstrap()

from scripts.ood_variance_perpos import _load_dirichletfm, _auroc, _perpos_feats  # noqa: E402
from scripts._bench_common import det_metrics  # noqa: E402
from scripts.heal_dirichlet import (  # noqa: E402
    make_bgmm_localizer,
    make_nll_localizer,
    train_localizer,
)
from scripts.bench_sflm_ebm import _load_gpt2, _gpt2_per_char_SE  # noqa: E402
from aitchinson_flow.training import build_training_datamodule  # noqa: E402
from aitchinson_flow.data.corruption import (  # noqa: E402
    corrupt_false_info,
    corrupt_token_ids,
    partially_shuffle_token_ids,
    build_vocab_by_len,
)

_ALPH = "abcdefghijklmnopqrstuvwxyz "  # text8 K=27
_CHAR2ID = {c: i for i, c in enumerate(_ALPH)}


def _decode(ids) -> str:
    return "".join(_ALPH[int(i)] if int(i) < len(_ALPH) else "?" for i in ids)


def _word_spans(row: torch.Tensor) -> list[tuple[int, int]]:
    """Fully-contained [a,b) letter-word spans of a token row (drop edge words)."""
    import re
    s = _decode(row)
    spans = [(m.start(), m.end()) for m in re.finditer(r"[a-z]+", s)]
    return [(a, b) for (a, b) in spans if a > 0 and b < len(s)]


@torch.no_grad()
def plausible_swap(model, windows, rate, by_len, *, t_nll, K, device,
                   n_cands=48, seed=0, chunk=64, freq_matched=False):
    """Model-guided plausible word swap.

    Selects the SAME slots ``corrupt_false_info`` would (shared RNG order) and
    replaces each with the same-length candidate that MINIMISES the denoiser NLL
    over the word's positions in context (excluding the original word). Returns
    (out_tokens (B,L), n_swapped)."""
    g = torch.Generator().manual_seed(seed)
    out = windows.clone()
    n_swapped = 0
    for i in range(windows.shape[0]):
        spans = _word_spans(windows[i])
        if not spans:
            continue
        n_sub = max(1, round(rate * len(spans)))
        order = torch.randperm(len(spans), generator=g).tolist()[:n_sub]
        base = windows[i:i + 1].to(device).long()          # (1,L) fixed context
        for idx in order:
            a, b = spans[idx]
            cands = by_len.get(b - a)
            if not cands:
                continue
            orig = _decode(windows[i, a:b])
            if freq_matched:
                # CONTROL for genericness: draw candidates from the ORIGINAL word's
                # own frequency band (a window around its rank in the freq-ordered
                # vocab), then pick the most-fluent among THOSE. The replacement is
                # a specific, similar-frequency word — so a detector can no longer
                # win by spotting "a generic high-frequency word in a specific slot".
                try:
                    r = cands.index(orig)
                except ValueError:
                    r = len(cands) // 2
                lo = max(0, r - n_cands // 2)
                cand_words = [w for w in cands[lo:lo + n_cands + 1] if w != orig]
            else:
                cand_words = [w for w in cands[:n_cands] if w != orig]
            if not cand_words:
                continue
            # build candidate-substituted windows (C,L) and score NLL over [a,b)
            C = len(cand_words)
            cand_ids = base.repeat(C, 1)                    # (C,L)
            for ci, w in enumerate(cand_words):
                for j, ch in enumerate(w):
                    cand_ids[ci, a + j] = _CHAR2ID[ch]
            nlls = []
            for s0 in range(0, C, chunk):
                cb = cand_ids[s0:s0 + chunk]
                Bc = cb.shape[0]
                beta = torch.ones(Bc, cb.shape[1], K, device=device)
                beta.scatter_(-1, cb.unsqueeze(-1), float(t_nll))
                x_t = beta / beta.sum(-1, keepdim=True)
                tt = torch.full((Bc,), float(t_nll), device=device)
                logits = model.forward(x_t, tt)            # (Bc,L,K)
                nll = -F.log_softmax(logits, -1).gather(
                    -1, cb.unsqueeze(-1)).squeeze(-1)       # (Bc,L)
                nlls.append(nll[:, a:b].mean(dim=1).cpu())  # mean over word chars
            nll_word = torch.cat(nlls)                      # (C,)
            best = cand_words[int(nll_word.argmin())]       # most fluent alternative
            for j, ch in enumerate(best):
                out[i, a + j] = _CHAR2ID[ch]
            n_swapped += 1
    return out, n_swapped


def train_logistic_head(model, clean_tok, corrupt_tok, t_eval, device, *,
                        max_fit=60000, C=1.0, seed=42, extra_pairs=None,
                        kind="logistic", mlp_steps=2000):
    """Full-dim energy head on standardised per-token features at t_eval, trained on
    clean(0) vs changed(1) tokens. Returns score(tok) -> (B,L) energy.

    ``kind``: ``logistic`` (the latent-split estimator; signed linear) |
    ``quad`` (low-rank SIGN-AGNOSTIC quadratic: departure from clean in a learned
    subspace) | ``mlp`` (general nonlinear). See the nonlinear block below for why the
    latter two are the right hypothesis class for the UNIFIED head.

    ``extra_pairs``: optional list of additional ``(clean_tok, corrupt_tok)`` pairs to
    pool into the SAME head — this is how the **unified** head is built (train on the
    union of replace / shuffle / falseinfo / plausible negatives). Each pair may have a
    different batch size (the plausible swap runs on a subset), which is why the
    negatives are collected per-pair rather than from one aligned tensor.

    Motivation: the specialists *anti-transfer* (paired mean shifts oppose,
    cos(d_fi, d_pl) = -0.49), yet their LEARNED directions are positively correlated
    (cos(w_fi, w_pl) = +0.40) — so a single w firing on both should exist. Nothing
    forced the specialists to find it; training on the union does.
    """
    from sklearn.linear_model import LogisticRegression
    fc = _perpos_feats(model, clean_tok, t_eval, device)  # (Nf,L,d) CPU
    d = fc.shape[-1]
    mu = fc.reshape(-1, d).mean(0)
    sigma = fc.reshape(-1, d).std(0).clamp_min(1e-6)

    def _proj(h):
        return (h - mu) / sigma

    pairs = [(clean_tok, corrupt_tok)] + list(extra_pairs or [])
    pos_parts, neg_parts = [_proj(fc.reshape(-1, d))], []
    for cl, ct in pairs:
        fk_i = _perpos_feats(model, ct, t_eval, device)
        lab_i = (ct != cl).reshape(-1)
        zk_i = _proj(fk_i.reshape(-1, d))
        pos_parts.append(zk_i[~lab_i])   # untouched tokens of a corrupted seq = valid
        neg_parts.append(zk_i[lab_i])    # the actually-changed tokens = corrupt
    zpos = torch.cat(pos_parts, 0)
    zneg = torch.cat(neg_parts, 0)

    def _samp(t, cap):
        return t[torch.randperm(t.shape[0])[:cap]] if t.shape[0] > cap else t
    Xp, Xn = _samp(zpos, max_fit).numpy(), _samp(zneg, max_fit).numpy()
    X = np.concatenate([Xp, Xn], 0)
    y = np.r_[np.zeros(len(Xp)), np.ones(len(Xn))]

    if kind != "logistic":
        # ---- NONLINEAR heads -------------------------------------------------
        # Why they can help *here* specifically, when SVGP saturated before: that
        # earlier negative was measured WITHIN one corruption. ACROSS corruptions the
        # geometry is nonlinear by construction — clean sits BETWEEN the two corruption
        # clusters (paired mean shifts oppose, cos(d_fi, d_pl) = -0.49), so the boundary
        # needed is a SHELL around clean, not a hyperplane. A signed linear head must
        # pick a side; these need not.
        #
        #   quad : score = sum_k (w_k . z)^2 - b   — a low-rank QUADRATIC form. It is
        #          SIGN-AGNOSTIC: it measures DEPARTURE from clean along the learned
        #          subspace, so it fires on both corruption directions at once. This is
        #          the minimal fix implied by the geometry (rank r ~ #corruption types).
        #          NB: this is NOT the BGMM density, which fails on plausible — that is a
        #          GENERATIVE density over the full space, where plausible words genuinely
        #          lie ON the manifold. This is DISCRIMINATIVE and restricted to the
        #          corruption subspace, where plausible IS displaced (its specialist gets
        #          0.904).
        #   mlp  : a general nonlinear boundary, the fallback if the quadratic form is
        #          too rigid.
        import torch.nn as nn
        dev = device
        Xt = torch.from_numpy(X).float()
        yt = torch.from_numpy(y).float()
        if kind == "quad":
            rank = 8

            class _Quad(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.W = nn.Linear(d, rank, bias=False)
                    self.b = nn.Parameter(torch.zeros(1))
                    self.s = nn.Parameter(torch.ones(1))

                def forward(self, z):
                    return self.s * (self.W(z) ** 2).sum(-1) - self.b
            net = _Quad().to(dev)
        else:
            net = nn.Sequential(nn.Linear(d, 256), nn.GELU(),
                                nn.Linear(256, 64), nn.GELU(),
                                nn.Linear(64, 1)).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        lossf = nn.BCEWithLogitsLoss()
        n_all, bs = Xt.shape[0], 4096
        g = torch.Generator().manual_seed(seed)
        for step in range(mlp_steps):
            idx = torch.randint(0, n_all, (bs,), generator=g)
            xb, yb = Xt[idx].to(dev), yt[idx].to(dev)
            out = net(xb).reshape(-1)
            loss = lossf(out, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 400 == 0 or step == mlp_steps - 1:
                with torch.no_grad():
                    acc = ((out > 0).float() == yb).float().mean().item()
                print(f"  [{kind}] step {step:>4} loss={loss.item():.4f} acc={acc:.3f}")
        net.eval()

        @torch.no_grad()
        def score_nl(tok, chunk=16):
            outs = []
            for i in range(0, tok.shape[0], chunk):
                h = _perpos_feats(model, tok[i:i + chunk], t_eval, device)
                b, Lq, _ = h.shape
                z = _proj(h.reshape(-1, d)).to(dev)
                e = net(z).reshape(b, Lq).cpu()
                outs.append(e)
            return torch.cat(outs, 0)
        print(f"  [{kind}] fit on {X.shape[0]} tok (pos={len(Xp)} neg={len(Xn)}) @ t={t_eval}")
        return score_nl

    clf = LogisticRegression(C=C, max_iter=2000).fit(X, y)
    print(f"  [logistic] fit on {X.shape[0]} tok (pos={len(Xp)} neg={len(Xn)}) @ t={t_eval}")

    @torch.no_grad()
    def score(tok, chunk=16):
        outs = []
        for i in range(0, tok.shape[0], chunk):
            h = _perpos_feats(model, tok[i:i + chunk], t_eval, device)
            b, Lq, _ = h.shape
            z = _proj(h.reshape(-1, d))
            e = torch.from_numpy(clf.decision_function(z.numpy())).float().reshape(b, Lq)
            outs.append(e)
        return torch.cat(outs)

    return score


def _seq_tok_auroc(score_clean, score_ood, changed):
    """(seq mean-pool AUROC, per-token AUROC, seq metrics, token metrics).

    The metric dicts add operating-point precision/recall/F1 at a clean-calibrated
    threshold (see _bench_common.det_metrics) — AUROC alone only says how well the
    score ranks, not what a deployed threshold actually flags."""
    cln_seq = score_clean.mean(1).numpy()
    ood_seq = score_ood.mean(1).numpy()
    lab = np.r_[np.zeros(score_clean.shape[0]), np.ones(score_ood.shape[0])]
    seq = _auroc(np.r_[cln_seq, ood_seq], lab)
    tok = float("nan")
    cm = changed.numpy().reshape(-1).astype(int)
    if changed.any() and (~changed).any():
        tok = _auroc(score_ood.numpy().reshape(-1), cm)
    m_seq = det_metrics(cln_seq, cln_seq, ood_seq)
    ood_tok = score_ood.numpy().reshape(-1)
    m_tok = det_metrics(score_clean.numpy().reshape(-1),
                        ood_tok[cm == 0], ood_tok[cm == 1])
    return seq, tok, m_seq, m_tok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--fit-seqs", type=int, default=512,
                    help="# clean seqs to fit the BGMM density detector")
    ap.add_argument("--n", type=int, default=64, help="# eval passages")
    ap.add_argument("--rate", type=float, default=0.15, help="word-swap rate")
    ap.add_argument("--t-eval", type=float, default=6.0,
                    help="path-time for the BGMM density features (falseinfo "
                         "separated best at t=6.0/7.5 in the sweep)")
    ap.add_argument("--t-nll", type=float, default=3.0,
                    help="path-time for the NLL detector AND candidate scoring")
    ap.add_argument("--n-cands", type=int, default=48,
                    help="# same-length candidates scored per slot for the "
                         "plausible swap (top-frequency words)")
    # BGMM knobs (match the detector/heal defaults)
    ap.add_argument("--bgmm-max-components", type=int, default=20)
    ap.add_argument("--bgmm-covariance-type", default="full")
    ap.add_argument("--bgmm-pca-dim", type=int, default=64)
    ap.add_argument("--bgmm-max-iter", type=int, default=500)
    # BLR (hinge energy) + GPT2-SE knobs
    ap.add_argument("--blr-t-eval", type=float, default=4.5,
                    help="path-time for the BLR hinge-energy features")
    ap.add_argument("--blr-train-rate", type=float, default=0.5,
                    help="corruption rate for the BLR hinge training negatives")
    ap.add_argument("--ref-lm", default="gpt2", help="HF causal LM for spilled energy")
    ap.add_argument("--lin-t-eval", type=float, default=4.5,
                    help="path-time for the full-dim LOGISTIC heads (latent-split t)")
    ap.add_argument("--unified-heads", default="logistic,quad,mlp",
                    help="hypothesis classes for the UNIFIED head, in increasing "
                         "capacity: logistic (signed linear) / quad (low-rank "
                         "SIGN-AGNOSTIC quadratic \u2014 departure from clean, the fix "
                         "implied by the opposed mean shifts) / mlp (general).")
    ap.add_argument("--unified-steps", type=int, default=2000,
                    help="training steps for the quad/mlp unified heads")
    ap.add_argument("--no-unified", action="store_true",
                    help="skip the UNIFIED head (trained on the union of "
                         "replace+shuffle+falseinfo+plausible) \u2014 the test of "
                         "whether ONE head can cover corruptions whose specialists "
                         "anti-transfer")
    ap.add_argument("--fit-plaus-seqs", type=int, default=128,
                    help="# fit windows to build PLAUSIBLE training negatives "
                         "(expensive model-guided swap; kept small)")
    ap.add_argument("--swap-mode", choices=["min_nll", "freq_matched"],
                    default="min_nll",
                    help="plausible-swap candidate pool: 'min_nll' (globally most-"
                         "fluent word — tends generic/high-freq) or 'freq_matched' "
                         "(most-fluent within the ORIGINAL word's frequency band — "
                         "controls for the genericness artifact).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = _load_dirichletfm(args.ckpt, device)
    K = cfg.text8_dataset.K

    dm, _ = build_training_datamodule(cfg)
    loaders = {"train": dm.train_dataloader, "val": dm.val_dataloader,
               "test": dm.test_dataloader}
    loader = loaders[args.split]() or dm.train_dataloader()
    seqs = []
    for b in loader:
        seqs.append(b["token_ids"].long())
        if sum(s.shape[0] for s in seqs) >= args.fit_seqs + args.n:
            break
    seqs = torch.cat(seqs)
    fit_tok = seqs[: args.fit_seqs]
    eval_tok = seqs[args.fit_seqs : args.fit_seqs + args.n]

    fit_txt = " ".join(_decode(row) for row in fit_tok)
    by_len = build_vocab_by_len(fit_txt)
    print(f"[plausible] split={args.split} fit={fit_tok.shape[0]} eval={eval_tok.shape[0]} "
          f"rate={args.rate} t_eval={args.t_eval} t_nll={args.t_nll} n_cands={args.n_cands}")

    # ---- detectors ----
    print("=== fit BGMM density detector (clean, one-class) ===")
    bgmm_score, _, n_eff = make_bgmm_localizer(
        model, fit_tok, args.t_eval, device,
        max_components=args.bgmm_max_components,
        covariance_type=args.bgmm_covariance_type, pca_dim=args.bgmm_pca_dim,
        max_iter=args.bgmm_max_iter, seed=args.seed,
    )
    nll_score, _ = make_nll_localizer(model, args.t_nll, device)
    print("=== train BLR hinge-energy detector ===")
    blr_score, _ = train_localizer(
        model, fit_tok, args.blr_t_eval, device, train_rate=args.blr_train_rate,
        margin=4.0, steps=600, lr=5e-2, seed=args.seed)
    print(f"=== load GPT-2 external-LM detectors ({args.ref_lm}) ===")
    gpt2_model, gpt2_tok = _load_gpt2(args.ref_lm, device)

    def se_score(tok):
        # the PAPER's cross-step spilled energy (Minut et al., ICLR 2026)
        return _gpt2_per_char_SE(gpt2_model, gpt2_tok, tok, chunk=16,
                                 score="spilled", attrib="boundary").cpu()

    def gpt2_nll_score(tok):
        # same LM, same-step per-token NLL — the honest per-token comparator (this is
        # what the repo previously reported under the name "spilled energy")
        return _gpt2_per_char_SE(gpt2_model, gpt2_tok, tok, chunk=16,
                                 score="nll", attrib="boundary").cpu()

    # ---- corruptions on the SAME slots ----
    print("=== build corruptions (random vs model-plausible, same slots) ===")
    rand_tok = corrupt_false_info(eval_tok.clone(), args.rate, by_len, seed=args.seed + 1)
    fm = args.swap_mode == "freq_matched"
    plaus_tok, n_sw = plausible_swap(
        model, eval_tok.clone(), args.rate, by_len, t_nll=args.t_nll, K=K,
        device=device, n_cands=args.n_cands, seed=args.seed + 1, freq_matched=fm)
    ch_rand = rand_tok != eval_tok
    ch_plaus = plaus_tok != eval_tok
    print(f"[plausible] random changed {int(ch_rand.sum())} toks; "
          f"plausible changed {int(ch_plaus.sum())} toks ({n_sw} words)")

    # ---- full-dim LOGISTIC heads (the latent-split estimator, held-out) ----
    # (1) trained on RANDOM false-info negatives (the deployable head); (2) trained
    # DIRECTLY on PLAUSIBLE negatives (best-case supervised head for plausible).
    print("=== train logistic head on RANDOM false-info negatives ===")
    fi_neg = corrupt_false_info(fit_tok.clone(), args.rate, by_len, seed=args.seed + 5)
    log_fi = train_logistic_head(model, fit_tok, fi_neg, args.lin_t_eval, device,
                                 seed=args.seed)
    print("=== build PLAUSIBLE training negatives (model-guided; subset) ===")
    plaus_fit = fit_tok[: args.fit_plaus_seqs]
    plaus_fit_neg, _ = plausible_swap(
        model, plaus_fit.clone(), args.rate, by_len, t_nll=args.t_nll, K=K,
        device=device, n_cands=args.n_cands, seed=args.seed + 7, freq_matched=fm)
    print("=== train logistic head on PLAUSIBLE negatives ===")
    log_pl = train_logistic_head(model, plaus_fit, plaus_fit_neg, args.lin_t_eval,
                                 device, seed=args.seed)

    # ---- THE UNIFYING HEAD: one head trained on the UNION of all corruption types ----
    # The specialists anti-transfer, but that does NOT mean no head can do both: their
    # LEARNED directions are positively correlated (cos(w_fi, w_pl) = +0.40), so a w
    # that fires on both exists — nothing forced a specialist to find it. Train on the
    # union and evaluate on false-info AND plausible to see whether it does.
    unified = {}
    if not args.no_unified:
        rp_neg = corrupt_token_ids(fit_tok.clone(), vocab_size=K,
                                   corrupt_rate=args.rate, seed=args.seed + 11)
        sh_neg = partially_shuffle_token_ids(fit_tok.clone(), shuffle_rate=args.rate,
                                             seed=args.seed + 13)
        extra = [(fit_tok, rp_neg), (fit_tok, sh_neg), (plaus_fit, plaus_fit_neg)]
        # Ladder of hypothesis classes for the ONE head that must cover corruptions whose
        # specialists ANTI-transfer. Linear may already suffice (the learned specialist
        # directions are positively correlated, cos=+0.40); if it does not, the geometry
        # says why — clean sits BETWEEN the corruption clusters, so a signed hyperplane is
        # the wrong shape and a sign-agnostic / nonlinear boundary is required.
        for kind in [k for k in args.unified_heads.split(",") if k.strip()]:
            print(f"=== train UNIFIED '{kind}' head on "
                  f"replace+shuffle+falseinfo+plausible ===")
            unified[kind] = train_logistic_head(
                model, fit_tok, fi_neg, args.lin_t_eval, device, seed=args.seed,
                extra_pairs=extra, kind=kind, mlp_steps=args.unified_steps)

    # ---- how "plausible" is plausible? mean denoiser NLL at the swapped chars ----
    nll_clean = nll_score(eval_tok)
    nll_rand = nll_score(rand_tok)
    nll_plaus = nll_score(plaus_tok)
    def _mean_at(score, mask):
        v = score[mask]
        return float(v.mean()) if v.numel() else float("nan")
    surprise = {
        "clean_mean_nll": float(nll_clean.mean()),
        "random_swapped_mean_nll": _mean_at(nll_rand, ch_rand),
        "plausible_swapped_mean_nll": _mean_at(nll_plaus, ch_plaus),
    }
    print(f"[plausible] denoiser NLL at swapped chars — random={surprise['random_swapped_mean_nll']:.3f} "
          f"plausible={surprise['plausible_swapped_mean_nll']:.3f} "
          f"(clean baseline={surprise['clean_mean_nll']:.3f})")

    # ---- all detector scores on clean / random / plausible ----
    bg_clean, bg_rand, bg_plaus = (bgmm_score(eval_tok), bgmm_score(rand_tok),
                                   bgmm_score(plaus_tok))
    blr_clean, blr_rand, blr_plaus = (blr_score(eval_tok), blr_score(rand_tok),
                                      blr_score(plaus_tok))
    lfi_clean, lfi_rand, lfi_plaus = (log_fi(eval_tok), log_fi(rand_tok),
                                      log_fi(plaus_tok))
    lpl_clean, lpl_rand, lpl_plaus = (log_pl(eval_tok), log_pl(rand_tok),
                                      log_pl(plaus_tok))
    uni_scores = {k: (fn(eval_tok), fn(rand_tok), fn(plaus_tok))
                  for k, fn in unified.items()}
    se_clean, se_rand, se_plaus = (se_score(eval_tok), se_score(rand_tok),
                                   se_score(plaus_tok))
    gn_clean, gn_rand, gn_plaus = (gpt2_nll_score(eval_tok), gpt2_nll_score(rand_tok),
                                   gpt2_nll_score(plaus_tok))

    rows = []
    print(f"\n{'detector':>14} {'corruption':>10} {'seq_AUROC':>10} {'token_AUROC':>12} "
          f"{'tokP@5':>7} {'tokR@5':>7} {'tokF1@5':>8}")
    dets = [("NLL", nll_clean, nll_rand, nll_plaus),
            ("BLR", blr_clean, blr_rand, blr_plaus),
            ("Logistic_fi", lfi_clean, lfi_rand, lfi_plaus),
            ("Logistic_plaus", lpl_clean, lpl_rand, lpl_plaus),
            ("BGMM", bg_clean, bg_rand, bg_plaus),
            ("GPT2_SE", se_clean, se_rand, se_plaus),
            ("GPT2_NLL", gn_clean, gn_rand, gn_plaus)]
    # the unifying heads — the rows that decide whether ONE head can cover corruptions
    # whose specialists anti-transfer (linear -> quadratic -> MLP, in increasing capacity)
    _UNI_LABEL = {"logistic": "ALL_linear", "quad": "ALL_quad", "mlp": "ALL_mlp"}
    for k, (c_, r_, p_) in uni_scores.items():
        dets.append((_UNI_LABEL.get(k, f"ALL_{k}"), c_, r_, p_))
    for det, cln, rnd, pls in dets:
        for name, ood, ch in [("random", rnd, ch_rand), ("plausible", pls, ch_plaus)]:
            seq, tok, m_seq, m_tok = _seq_tok_auroc(cln, ood, ch)
            p5 = m_tok["prf"].get("0.05", {})
            print(f"{det:>14} {name:>10} {seq:>10.3f} {tok:>12.3f} "
                  f"{p5.get('precision', float('nan')):>7.3f} "
                  f"{p5.get('recall', float('nan')):>7.3f} "
                  f"{p5.get('f1', float('nan')):>8.3f}")
            rows.append({"detector": det, "corruption": name,
                         "seq_auroc": seq, "token_auroc": tok,
                         "prf_seq": m_seq, "prf_token": m_tok})

    # ---- example: clean vs plausible swap text (to eyeball fluency) ----
    ex = []
    for b in range(min(2, eval_tok.shape[0])):
        ex.append({
            "clean": _decode(eval_tok[b][:120]),
            "random": _decode(rand_tok[b][:120]),
            "plausible": _decode(plaus_tok[b][:120]),
            "plausible_changed": [bool(x) for x in ch_plaus[b][:120].tolist()],
        })
        print(f"\n--- example {b} ---")
        print(f"  clean    : {ex[-1]['clean']!r}")
        print(f"  random   : {ex[-1]['random']!r}")
        print(f"  plausible: {ex[-1]['plausible']!r}")

    out_path = Path(args.out or (Path(args.ckpt).parent / "plausible_swap.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "ckpt": args.ckpt, "split": args.split, "n": int(eval_tok.shape[0]),
        "rate": args.rate, "t_eval": args.t_eval, "t_nll": args.t_nll,
        "blr_t_eval": args.blr_t_eval, "ref_lm": args.ref_lm,
        "swap_mode": args.swap_mode, "lin_t_eval": args.lin_t_eval,
        "n_cands": args.n_cands, "bgmm_n_effective": n_eff,
        "detector": "plausible_vs_random_swap",
        "detector_long": "random vs model-plausible word swap on identical slots; "
                         "BGMM density + NLL surprise; seq & per-token AUROC",
        "surprise": surprise, "rows": rows, "examples": ex,
    }, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
