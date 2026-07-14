"""Shared utilities for the reproducible OOD-detection + healing benchmark.

Small, dependency-light helpers reused by run_bench_ood.py / run_bench_heal.py /
bench_aggregate.py / plot_heatmaps_all.py:

  * provenance   — git_sha(), ckpt_md5(), gpu_name(), now_iso(), write_manifest()
  * DETECTOR_KEYS — per-detector key-map (seq/token AUROC field + example score key)
    so the aggregator/plotter can read every detector's differently-named output.
  * det_metrics() — AUROC (threshold-free ranking) PLUS the operating-point metrics
    precision / recall / F1 at a CLEAN-CALIBRATED threshold, so a detector's
    deployable behaviour is reported, not just its ranking.
  * heal_style_examples() — factored from ood_bgmm_perpos.py: clean / replace30 /
    falseinfo30 per-token examples with a CALIBRATED flag mask (score > clean
    quantile at example_fpr), matching heal_dirichlet's clean/corr/flag dump.
"""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from aitchinson_flow.data.corruption import (
    corrupt_token_ids, corrupt_false_info, partially_shuffle_token_ids,
)

_ALPH = "abcdefghijklmnopqrstuvwxyz "  # text8 K=27 decode (a-z + space)
_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #
def git_sha() -> str:
    """Current commit SHA (mirrors scripts/run_experiment.py:_git_sha)."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(_ROOT),
                             capture_output=True, text=True, check=True)
        return out.stdout.strip() or "unknown"
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def git_dirty() -> bool:
    """True if the working tree has uncommitted changes (so git_sha alone is
    insufficient to reproduce — pair it with the repro patch)."""
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=str(_ROOT),
                             capture_output=True, text=True, check=True)
        return bool(out.stdout.strip())
    except (subprocess.SubprocessError, OSError):
        return False


def save_repro_patch(out_dir: str | Path) -> dict:
    """Capture the EXACT code that ran, so a dirty tree is still reproducible:
      * <out_dir>/repro.patch     — `git diff HEAD` (tracked modifications)
      * <out_dir>/new_scripts/    — copies of UNTRACKED scripts (new files the
                                    diff can't see, e.g. run_bench_ood.py)
    Reproduce with:
        git checkout <git_sha> && git apply repro.patch
        cp new_scripts/* scripts/ && <arm cmd from manifest>
    Returns {"patch": name|None, "new_scripts": [names]}.
    """
    import shutil
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = {"patch": None, "new_scripts": []}
    try:
        diff = subprocess.run(["git", "diff", "HEAD"], cwd=str(_ROOT),
                             capture_output=True, text=True, check=True).stdout
        if diff.strip():
            (out / "repro.patch").write_text(diff)
            res["patch"] = "repro.patch"
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "scripts"],
            cwd=str(_ROOT), capture_output=True, text=True, check=True).stdout.split()
        snap = out / "new_scripts"
        for f in untracked:
            if f.endswith(".py"):
                snap.mkdir(exist_ok=True)
                shutil.copy2(_ROOT / f, snap / Path(f).name)
                res["new_scripts"].append(Path(f).name)
    except (subprocess.SubprocessError, OSError):
        pass
    return res


def finalize_manifest(out_dir: str | Path) -> Path | None:
    """Post-hoc: stamp an existing manifest.json with git_dirty + the repro patch
    (+ untracked benchmark scripts), so a manifest written mid-run against a dirty
    tree becomes fully reproducible. Idempotent."""
    import json
    p = Path(out_dir) / "manifest.json"
    if not p.exists():
        return None
    rec = json.loads(p.read_text())
    rec["git_dirty"] = git_dirty()
    cap = save_repro_patch(out_dir)
    rec["repro_patch"] = cap["patch"]
    rec["new_scripts"] = cap["new_scripts"]
    rec["repro_recipe"] = (
        f"git checkout {rec.get('git_sha', '?')} && "
        f"git apply {cap['patch'] or '(no-patch)'} && "
        f"cp new_scripts/* scripts/ && <arm cmd from arms[]>")
    p.write_text(json.dumps(rec, indent=2))
    return p


def code_fingerprint() -> str:
    """Content hash of the benchmark code (scripts/ + the package).

    `--resume` must skip an arm ONLY if it was produced by the same command AND the
    same code. The command line alone is not enough: editing a detector changes what
    an arm computes while its CLI stays byte-identical, so a cmd-only check would
    silently keep stale results (this bit us — arms run before word-level metrics were
    added carried an identical cmd). Comparing this fingerprint makes a code change
    invalidate the cached arms automatically.
    """
    h = hashlib.md5()
    files = sorted(list((_ROOT / "scripts").glob("*.py"))
                   + list((_ROOT / "src").rglob("*.py")))
    for f in files:
        try:
            h.update(f.read_bytes())
        except OSError:
            continue
    return h.hexdigest()


def ckpt_md5(path: str, chunk: int = 1 << 20) -> str:
    """md5 of a checkpoint file (streamed; '' if unreadable)."""
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(chunk), b""):
                h.update(blk)
        return h.hexdigest()
    except OSError:
        return ""


def gpu_name() -> str:
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name()
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_manifest(out_dir: str | Path, record: dict) -> Path:
    """Write/overwrite <out_dir>/manifest.json with a provenance record."""
    import json
    p = Path(out_dir) / "manifest.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record, indent=2))
    return p


# --------------------------------------------------------------------------- #
# per-detector key-map (from the detector-schema survey)
# --------------------------------------------------------------------------- #
# EVALUATION PROTOCOL — each model is scored in the unit it ACTUALLY operates in,
# and the cross-model comparison happens at a common unit:
#
#   * flow-matching detectors (NLL / BLR / BGMM)  -> native unit = CHARACTER
#   * GPT-2 baselines (spilled energy, NLL)       -> native unit = BPE TOKEN
#   * BOTH are additionally pooled up to WORDS    -> the fair head-to-head
#
# No score is ever attributed DOWNWARD (BPE -> char): GPT-2 has no per-character
# opinion, and the old "spread the token's score uniformly over its chars" rule
# smeared localization. Pooling UP is exact, so words are the comparison unit.
#
# SEQUENCE level is already comparable across tokenizers *provided* the total is
# normalised per CHARACTER (the standard bits-per-character convention). With
# boundary attribution, mean-over-chars == sum(BPE scores)/n_chars, so `auroc_seq_*`
# is exactly that. Do NOT use a mean over BPE tokens for the sequence score: corrupted
# text tokenises into MORE BPE tokens, which dilutes the mean and is not comparable.
#
# label -> {seq, token (native-unit AUROC), unit, word_max/word_mean, score, file, subdir}
DETECTOR_KEYS: dict[str, dict] = {
    "NLL": {"seq": "auroc_seq_nll", "token": "auroc_token_nll", "unit": "char",
            "score": "NLL_t", "subdir": "nll", "file": "denoiser_nll_sweep.json",
            "detector_label": "denoiser NLL"},
    "BLR": {"seq": "auroc_seq_energy", "token": "auroc_token_energy", "unit": "char",
            "score": "E_t", "subdir": "blr", "file": "bayes_linear_sweep.json",
            "detector_label": "BayesLin energy (replace-trained)"},
    "BLR_ADV": {"seq": "auroc_seq_energy", "token": "auroc_token_energy",
                "unit": "char", "score": "E_t", "subdir": "blr_adv",
                "file": "bayes_linear_adv_sweep.json",
                "detector_label": "BayesLin adv-mix"},
    "BLR_FI": {"seq": "auroc_seq_energy", "token": "auroc_token_energy",
               "unit": "char", "score": "E_t", "subdir": "blr_fi",
               "file": "bayes_linear_fi_sweep.json",
               "detector_label": "BayesLin falseinfo-only"},
    "BGMM": {"seq": "auroc_seq_gmm", "token": "auroc_token_gmm", "unit": "char",
             "score": "GMMNLL_t", "subdir": "bgmm", "file": "bgmm_perpos_sweep.json",
             "detector_label": "BGMM density"},
    # The PAPER's method (Minut et al., ICLR 2026): cross-step ΔE. Strong at sequence
    # level, but not a localizer (it pairs a step-(i-1) logit with a step-i logsumexp).
    # Native token unit = BPE (`auroc_token_se_bpe`), never the char-attributed number.
    "GPT2_SE": {"seq": "auroc_seq_se", "token": "auroc_token_se_bpe", "unit": "bpe",
                "token_char_attributed": "auroc_token_se",
                "score": "SE_t", "subdir": "gpt2_se",
                "file": "gpt2_spilled_energy_sweep.json",
                "detector_label": "GPT-2 spilled energy (cross-step ΔE)"},
    # Same LM, same slices, SAME-STEP per-token NLL — the honest localization
    # comparator. This is what the repo previously mislabelled as "spilled energy".
    "GPT2_NLL": {"seq": "auroc_seq_se", "token": "auroc_token_se_bpe", "unit": "bpe",
                 "token_char_attributed": "auroc_token_se",
                 "score": "SE_t", "subdir": "gpt2_nll",
                 "file": "gpt2_nll_sweep.json",
                 "detector_label": "GPT-2 per-token NLL"},
}


# --------------------------------------------------------------------------- #
# detection metrics: AUROC + operating-point precision / recall / F1
# --------------------------------------------------------------------------- #
# Target false-positive rates on CLEAN data at which P/R/F1 are reported. Same
# convention as heal_dirichlet.calibrate_threshold, so OOD precision/recall is
# directly comparable with healing's loc_precision / loc_recall.
PRF_FPRS: tuple[float, ...] = (0.01, 0.05, 0.10)


def prf_at_threshold(neg: np.ndarray, pos: np.ndarray, thr: float) -> dict:
    """Precision / recall / F1 of ``score > thr`` (positive == corrupt).

    ``neg``/``pos`` are the NEGATIVE (clean) and POSITIVE (corrupt) score arrays
    of the evaluation set. Precision depends on the positive base rate, so the
    caller should read it next to ``prevalence`` (see det_metrics)."""
    tp = float((pos > thr).sum())
    fp = float((neg > thr).sum())
    fn = float((pos <= thr).sum())
    prec = tp / max(tp + fp, 1.0)
    rec = tp / max(tp + fn, 1.0)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"threshold": float(thr), "precision": prec, "recall": rec, "f1": f1,
            "tp": int(tp), "fp": int(fp), "fn": int(fn)}


def det_metrics(cal_neg: np.ndarray, neg: np.ndarray, pos: np.ndarray,
                fprs: tuple[float, ...] = PRF_FPRS) -> dict:
    """Full detection report for one corruption cell: ranking + operating points.

    AUROC/AP are threshold-free (how well the score RANKS corrupt above clean).
    P/R/F1 need a threshold, and the only GT-free way to pick one is to calibrate
    it on CLEAN data — ``cal_neg`` (clean reference scores) at each target FPR.
    ``f1_best`` is the max-F1 over ALL thresholds: the ceiling an oracle threshold
    would reach, which bounds how much better calibration could ever make you.

    Parameters
    ----------
    cal_neg : clean scores used ONLY to pick thresholds (never scored against).
    neg     : negative (uncorrupted) scores of the evaluation set.
    pos     : positive (corrupted) scores of the evaluation set.

    Returns {auroc, ap, prevalence, f1_best, precision_at_f1_best,
             recall_at_f1_best, threshold_at_f1_best, prf: {"<fpr>": {...}}}.
    All-NaN if either class has <2 members (e.g. rate=1.0 leaves no clean tokens).
    """
    from sklearn.metrics import average_precision_score, precision_recall_curve
    from scripts.ood_variance_perpos import _auroc

    neg = np.asarray(neg, dtype=float).reshape(-1)
    pos = np.asarray(pos, dtype=float).reshape(-1)
    nan = float("nan")
    if len(pos) < 2 or len(neg) < 2:
        return {"auroc": nan, "ap": nan, "prevalence": nan, "f1_best": nan,
                "precision_at_f1_best": nan, "recall_at_f1_best": nan,
                "threshold_at_f1_best": nan, "prf": {}}

    scores = np.r_[neg, pos]
    labels = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    auroc = _auroc(scores, labels)
    ap = float(average_precision_score(labels, scores))
    prevalence = float(len(pos) / len(scores))  # = AP of a random detector

    # max-F1 over every threshold the data realises (oracle operating point)
    p, r, thr_grid = precision_recall_curve(labels, scores)
    f1_grid = 2 * p * r / np.maximum(p + r, 1e-9)
    i = int(np.nanargmax(f1_grid))
    f1_best = float(f1_grid[i])
    # precision_recall_curve returns len(thr)=len(p)-1 (last point is recall=0)
    thr_best = float(thr_grid[i]) if i < len(thr_grid) else float(np.max(scores))

    cal = np.asarray(cal_neg, dtype=float).reshape(-1)
    prf = {}
    for f in fprs:
        thr = float(np.quantile(cal, 1.0 - f))
        prf[f"{f:g}"] = {"target_fpr": float(f), **prf_at_threshold(neg, pos, thr)}

    return {"auroc": auroc, "ap": ap, "prevalence": prevalence,
            "f1_best": f1_best, "precision_at_f1_best": float(p[i]),
            "recall_at_f1_best": float(r[i]), "threshold_at_f1_best": thr_best,
            "prf": prf}


# --------------------------------------------------------------------------- #
# WORD-level pooling — the common unit for char-model vs BPE-LM comparison
# --------------------------------------------------------------------------- #
# Our detector scores CHARACTERS; GPT-2 scores BPE TOKENS. Comparing them is only
# ill-posed in one direction: pooling scores UP to a coarser unit is well-defined
# (a segment's score is an aggregate of its parts), while spreading a BPE score DOWN
# onto characters is not — GPT-2 simply has no per-character opinion, and the old
# "spread uniformly over the token's chars" rule smeared localization.
#
# So we never attribute downward. Both detectors are lifted to a COMMON segmentation.
# The natural one for text8 is the WORD (whitespace-delimited):
#   * exact for both sides — GPT-2's pre-tokenizer splits on whitespace, so a BPE
#     token never crosses a word boundary (verified on our data);
#   * it is the unit the corruptions actually use — `falseinfo` and `plausible` swap
#     WHOLE WORDS — so "which word is wrong" is the claim we are really making.
# BPE-granularity is the secondary check; per-CHARACTER localization stays a
# capability of the char-level model that a BPE LM structurally cannot have.
_SPACE_ID = 26  # text8 CHAR2ID: 'a'..'z' = 0..25, ' ' = 26


def word_spans(token_ids: "torch.Tensor", space_id: int = _SPACE_ID) -> list:
    """Whitespace-delimited word spans per sequence → list[B] of [(start, end), ...].

    Computed from the sequence BEING SCORED (clean or corrupted), because a
    corruption can move spaces and thus re-segment the text.
    """
    out = []
    for row in token_ids.tolist():
        spans, start = [], None
        for i, t in enumerate(row):
            if t == space_id:
                if start is not None:
                    spans.append((start, i))
                    start = None
            elif start is None:
                start = i
        if start is not None:
            spans.append((start, len(row)))
        out.append(spans)
    return out


def pool_to_words(char_scores, spans: list, op: str = "max") -> list:
    """Pool per-char scores up into per-word scores. list[B] of np.ndarray(n_words,).

    ``op`` matters and should not be chosen to flatter a result:
      * ``max``  — "is ANY character here surprising?"  Suits sharp, local damage
                   (a single replaced character).
      * ``mean`` — "is this word surprising ON AVERAGE?"  Suits weak, distributed
                   signal spread over the whole word (this is the regime false-info
                   sits in: the word is lexically valid, only contextually wrong).
    Report both.
    """
    if op not in ("max", "mean"):
        raise ValueError(f"op must be 'max' or 'mean', got {op!r}")
    arr = (char_scores.detach().cpu().numpy()
           if hasattr(char_scores, "detach") else np.asarray(char_scores))
    out = []
    for j, sp in enumerate(spans):
        if not sp:
            out.append(np.zeros(0, dtype=float))
            continue
        vals = [arr[j, s:e].max() if op == "max" else arr[j, s:e].mean()
                for (s, e) in sp]
        out.append(np.asarray(vals, dtype=float))
    return out


def word_labels(changed, spans: list) -> list:
    """1 for each word containing at least one corrupted character."""
    ch = (changed.detach().cpu().numpy()
          if hasattr(changed, "detach") else np.asarray(changed))
    return [np.array([int(bool(ch[j, s:e].any())) for (s, e) in sp], dtype=int)
            for j, sp in enumerate(spans)]


def pool_segments_to_words(seg_scores: list, seg_spans: list, wspans: list,
                           op: str = "max") -> list:
    """Pool arbitrary SEGMENT scores (e.g. GPT-2 BPE tokens, each with a char span)
    up into per-word scores. list[B] of np.ndarray(n_words,).

    Use this for the BPE LM instead of pooling its char-ATTRIBUTED scores: with
    boundary attribution most characters carry no score at all, so a char-level
    mean-pool would divide by the character count and silently dilute the word score
    with zeros. Pooling segment→word keeps the LM in its native units the whole way
    up, which is the entire point of choosing a common coarser unit.

    A BPE token is assigned to the word its span overlaps (GPT-2's pre-tokenizer
    splits on whitespace, so a token never straddles two words).
    """
    if op not in ("max", "mean"):
        raise ValueError(f"op must be 'max' or 'mean', got {op!r}")
    out = []
    for j, wsp in enumerate(wspans):
        vals = (seg_scores[j].tolist() if hasattr(seg_scores[j], "tolist")
                else list(seg_scores[j]))
        buckets: list[list[float]] = [[] for _ in wsp]
        for v, (a, b) in zip(vals, seg_spans[j]):
            for k, (ws, we) in enumerate(wsp):
                if not (b <= ws or a >= we):  # overlap
                    buckets[k].append(float(v))
                    break
        res = [(max(bk) if op == "max" else sum(bk) / len(bk)) if bk else 0.0
               for bk in buckets]
        out.append(np.asarray(res, dtype=float))
    return out


def word_metrics_from_segments(clean_vals, clean_spans, clean_tok,
                               corr_vals, corr_spans, corr_tok, changed,
                               op: str = "max", fprs: tuple = PRF_FPRS) -> dict:
    """Word-level report for a SEGMENT-scoring model (the BPE LM). Same contract as
    :func:`word_metrics`, so the two are directly comparable."""
    cws, ows = word_spans(clean_tok), word_spans(corr_tok)
    cw = pool_segments_to_words(clean_vals, clean_spans, cws, op)
    ow = pool_segments_to_words(corr_vals, corr_spans, ows, op)
    ol = word_labels(changed, ows)
    return _word_report(cw, ow, ol, fprs, op)


def _word_report(cw: list, ow: list, ol: list, fprs: tuple, op: str) -> dict:
    """Shared tail of the word-level metric: flatten, calibrate on clean, score."""
    nan = float("nan")
    cal = (np.concatenate([w for w in cw if len(w)])
           if any(len(w) for w in cw) else np.zeros(0))
    ow_flat = (np.concatenate([w for w in ow if len(w)])
               if any(len(w) for w in ow) else np.zeros(0))
    ol_flat = (np.concatenate([lb for lb, w in zip(ol, ow) if len(w)])
               if any(len(w) for w in ow) else np.zeros(0, dtype=int))
    tok = ({"auroc": nan} if not (ol_flat.any() and (~ol_flat.astype(bool)).any())
           else det_metrics(cal, ow_flat[ol_flat == 0], ow_flat[ol_flat == 1], fprs))
    c_seq = np.array([w.mean() if len(w) else nan for w in cw])
    o_seq = np.array([w.mean() if len(w) else nan for w in ow])
    seq = det_metrics(c_seq, c_seq, o_seq, fprs)
    return {"op": op, "word": tok, "seq": seq, "n_words": int(len(ow_flat))}


def word_metrics(clean_char, clean_tok, corr_char, corr_tok, changed,
                 op: str = "max", fprs: tuple = PRF_FPRS) -> dict:
    """Word-level detection report (AUROC + P/R/F1), the common-unit comparison.

    Sequence score = mean over the sequence's word scores. Threshold is calibrated
    GT-free on CLEAN words, as everywhere else in this bench.
    """
    cws, ows = word_spans(clean_tok), word_spans(corr_tok)
    cw = pool_to_words(clean_char, cws, op)
    ow = pool_to_words(corr_char, ows, op)
    ol = word_labels(changed, ows)
    return _word_report(cw, ow, ol, fprs, op)


# --------------------------------------------------------------------------- #
# healing-style per-token examples (shared across detectors)
# --------------------------------------------------------------------------- #
def make_adversarial_negatives(fit_tok, *, K, by_len, schemes, rate, seed=0,
                               model=None, t_nll=3.0, device="cuda", n_cands=48,
                               plausible_seqs=None):
    """Build a SYNTHETIC ADVERSARIAL training set: one corrupted copy of the clean
    windows per corruption scheme in ``schemes`` (replace / shuffle / falseinfo /
    both / **plausible**), so a discriminative head trained on the pool learns a
    *general* corrupt-token boundary instead of a single-corruption one.

    ``plausible`` = the model-guided min-NLL word swap (the hardest corruption: the
    planted word is chosen to MINIMISE the model's own surprise). It needs ``model``,
    and it is expensive (``n_cands`` forward passes per swapped word), so it is
    generated on the first ``plausible_seqs`` windows only (default: all of them).

    **Why include it.** The specialist heads *anti-transfer*: measured paired mean
    shifts give cos(d_falseinfo, d_plausible) = -0.49 — plausible displaces the
    representation the OPPOSITE way from replace/false-info, so a head trained on one
    scores the other as "cleaner than clean". BUT the LEARNED directions are positively
    correlated (cos(w_fi, w_pl) = +0.40), so a weight vector that fires on BOTH exists.
    Nothing forced the specialists to find it; training on the union does.

    Returns list[(scheme, corrupt_tok (B,L), changed_mask (B,L) bool)].
    ``by_len`` (from build_vocab_by_len) is required for 'falseinfo'/'plausible'.
    """
    out = []
    for i, scheme in enumerate(schemes):
        s = seed + 17 * i
        if scheme == "replace":
            ct = corrupt_token_ids(fit_tok.clone(), vocab_size=K, corrupt_rate=rate, seed=s)
        elif scheme == "shuffle":
            ct = partially_shuffle_token_ids(fit_tok.clone(), shuffle_rate=rate, seed=s)
        elif scheme in ("falseinfo", "wordswap"):
            ct = corrupt_false_info(fit_tok.clone(), rate, by_len, seed=s)
        elif scheme == "both":
            ct = corrupt_token_ids(fit_tok.clone(), vocab_size=K, corrupt_rate=rate, seed=s)
            ct = partially_shuffle_token_ids(ct, shuffle_rate=rate, seed=s + 1)
        elif scheme == "plausible":
            if model is None:
                raise ValueError("scheme 'plausible' needs the model (min-NLL swap)")
            from scripts.ood_plausible_swap import plausible_swap
            sub = fit_tok if plausible_seqs is None else fit_tok[:plausible_seqs]
            ct_sub, _ = plausible_swap(model, sub.clone(), rate, by_len, t_nll=t_nll,
                                       K=K, device=device, n_cands=n_cands, seed=s)
            out.append((scheme, ct_sub, ct_sub != sub))
            continue  # note: a SUBSET of the windows, so append it directly
        else:
            continue
        out.append((scheme, ct, ct != fit_tok))
    return out


def _decode(ids, max_pos: int) -> str:
    return "".join(_ALPH[int(i)] if int(i) < len(_ALPH) else "?"
                   for i in ids[:max_pos])


def heal_style_examples_bpe(bpe_fn, pos_tok, ex_tok, *, K, by_len, score_key,
                            example_fpr=0.05, max_cells=64, seed=7, verbose=True):
    """Healing-style examples for a SEGMENT scorer (the BPE LM), in ITS OWN units.

    Same contract as :func:`heal_style_examples`, but one cell = one **BPE token**,
    not one character. This is what makes the GPT-2 heatmaps honest: a red box marks
    a *BPE token that overlaps a corrupted character*, and the flag marker is that
    token's own score against a clean-calibrated threshold. Rendering GPT-2 on
    character cells would require attributing its score down onto characters — the
    very smearing artifact the boundary/BPE split exists to remove.

    ``bpe_fn(token_ids) -> (scores: list[B] of (n_j,), spans: list[B] of [(s,e)])``.

    Emits ``cell_labels`` (the token strings) and ``unit="bpe"`` so the plotter labels
    the axis with BPE tokens rather than characters.
    """
    sc_clean, _ = bpe_fn(pos_tok)
    flat = np.concatenate([s.numpy() for s in sc_clean if len(s)])
    flag_thr = float(np.quantile(flat, 1.0 - example_fpr))
    if verbose:
        print(f"  [examples/bpe] flag thr={flag_thr:.4f} @ fpr={example_fpr} "
              f"(key={score_key}, unit=BPE token)")
    variants = [
        ("clean", ex_tok.clone()),
        ("replace30", corrupt_token_ids(ex_tok.clone(), vocab_size=K,
                                        corrupt_rate=0.3, seed=seed)),
    ]
    if by_len is not None:
        variants.append(
            ("falseinfo30", corrupt_false_info(ex_tok.clone(), 0.3, by_len, seed=seed)))
    examples = []
    for tag, tk in variants:
        sc, sp = bpe_fn(tk)
        changed = (tk != ex_tok).numpy()
        for b in range(tk.shape[0]):
            vals = sc[b].tolist()[:max_cells]
            spans = sp[b][:max_cells]
            text = _decode(tk[b].tolist(), 10_000)
            labels = [text[s:e] for (s, e) in spans]
            corr = [bool(changed[b, s:e].any()) for (s, e) in spans]
            flg = [bool(v > flag_thr) for v in vals]
            examples.append({
                "which": tag, "idx": b, "flag_thr": flag_thr, "unit": "bpe",
                "clean_text": _decode(ex_tok[b].tolist(), 10_000),
                "text": text,
                "cell_labels": labels,
                "cell_spans": [[int(s), int(e)] for (s, e) in spans],
                score_key: [round(float(v), 4) for v in vals],
                "corrupted": corr,
                "flagged": flg,
            })
            if verbose and b == 0:
                print(f"  --- {tag} #{b} (BPE cells) ---")
                if tag != "clean":
                    print(f"    truec: {''.join('^' if x else ' ' for x in corr)}")
                print(f"    flag : {''.join('*' if x else ' ' for x in flg)}")
    return examples, flag_thr


def heal_style_examples(score_fn, pos_tok, ex_tok, *, K, by_len, score_key,
                        example_fpr=0.05, max_pos=120, seed=7, verbose=True):
    """Produce clean / replace30 / falseinfo30 per-token examples with a
    calibrated flag mask, for ANY detector.

    Parameters
    ----------
    score_fn : callable token_ids(B,L) -> per-token score (B,L), higher == more OOD.
    pos_tok  : clean sequences used ONLY to calibrate the flag threshold.
    ex_tok   : the (few) clean passages to display.
    score_key: JSON key for the per-token score array (e.g. "NLL_t", "SE_t").

    Returns (examples: list[dict], flag_thr: float). Each example dict has:
    which, idx, clean_text, text, <score_key>, corrupted, flagged.
    """
    clean_all = score_fn(pos_tok).numpy().reshape(-1)
    flag_thr = float(np.quantile(clean_all, 1.0 - example_fpr))
    if verbose:
        print(f"  [examples] flag thr={flag_thr:.4f} @ fpr={example_fpr} "
              f"(key={score_key})")
    variants = [
        ("clean", ex_tok.clone()),
        ("replace30", corrupt_token_ids(ex_tok.clone(), vocab_size=K,
                                        corrupt_rate=0.3, seed=seed)),
    ]
    if by_len is not None:
        variants.append(
            ("falseinfo30", corrupt_false_info(ex_tok.clone(), 0.3, by_len, seed=seed)))
    examples = []
    for tag, tk in variants:
        sc = score_fn(tk)
        changed = (tk != ex_tok)
        flagged = sc > flag_thr
        for b in range(tk.shape[0]):
            ch = changed[b][:max_pos]
            fl = flagged[b][:max_pos]
            examples.append({
                "which": tag, "idx": b, "flag_thr": flag_thr, "unit": "char",
                "clean_text": _decode(ex_tok[b].tolist(), max_pos),
                "text": _decode(tk[b].tolist(), max_pos),
                score_key: [round(float(x), 4) for x in sc[b][:max_pos].tolist()],
                "corrupted": [bool(x) for x in ch.tolist()],
                "flagged": [bool(x) for x in fl.tolist()],
            })
            if verbose and b == 0:
                print(f"  --- {tag} #{b} ---")
                if tag != "clean":
                    print(f"    truec: {''.join('^' if x else ' ' for x in ch.tolist())}")
                print(f"    flag : {''.join('*' if x else ' ' for x in fl.tolist())}")
    return examples, flag_thr
