# Heal-from-context proof of concept — insulin (in-distribution protein)

Goal: demonstrate that the frozen full-text8 L256 `DirichletFM` healer can **recover
corrupted tokens using context**, on a real biomedical Wikipedia article. Driver:
`scripts/heal_protein_poc.py`; outputs in `heal_poc_insulin/` (JSON + log + the exact
article extract used). Reuses the `heal_out_best` machinery verbatim (NLL localizer →
threshold calibrated on generic text8 → `inpaint` → `score_healing`).

- **Checkpoint:** `runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM/epoch_best.pt` (epoch 5, best val — the healing checkpoint; see `RESULTS_OOD_HEAL_EPOCH_BEST.md`).
- **Demo set:** 32 windows (L=256) of the insulin article (211 windows total; "insulin" appears ~248×), 143 of which contain the name.
- **Fit/calibration:** generic text8 **test** split (never the article). Operating point `fpr=0.02`, `thr=1.239`, chosen GT-free by max cal-F1.
- **Corruption:** rate 0.15, mean over 3 seeds. Ran on an 8 GB laptop GPU (~17 min).

## Why insulin, not BRCA1

The task started as "heal the BRCA1 article." BRCA1 is a **bad probe**: `brca` occurs
**0×** in the text8 train split (only 4× in validation, in the *Mutation* article's tail),
and text8 spells digits out, so "BRCA1" → "brca one" — a token the model never saw and
**cannot** heal from context. Insulin is in-distribution: **347** train occurrences,
pure-letter name, dense standalone article. (Peers by train count: hemoglobin 101,
collagen 65.)

## Headline: the healer fixes *noise*, not *plausible falsehoods*

| track | corruption | ~%chars | loc_P | loc_R | fix | dmg | **net/corrupt** |
|-------|-----------|--------:|------:|------:|----:|----:|-----------------|
| **A — char noise** | random char substitution (typos/non-words) | 15.1% | 0.785 | 0.787 | 0.581 | 0.037 | **+0.374** ±0.004 |
| **C — false information** | 15% of **words** → different real **same-length** word | 11.3% | 0.513 | **0.196** | **0.040** | 0.023 | **−0.142** ±0.011 |

Same corruption *volume*, opposite result:

- **A works** — +0.374 net recovery, in line with the text8-test baseline (**+0.407**, `epoch_best`). A valid "heals corrupted tokens from context" PoC on a **held-out** biomedical article. Visible denoising: `producxd→produced`, `metabozism→metabolism`, `carbohydratss→carbohydrates`.
- **C fails, and does slight net *harm*** — the false words are lexically valid, so the NLL localizer catches only **20%** (recall 0.20 vs A's 0.79), heals almost none back to truth (fix 0.04), and its few false-positive edits on clean tokens outnumber the fixes ⇒ **net-negative**. The injected falsehoods pass straight through healing:

  ```
  clean : ... is a peptide hormone produced   by beta cells ... encoded in humans ...
  healed: ... of a artwork hormone snapcase by beta cells ... arsenia in humans ...
                    ^^^^^^^        ^^^^^^^^                    ^^^^^^^
  ```
  → **context-healing corrects corruption; it does not detect or correct fluent hallucinations.**

## Track B — whole-word erasure (targeted name)

Scrambling **every** letter of "insulin" (no surviving character evidence), then
localize + inpaint. **0 / 3 windows fully restored.** The localizer flags the damage
(8/10, 5/10, 3/5 positions) but the inpainter fills a *plausible substitute*, not the
exact word:

| window | name chars corrupted | flagged | healed → | restored? |
|-------:|---------------------:|--------:|----------|:---------:|
| 0 | 10 | 8 | `ig vjcn`→`is main`, `ikvedhn`→`isle in` | ✗ |
| 2 | 10 | 5 | `…concentrations of iettvjn in the blood`→`…of between in the blood` | ✗ |
| 3 | 5 | 3 | `low iznhian in the blood`→`low iension in the blood` | ✗ |

So B maps the **boundary**: with all character evidence destroyed, a char-level model
cannot reconstruct a *specific* erased word from context alone — it produces a locally
fluent filler instead.

## Verdict (the claim to make)

> The healer recovers **corrupted** tokens from context + residual character evidence
> (Track A: +0.374 net on held-out insulin, ≈ the +0.407 text8-test baseline). It does
> **not** (B) reconstruct a *fully erased* specific word from context alone, nor (C)
> detect/correct *plausible false information* (net −0.142).

This is the defensible, ML-for-bio-appropriate framing: **denoising ≠ fact-checking**.
Relates to the capstone claim ledger — do not overclaim "heals anything from context."

## Reproduce

```bash
uv run python -u scripts/heal_protein_poc.py \
  --ckpt runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM/epoch_best.pt \
  --article-json heal_poc_insulin/insulin_article_extract.json --name insulin \
  --corrupt-rate 0.15 --n-demo 32 --n-seeds 3 --nfe 100 \
  --out heal_poc_insulin/heal_insulin_poc.json
```
