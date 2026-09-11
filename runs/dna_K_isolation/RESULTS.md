# EqM_OneHot vocab-size (K) isolation: DNA (K=4) vs text8 (K=27)

**Question.** Is the vocabulary size K the reason `EqM_OneHot` produces
unigram-correct but bigram/trigram-broken gibberish on text8 (K=27)? Does it
work on a small-K modality (DNA, K=4)?

**Setup (only K changed).** Identical `EqM_OneHot` recipe to the text8 local
baseline — `dirichlet_sampling=False`, `gradient_lambda=3.0`, `gamma_power=0.5`,
d512/6L/8H, batch 16, **L=40**, 10k train windows, 10 epochs, NAG-GD sampling
(200 steps). DNA = human non-TATA promoters (`genomic-benchmarks`), tokenized
`{A,C,G,T}`→0..3, windowed within-sequence at L=40 (10k/5k/5k), routed through
the *same* `CharWindowDataset`/`CorruptingCollate`/CLR path via `DNADataModule`.
Generation scored by the canonical scorecard (`scripts/eval_all.py`), KL vs each
corpus's **own** n-gram reference.

## Headline (looks like a huge win for small K)

| metric          | text8 K=27 | DNA K=4 |
|-----------------|-----------:|--------:|
| KL_uni          | 0.0693     | 0.0069  |
| **KL_bi**       | **1.5694** | **0.0307** |
| **KL_tri**      | **6.9696** | **0.0685** |
| H_gen / H_gt    | 2.78 / 2.85 | 1.376 / 1.379 |
| H_ratio         | 0.975      | 0.998   |
| per_pos_entropy | 2.311      | 1.302   |
| collapsed       | False      | False   |

DNA sample[0]: `CTTCCATTTGATTTGCGATCCCCTGAGGTGACCAAGTTCC`

## But it's the confound, not K (the decisive diagnostic)

`scripts/diag_structure_content.py` measures how much **sequential structure**
each corpus has and compares the model to a **256-sample i.i.d.-from-marginal**
("no-structure") baseline under the *same* estimator (so finite-sample bias
cancels). 8-seed baselines:

| corpus       | bigram structure present | no-structure floor KL_bi | model KL_bi | verdict |
|--------------|-------------------------:|-------------------------:|------------:|---------|
| text8 (K=27) | 0.485 nats               | 1.673 ± 0.038            | 1.569       | ~2.7σ **below** floor → captured ~20% of structure |
| DNA (K=4)    | 0.019 nats               | 0.0219 ± 0.0026          | 0.0307      | ~3σ **above** floor → captured **none** |

(trigram: text8 structure 1.344 nats, floor 7.196 ± 0.112, model 6.970 → just
below; DNA structure 0.051 nats, floor 0.0616 ± 0.0062, model 0.0685 → at/above.)

Two confounds fully explain the headline drop:
1. **Data structure.** DNA promoters have ~25× less bigram and ~26× less
   trigram structure than English text8 (adjacent bases are near-independent;
   unigram entropy is 99.4% of max ln4). There is almost nothing to capture.
2. **Metric sensitivity vs K.** At a fixed 256 samples the n-gram-KL noise floor
   is ~1.67 at K=27 (729 bigram bins, severe undersampling) vs ~0.022 at K=4 (16
   bins). The same "match marginals, capture no structure" behaviour scores a
   tiny KL at K=4 and a huge KL at K=27, automatically.

## Verdict

**K is not the bottleneck.** `EqM_OneHot`'s structure-capturing ability is *not*
better at K=4 — measured against the right no-structure baseline it does
marginally **better at K=27** (where structure exists) and **nothing at K=4**.
The failure mode (unigram-correct, sequentially-unstructured generation) is the
same at both K; small K merely **masks** it because the data is near-independent
and the KL metric is far less sensitive there. The dramatic 1.57→0.031 drop is
an artifact, not a cure.

To actually isolate K you need the structure-matched control that was deferred:
a synthetic K-sweep (e.g. K∈{4,8,16,27}) from a Markov source at **matched
per-token conditional entropy** — vary K with sequential structure held fixed.

## Repro
```
uv run python scripts/prep_dna_windows.py --L 40 --out data_cache/dna/promoters_L40.pt
uv run python scripts/exp_dna_eqm_onehot.py --windows-path data_cache/dna/promoters_L40.pt \
    --out-dir runs/dna_K_isolation/EqM_OneHot_dna --epochs 10
uv run python scripts/diag_structure_content.py
```
Artifacts: `EqM_OneHot_dna/{epoch_final.pt,eval_all.json}`, `data_cache/dna/promoters_L40.pt`.
