# Reproducing the paper's results

Every number in the capstone paper traces to a git-tracked artifact in this
repository; `docs/paper-map.md` is the map and `claims.tsv` +
`check_claims.py` make it executable. This file says exactly what a reader
can do at each cost level, and what they cannot.

Tier summary, per claim class:

| Tier | What you do | Cost | Covers |
|---|---|---|---|
| T1 | re-derive every printed number from committed artifacts | < 1 s, CPU | **937 of 943 matrix rows** |
| T2 | re-run an evaluation from a pinned checkpoint | 9.4 GPU-h (whole detection+repair bench) | detection, repair, latent chapters; 12 of 13 generation arms |
| T3 | retrain from scratch | GPU-weeks | everything except the rows listed under "Permanently T1" |

## T1 — check a claim in five minutes (no GPU, no venv)

```bash
git clone <this repo> && cd aitchinson-flow
python3 check_claims.py claims.tsv        # needs only the stdlib
```

Expected: `937 pass, 6 weak, 0 fail, 0 error (943 rows)`. The 6 weak rows are
values whose only sources are markdown documents (`SAMPLER_FINDINGS.md`,
`RESULTS.md`) — they are stated honestly in the row notes and were verified by
hand. To check one specific number, find its row in `claims.tsv` (grep the
paper location, e.g. `results.tex:441`) and open the named artifact at the
named pointer.

Figures: the nine data figures re-render from the committed JSONs on CPU via
`scripts/paper_figs/fig_*.py` (after 2026-08-20 the default data roots are the
correct `bench_*_final` trees; `fig:heal-compare` reproduces byte-identically
modulo the PDF timestamp).

Caveat for non-Python tooling: several sweep artifacts contain bare `NaN`
(the undefined-cell value, kept deliberately). Python and `jq` parse them;
strict RFC-8259 parsers reject them.

## T2 — re-run an evaluation (GPU)

The model of record is
`runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM_converge/epoch_final.pt`
(md5 `9946dce94cb2051de0e288e4caa3ffbd`; verify with
`md5sum` against `bench_ood_final/manifest.json`).

**Copy the argv from the manifests** (`bench_ood_final/manifest.json`,
`bench_heal_final/manifest.json`, per arm), do not trust script defaults from
memory. The manifests also record per-arm wall-clock: the full detection bench
is 2.3 GPU-h and the full repair bench 7.1 GPU-h on an 8 GB laptop GPU
(RTX PRO 1000 Blackwell); the training A100 is not needed for evaluation.
Evaluation code state: every `_final` arm records `code_fp`
`db3232d888a8c6f48727a3e5082f82ec`, which is the hash of the **clean** tree at
commit `12c9ceb` — `git checkout 12c9ceb` provably restores the evaluation
code (the fingerprint covers `scripts/*.py` and `src/**/*.py`; it does not
cover `scripts/paper_figs/` or the `drive_*.sh` drivers).

Environment: `uv sync` against the committed `uv.lock` (Python 3.13,
torch 2.11.0, pinned hashes; the lock predates both benchmark runs and has not
changed since). The manifests written after 2026-08-20 also record
python/torch/CUDA/GPU per run.

### Permanently T1 (checkpoints are gone; the numbers remain committed)

- `tab:gen`'s Hyperspherical-flow (SFLM) row — `epoch_final.pt` was lost; the
  surviving `epoch_10/20.pt` are a different configuration than the one
  evaluated. Do not let `scripts/_ensure_ckpt.py` "re-create" it: that trains
  a new model.
- `tab:gen`'s "EqM, VAE latent" row — the frozen VAE
  (`runs/vae_a100_20g_L256/`, config preserved) lost its weights, and
  `EqMAE.__init__` loads that path at construction.
- `tab:hilbert-null`'s two deterministic-clr reference rows
  (`runs/comp_ref_det_*`) — their `epoch_final.pt` was removed by the cleanup
  loop in `RUNBOOK_COMPOSITIONAL_EQM_TEST.md` §6. **Never run that section.**
- the ten `app:budget` ladder numbers and the Phase-R Langevin sweep — their
  dev-scale checkpoints were never kept.

## T3 — retrain

The paper's backbone trained in four legs (all commands live in the tracked
drivers, in order): `drive_fulltext8.sh` (10 epochs, wall-clock capped, B=48,
~9.8 h/epoch on an A100 80GB MIG 2g.20gb) → `drive_fulltext8_continue.sh`
(resumed at B=48→24 after OOM; its arm directory was later removed — re-run
this leg from `DirichletFM/epoch_best.pt`) → `drive_fulltext8_converge2.sh` +
tail (8-epoch cosine anneal at B=16, lr 5.2e-4→1.5e-4, **no early stop**, so
its `epoch_final.pt` is a genuine annealed final). Seed 42 throughout; the
benchmark arms are `scripts/train_for_sflm_bench.py --scale a100_20g_L256`
(batch 8, 10,000 windows, 10 epochs — NOT the 50-epoch budget an older
runbook describes). Dev-scale cells: `scripts/run_sweep.py` +
`run_matched_recovery.sh` + `run_dsm_rescaled.sh`.

## Data availability

| Input | Source | Pinned? |
|---|---|---|
| text8 corpus | HF `afmck/text8`, fetched at run time (nothing local needed); splits are the standard 90M/5M/5M and windowing is deterministic (non-overlapping length-L from offset 0) | `config.py` pins `revision="58c74e966ccc66eab2de6b52fad5b14ddb259fa4"` (2026-08-20) |
| corpus-derived constants | `runs/text8_unigram_stats.json` (committed) | yes |
| insulin article (repair track, deliberately in-distribution) | committed verbatim at `heal_poc_insulin/insulin_article_extract.json` (MediaWiki page 14895) | text yes; the revid was never captured (the API call omitted `prop=revisions`); retrieval bounded ≤ 2026-07-11 by commit `ebe9fb3` |
| semaglutide article (transfer track, out-of-domain) | committed at `bench_ood_final/transfer/semaglutide_article_extract.json` | yes — revid 1369737560 (2026-08-16); whole-word count in all three text8 splits is 0 (`runs/text8_unigram_stats.json`) |
| GPT-2 baseline | HF `gpt2` at run time | no `revision=` pinned (upstream is stable) |
| `data/`, `data_cache/` | GPT-2/HaluEval caches and DNA windows for retired or unreported tracks | regenerable except the HaluEval cache (see `scripts/legacy/`); backs nothing in the paper |

## Checkpoint durability (unresolved, author decision pending)

The T2-enabling checkpoint set is ~10 GiB (2.07 GiB pinned pair + 12 arm
finals + dev-scale finals) out of ~18 GiB kept in `runs/`. All of it is
excluded from git by `.gitignore` and exists only on the author's machine
(one physical disk). Until an external deposit with checksums exists, treat
every T2 path above as available-locally-only; the T1 path does not depend on
any checkpoint.
