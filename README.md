# Aitchison-Flow — flow matching on the simplex for character-level text

Experiments repository for the capstone paper *Flow Matching on the Simplex
for Character-Level Text: Generation, Healing, and the Limits of the
Equilibrium Route* (the sibling `capstone-paper` repository). Character-level
**text8** (K=27). Nine budget-matched generative arms, a frozen-backbone
detector suite, a localize-then-inpaint repair loop, and the Equilibrium
Matching negative result.

**Start here → [`REPRODUCE.md`](REPRODUCE.md)** (what you can check at which
cost) and **[`docs/paper-map.md`](docs/paper-map.md)** (every table and figure
→ the artifact behind it).

## Check a claim in five minutes

```bash
python3 check_claims.py claims.tsv     # stdlib only, <1 s
```

`claims.tsv` maps every number the paper prints to a tracked artifact and a
JSON pointer; the checker re-reads each one. Expected:
`939 pass, 6 weak, 0 fail` — the 6 weak rows name markdown sources honestly.

## The three objectives, as the paper concludes them

| # | Objective | Verdict | Evidence |
|---|---|---|---|
| 1 | Unconditional generation | **Dirichlet Flow Matching works**: the one arm of nine that moves past unigram statistics; retrained on the full corpus it generates mostly word-like English at KL_bi 0.130 (`tab:gen`). **Equilibrium Matching fails structurally** — no variant generates, and no budget, loss, geometry, sampler, or conditioning we varied repairs it (the paper's negative result). | `runs/sflm_bench_a100_20g_L256/*/eval_all.json`; negative result: `runs/compu_*`, `runs/dsmx_*` |
| 2 | Conditional recovery | **Transport recovery works**: the selected model recovers +0.282 token accuracy beyond the perturbed input at α=0.5 (`tab:recovery`). **EqM recovery is a no-op** (Δ between −0.009 and +0.005): generation and recovery are one failure, not two. | `runs/.../recovery_fine/`; `runs/sflm_bench_a100_20g_L256/EqM*/recovery.json` |
| 3 | Detection + repair | **Works, on the frozen Dirichlet FM backbone.** The training-free denoiser NLL is the strongest localizer (word AUROC 0.981 replace / 0.967 shuffle vs GPT-2 ≤ 0.75); false information is a sequence-triage signal (the supervised head leads slightly, 0.812 vs 0.807); repair recovers geometry (+0.432 net on replace) and never fixes fluent falsehoods; the transfer verifier holds 0.882 vs GPT-2's 0.847 on an article provably absent from training. The EqM energy contributed nothing: the free OOD score was never there to read. | **`bench_ood_final/`**, **`bench_heal_final/`**, `heal_poc_insulin_final/` (manifests pin the checkpoint by md5) |

The evidence trees of record are `bench_ood_final/` and `bench_heal_final/`
(checkpoint `DirichletFM_converge/epoch_final.pt`, md5 `9946dce9…`). The
unsuffixed `bench_ood/` and `bench_heal/` are the superseded epoch_best
generation, kept for the record — their banner says how to tell the eras
apart at a glance.

## Paper chapter → artifacts

| Paper chapter | Backed by |
|---|---|
| §4.2 Generation, recovery, latent space | `runs/sflm_bench_a100_20g_L256/`, `runs/sflm_bench_a100_20g_L256_d1280L14_full/`, `bench_ood_final/latent_split_*` |
| §4.3 The negative result | `runs/compu_*`, `runs/dsmx_*`, `runs/band_*`, `runs/comp_*` + `RUNBOOK_COMPOSITIONAL_EQM_TEST.md` |
| §4.4 OOD detection | `bench_ood_final/` (9 manifest arms + `manifest_addenda.json` extras incl. `transfer/`, `tsweep/`) |
| §4.5 Repair | `bench_heal_final/`, `heal_poc_insulin_final/` (+ the input article in `heal_poc_insulin/`) |
| Appendix theory & budget | `runs/compu_mse_det/` probes, the `runs/{baseline_5ep,ep25_default,data_*,bb_*}` ladder, `runs/tc_*`, `runs/text8_unigram_stats.json` |
| Appendix config (`tab:config`) | `train_meta.json` files, both manifests, `src/aitchinson_flow/config.py`, `runs/vae_a100_20g_L256/config.json` |

## Canonical documents

| Doc | Status |
|---|---|
| [REPRODUCE.md](REPRODUCE.md) | **the entry point**: per-tier commands, costs, data availability |
| [docs/paper-map.md](docs/paper-map.md) | claim → artifact map, naming traps, value decoys |
| [CLAUDE.md](CLAUDE.md) | architecture + load-bearing implementation notes (read before editing EqM) |
| [bench_ood_final/RESULTS.md](bench_ood_final/RESULTS.md) | the detector benchmark writeup that matches the paper cell for cell |
| [NOTE_EQUILIBRIUM_FAILURE_CLASS.md](NOTE_EQUILIBRIUM_FAILURE_CLASS.md) | the negative result's theory (Part A and B.3 are current; see its banner) |
| [EQUILIBRIUM_SETTING.md](EQUILIBRIUM_SETTING.md) | the equilibrium lineage (Hopfield/Boltzmann/DEQ) behind the negative result |
| [docs/bayes_linear_ood.md](docs/bayes_linear_ood.md) | the Bayesian-linear head, derived more fully than in the paper (see its banner for the two-pass correction) |
| [FISHER_FM_PLAN.md](FISHER_FM_PLAN.md) | the ninth arm's plan, discharged as written; the repo's authoritative Cheng-vs-Davis distinction |
| [SAMPLER_FINDINGS.md](SAMPLER_FINDINGS.md) | the sampler diagnosis behind the appendix's numbers |
| [RUNBOOK_COMPOSITIONAL_EQM_TEST.md](RUNBOOK_COMPOSITIONAL_EQM_TEST.md) | reproduction path for `tab:hilbert-null` — **do not run its §6** |

Everything else at the root (`RESULTS*.md`, `EVAL_ASSESSMENT.md`, `REPORT.md`,
`TRAINING_*`, `CAPSTONE_*`, `SESSION_*`, the remaining notes) is **historical**:
written while the results were still moving, each now carries a dated banner
saying what the paper superseded or reversed. They stay in place because code
and configs cite them by name and section. `docs/archive/` holds the ones
nothing pins. `runs/DECISION_LOG.md` is the append-only diary.

> **Eval protocol for the external-LM baseline** (details: `CLAUDE.md` §"Spilled
> energy"): our detectors score **characters**, GPT-2 scores **BPE tokens**, and
> the head-to-head happens at the **word** level. A BPE score is never
> attributed *down* onto characters; pooling *up* is exact. The repo's "spilled
> energy" was a per-token NLL until 2026-07-13; the real cross-step ΔE now runs
> as `gpt2_se`, with `gpt2_nll` as the fair localization comparator.

## Repo layout

```
claims.tsv, check_claims.py # the executable traceability matrix
REPRODUCE.md                # tiers, costs, data availability
main.py                     # train EqM (default model); see CLAUDE.md
src/aitchinson_flow/        # package: config, models/, data/, training/, geometry, sampling
scripts/                    # canonical entry points (eval/bench/train)
scripts/legacy/             # archived one-off / diagnostic scripts
sweeps/ (+ archive/)        # 12 active sweep configs + phase-numbered history
runs/                       # experiment outputs (weights untracked; every JSON tracked)
bench_ood_final/, bench_heal_final/   # the paper's evidence trees (never rename)
docs/ (+ archive/)          # method notes, paper-map, superseded docs
```

## Key commands (run via `uv run`)

```bash
python3 check_claims.py claims.tsv                               # verify the paper, CPU
uv run python scripts/run_bench_ood.py                           # re-run the detection bench (GPU, ~2.3 h)
uv run python scripts/run_bench_heal.py                          # re-run the repair bench (GPU, ~7.1 h)
uv run python scripts/eval_generation.py --scale a100_20g_L256   # generation metrics
uv run ruff check && uv run ruff format                          # lint / format
```

The L=256 training recipe and its four-leg history are in `REPRODUCE.md` §T3.
There is no test suite; `check_claims.py` is the tripwire.
