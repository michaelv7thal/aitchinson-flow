# Aitchison-Flow — continuous flow-matching on the simplex for text8

Capstone research code. Character-level **text8** (K=27 vocab). Studies whether
continuous flow-matching / energy-based models on the probability simplex can do
unconditional generation, conditional recovery, and OOD detection — and *why*
they do or don't. See **[CLAUDE.md](CLAUDE.md)** for the architecture and the
load-bearing implementation details.

## The three objectives (and the honest verdicts)

| # | Objective | Verdict | Where the evidence lives |
|---|---|---|---|
| 1 | Unconditional generation | **Fails** — matches low-order n-gram stats, not words | `EVAL_ASSESSMENT.md` §Obj1; `NOTE_WHY_UNCONDITIONAL_FAILS.md` |
| 2 | Conditional recovery | **Fails (no-op)** — descent returns the corrupted input (Δ@α≈0); only a marginal, non-competitive bump at mid-α (Δ@.50 ≈ +0.06) | `NOTE_EQUILIBRIUM_FAILURE_CLASS.md` §B; `EVAL_ASSESSMENT.md` §Obj2 |
| 3 | OOD detection | **Works** — native EBM energy fails on order, but a full detector benchmark on frozen `DirichletFM` shows: training-free **NLL wins replace/shuffle at the fair (word) unit** (0.981/0.963 vs GPT-2's 0.738/0.746); **false-info localizes to the word** (0.813 — but **GPT-2's NLL beats us**, 0.853); **plausible errors need supervision** — and one **sign-agnostic (quadratic)** head covers *all* corruptions, though specialists still win their own; **healing recovers geometry, not meaning** | **`RESULTS.md` §Objective 3** (`bench_ood/`, `bench_heal/`); `EVAL_ASSESSMENT.md` §Obj3 (SVGP-era, superseded) |

**Start here → [`EVAL_ASSESSMENT.md`](EVAL_ASSESSMENT.md)**: which eval to report
per objective, the headline numbers, *what works / what fails and WHY* (linked to
theory), peer-comparable BPC, and the run triage.

## Canonical documents

| Doc | What |
|---|---|
| [CLAUDE.md](CLAUDE.md) | Architecture, conventions, load-bearing fixes (read before editing EqM) |
| **[EVAL_ASSESSMENT.md](EVAL_ASSESSMENT.md)** | **Eval reference — the meaningful metrics + theory + peer comparability** |
| **[RESULTS.md](RESULTS.md) §Objective 3** | **OOD-detection + healing benchmark** — 3 claims, detector comparison, plausible boundary, real-text transfer, healing (reproducible in `bench_ood/` + `bench_heal/`, each with a manifest + repro patch) |
| [SESSION_OOD_HEALING.md](SESSION_OOD_HEALING.md) | Session findings/decisions log for the Obj-3 benchmark — corrections (in-sample vs held-out, SVGP saturation, **the spilled-energy fix + char-vs-BPE eval protocol**), bugs fixed, open threads |

> **Eval protocol for the external-LM baseline** (see `CLAUDE.md` §"Spilled energy"): our
> detectors score **characters**, GPT-2 scores **BPE tokens**, and the head-to-head happens
> at the **word** level. A BPE score is never attributed *down* onto characters (ill-posed);
> pooling *up* is exact. Character corruption *shatters* GPT-2's tokenization (+70% tokens at
> 15% noise), so words are the only unit stable across both tokenizations. The repo's
> "spilled energy" was a per-token **NLL** until 2026-07-13; the real cross-step ΔE (Minut
> et al., ICLR 2026) now runs as `gpt2_se`, with `gpt2_nll` as the fair localization comparator.
| [CLUSTER_RUNBOOK_L256.md](CLUSTER_RUNBOOK_L256.md) | Hand-off recipe for the L=256 GPU run on the 20 GB cluster |
| [REPORT.md](REPORT.md) | Phase A–H technical report |
| [CAPSTONE_PLAN.md](CAPSTONE_PLAN.md) / [CAPSTONE_EXPERIMENTS.md](CAPSTONE_EXPERIMENTS.md) | Strategy + follow-up tracks |
| [TRAINING_PROTOCOL_v2.md](TRAINING_PROTOCOL_v2.md) | Operational protocol (supersedes v1) |
| [NOTE_WHY_EBM_INIT_STUCK.md](NOTE_WHY_EBM_INIT_STUCK.md) / [NOTE_WHY_UNCONDITIONAL_FAILS.md](NOTE_WHY_UNCONDITIONAL_FAILS.md) | The theory: training- and sampling-time failure mechanisms |
| **[NOTE_EQUILIBRIUM_FAILURE_CLASS.md](NOTE_EQUILIBRIUM_FAILURE_CLASS.md)** | **Theory consolidated**: SFM≠EBM, the corrected single failure mode (recovery also fails), and *which model class* it applies to |
| [EQUILIBRIUM_SETTING.md](EQUILIBRIUM_SETTING.md) | Background: the equilibrium setting, its literature lineage, and why it fails for text (with full sources) |
| [SFLM_EBM_FINDINGS.md](SFLM_EBM_FINDINGS.md) / [DFM_SVGP_FINDINGS.md](DFM_SVGP_FINDINGS.md) / [SAMPLER_FINDINGS.md](SAMPLER_FINDINGS.md) | Per-line findings |
| [RESULTS_LATENT_FINAL.md](RESULTS_LATENT_FINAL.md) | Latent-EqM results (latest) |
| [POSITIONING.md](POSITIONING.md) / [PROPOSAL_COMPOSITIONAL_EQM.md](PROPOSAL_COMPOSITIONAL_EQM.md) / [RESEARCH_FINDINGS.md](RESEARCH_FINDINGS.md) | Framing, the compositional proposal, the BPC-baseline synthesis |

Other root `*.md` (RESULTS*.md, TRAINING_PLAN*.md, RUNBOOK_*, CLUSTER_TRAINING_PLAN.md,
CAPSTONE_SUMMARY.md, SESSION_SUMMARY.md) are **historical** — kept in place because
they are cross-referenced from source comments. `docs/archive/` holds the two that
were safe to move. `runs/DECISION_LOG.md` is the append-only source-of-truth diary.

## Repo layout

```
main.py                     # train EqM (default model); see CLAUDE.md
src/aitchinson_flow/        # package: config, models/, data/, training/, geometry, sampling
scripts/                    # canonical entry points (eval/bench/train) ...
scripts/legacy/             #   ... + archived one-off / diagnostic / broken scripts
sweeps/                     # 8 active sweep configs ...
sweeps/archive/             #   ... + phase-numbered / POC configs
runs/                       # experiment outputs (checkpoints untracked); DECISION_LOG.md
docs/archive/               # superseded planning docs
```

## Key commands (run via `uv run`)

```bash
uv run python main.py                                            # train EqM
uv run python scripts/eval_generation.py --scale a100_20g       # Obj1: KL / H_ratio / BPC
uv run python scripts/recovery_check.py --ckpt <eqm_ckpt>       # Obj2: recovery Δ@α
uv run python scripts/bench_sflm_ebm.py --scale a100_20g --ref-lm gpt2   # Obj3: OOD bench
uv run ruff check && uv run ruff format                          # lint / format
```

The L=256 (publication-length) benchmark runs on the 20 GB cluster — see
[CLUSTER_RUNBOOK_L256.md](CLUSTER_RUNBOOK_L256.md). There is no test suite.
