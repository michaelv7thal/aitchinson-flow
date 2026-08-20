# Capstone Experiment Runbook — for an autonomous Claude Code session (A100, 20 GB MIG)


> **Superseded as a plan, 2026-06-06. Useful as a record of what was intended and of the operational policies, which are still accurate. Two of its rules were not followed and one of its P0 deliverables was dropped.**
>
> - **The 3-seed policy did not happen.** §1 requires headline generation, recovery and OOD to be 3-seed with mean ± std. Every number in the paper is single-seed; `chapters/results.tex` says so twice. Do not read any paper table as a 3-seed mean.
> - **The peer-comparable BPC was abandoned.** E1's BPC half and §2 gap 7 aimed at a `DFM.elbo_bpc` number against the SEDD/D3PM frontier. The paper reports no BPC for any of our models.
> - **The SVGP workstream was superseded, not dropped.** P0 items E4a and E4f ran (`runs/dfm_svgp_L256/`, `runs/sflm_bench_a100_20g_L256/DFM_SVGP/`) but the paper's detector is `BayesLinHead`. SVGP appears nowhere in the paper.
> - Never run at all: E4d (ELBO-as-OOD-score), E4e (constant-character and valid-permutation controls), E2d (field-geometry curl probe), E5a at L=256, E6a/E6b (Qwen auditor — retired), X0's physical archival (done differently in the 2026-08-20 cleanup), and E7's `results/RESULTS.md`.
> - E3b's stated prediction is wrong: it expects the healing curve to peak at r = 0.2–0.3; the measured peak is α = 0.6 for all three budgets.
>
> Still correct and worth keeping: the memory-fallback ladder and its rung order, the collapse guard (KL_uni < 0.05 ∧ KL_bi > 1.0, the paper's unigram-collapse signature in operational form), and the sign-inversion guard.
> **Do not reuse §3's X0 legacy list as a deletion list.** It was written three months before the paper and marks families the paper later cashed in.

**Audience:** a Claude Code agent running unattended on the cluster, 24/7, processing
experiments one at a time. **Goal:** by end of week, produce a complete, principled,
reproducible set of results for the DAS capstone (pass/fail), with the publishable
extensions completed only after the pass-critical set is done.

**Repo:** `aitchinson_flow` (model classes under `src/aitchinson_flow/...`, scripts
under `scripts/`). Reference docs already in the repo: `CLUSTER_RUNBOOK_L256.md`,
`CLUSTER_TRAINING_PLAN.md`, `DFM_SVGP_FINDINGS.md`, `NOTE_WHY_EBM_INIT_STUCK.md`,
`NOTE_WHY_UNCONDITIONAL_FAILS.md`, the evaluation-assessment and research-findings
notes. **Trust those for script names and config knobs; this runbook is the
execution plan layered on top.**

---

## 0. How to use this runbook

Work through phases **in order**. Within a phase, run experiments in listed order.
Each experiment has a **tier**:

- **P0 — Pass-critical.** Must complete. These alone constitute a defensible capstone.
- **P1 — Strengthening.** Threats-to-validity and robustness. Do after all P0 done.
- **P2 — Publish-upside.** For the supervisor's paper. Do only after P0+P1 done.

**Stop-and-be-safe rule:** if the queue is ever at risk (time, repeated OOMs), finishing
all **P0** experiments and Phase 7 aggregation is sufficient to pass. Never start a P2
experiment while a P0 experiment is unfinished or failed.

---

## 1. Global policies (apply to every experiment)

**Environment.** Prefix every GPU command with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Run from repo root in the project
venv. Log `git rev-parse HEAD`, the resolved config, GPU name, seed, wall-time, and
`torch.cuda.max_memory_allocated()` for every run.

> **Repo readiness note (this run).** The repo was hardened before this runbook:
> the X1/X2/X3 harness and ALL experiment scripts now **exist** (E2a→`ablate_training_signal.py`,
> E2b→`ablate_sampler.py`, E2d→`probe_field_geometry.py`, E2e→`ablate_hyperparams.py`,
> E4d→`eval_ood_baselines.py`, E5a→`train_logit_kl_flow.py`, E6a→`cache_llm_features.py`,
> E6b→`train_eqm_auditor.py`, E7→`aggregate_results.py`); so "Create scripts/…" below
> means "run the existing script". The memory-fallback ladder, 3-seed support,
> idempotent resume, `--full-split`, **early stopping (`--early-stop-patience`) and
> the 36 h cap (`--max-hours`)**, and the E1b length override (`--length`) are
> **implemented** in `train_for_sflm_bench.py`; the DFM BPC is now a genuine
> variational bound (`DFM.elbo_bpc`); the `<0.5` identity-artifact guard is wired
> into the eval/bench. Only **E5b** (Fisher-Rao/SFM √p) has no dedicated model — it
> runs via the existing `SFLM` arm; a √p-reparameterized model is the contingency.
> See `RESULTS_README.md` and the per-policy notes below.

**Scale (fit 20 GB).** Primary L=256 training scale: **`--scale a100_20g_L256`
= d1024 / 10 layers / 16 heads / batch 8 / L=256** (~127M params; this is what
actually trained DFM / SFLMEBM_FM / the SVGP). `--scale cluster` is **L=40** — do
**not** use it for any L=256 deliverable (E1 etc.). The cheap forward-only families
can use a larger width only if it stays < 18 GB; otherwise stay at the L256 scale.

**Memory fallback ladder — IMPLEMENTED in `train_for_sflm_bench.py`.** On a CUDA
OOM *or* the MIG-restricted NVML allocator assert (matched by message), each arm
auto-escalates: (1) enable gradient checkpointing (`cfg.transformer.grad_checkpointing`,
applied *before* batch-halving since it preserves the effective batch); (2) halve
batch; (3) quarter batch; (4) drop to **L=128** (records `length_fallback=128` in
`train_meta.json`); if it still fails, write `FAILED.json` with the traceback and
**continue the queue**. The Hutchinson `second_order=hutchinson` rung is **not**
implemented — L=128 is the last resort. No OOM ever silently produces a number or
aborts the chain.

**Data (the highest-leverage knob — you have unlimited time, not unlimited memory).**
Use **`--full-split`** to train on the entire `afmck/text8` train split with **lazy**
CLR features (`cfg.text8_dataset.lazy_features`, memory scales with batch not corpus),
or `--max-train-windows N` for a bounded middle ground. More data is the single
biggest lever for closing the BPC gap and costs time, not memory. **Early stopping
on val + the 36 h cap are IMPLEMENTED**: `--early-stop-patience N` (stops after N
val-evals with no improvement and restores the best checkpoint as `epoch_final.pt`;
implies `--val-eval`) and `--max-hours 36`. Do **not** increase model size to chase BPC.

**Seeds.** `--seeds 42,43,44` (implemented). Report mean ± std. Seed 42 → canonical
`runs/sflm_bench_<scale>/<arm>/`; 43/44 → `<arm>/seed<seed>/`. (Mechanism ablations
may use 1 seed if noted; headline generation/recovery/OOD must be 3-seed.)

**Standard eval protocol (peer comparability).** Standard split (last 5M = test);
context **L=256**; **BPC reported only via a real bound** — `DFM.elbo_bpc` (`n_mc=8`),
the D3PM uniform-process variational bound. Its honest peer is **D3PM-uniform ≈1.61**;
SEDD 1.32 / D3PM-absorb 1.45 / MDLM ≤1.38 / SFM 1.39 are absorbing/score methods →
**reference only, not a head-to-head**. For EqM/EqMLatent/SFLM the `bpd()` identity-path
number is invalid → `generation_metric_valid=False`, emit `—`. The `<0.5` guard is
**implemented** (bench + eval_generation + eval_all): any non-finite or `<0.5` text8
BPC is refused as the identity/near-clean artifact (this also correctly demotes
DirichletFM's ~0.36).

**Single eval module (USE — see X1, now built as `scripts/eval_all.py`).** Every arm is scored by the *same* code
computing: `KL_uni, KL_bi, KL_tri, H_ratio`, per-position entropy, BPC-or-`—`, 64
samples, and (for OOD arms) the full AUROC table. This guarantees cross-arm
comparability and prevents metric drift.

**Manifest (USE — see X2, now built as `scripts/manifest.py` + `scripts/run_experiment.py`).** A single `results/manifest.jsonl`. Every experiment
appends one record on completion: `{exp_id, status, run_dir, seeds, git_sha,
config_hash, gpu, wall_time_s, peak_mem_gb, length_fallback, second_order, metrics{...},
artifacts[...], notes}`. **Idempotency:** before running any experiment, check the
manifest; if a completed record with the same `exp_id` exists, **skip it** (so the queue
resumes cleanly after any interruption). Write the record the moment a run finishes — do
not batch, so a later crash never loses earlier results.

**Failure handling.** On any non-OOM failure: log the traceback, write a `failed` record,
**continue to the next experiment** (never block the queue). Retry a failed run **once**
with batch halved before marking it failed.

**Sanity guards (catch silent badness).**
- *Collapse guard:* if an arm has `KL_uni < 0.05` **and** `KL_bi > 1.0`, flag
  `collapsed=true` (this is the unigram-collapse signature, not quality).
- *Sign-inversion guard:* for every OOD score, record both `AUROC` and `|AUROC−0.5|+0.5`,
  and set `sign_inverted=true` if raw AUROC < 0.45. (Native EBM energies invert.)
- *Smoke first:* before any full run of a **new or modified** script, run it at
  `--quick` / tiny scale (≤2 min) and confirm it writes a well-formed manifest record.

---

## 2. Gaps this runbook closes (never run, or never run at L=256)

1. Generative-likelihood OOD baseline (PF-ODE / DFM-ELBO as an OOD score) — the
   "did you just try the density?" question. **(P0, E4d)**
2. Constant-character / low-complexity OOD control (Kirichenko probe). **(P1, E4e)**
3. Valid-permutation OOD control (coherence vs n-gram statistics). **(P1, E4e)**
4. Clean **sampler-only** ablation on one fixed checkpoint across {NAG, Euler-det, SDE,
   annealed Langevin}. **(P0, E2b)**
5. Full **training-signal-class matrix** ({FM-velocity, x1-point-L2, CE/KL-distributional,
   DSM} × {Det-CLR, Dirichlet}) under one shared backbone. **(P0, E2a)**
6. L=256 **DFM-SVGP** (the empty bench rows) and the **4-arm hinge-vs-FM** ablation.
   **(P0, E4a / E4f)**
7. **L=256 DFM BPC** vs the published frontier. **(P0, E1)**
8. **Field-geometry probe** (curl fraction; `cos(g,g*)` across γ). **(P1, E2d)**
9. **Logit-KL Flow** (the one untested lever that may close the generation gap).
   **(P2, E5a)**
10. **LLM-auditor parity** check. **(P2, E6)**

---

## 3. Phase 0 — Repo hygiene & infrastructure (do first; ~half a day, mostly fast)

### X0 — Inventory & archive (cleanup)
Snapshot the current `runs/` tree to `results/runs_inventory_pre.json` (path, config,
key metric if present). Then, per the triage in the evaluation-assessment note:
- Move **Archive (legacy)** runs to `runs/_archive/` (do **not** delete): the GPT-2
  auditor line (`aud_gpt2_*`, `dfm_auditor_*`, `hilbert_uq_*`, `hal_gpt2_*`), early
  phases (`dphase*`, `hilbert_*`, `dsm_vs_eqm_poc`), ablations (`data_*`, `tc_*`,
  `bb_*`, `latent_d*`), smoke/poc (`*_smoke`, `*_poc`, `capstone/`), alt-continuous
  (`fmclr_*`, `lkflow_*` — except any you are about to regenerate), `ng_*`,
  `baseline_*`, `ep*_default`.
- Keep **Failed/incomplete** runs that are *evidence* (e.g. `sflm_bench_a100_20g_L256`
  crash) in place with a `README` noting "kept as 20 GB-limit evidence."
- Move one-off / phase-specific scripts and the GPT-2 auditor line to `scripts/legacy/`.
Write `results/ARCHIVE_MANIFEST.json` listing everything moved and why.

### X1 — Build the unified eval module
Create `scripts/eval_all.py` (or refactor `eval_generation.py`) exposing
`evaluate(ckpt, model_kind, split='test', n_samples=64) -> dict` that returns the full
metric suite (§1 "Standard eval"), with the `generation_metric_valid` guard and the
collapse/sign-inversion flags. All later phases call this — **no ad-hoc metric code
elsewhere.** Smoke-test on the existing `dfm_svgp_pure50_lr3e4` checkpoint and confirm
it reproduces KL_bi ≈ 0.45.

### X2 — Build the manifest + runner harness
Create `scripts/manifest.py` (append/lookup/skip-if-done) and a thin
`scripts/run_experiment.py <exp_id>` wrapper that: looks up `exp_id`, skips if done,
runs the experiment, captures wall-time/peak-mem/git-sha, applies the failure/retry
logic, and writes the record. The 24/7 driver is just a loop over the registry in §4
order calling `run_experiment.py`.

### X3 — README mapping
Write `RESULTS_README.md`: a table mapping **objective → canonical script → output
artifact → manifest exp_id**, plus the "how to resume" note (idempotent re-run).

---

## 4. Experiment registry (the DAG)

Run in this order. `time` is a rough wall-clock estimate per run *before* the 3-seed
multiplier; multiply ×3 for headline rows. "Dep" = must finish first.

| ID | Tier | Phase | Name | Dep | Est. time |
|---|---|---|---|---|---|
| X0–X3 | P0 | 0 | Cleanup + eval module + manifest + README | — | ~0.5 day |
| **E1** | **P0** | 1 | Generation @ L=256, all families, 3 seeds, full eval + DFM BPC | X1 | 6–36 h ea |
| **E2a** | **P0** | 2 | Training-signal-class matrix (4 targets × 2 data recipes) | E1 | 6–24 h ea |
| **E2b** | **P0** | 2 | Sampler-only ablation on one fixed EqM checkpoint | E1 | 1–3 h |
| **E2c** | **P0** | 2 | Collapse quantification (n-gram KL + per-pos entropy, all arms) | E1 | folded into eval |
| E2d | P1 | 2 | Field-geometry probe (curl fraction; cos(g,g*) vs γ) | E1 | 1–3 h |
| **E3a** | **P0** | 3 | Recovery sweep Δ@α, recipes + DFM, 3 seeds | E1 | 1–3 h ea |
| **E3b** | **P0** | 3 | Healing curve (recovery vs corruption rate) | E1 | 1–2 h |
| **E4a** | **P0** | 4 | DFM-SVGP hinge @ L=256 + corruption ladder AUROC | E1(DFM) | 0.5–2 h |
| **E4b** | **P0** | 4 | Native-energy OOD, all EBM families @ L=256 | E1(EBMs) | 0.5–2 h ea |
| **E4c** | **P0** | 4 | DFM denoiser-NLL OOD | E1(DFM) | 0.5–1 h |
| **E4d** | **P0** | 4 | Generative-likelihood OOD baseline (DFM-ELBO ranking) | E1(DFM) | 1–2 h |
| **E4f** | **P0** | 4 | 4-arm hinge-vs-FM ablation @ L=256 | E1(DFM) | 2–6 h |
| **E4g** | **P0** | 4 | External baseline: GPT-2 spilled-energy reference | — | 0.5–1 h |
| E4e | P1 | 4 | OOD controls: constant-char + valid-permutation | E4a–c | 0.5–1 h |
| E1b | P1 | 1 | Sequence-length sweep (L∈{40,128,256}) for one family | E1 | 6–24 h ea |
| E2e | P1 | 2 | Hyperparam ablations: smoothing ε; OOD feature t | E1 | 0.5–2 h ea |
| **E5a** | **P2** | 5 | Logit-KL Flow Matching baseline @ L=256 | E1 | 6–36 h |
| E5b | P2 | 5 | Fisher-Rao / SFM √p reparameterization (contingency) | E5a | 6–36 h |
| E6a | P2 | 6 | Cache Qwen2.5-1.5B (4-bit) logits + hidden states | — | 2–6 h |
| E6b | P2 | 6 | EqM-energy auditor parity vs SVGP + spilled energy (WikiText-2) | E6a | 1–3 days |
| **E7** | **P0** | 7 | Aggregate all tables + figures + provenance | all P0 | ~1 h |

---

## 5. Phase 1 — Canonical generation @ L=256 (P0)

### E1 — Train every family at shared scale, 3 seeds, full eval
Use `scripts/train_for_sflm_bench.py --scale a100_20g_L256 --seeds 42,43,44`
(**L=256**; `--scale cluster` is L=40 — wrong for this phase) as the shared-scale
trainer where it supports the family; for families it doesn't cover, train via the
existing per-family path with each family's **known-good recipe**. Add `--full-split`
to close the data gap. Required arms:

| Arm | Recipe notes | BPC |
|---|---|---|
| **DFM** | the working generator; compute the **variational ELBO BPC** (`DFM.elbo_bpc`, `n_mc=8`); peer = D3PM-uniform ≈1.61 | **report** |
| EqM (CLR) | tuned recipe: `cfg.eqm.gradient_lambda=3.0`, `dirichlet_sampling=True` (per `dphase4_lambda_recalib`); sampler per its default | `—` (invalid) |
| EqM-latent | the AE-latent variant (`d128`, tied embed, euler) | `—` |
| FMonCLR | naive linear FM on CLR — the control | `—` (compute DFM-style ELBO only if a valid bound exists; else `—`) |
| SFLM | time-conditioned hyperspherical (`models/sflm.py`) | `—` unless a valid bound is implemented |
| SFLMEBM | hyperspherical EBM (`models/sflm_ebm.py`) | `—` |
| DSM | denoising score matching on CLR (the escape-class contrast) | report if a valid bound exists |

For each arm × 3 seeds: run `eval_all.py`, save 64 samples, write manifest. Apply the
data policy (full train split, early stopping, 36 h cap) and the memory fallback ladder
for the second-order arms.

**Success:** every arm has a 3-seed metric record; DFM has a real BPC; EBM arms show
`—` for BPC; the collapse guard fires on EqM/FMonCLR and not on DFM (this is itself a
result). **Deliverable:** the master generation table overlaid on the published frontier
(SEDD 1.32, SFM 1.39, D3PM-abs 1.45; AR 1.13–1.18; Plaid 1.12).

### E1b (P1) — Sequence-length sweep
For one family (DFM and EqM), train at L∈{40,128,256} at matched step count via the
`--length` override (e.g. `--scale a100_20g_L256 --length 128`; output dir gets an
`_L128` suffix so cells don't collide) to reproduce and quantify the length effect
(expected: KL_bi improves with L). Documents a threat to
validity (short-context confound) directly.

---

## 6. Phase 2 — The causal mechanism (P0; this is what makes the negative result airtight)

### E2a — Training-signal-class matrix (the money experiment)
Run `scripts/ablate_training_signal.py` (now built; `dsm_clr_ablation` is a
`runs/` output dir, not a script). **Fix backbone, data, and sampler**; vary only
the **training target** × **data recipe**:

Targets: `{FM-velocity-L2, x1-point-L2, CE/KL-distributional-x1-on-simplex, DSM-epsilon}`.
Data recipe: `{Det-CLR, Dirichlet}`.

This is an 8-cell grid (1 seed per cell acceptable for the mechanism; 3 seeds on the two
diagonal cells that anchor the claim). Report `KL_uni, KL_bi, Δ@.50, collapsed` per cell.
**Predicted pattern:** FM-velocity and x1-point-L2 collapse (low KL_uni, high KL_bi,
Δ=0); CE/KL-distributional and DSM escape; Dirichlet thickening rescues FM *recovery*
(Δ) but not generation. This isolates **distributional-vs-point prediction** as the
mechanism — the cleanest single contribution.

### E2b — Sampler-only ablation
Create `scripts/ablate_sampler.py`. Take **one fixed trained EqM checkpoint** and vary
**only the sampler**: `{NAG-GD, Euler-det (use_grad=True and use_grad=False), SDE/Langevin
(cfg.eqm.sampler="sde"), annealed Langevin}`. Report generation metrics for each.
**Predicted:** NAG ≈ Euler(use_grad=True) ≈ 1.38; Euler-on-raw-f plateaus ≈1.68; SDE
helps somewhat but does not reach DFM. Confirms **the sampler is not the lever** and
closes the "you just needed a better sampler" objection.

### E2c — Collapse quantification
No separate run — ensure `eval_all.py` emits `KL_uni/bi/tri`, per-position entropy, and
the `collapsed` flag for **every** arm from E1/E2a. Produces the standardized collapse
table.

### E2d (P1) — Field-geometry probe
Create `scripts/probe_field_geometry.py`. For a collapsing arm (EqM/FMonCLR) and a
working arm (DFM, treating its implied velocity), estimate at γ∈{0.1,0.25,0.5,0.75,1.0}:
(a) the **rotational/curl fraction** of the learned field via a Helmholtz-style
decomposition or a finite-difference antisymmetric-Jacobian estimate (Hutchinson, no
second-order needed); (b) `cos(g_θ, g*)` against the analytic target. **Predicted:**
large curl fraction / low cos at the data basin for the collapsing arm; flat-field
signature near γ→1. Quantifies the conservative-vs-not and flat-basin geometry.

---

## 7. Phase 3 — Conditional recovery (P0; the positive result)

### E3a — Recovery sweep Δ@α
`scripts/recovery_check.py` (mode B), α∈{0.1,0.3,0.5,0.7,1.0}, n≥256, 3 seeds, for:
compositional-MSE, compositional-Hilbert, deterministic-CLR (ref), and DFM. Headline =
**Δ@.50** + the per-α curve. **Lead with Δ, not KL** (KL rewards the deterministic ref's
unigram overfit). **Predicted:** compositional Δ@.50 ≈ +0.06; deterministic ≈ +0.00.

### E3b — Healing curve
`scripts/eval_healing.py`: recovery vs corruption rate r∈{0.1,…,1.0} for the best
recipe. Produces the "local structure exists only in a mid-corruption band" figure
(expected peak around r=0.2–0.3). Bridges generation and per-token OOD.

---

## 8. Phase 4 — OOD detection (P0; the discriminative-vs-native result + missing baselines)

Use one **fixed held-out eval set** for all OOD experiments (same clean windows; same
corruption seeds) so AUROC is comparable across detectors. Corruption types:
`substitution`, `shuffle` (histogram-preserving — **the discriminating axis**),
`random-simplex`. Report AUROC at r∈{0.1,0.3,0.5,0.7,1.0}. **Headline the shuffle axis.**

### E4a — DFM-SVGP hinge @ L=256 (fills the empty bench rows)
`scripts/fit_dfm_svgp_hinge.py --ckpt <L256 DirichletFMSvgp Stage-1>` (e.g.
`runs/dfm_svgp_L256/epoch_final.pt` — **NOT** the §5 `DFM` arm, which is a
different class with no SVGP/pooler) then
`scripts/sweep_dfm_svgp_corruption.py --ckpt <…with_svgp> --n 500`. Knobs from
`DFM_SVGP_FINDINGS.md`: `pooling=mean, d_embed=256, n_inducing=128, kernel=matern52,
t_eval≈4.5, margin_energy=2.0`. **Report both the trained-mean (`prob`) AUROC and the
predictive-variance AUROC**, and state plainly that the variance is uninformative
(concentration of measure) — the signal is in the mean. **Predicted:** shuffle ≈0.93,
random ≈1.0; variance AUROC ≈0.50.

### E4b — Native-energy OOD, all EBM families
`scripts/eval_ood.py` for EqM, EqMLatent, SFLMEBM: sequence-level `E(x)`, per-position
`‖∇E‖`, per-position uncertainty `U_pos_mean`. **Report AUROC and `sign_inverted`.**
**Predicted:** shuffle ≈0.50 (chance); raw energy frequently sign-inverted on random;
`U_pos_mean` good on substitution (~0.99) but failing on shuffle (~0.52).

### E4c — DFM denoiser-NLL OOD
`scripts/eval_ood.py` for the DFM denoiser proxy `−log p_{1|t≈1}(x|x)`. **Predicted:**
dominates sequence-level (rand/subst/shuffle ≈ 1.00/1.00/0.99). **This is the corrected
finding — DFM *does* produce a strong sequence-level score** (supersedes the earlier
"DFM cannot" claim).

### E4d — Generative-likelihood OOD baseline (the missing baseline)
Create `scripts/eval_ood_baselines.py`. Use the **DFM MC-ELBO as a density/OOD score**
(rank inputs by estimated log-likelihood; lower → OOD). Report AUROC on all corruption
types. **Predicted (Nalisnick):** unreliable relative to the discriminative head/denoiser
proxy — possibly near chance or sign-inconsistent. This is what **justifies the
discriminative pivot** and closes the "did you try the actual density?" question.

### E4e (P1) — OOD controls
In the same `eval_ood_baselines.py`: add (i) **constant-character / low-complexity**
sequences (`"aaaa…"`, all-spaces) and check no detector ranks them as in-distribution
(Kirichenko low-complexity probe); (ii) a **valid-permutation control** (structurally
plausible reorderings) and check whether any detector flags it — separates "captures
coherence" from "is a fancy n-gram classifier."

### E4f — 4-arm hinge-vs-FM ablation
`scripts/ablate_hinge_vs_fm.py` at L=256: **A** FM+hinge, **B** random-features+hinge,
**C** end-to-end-hinge (no FM pretraining), **D** FM+linear-probe. Train all on
*replace-only* negatives; evaluate replace→shuffle **transfer**. **Predicted:** A ≈ D ≫ B
on shuffle; C fails the replace→shuffle transfer. Establishes that **the FM
representation (not the OOD head) carries the cross-corruption signal** — the honest
justification for FM-first over a direct discriminator.

### E4g — External baseline
Recompute / surface the **GPT-2 spilled-energy** reference (the bench's `gpt2_baseline`).
**Correct the citation in all outputs to arXiv:2602.18671** (Minut, Dewidar, Masi, ICLR
2026). Frame honestly: small from-scratch FM **with** outlier exposure vs zero-shot LLM
method — practical dominance, not a controlled method-vs-method claim.

---

## 9. Phase 5 — The geometry lever (P2; publishable upside)

### E5a — Logit-KL Flow Matching baseline
Use the existing `src/aitchinson_flow/models/logitkl_flow.py` (registered `LogitKLFlow`,
config `cfg.logitkl`) via `scripts/train_logit_kl_flow.py` (now built). Recipe (Sevriugov
& Oseledets, arXiv:2411.16821): token→logit vector; **regress clean logits `l_1`** (not
velocity); **deterministic ODE for t<0.28, stochastic re-noising for t≥0.28**. Train at
L=256, full eval + BPC. **This is the one untested lever that may close the generation
gap and is the strongest positive result for the paper.** If it reaches DFM-level KL_bi,
that is the headline upside; if not, it is still a valuable additional negative.

### E5b — Fisher-Rao / SFM (contingency)
Only if E5a does not close the gap. The repo's `SFLM` arm (`models/sflm.py`,
time-conditioned hyperspherical flow) is the closest existing model — run it via
`train_for_sflm_bench.py --only SFLM` first. A dedicated `√p`-sphere
reparameterization with a Riemannian geodesic sampler (SFM, arXiv:2405.16441) is
a **new model class** and remains the contingency-only build (not yet implemented,
since it is gated on E5a's result). Direct text8 BPC target to beat: 1.39.

---

## 10. Phase 6 — LLM-auditor parity (P2; clearly beyond pass)

### E6a — Cache LLM features
`scripts/cache_llm_features.py`: Qwen2.5-1.5B in 4-bit (bnb-nf4), precompute
`(token_ids, top-k logits, last-hidden-state)` on a WikiText-2 subset, memory-map to
disk. **The LLM is then never loaded during training** → fits 20 GB trivially.

### E6b — EqM-energy auditor parity
Run `scripts/train_eqm_auditor.py --cache <E6a output>` (now built): trains the
EqM-style energy auditor on the cached features (fuses detached `h_LLM` with a token
embedding; contrastive hinge on valid/corrupted pairs) and reports per-token and
per-sequence AUROC **alongside a training-free spilled-energy baseline** read off the
cached top-k logits, so the two are directly comparable. **Parity targets** vs spilled
energy on WikiText-2 corruption: seq AUROC ≥0.99, tok AUROC ≥0.97.
**This is a parity check, not a new SOTA claim** — its value is "EqM energy is a drop-in
for the SVGP head and is *also* a generator." Run only if all P0+P1 are done.

---

## 11. Phase 7 — Aggregation & report-ready outputs (P0; do at the end)

### E7 — Aggregate
Run `scripts/aggregate_results.py` (now built) reading `results/manifest.jsonl` and emitting
`results/RESULTS.md` + `results/figs/`:
- **Table 1 — Generation @ L=256** (KL_uni/bi/tri, H_ratio, BPC-or-`—`, sample) overlaid
  on the published frontier.
- **Table 2 — Training-signal-class matrix** (the mechanism).
- **Table 3 — Recovery** (Δ@α curve + headline Δ@.50).
- **Table 4 — OOD** (every detector × corruption type; shuffle axis highlighted; variance
  AUROC shown as uninformative; likelihood baseline shown as unreliable).
- **Fig 1** corruption-ladder AUROC; **Fig 2** healing curve; **Fig 3** field-geometry
  (curl/cos vs γ); **Fig 4** BPC vs frontier.
- Every cell annotated with **provenance**: `exp_id`, run_dir, seeds, git_sha. Any
  `collapsed`/`sign_inverted`/`length_fallback` flags surfaced in footnotes.

---

## 12. Definition of done (acceptance criteria)

**Pass-critical (all must be true):**
- [x] Infra in place: eval module (`scripts/eval_all.py`), manifest harness
      (`scripts/manifest.py` + `scripts/run_experiment.py`), `RESULTS_README.md`, and
      the E2a/E2b/E2d/E4d/E5a/E6a/E7 scripts all **exist** (built during repo hardening);
      memory-fallback ladder + 3-seed + idempotent resume + `--full-split` implemented.
      (Legacy-run archival is the one remaining cleanup item.)
- [ ] E1 complete: all families trained @ L=256, 3 seeds, full eval; **DFM has a real
      variational ELBO BPC** (`DFM.elbo_bpc`; peer D3PM-uniform ≈1.61); EBM/identity-path
      BPC shown as `—` (`<0.5` guard active); collapse flags recorded.
- [ ] E2a + E2b + E2c complete: the training-signal-class matrix and sampler-only
      ablation both run, with the predicted collapse/escape pattern documented (or any
      deviation explained).
- [ ] E3a + E3b complete: Δ@α recovery sweep (3 seeds) + healing curve.
- [ ] E4a–E4d, E4f, E4g complete: DFM-SVGP @ L=256, native-energy OOD, DFM denoiser-NLL,
      **generative-likelihood baseline**, 4-arm hinge-vs-FM, GPT-2 reference — all on the
      shared eval set, shuffle axis reported.
- [ ] E7 complete: `RESULTS.md` with all four tables + four figures, every number with
      provenance.

**Strengthening (do after pass-critical):** E1b, E2d, E2e, E4e.

**Publish-upside (do only after P0+P1):** E5a, (E5b), E6a, E6b.

---

## 13. Notes for the agent

- The **negative results are the contribution** — do not "fix" a collapse to make a
  table look better; document it. A clean, flagged collapse is a positive deliverable.
- Prefer the **Hutchinson divergence-trace** over exact second-order whenever memory is
  tight; it generalizes across families and avoids OOM.
- If a P2 experiment threatens the queue, **drop it** — P0 + E7 is a complete, passing
  capstone on its own.
- Keep `results/manifest.jsonl` append-only and idempotent; it is the source of truth and
  the resume point after any interruption.
