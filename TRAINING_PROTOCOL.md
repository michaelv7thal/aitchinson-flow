# Training Protocol — Aitchison-Flow capstone (multi-session execution)


> **HISTORICAL — Phases A–J, executed 2026-05-07 onward; A resolved, B and F–H superseded or retired. Not an instruction set. Read for the record, not the plan.**
>
> Its **Phase A verdict table (§6) is still accurate and load-bearing**: W1 sampler swap negative (Euler 1.682 vs NAG 1.382), W4 FMonCLR 1.559 near EqM rather than near Discrete FM's 0.148, W3 EqM sequence energy at chance. Those findings are what the paper's negative result rests on.
>
> **Phase B (Logit-KL Flow) landed and is in the paper** as the third family of the noise-injection sweep in the appendix §Sampling-time failure; the authoritative configuration is `LogitKLFlowConfig` in `src/aitchinson_flow/config.py`, not the YAML sketched here.
>
> **Phases F/G/H (the GPT-2/Qwen auditor track) are retired** — the trained auditor added nothing beyond a linear probe or spilled energy (see `CLAUDE.md`). The paper contains no auditor, no WikiText-2 and no TriviaQA. This document is nevertheless the only specification of what `runs/aud_gpt2_*` and `runs/dfm_auditor_*` contain.
>
> **Phase E's peer-comparable-BPC goal was dropped.** The final paper reports no BPC for any of our own models; `tab:config` marks `bpd()` "diagnostic only".
>
> Dead pointers: the boot command `cd /home/renku/work/aitchinson-flow` is a stale cluster path; `sweeps/phaseB_logitkl.yaml` moved to `sweeps/archive/`; `sweeps/phase{D_loss,G_auditor_trivia,H_auditor_gen,I_sfm,J_longrun}.yaml`, `src/aitchinson_flow/data/trivia.py`, `scripts/generate_audited.py`, `scripts/eval_auditor_wiki.py`, `runs/best_{so_far,lkflow,auditor}.pt` and `data/wiki_cache_gpt2.pt` were never created or no longer exist.

> **Purpose.** Lead document for the executing claude session(s). Specifies
> *what* to train, *in what order*, *with what configs*, *what to watch*, and
> *when to stop or pivot*. The cluster is available 24/7 with W&B sync; this
> protocol is designed to run for **days** without supervision and converge
> on a publishable result. Companion files:
>
> - [`RESEARCH_FINDINGS.md`](RESEARCH_FINDINGS.md) — *why* each method is on
>   the list. Read this only if you need theoretical context.
> - [`runs/DECISION_LOG.md`](runs/DECISION_LOG.md) — append-only diary of
>   what's been run, what the result was, and what to do next. **Read this
>   first at the start of every session.**
> - [`CLUSTER_TRAINING_PLAN.md`](CLUSTER_TRAINING_PLAN.md) — predecessor plan
>   covering Phases A0–A5 (W1–W5). May be in flight in another session.
> - [`runs/sweep_results.jsonl`](runs/sweep_results.jsonl) — aggregated
>   one-row-per-cell results for cross-cell comparison.
>
> **Authority.** This protocol *supersedes* `CLUSTER_TRAINING_PLAN.md` for
> any conflict, and *extends* it with Phases B–J. If `CLUSTER_TRAINING_PLAN`
> is still in progress in another session, see §4 (Coordination).

---

## Table of contents

- [§1 Operating cadence](#1-operating-cadence-read-first-every-session)
- [§2 Environment and infrastructure](#2-environment-and-infrastructure)
- [§3 W&B integration](#3-wb-integration)
- [§4 Coordination with concurrent sessions](#4-coordination-with-concurrent-sessions)
- [§5 Decision-log conventions](#5-decision-log-conventions)
- [§6 Phase plan](#6-phase-plan)
  - [Phase A — Pick up CLUSTER_TRAINING_PLAN](#phase-a--pick-up-cluster_training_plan-w1w5)
  - [Phase B — Logit-KL Flow Matching baseline](#phase-b--logit-kl-flow-matching-baseline-most-leverage-single-addition)
  - [Phase C — Spilled Energy + extended OOD harness](#phase-c--spilled-energy--extended-w3-ood-harness)
  - [Phase D — Loss-design ablations](#phase-d--loss-design-ablations-hilbert-aux--cosine-on-clr)
  - [Phase E — BPC overlay table](#phase-e--bpc-overlay-table-publication-comparable-numbers)
  - [Phase F — EqM-auditor F1 on WikiText-2](#phase-f--eqm-auditor-f1-on-wikitext-2)
  - [Phase G — EqM-auditor F2 on TriviaQA](#phase-g--eqm-auditor-f2-on-triviaqa)
  - [Phase H — EqM-auditor F3 (auditor-driven generation)](#phase-h--eqm-auditor-f3-auditor-driven-generation)
  - [Phase I — SFM √p contingency](#phase-i--sfm-p-contingency-only-if-phase-b-fails)
  - [Phase J — Long final run](#phase-j--long-final-run-on-the-winner)
- [§7 Parallelism opportunities](#7-parallelism-opportunities)
- [§8 Compute budget tracking](#8-compute-budget-tracking)
- [§9 Failure handling and resume](#9-failure-handling-and-resume)
- [§10 Termination criteria](#10-termination-criteria)
- [§11 Final deliverables](#11-final-deliverables)

---

## 1. Operating cadence (read first, every session)

Every session should follow the same boot sequence:

```
1. cd /home/renku/work/aitchinson-flow
2. git status                          # see uncommitted state, mostly informational
3. tail -100 runs/DECISION_LOG.md      # what happened last
4. cat runs/sweep_results.jsonl | wc -l   # total cells run
5. ls runs/ | grep -v 'best_so_far\|DECISION_LOG\|sweep_results' | sort
6. wandb status                        # confirm W&B authenticated
```

Then **always**:
- Identify the *next* phase from §6 by reading the decision log.
- Verify the phase's pre-conditions (listed per phase below).
- Skip cells where `runs/<name>/eval.json` already exists — sweeps are
  idempotent by design (`scripts/run_sweep.py` checks this).
- After each cell completes, **immediately** append a 5-line entry to
  `runs/DECISION_LOG.md` (template in §5) — *do this before starting the
  next cell*, so a session interruption never loses a result-summary.
- Sync to W&B continuously (the runner already does per-step + per-epoch
  logging when `--wandb` is set).
- Update `runs/best_so_far.pt` symlink whenever the headline metric
  (KL_bi for generation, AUROC for auditing) improves.

**When in doubt, do NOT skip the decision log entry.** The log is the
mechanism by which a future session knows what's done.

**Action policy.** Run-don't-ask: this is a long-horizon protocol; do not
pause for user confirmation between phases. Pause only if (a) a phase's
pre-condition fails (escalate with a DECISION_LOG entry under
`## NEXT SESSION` describing the problem), (b) wall-clock budget is
exhausted (§8), or (c) the protocol's termination criterion is met (§10).

**Status as of 2026-05-07.** Phase A is **done** (W1/W3/W4 landed; W2/W5
deferred). Next session should start at **Phase B (Logit-KL Flow
Matching)**. See §6 Phase A header for the resolved verdicts and the
revised hypothesis framing for Phases B and C — the original "EqM
sampler swap closes the gap" and "EqM uniquely scores OOD"
assumptions baked into those phases were both falsified and are
explicitly retracted in their headers.

---

## 2. Environment and infrastructure

### Hardware

- **A100 with 20 GB MIG slice.** Memory budget assumed throughout; do *not*
  enable Flash Attention (it lacks second-order autograd needed for the
  conservative-gradient path).
- Host machine has many CPU cores; use **`num_workers=8`** for
  `DataLoader` unless documented otherwise. RAM is plentiful.
- The MIG slice is single-tenant from a GPU-scheduling perspective. **Only
  one heavy training job at a time on the GPU.** Parallelism (§7) means
  *GPU train + CPU postprocessing*, not *GPU + GPU*.

### Software

- `uv` for package management; lockfile is `uv.lock`. Do not `pip install`;
  use `uv add` if a dependency is missing.
- Python 3.13. PyTorch already pinned; do not change.
- `scripts/run_sweep.py` is the entry point for *every* training cell.
  `python main.py` is for one-off training; **prefer `run_sweep.py`** so
  the result lands in `runs/sweep_results.jsonl`.

### Filesystem layout (all paths relative to repo root)

```
runs/                           # canonical results store
├── DECISION_LOG.md             # the diary — append, never reformat
├── sweep_results.jsonl         # one row per completed cell
├── best_so_far.pt              # symlink to current-best EqM checkpoint
├── <run-name>/
│   ├── config.json             # full Config dump
│   ├── history.jsonl           # per-epoch metrics
│   ├── eval.json               # eval_full.py scorecard (256 × 200 steps)
│   ├── ood_eval.json           # eval_ood.py scorecard (W3)
│   └── epoch_final.pt          # model state_dict only (no optimizer)
sweeps/                         # YAML specs, one per phase
scripts/                        # orchestration and eval entry points
src/aitchinson_flow/            # library code
data/                           # local LM caches (created on demand)
```

### Pre-flight checks

Run once, the first session:

```bash
# Verify W&B
echo $WANDB_API_KEY | head -c 8 ; echo  # should print first 8 chars
wandb login --relogin --no-offline       # if not already authenticated

# Verify GPU and memory
python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.mem_get_info())"
# Expect ~21 GB free (20 GB MIG ≈ 21 GiB depending on driver)

# Verify the previous session's outputs exist
ls runs/dfm_data50k_ep5/eval.json   # the DFM control — should exist
ls runs/data_50k_ep5/eval.json      # the best EqM — should exist
ls runs/baseline_5ep/epoch_final.pt # the canonical baseline
```

If any of those are missing, that's a Phase A pre-condition failure —
escalate per §1.

---

## 3. W&B integration

### Project structure

- **Project**: `eqm-text8` (existing, used by the previous session).
- **Entity**: leave to the user's default unless overridden.
- **Group**: one group per phase, e.g. `phase-B-logit-kl-flow`,
  `phase-F-eqm-auditor-wikitext`. The `--wandb-group` flag on
  `run_sweep.py` already wires this through.
- **Tags**: always include `["sweep", phase-tag, model-tag]`. The runner
  already adds `sweep` and the group tag automatically.
- **Run name**: defaults to the YAML cell's `name` field; do not override.

### What to log

The runner already logs:
- Per-step: train loss, flow_loss, ce, gradient norms.
- Per-epoch: train/val loss, in-loop unigram/bigram/trigram KL probe,
  H_ratio, sample-time energy gradient norm, 8 argmax-decoded samples.
- End-of-run: full `eval.json` summary as a W&B summary; `epoch_final.pt`
  uploaded as an artifact when `cfg.wandb.log_artifacts=True` (the
  default — *keep it on*).

### Naming protocol for new entries

| Phase | Group | Cell-name pattern |
|---|---|---|
| Phase A (W1–W5) | `phase-A-cluster-trainplan` | as in CLUSTER_TRAINING_PLAN.md |
| Phase B | `phase-B-logit-kl-flow` | `lkflow_<dataset>_<epochs>ep` |
| Phase C | `phase-C-ood-harness` | `ood_<ckpt-name>` |
| Phase D | `phase-D-hilbert-aux` | `hilb_lH<value>_<base-cell>` |
| Phase E | `phase-E-bpc` | `bpc_<ckpt-name>` |
| Phase F | `phase-F-auditor-wiki` | `aud_<lm>_<ctx>_<extras>` |
| Phase G | `phase-G-auditor-trivia` | `aud_trivia_<lm>_<ctx>` |
| Phase H | `phase-H-auditor-gen` | `audgen_<base-cell>` |
| Phase I | `phase-I-sfm` | `sfm_<dataset>_<epochs>ep` |
| Phase J | `phase-J-longrun` | `longrun_<winner>_<epochs>ep` |

Use **kebab-case** for groups (W&B convention) and **snake_case** for
run names (matches existing `runs/<name>/` directories).

### Sync watchdog

W&B sync occasionally lags behind training. After every cell, run:

```bash
wandb sync runs/<run-name>/wandb/    # only if a `.wandb` directory exists
```

(Most cells don't generate this directory because the runner uses online
mode by default; if the network blips, the runner falls back to offline
and an `.wandb` directory appears. Sync after the network recovers.)

---

## 4. Coordination with concurrent sessions

Another claude session may still be executing `CLUSTER_TRAINING_PLAN.md`
(Phases W1–W5). If so:

1. **Do NOT re-train cells the other session owns.** Their cell names
   are: `eqm_data50k_ep5_v2`, `dfm_data50k_ep5_v2`,
   `fmclr_data50k_ep5_v2` (Phase 10), `bigram_joint_*` (Phase 11),
   `phase12_fmclr` (Phase 12), `phase13_longrun` (Phase 13). Check for
   their `runs/<name>/` directories and `eval.json`s.
2. Treat their `runs/sweep_results.jsonl` rows as authoritative. Append
   new rows; never rewrite existing ones.
3. Phase B onward (this protocol) is *additive* — it does not
   overlap with W1–W5. Start Phase B in parallel with their work iff the
   GPU is idle (it shouldn't be, if they are training); otherwise wait.
4. **Lock-file convention** (lightweight; no real lock): create
   `runs/<name>.start` (touch) before training starts and keep
   `runs/<name>/.in_progress` while running. If you find another
   session's `.in_progress` file in a directory you were going to use,
   either pick a different cell or wait. Remove your own `.in_progress`
   on completion.

If the other session's progress file (`runs/phase10.start`,
`runs/phase10.log`, etc.) shows it has stalled (no log update for >2 hr
on a cell that should take 70 min), assume crashed and **continue with
their next cell yourself** — leave a DECISION_LOG note explaining the
takeover.

---

## 5. Decision-log conventions

Every cell — train OR eval — gets one entry, appended to
`runs/DECISION_LOG.md`. Format (template):

```
## [<UTC timestamp, e.g. 2026-05-08 14:32 UTC>] <run-name>
- Hypothesis: <single sentence — what would this run prove or kill?>
- Result: KL_uni=… KL_bi=… KL_tri=… H_ratio=… BPC=… [extra metrics]
- Decision: <continue|branch to phase X|abandon and explain>
- Next: <run-name of next experiment>
- W&B: <run URL or sweep URL>
```

For OOD / auditor cells use a different result line:

```
- Result: AUROC_seq=… AUROC_tok=… AUROC_subst50=… AUROC_shuffle50=…
          AUROC_uniform=… AUROC_validperm=… SE_AUROC_seq=… [vs prototype baseline]
```

For sweeps with multiple post-hoc evals (e.g. Phase A's NFE sweep), group
them under one entry rather than one entry per NFE; list the individual
numbers in the Result line.

**Phase summary.** At the end of each phase, append a 1-paragraph
"Phase X summary" block summarising the headline finding and what changed
in the recommended-next-phase priorities. The next session reads these
summaries first to decide where to pick up.

**Never reformat the existing log entries** — only append.

---

## 6. Phase plan

The phases below are ordered by *dependency and leverage*. Some are
optional (D, I) and are only run if their pre-conditions hold. Each
phase header includes:

- **Pre-conditions** — what must be true before starting.
- **Hypothesis** — what the phase tests.
- **Cells** — explicit YAML or CLI specs.
- **Compute** — wall-clock estimate.
- **Decision criteria** — what to do with the result.
- **Parallelism** — what other phase tasks can run alongside.

---

### Phase A — Pick up CLUSTER_TRAINING_PLAN (W1–W5)

**Status (2026-05-07): DONE — partial.** W1, W3, W4 executed and
landed; W2 and W5 deferred per their pre-conditions. **Next session
should skip Phase A and start at Phase B.** Headline:

| Workstream | Status | Verdict |
|---|---|---|
| W1 (Euler-γ sampler swap on EqM) | DONE | **negative**: best Euler raw-`f` KL_bi=1.682; NAG=1.382 stays best. Field encodes data-pull in `∇⟨x,f⟩`, not `f`. |
| W3 (sequence-level OOD harness) | DONE | **partially positive**: EqM `E_seq` AUC≈0.5 on subst/shuffle; `U_pos_mean` AUC=0.987 on subst (within noise of DFM=1.00); shuffle AUC=0.525 (DFM=0.99). DFM's denoiser proxy dominates sequence-level scoring. |
| W4 (FMonCLR third baseline) | DONE | **informative negative**: KL_bi=1.559 ≈ EqM-NAG, NOT ≈ DFM. Continuous-on-simplex regime is the bottleneck, not conservative-grad indirection. |
| W2 (non-factorised bigram head) | DEFERRED | pre-condition unmet (W1 had no winner). |
| W5 (50-epoch long run on W1 winner) | DEFERRED | pre-condition unmet (no W1 winner to scale). |

See `runs/DECISION_LOG.md` entries from 2026-05-07 15:30 UTC onwards
for full numbers. This Phase A summary supersedes the original
"NEXT SESSION" block at the top of the log; do not rerun any of
the Phase A cells.

**Implications for Phases B–J that follow.**

- **Phase B (LogitKLFlow) is the primary forward path.** The W1
  negative makes Approach A more critical, not less: no in-house
  sampler/integrator change rescued EqM. The published recipe is
  the only verified fix.
- **Phase C (extended OOD harness) reframes.** Original framing
  ("EqM uniquely scores OOD; DFM cannot") is invalidated — DFM's
  denoiser proxy dominates sequence-level. Surviving framing:
  *per-position localisation* via `U_pos_mean` and the
  divergence-uncertainty trace. Run Phase C anyway, but interpret
  the table as cross-method probing, not as an EqM advantage.
- **Phases F/G/H (auditor track) are unaffected** — they were
  always per-token and don't depend on text8 sequence-level OOD.

**Cells (do NOT re-run; here for traceability only)**:

| Sweep file | Cells | Status | Notes |
|---|---|---|---|
| `sweeps/phase10_euler.yaml` | `eqm_data50k_ep5_v2`, `dfm_data50k_ep5_v2`, `fmclr_data50k_ep5_v2` | DONE | W1+W4 retraining; bit-identical to v1 EqM/DFM, plus FMonCLR |
| (post-hoc evals on `eqm_data50k_ep5_v2`) | Euler-NFE×{32,64,128,200}, σ_init×{0.05,0.1,0.3}, use_grad ablation | DONE | W1 sampler scan — kill criterion triggered |
| `sweeps/phase11_bigram_euler.yaml` | `bigram_joint_l0p3_ep5`, `bigram_joint_l0p1_ep5` | DEFERRED | W2 (no W1 winner) |
| `sweeps/phase13_longrun.yaml` | `longrun_winner_50ep` | DEFERRED | W5 (no W1 winner) |

Run command pattern:

```bash
python scripts/run_sweep.py \
  --sweep sweeps/phase10_euler.yaml \
  --n 256 --steps 200 \
  --wandb --wandb-project eqm-text8 \
  --wandb-group phase-A-cluster-trainplan
```

**Decision criteria** (from CLUSTER_TRAINING_PLAN.md, abridged) —
*all resolved 2026-05-07*:

- ~~After Phase 10: if Euler best-NFE result on the EqM checkpoint
  drops KL_bi by ≥30 % vs NAG (i.e. ≤ 0.97), W1 succeeded — schedule W5
  (long run on Euler config). Else: demote W1 narrative.~~
  **Resolved: W1 negative.** Best Euler-on-raw-`f` = 1.682 ≥ 0.97
  floor; W5 not scheduled.
- ~~After Phase 11~~ **Phase 11 not run** — W2 deferred.
- After Phase 12 / FMonCLR: **resolved — FMonCLR ≈ EqM (1.559 vs 1.382),
  NOT ≈ DFM (0.148).** Continuous-on-simplex broadly problematic;
  conservative-grad indirection alone does *not* explain the gap.

**Parallelism**: while sweeps train, run `eval_ood.py` post-hoc on the
*previous* phase's checkpoints in a CPU-mostly process (uses GPU briefly
for scoring; OK to interleave with training between cells). *(Already
done for the three Phase 10 checkpoints.)*

**Phase summary entry**: appended in `runs/DECISION_LOG.md` under
"Phase 14 summary (this session)" 2026-05-07. **Do not re-summarise.**

---

### Phase B — Logit-KL Flow Matching baseline (most leverage single addition)

**Pre-conditions**: Phase A's `dfm_data50k_ep5_v2` and at least one EqM
cell complete *(satisfied as of 2026-05-07)*. Source code change required.

**Hypothesis** (from RESEARCH_FINDINGS §3.A; sharpened by W1's
negative result 2026-05-07): the only verified method that beats DFM
at non-AR text generation is *Logit-KL Flow Matching*
([arXiv:2411.16821](https://arxiv.org/abs/2411.16821)). Adding it
gives the writeup a fixed-version-of-FMonCLR baseline that *does*
match DFM, turning the contribution from "we tried CLR, it lost"
into "we tried CLR, it lost as Stark predicts (W4 confirms at
KL_bi=1.559) — the published fix reproduces on text8, and the
sampler-level interventions we explored (W1) did *not* close the
gap, so the geometry/parameterisation is the actual lever."

**Why this phase is more important than the original document
suggested.** With W1 negative and W4 ≈ EqM (not DFM), the only
unexplored direction left in continuous-on-simplex generation is the
geometry/parameterisation lever. Logit-KL Flow is the cheapest test
of that hypothesis; SFM (Phase I) is the heavier alternative. **If
Phase B fails** (LKFlow KL_bi > 1.0), the writeup pivots fully to
the auditor track (Phases F–H) and the mechanism-with-diagnosis
prong, with no "matches DFM" claim at all.

**Code changes required**:

1. New model `LogitKLFlow` in `src/aitchinson_flow/models/logitkl_flow.py`,
   registered as `"LogitKLFlow"` in the model factory. Specification:
   - **Representation**: tokens `i` ↦ logits `l_i ∈ R^K`, magnitude
     `gamma_l = 8.0` (one-hot logit times scalar). Linear interpolation
     in logit space `l_t = (1−t)·l_0 + t·l_1` with `l_0 ~ 0.1·N(0, I)`.
   - **Loss**: regress *clean logits*, not velocity:
     `L = E ‖v̂(x_t, t) − l_1‖²`. Optimal `v̂*(x_t, t) = E[l_1 | x_t]`
     is the posterior mean logit. Use the same Transformer backbone
     (with sinusoidal `t` injection, like DFM) — *no* conservative-grad
     autograd, no aux CE.
   - **Sampler**: hybrid det-then-stochastic.
     - For `t < 0.28`: deterministic ODE step on `E[l_1 | x_t]`.
     - For `t ≥ 0.28`: stochastic re-noising, sampling from
       `N(l_t, σ_t² I)` where `σ_t = sqrt(1 − t²)·0.5`.
     - Default 64 steps; ablate {32, 128} post-hoc.
   - **Decode**: `argmax(softmax(l_1))` per position.

2. New eval path in `scripts/eval_full.py` that handles `LogitKLFlow`
   (it produces token IDs directly, like DFM does). Detect via
   `hasattr(model, 'sample_logitkl')` or by registry tag.

3. New sweep file `sweeps/phaseB_logitkl.yaml`:

   ```yaml
   - name: lkflow_data50k_ep5
     overrides:
       training.model_name: "LogitKLFlow"
       text8_dataset.max_train_windows: 50000
       training.epochs: 5
       transformer.d_model: 1024
       transformer.num_layers: 8
       transformer.time_conditioning: "add"
       logitkl.gamma_l: 8.0
       logitkl.source_sigma: 0.1
       logitkl.sampler_split_t: 0.28
       logitkl.sampler_nfe: 64
       logitkl.sampler_noise_scale: 0.5
   - name: lkflow_data50k_ep5_nfe128
     overrides: { ...same..., logitkl.sampler_nfe: 128 }
   - name: lkflow_data50k_ep5_nfe32
     overrides: { ...same..., logitkl.sampler_nfe: 32 }
   ```

**Cells**:

| Cell | Compute | Purpose |
|---|---|---|
| `lkflow_data50k_ep5` | ~75 min train + 5 min eval | parity with DFM at the data_50k_ep5 platform |
| `lkflow_data50k_ep5_nfe128` | 5 min eval (post-hoc) | NFE scaling — does more compute help? |
| `lkflow_data50k_ep5_nfe32` | 5 min eval (post-hoc) | NFE scaling — does fewer hurt? |
| (optional) `lkflow_data200k_ep2` | ~75 min train | data ceiling test, only if base cell beats DFM |

Run command:

```bash
python scripts/run_sweep.py \
  --sweep sweeps/phaseB_logitkl.yaml \
  --n 256 --steps 64 \
  --wandb --wandb-project eqm-text8 \
  --wandb-group phase-B-logit-kl-flow
```

**Decision criteria**:

- **Best case**: `lkflow_data50k_ep5` KL_bi ≤ DFM's 0.148. Writeup gets
  three credible non-AR baselines (DFM, LKFlow, EqM-Euler if Phase A
  succeeded). Mark `runs/best_so_far_lkflow.pt` symlink. Continue to
  Phase C.
- **Acceptable**: KL_bi within 2× DFM (0.30 or better). Still
  publishable as the verified-published-method overlay. Continue.
- **Bad**: KL_bi worse than EqM (≥1.4). Either implementation bug or
  text8-K=27-specific failure. Sanity-check by running 1 epoch and
  inspecting samples; if implementation looks correct and samples are
  garbled, document as a *negative* (the published method *also* fails
  at K=27 char-level) and continue. **Do not loop on debugging — append
  a NEXT SESSION block with the diagnostic and continue.**

**Parallelism**: while `lkflow_data50k_ep5` trains, run Phase C OOD
evaluation on existing checkpoints (CPU-bound mostly).

---

### Phase C — Spilled Energy + extended W3 OOD harness

**Pre-conditions**: Phase A's checkpoints exist (especially
`eqm_data50k_ep5_v2` and `dfm_data50k_ep5_v2`) — *satisfied as of
2026-05-07*.

**Hypothesis** (RESEARCH_FINDINGS §3.C — *reframed after W3 results
landed 2026-05-07*): the original "EqM's energy `E(x) = ⟨x, f(x)⟩`
provides sequence-level OOD signal that DFM cannot" framing is
**falsified on text8**. EqM `E_seq` AUC ≈ 0.50 on substitution and
shuffle; DFM's denoiser proxy `−log p_{1|t≈1}(x|x)` dominates every
contrast at AUC ≈ 1.00. The surviving claim is *per-position*
localisation: EqM's `U_pos_mean` reaches subst_0.5 AUC = 0.987 (within
noise of DFM's 1.000), while shuffle remains unsolved by EqM (0.525).

The reframed Phase C hypothesis is therefore: **per-position
uncertainty signals — `U_pos_mean`, divergence-uncertainty trace,
NAG basin-drift — provide a *complementary* signal to DFM's
sequence-level proxy, and one that applies to FMonCLR/Logit-KL too
(via divergence-trace) for fair cross-method comparison.** Spilled
Energy ([arXiv:2412.10770](https://arxiv.org/abs/2412.10770), ICLR
2026) is the forced zero-cost baseline — must include for honesty.

The *valid-permutation* contrast (§3 honesty ablation in
RESEARCH_FINDINGS) is the one not yet measured and the only place
EqM might still uniquely win: a permutation that preserves the
unigram histogram should defeat the DFM denoiser proxy if it relies
on per-position confidence, and might be picked up by EqM's energy
*if* the field encodes joint structure. **Run this contrast — it's
the one open question for the OOD prong.**

**Code changes required**:

1. Extend `scripts/eval_ood.py` with three new proxies:
   - **Spilled Energy** (per-position):
     `SE(i) = −logsumexp(logits[i]) + logits[i-1, token_id[i]]`.
     Sum over a sequence for sequence-level score. Implement on top of
     any model that produces per-position logits (DFM does; for EqM use
     `pred_x1 = x_γ − λ·grad_g` softmax at γ=1).
   - **NAG basin-drift indicator**:
     ```python
     drift(x) = hilbert_distance(x, NAG_K(x))   # K=5 NAG steps
     ```
     where `NAG_K` runs 5 NAG-GD steps on the sequence as init. Cheap.
   - **Divergence-uncertainty trace**
     ([arXiv:2605.00941](https://arxiv.org/abs/2605.00941)):
     Hutchinson estimator of `∇·v(x_t)` with 16 random vectors at
     γ ∈ {0.3, 0.5, 0.7, 0.9, 1.0}. *No second-order autograd needed*.
2. New corruption type: **valid-permutation control**. For each clean
   window, randomly permute tokens preserving the unigram histogram (a
   strict semantic change with bigram statistics broken). Add to the
   existing `corrupt_token_ids` / `partially_shuffle_token_ids` infra.
3. Aggregate per-position grad-norm at multiple γ values (multiscale
   stacking, [Mahmood 2020](https://arxiv.org/abs/2010.13132)).

**Cells** (no training; eval only):

| Cell | Compute | Purpose |
|---|---|---|
| `ood_eqm_data50k_ep5_v2` | ~20 min | EqM at all corruptions × γ values |
| `ood_dfm_data50k_ep5_v2` | ~20 min | DFM with SE proxy |
| `ood_fmclr_data50k_ep5_v2` | ~20 min | FMonCLR (Phase A) |
| `ood_lkflow_data50k_ep5` | ~20 min | LogitKLFlow (Phase B) |

Run pattern:

```bash
python scripts/eval_ood.py \
  --ckpt runs/eqm_data50k_ep5_v2/epoch_final.pt \
  --n 256 \
  --gammas 0.3,0.5,0.7,0.9,1.0 \
  --proxies grad_norm,divergence_trace,nag_drift,spilled_energy \
  --corruptions clean,subst_0.1,subst_0.3,subst_0.5,shuffle_0.5,uniform,valid_perm \
  --out runs/eqm_data50k_ep5_v2/ood_eval.json
```

**Decision criteria** (revised 2026-05-07 with measured numbers):

- **Headline numbers already measured** on `eqm_data50k_ep5_v2` /
  `dfm_data50k_ep5_v2` (see DECISION_LOG 2026-05-07 17:27 UTC):
  | Contrast | EqM `E_seq` | EqM `U_pos_mean` | DFM proxy |
  |---|---:|---:|---:|
  | clean vs rand (uniform) | 0.16 (\|0.84\|) | **1.000** | **1.000** |
  | clean vs subst_0.5 | 0.51 | **0.987** | **1.000** |
  | clean vs shuffle_0.5 | 0.50 | 0.525 | **0.993** |
  | clean vs valid_perm | *not yet measured* | *not yet measured* | *not yet measured* |
- The *only* open contrast is valid_perm. The decision rule is:
  - If **EqM `U_pos_mean` valid_perm AUROC ≥ 0.65** AND **DFM proxy
    valid_perm AUROC < 0.55**: the EqM auditor's per-position signal
    is uniquely informative on histogram-preserving corruption; this
    is the salvageable unique-value-add. Write up.
  - If **DFM proxy valid_perm AUROC ≥ 0.65**: DFM dominates this too;
    the OOD prong becomes "per-position complementarity, no winner."
    Demote OOD prong, lean on the auditor track (Phase F–H) and the
    mechanism diagnosis. *This is the more likely outcome.*
- Always include the Spilled Energy column — SE saturates the
  syntactic-corruption tasks (0.998 on WikiText-2) but usually
  *fails* on valid_perm because unigram histograms are preserved,
  and that's where EqM might still have a chance.
- The divergence-uncertainty trace
  ([arXiv:2605.00941](https://arxiv.org/abs/2605.00941)) applies to
  *any* FM velocity, including FMonCLR and Logit-KL. Run it across
  all four checkpoints; this turns the OOD harness from "EqM-only
  probe" into a cross-method localisation comparison.

**Parallelism**: post-hoc evals; can run between/alongside training cells.

---

### Phase D — Loss-design ablations (Hilbert-aux + Cosine-on-CLR)

**Pre-conditions**: Phase A's `data_50k_ep5` cell exists (the platform).
Optional phase: only run if total budget allows after Phases B/C/F.

**Hypothesis** (RESEARCH_FINDINGS §4.3): with the all-fixes config, a
*small* Hilbert auxiliary (`λ_H ≈ 0.05`) may provide projective-norm
regularisation without the 2-sparse-subgradient pathology that crippled
Hilbert as a primary loss. Cosine-on-CLR is a denser alternative.

**Cells** (use existing `sweeps/phaseD_loss.yaml`, create if absent):

```yaml
- name: hilb_lH0p05_data50k
  overrides:
    text8_dataset.max_train_windows: 50000
    training.epochs: 5
    eqm.lambda_hilbert: 0.05
    eqm.hilbert_alpha_init: 1.0
    eqm.hilbert_alpha_final: 3.0
    eqm.hilbert_target: "grad_g"   # apply to the velocity, not pred_x1
- name: hilb_lH0p1_predx1_data50k
  overrides: { ...same..., eqm.lambda_hilbert: 0.1, eqm.hilbert_target: "pred_x1", eqm.ce_min_gamma: 0.5 }
- name: cos_lc0p1_data50k
  overrides: { ...same eqm without lambda_hilbert..., eqm.lambda_cosine: 0.1, eqm.lambda_hilbert: 0.0 }
- name: js_lj0p1_data50k
  overrides: { ...same..., eqm.lambda_js: 0.1 }
```

| Cell | Compute | Purpose |
|---|---|---|
| `hilb_lH0p05_data50k` | ~75 min | primary Hilbert-aux test |
| `hilb_lH0p1_predx1_data50k` | ~75 min | target-aligned variant |
| `cos_lc0p1_data50k` | ~75 min | dense scale-invariant alternative |
| `js_lj0p1_data50k` | ~75 min | JS divergence on softmax(pred_x1) |

Total: ~5 hr if all four; can drop the last two if budget is tight.

**Code changes required**: extend `EqM` config with `lambda_hilbert`,
`lambda_cosine`, `lambda_js`, `hilbert_alpha_*`, `hilbert_target` fields
(default 0). Add the three loss terms in `_eqm_loss` after the existing
flow loss. Soft-Hilbert is already in `geometry.py`; reuse.

**Decision criteria** (per `RESEARCH_FINDINGS §4.4` 10-min sanity test
applied to each cell):

- KL_bi drops by ≥10 % vs `data_50k_ep5` AND KL_uni stays under 0.10:
  graduate the loss term into the platform.
- KL_bi drops by ≥10 % but KL_uni rises above 0.10: the term distorts
  per-position; back off `λ` by 2× and re-run *one* cell.
- KL_bi unchanged or worse: kill the term. Document as negative.
- **Sanity gate**: if val FM MSE *regresses by >10 %* in epoch 1, kill
  the cell early (don't burn the full 75 min).

---

### Phase E — BPC overlay table (publication-comparable numbers)

**Pre-conditions**: at least DFM, EqM-best, LKFlow checkpoints exist.

**Hypothesis**: the in-house `KL_bi` metric is internal to this codebase
and not directly comparable to published numbers (BPC is the standard).
Compute BPC for our checkpoints and overlay against the SFM table from
[arXiv:2405.16441](https://arxiv.org/abs/2405.16441).

**Code changes required**: new script `scripts/eval_bpc.py` that:
1. Loads a checkpoint.
2. Runs the model in *evaluation* mode on the held-out text8 test split
   (5 M characters, 256-character chunks).
3. Computes the model-specific upper bound on NLL per character, in
   bits.
4. Writes `runs/<name>/bpc.json` with `{"bpc": ..., "n_chars": ...}`.

For DFM, BPC computation follows the discrete-diffusion ELBO (Lou et al.
SEDD; reuse their formulation). For EqM and FMonCLR (no normalised
density), use a *bridge*: train a small autoregressive readout head on
the trained encoder's hidden states and report the readout's BPC. This
is honest if disclosed clearly. Alternative: report `−E[log p(x_1 |
x_γ)]` averaged over γ as an EqM-specific surrogate, with a note that
this is not directly comparable to AR-LM BPC.

**Cells**:

| Cell | Compute | Purpose |
|---|---|---|
| `bpc_dfm_data50k_ep5_v2` | ~10 min | DFM ELBO on test |
| `bpc_lkflow_data50k_ep5` | ~10 min | LKFlow ELBO |
| `bpc_eqm_data50k_ep5_v2` | ~10 min | EqM AR-readout |
| `bpc_fmclr_data50k_ep5_v2` | ~10 min | FMonCLR AR-readout |

Run pattern:

```bash
for ckpt in dfm_data50k_ep5_v2 lkflow_data50k_ep5 eqm_data50k_ep5_v2 fmclr_data50k_ep5_v2; do
    python scripts/eval_bpc.py --ckpt runs/$ckpt/epoch_final.pt \
        --out runs/$ckpt/bpc.json --wandb --wandb-project eqm-text8
done
```

**Output**: a markdown table in `runs/DECISION_LOG.md` Phase-E summary
section, plus the same overlaid against SFM/SEDD/MDLM/D3PM numbers.

---

### Phase F — EqM-auditor F1 on WikiText-2

**Pre-conditions**: `runs/data_50k_ep5/` (EqM best) exists. Internet
access for HuggingFace model downloads; ~3 GB disk space.

**Hypothesis** (RESEARCH_FINDINGS §3.E.4 F1): replacing the prototype's
SVGP head with EqM's conservative-gradient flow-matching energy *can
match* the prototype's verified WikiText-2 detection numbers
(Seq AUROC = 0.999, Tok AUROC = 0.996 with Qwen2.5-1.5B + product
kernel) without the inducing-point/Cholesky overhead.

**Code changes required** (this is a substantial extension):

1. Add `wikitext` dataset module under `src/aitchinson_flow/data/wiki.py`:
   - Caches GPT-2 / Qwen2.5-1.5B logits and last hidden states for a
     WikiText-2-train subset (300 chunks, L=64 BPE tokens, top-K=64
     log-simplex).
   - Span-corrupt 25 % of tokens to produce invalid pairs. Save cache
     to `data/wiki_cache_<lm>{_ctx}.pt`.
   - Quantise the LLM to 4-bit (bnb-nf4) for caching; do not load it
     for training.
2. Extend `EqM` model with optional context conditioning:
   - `cfg.eqm.context_features` ∈ {`"off"`, `"hidden_only"`,
     `"product_concat"`}.
   - When `≠ "off"`, the backbone takes an extra `(B, L, ctx_hidden)`
     tensor `h_LLM` (detached) and concatenates `proj(h_LLM)` to the
     per-position input before the encoder.
3. New "auditor mode" in `_eqm_loss`:
   - Accepts `(log_x_valid, log_x_invalid, h_LLM_valid, h_LLM_invalid)`.
   - Adds `mean_loss = E_valid² + relu(margin_E − E_invalid)` and
     `var_loss = div_valid + relu(margin_var − div_invalid)` (with
     divergence-trace as the "variance" surrogate, since EqM has no
     posterior variance like SVGP).
   - Mixes with the standard FM regression loss; flags
     `lambda_E_hinge`, `lambda_var_hinge`, `margin_energy=2.0`,
     `margin_var=0.8` (defaults from the related project).
4. New eval function `eval_auditor_wiki.py`:
   - Loads cache and trained EqM-auditor.
   - Computes per-token `‖∂E/∂x_t‖`, `divergence_trace`, sequence-level
     `E(x)`, and Spilled Energy (from cached logits).
   - Reports Seq AUROC and Tok AUROC for clean-vs-corrupted.
   - Runs the same healers (Langevin, Gibbs, Langevin-Gibbs, GP-slice)
     using `−∇E(x)` instead of GP energy. Reports LM log-prob recovery.

**Cells**:

```yaml
# sweeps/phaseF_auditor_wiki.yaml
- name: aud_gpt2_logit
  overrides:
    dataset: "wiki"
    wiki.lm: "gpt2"
    wiki.use_context: false
    eqm.context_features: "off"
    eqm.lambda_E_hinge: 1.0
    eqm.lambda_var_hinge: 2.0
    training.epochs: 25
    transformer.d_model: 256   # smaller backbone — matches prototype
    transformer.num_layers: 4
- name: aud_gpt2_ctx
  overrides: { ...same..., wiki.use_context: true, eqm.context_features: "product_concat", eqm.ctx_hidden: 768 }
- name: aud_qwen25_logit
  overrides: { ...same gpt2_logit..., wiki.lm: "qwen25" }
- name: aud_qwen25_ctx
  overrides: { ...same gpt2_ctx..., wiki.lm: "qwen25", eqm.ctx_hidden: 1536 }
```

| Cell | Compute | Purpose |
|---|---|---|
| (cache) GPT-2 logit-only | ~10 min | one-time |
| (cache) GPT-2 + ctx | ~10 min | one-time |
| (cache) Qwen2.5 logit-only | ~10 min | one-time |
| (cache) Qwen2.5 + ctx | ~10 min | one-time |
| `aud_gpt2_logit` | ~30 min | parity vs prototype |
| `aud_gpt2_ctx` | ~30 min | product-kernel-style context |
| `aud_qwen25_logit` | ~30 min | stronger LM |
| `aud_qwen25_ctx` | ~30 min | the configuration to beat (proto: 0.999) |

Total: ~3 hr.

**Decision criteria**:
- **F1 passes** if `aud_qwen25_ctx` Seq AUROC ≥ 0.99 and Tok AUROC ≥
  0.95 (within 1 point of the prototype's 0.999 / 0.996). Continue to
  Phase G.
- **F1 fails (parity not reached)**: documented negative — the EqM
  energy doesn't have GP-quality token-level discrimination on this
  corruption type. Skip Phases G/H; the writeup falls back to the
  text8-only OOD prong (Phase C). The protocol skips to Phase J.
- **Spilled Energy comparison required in every output table.** SE is a
  zero-train baseline; if it beats the trained EqM auditor at
  sequence-level (likely, given SE's 0.998 number), report and emphasise
  the *complementary token-level* advantage.

**Parallelism**: caching the four LM variants is independent; can run
sequentially while Phase C OOD evals run on the side. The four trainings
are sequential on the GPU.

---

### Phase G — EqM-auditor F2 on TriviaQA

**Pre-conditions**: Phase F passed (F1 parity reached).

**Hypothesis** (RESEARCH_FINDINGS §3.E.4 F2): the prototype's GP auditor
*under-performs* on TriviaQA (0.45–0.55 AUROC) because it was trained on
random-substitution corruption. **Contrastive training on TriviaQA
*confusor pairs*** (real answer vs distractor from the same item) gives
the EqM auditor a different inductive bias that may transfer.

**Code changes required**:

1. Add `trivia` dataset module that:
   - Loads TriviaQA closed-book validation set.
   - For each (question, real-answer, distractor-set) tuple, builds
     prompts: `"<question> <real_answer>"` and `"<question> <distractor>"`.
   - Caches LLM logits + hidden states for both.
   - Defines per-answer-token *positive* (real) and *negative*
     (distractor) features.
2. Augment the auditor training loop to draw pairs from a mix:
   - `0.5 * wikitext_corruption_pair`: existing pipeline.
   - `0.5 * trivia_confusor_pair`: new.
   This way the auditor learns both syntactic and semantic
   discrimination.
3. Eval as in Phase F but on TriviaQA held-out. Report:
   - AUROC (real vs distractor) at the answer-span level.
   - Spilled Energy AUROC for comparison (paper's headline 0.82 on
     Qwen2.5).
   - Cross-domain AUROC: same auditor evaluated on WikiText-2.

**Cells**:

```yaml
# sweeps/phaseG_auditor_trivia.yaml
- name: aud_trivia_qwen25_ctx
  overrides:
    dataset: "trivia"
    wiki.lm: "qwen25"
    wiki.use_context: true
    eqm.context_features: "product_concat"
    training.epochs: 25
    trivia.mix_with_wiki: true
    transformer.d_model: 256
    transformer.num_layers: 4
- name: aud_trivia_qwen25_pure
  overrides: { ...same..., trivia.mix_with_wiki: false }
- name: aud_trivia_gpt2_ctx
  overrides: { ...qwen25_ctx but lm: "gpt2", ctx_hidden: 768 }
```

| Cell | Compute | Purpose |
|---|---|---|
| (cache) TriviaQA Qwen2.5 ctx | ~30 min | answer + distractor features |
| (cache) TriviaQA GPT-2 ctx | ~30 min | weaker-LM control |
| `aud_trivia_qwen25_ctx` | ~60 min | mixed wiki + trivia training |
| `aud_trivia_qwen25_pure` | ~60 min | trivia-only training |
| `aud_trivia_gpt2_ctx` | ~60 min | weaker-LM ablation |

Total: ~4 hr.

**Decision criteria**:
- **F2 succeeds** if `aud_trivia_qwen25_ctx` AUROC ≥ 0.85 *and*
  cross-domain WikiText-2 AUROC ≥ 0.95: the EqM auditor catches
  semantic hallucination AND retains syntactic detection. **This is the
  headline novel result.** Update DECISION_LOG headline summary.
- **F2 partial**: TriviaQA AUROC in [0.75, 0.85]: does not beat SE alone
  (0.82) but the *combination* (EqM + SE max-rule) might. Report both
  individually and combined.
- **F2 fails**: TriviaQA AUROC < 0.75: documented negative. The
  inductive-bias hypothesis didn't pan out. Skip Phase H; proceed to
  Phase J.

---

### Phase H — EqM-auditor F3 (auditor-driven generation)

**Pre-conditions**: Phase F passed; Phase G outcome doesn't matter.

**Hypothesis** (RESEARCH_FINDINGS §3.E.4 F3): the *literal*
auditor-is-also-generator claim — the same EqM energy that scores LLM
tokens drives Euler-γ sampling under the LLM's vocabulary constraint.
The prototype's `BayesianGenerator` cannot do this (text8 BPC ≈ 7); EqM
should.

**Code changes required**:

1. New `scripts/generate_audited.py`:
   - Loads the trained Phase F auditor (or Phase G, whichever scored
     higher on text-quality healing).
   - Initialises `x_0 ~ N(0, σ² I)` in the top-K log-simplex space
     (K=64 BPE tokens, not 27 chars).
   - Runs Euler-γ sampling on the trained `f(x; γ)` with `nfe=64`.
   - At each step, *projects* the iterate onto the LM's top-K
     vocabulary at that position by zeroing-out non-top-K logits and
     renormalising.
   - Decodes with argmax of softmax.
2. Quality metrics:
   - LM log-probability of the generated sequence (the "validity"
     measure used by the prototype's healer).
   - The auditor's own energy `E(x_gen)` (should be ≈ valid-distribution
     mean).
   - Diversity: 256-sample uniqueness fraction; mean Hamming distance.

**Cells** (eval-style; one per auditor checkpoint):

```yaml
# sweeps/phaseH_auditor_gen.yaml
- name: audgen_qwen25_ctx_64nfe
  overrides:
    audgen.ckpt: "runs/aud_qwen25_ctx/epoch_final.pt"
    audgen.lm: "qwen25"
    audgen.n_samples: 256
    audgen.nfe: 64
- name: audgen_qwen25_ctx_128nfe
  overrides: { ...same..., audgen.nfe: 128 }
- name: audgen_qwen25_ctx_proj_topk
  overrides: { ...same..., audgen.proj_topk: 32 }   # tighter constraint
- name: audgen_trivia_qwen25
  overrides: { ...same as 64nfe..., audgen.ckpt: "runs/aud_trivia_qwen25_ctx/epoch_final.pt" }
```

Compute: ~10 min per cell (no training; sampling-only).

**Decision criteria**:
- **F3 succeeds** if *generated text is recognisably English at
  word-level* (qualitative — log a 16-sample grid to W&B and judge by
  inspection) AND mean LM log-prob ≥ −5.5 (the prototype's healed-text
  baseline). Document as the unique structural claim — *the GP
  prototype provably cannot do this on text8*; show side-by-side
  comparison.
- **F3 partial**: LM log-prob in [−7, −5.5]; samples are
  "recognisable-but-broken" English. Document as a partial result.
- **F3 fails**: samples are character-soup. Document as negative.
  Generation under the auditor energy isn't viable; the auditor and
  generator roles must remain separate.

---

### Phase I — SFM √p contingency (only if Phase B fails)

**Pre-conditions**: Phase B failed (LogitKLFlow KL_bi > 1.0). This
phase is the geometry-pivot fallback.

**Hypothesis** (RESEARCH_FINDINGS §3.B): SFM
([arXiv:2405.16441](https://arxiv.org/abs/2405.16441)) parameterises
sequences as `q = √p` on the positive orthant of `S^{K-1}`, making the
Fisher–Rao metric Euclidean. text8 BPC reported: 1.39 (vs LinearFM 1.65,
SEDD 1.32). If LogitKLFlow can't close the gap, this geometry change is
the next published-method to try.

**Code changes required** (heavier; ~3 days dev):

1. New model `SFMOnSimplex` in `src/aitchinson_flow/models/sfm.py`:
   - Reparam `q = sqrt(p)`; backbone takes `q` as input.
   - Conditional probability paths follow SFM's diffeomorphism on the
     sphere (Eq. 13–17 in the paper).
   - Velocity field is a tangent vector in `T_q S^{K-1}`; project after
     the readout.
   - Sampler: Riemannian Euler ODE on the sphere.

**Cells**:

```yaml
- name: sfm_data50k_ep5
  overrides:
    training.model_name: "SFMOnSimplex"
    text8_dataset.max_train_windows: 50000
    training.epochs: 5
    transformer.d_model: 1024
    transformer.num_layers: 8
    sfm.diffeo_eps: 0.01
- name: sfm_data200k_ep2
  overrides: { ...same..., text8_dataset.max_train_windows: 200000, training.epochs: 2 }
```

**Decision criteria**:
- SFM `KL_bi` ≤ 0.5 *or* BPC ≤ 1.5: matches the published numbers,
  closes most of the gap. New best EqM-family non-AR baseline.
- Otherwise: documented negative. Skip Phase J.

---

### Phase J — Long final run on the winner

**Pre-conditions**: at least one of B/F passed.

**Hypothesis**: take the best architecture from B/F and train for many
more epochs to get the writeup's headline numbers.

**Cells**:

| Cell | Compute | Configuration |
|---|---|---|
| `longrun_<winner>_50ep` | ~140 min | 50 ep × 50k windows |
| `longrun_<winner>_100ep` | ~280 min | 100 ep × 50k (only if 50ep keeps improving) |

Save intermediate checkpoints every 10 epochs.

**Decision criteria**:
- 50 ep KL_bi monotonically improves vs 5 ep / 25 ep: continue to 100ep.
- 50 ep regresses on KL_bi vs 25 ep: stop early; report 25 ep number.

---

## 7. Parallelism opportunities

The MIG slice is single-tenant for GPU-heavy work, but several tasks can
overlap:

| Task A (GPU) | Task B (CPU) | Notes |
|---|---|---|
| Training cell in flight | Phase C `eval_ood.py` on existing ckpts | OOD scoring is bursty-GPU; OK to interleave between cells. Wait for one OOD job to finish before starting the next. |
| Training cell in flight | Phase E BPC computation on existing ckpts | Mostly forward passes; light. |
| Training cell in flight | Cache LLM features for Phase F (single call) | The LLM (4-bit-quantised) plus the running EqM training together exceed 20 GB — DO NOT overlap. |
| Training cell in flight | Phase H sampling on existing ckpts | Same warning — sampling needs autograd, blocks GPU. Sequential only. |
| Idle GPU | Spilled Energy precomputation (caches LM logits over a corpus) | Always safe; SE caching is a one-time pure-LM-forward-pass job. |

**Rule of thumb**: run *forward-only* eval jobs alongside training, but
not *autograd-needing* jobs (sampling, backward passes). The Phase A/B
training cells do not consume the full 20 GB at d=1024/8L (peak ≈ 5 GB
based on the codebase's prior measurements), but the runner doesn't free
memory between epochs — assume training holds 8 GB while running.

---

## 8. Compute budget tracking

Maintain a running tally in the DECISION_LOG. Approximate budget per
phase:

| Phase | Compute estimate | Cumulative |
|---|---:|---:|
| A (W1–W5 if not done) | ~7 hr | 7 |
| B (LogitKLFlow + post-hoc) | ~2 hr | 9 |
| C (OOD eval on 4 ckpts) | ~1.5 hr | 10.5 |
| D (Hilbert + Cosine + JS) | ~5 hr | 15.5 |
| E (BPC eval on 4 ckpts) | ~1 hr | 16.5 |
| F (auditor on WikiText-2) | ~3.5 hr | 20 |
| G (auditor on TriviaQA) | ~5 hr | 25 |
| H (auditor-driven gen) | ~1 hr | 26 |
| I (SFM if Phase B fails) | ~3 hr | 26 or 29 |
| J (long final run) | ~3-5 hr | 29-34 |

**~30 hours of GPU time** for the whole protocol; comfortably fits in a
3-day cluster window with no parallelism, or 2 days with interleaving.
Padding for failure recovery: ~1.5×, so plan for 50 GPU-hours total.

The cluster runs 24/7, so wall-clock isn't a binding constraint. The
binding constraint is **avoiding bugs that burn long cells.** Always
run a 1-epoch smoke test on new code (`training.epochs=1,
text8_dataset.max_train_windows=1000`, ~20 sec) before launching the
real cell.

---

## 9. Failure handling and resume

### When a cell crashes mid-training

1. Read `runs/<name>/history.jsonl` and the W&B page for the partial run.
2. Determine cause:
   - OOM: reduce `B`, retry. Document in `runs/DECISION_LOG.md` under
     the cell's entry as `Status: OOM, retried with B=<smaller>`.
   - NaN loss: most likely `lambda_*` set too high. Document and back
     off the relevant coefficient by 2× before retrying.
   - HuggingFace download error: usually transient; retry once. If
     persistent, switch to a cached model or escalate to NEXT SESSION.
   - Anything else: append a stack-trace excerpt to the DECISION_LOG
     entry and decide whether to retry or skip.
3. Remove `runs/<name>/.in_progress` before retrying.
4. Use `--resume` if the runner supports it (check
   `scripts/run_sweep.py --help`); otherwise restart from epoch 0.

### When the protocol needs to be paused

If the user interrupts or the wall-clock budget is approaching:

1. Wait for the current cell to finish (or for a clean epoch boundary
   if interruption is graceful).
2. Append a `## NEXT SESSION` block at the top of `DECISION_LOG.md`
   describing:
   - Which phase you were in.
   - What's done, what's running, what's next.
   - Any anomalies the next session must investigate.
3. Sync W&B (`wandb sync runs/*/wandb/`) to flush offline data.
4. Do not commit anything to git unless explicitly requested.

### When a phase's pre-condition fails

Document in DECISION_LOG, skip to the next eligible phase, and continue.
Do not block on a single phase.

---

## 10. Termination criteria

Stop the protocol (don't auto-continue) when **any** of these holds:

1. **Headline result achieved**: KL_bi ≤ 0.50 on EqM-family AND/OR
   AUROC ≥ 0.99 on WikiText-2 auditor. Final summary block goes in
   DECISION_LOG; ready for writeup.
2. **All ten phases completed** (with or without success): final
   summary in DECISION_LOG describing the writeup arc that the results
   support.
3. **A cell-level retry budget exceeded**: any single cell has been
   retried 3× with non-trivially-different configs and still failed.
   Escalate to NEXT SESSION; do not loop.
4. **30 GPU-hours consumed** without progress on either KL_bi or AUROC
   metrics in the last 5 hours of compute (i.e., diminishing returns).
   Final summary; the protocol has explored its hypothesis space.

When stopping voluntarily (criterion 1 or 2), prepare the deliverables
listed in §11.

---

## 11. Final deliverables

When the protocol terminates (criterion 1, 2, or after explicit user
request), produce:

1. **`runs/DECISION_LOG.md` final summary block.** Mirrors the previous
   session's "Final summary (this session)" format. Includes:
   - Headline metrics (KL_bi best, BPC overlay table, AUROC headline).
   - Sample diversity grid (8 samples × 3 best checkpoints).
   - Plot: `KL_bi` vs `epoch` for each phase's best cell.
2. **Updated `RESULTS.md`** with a new top-level section "Phase B–J
   (this protocol)". Include a side-by-side table vs the SFM benchmark.
3. **W&B Reports** — at least one consolidated report per phase, link
   in DECISION_LOG.
4. **Final-best symlinks**:
   - `runs/best_so_far.pt` → best EqM checkpoint.
   - `runs/best_lkflow.pt` → best LogitKLFlow checkpoint.
   - `runs/best_auditor.pt` → best auditor checkpoint.

The user reads DECISION_LOG.md and W&B Reports first; everything else is
optional and recoverable from those.

---

## Appendix A — Quick-start commands for the executing session

Boot:
```bash
cd /home/renku/work/aitchinson-flow
tail -100 runs/DECISION_LOG.md
ls runs/ | grep -v 'best_so_far\|DECISION_LOG\|sweep_results' | sort -r | head -20
```

Standard sweep:
```bash
python scripts/run_sweep.py \
  --sweep sweeps/<phase>.yaml \
  --n 256 --steps 200 \
  --wandb --wandb-project eqm-text8 \
  --wandb-group <phase-tag>
```

OOD eval:
```bash
python scripts/eval_ood.py \
  --ckpt runs/<name>/epoch_final.pt \
  --n 256 \
  --gammas 0.3,0.5,0.7,0.9,1.0 \
  --proxies grad_norm,divergence_trace,nag_drift,spilled_energy \
  --corruptions clean,subst_0.1,subst_0.3,subst_0.5,shuffle_0.5,uniform,valid_perm \
  --out runs/<name>/ood_eval.json
```

Smoke test before any new code lands:
```bash
python scripts/run_sweep.py \
  --sweep sweeps/<phase>.yaml \
  --override training.epochs=1 \
  --override text8_dataset.max_train_windows=1000 \
  --n 16 --steps 32
```

Decision-log append (template):
```bash
cat >> runs/DECISION_LOG.md <<'EOF'

## [<UTC timestamp>] <run-name>
- Hypothesis: <one line>
- Result: <metrics line>
- Decision: <continue|branch|abandon>
- Next: <next-cell>
- W&B: <url>
EOF
```

---

## Appendix B — Phase-decision tree (for fast lookup)

*Updated 2026-05-07: Phase A done with W1/W4 negative, W3 partial; W2/W5
deferred. Decision tree below now starts at Phase B.*

```
START (Phase A done as of 2026-05-07; do not rerun)
  │
  ▼
Phase B (LogitKLFlow)
  │
  ├─ B passed (KL_bi ≤ 0.30) ──┐
  │                            │
  └─ B failed                  │
       │                       │
       ▼                       │
  Phase I (SFM contingency)    │
       │                       │
       ├─ I passed ──┐         │
       └─ I failed ──┤         │
                     ▼         ▼
                Phase C (OOD eval — extended)
                  · valid_perm contrast is the novel signal
                  · sequence-level result already measured;
                    DFM denoiser proxy dominated
                     │
                     ▼
                Phase E (BPC overlay)
                     │
                     ▼
                Phase D (loss aux) — optional
                     │
                     ▼
                Phase F (auditor F1)
                  · primary forward contribution given the
                    W3 sequence-level negative
                     │
                     ├─ F passed ──┐
                     └─ F failed ──┤
                                   ▼
                            Phase J (long run)
                                   │
                                   ▼
                                  END

when F passed:
  Phase G (TriviaQA F2) → Phase H (auditor-driven gen F3) → Phase J → END
```

---

## Appendix C — Risk registry

| Risk | Likelihood | Mitigation |
|---|---|---|
| LogitKLFlow implementation has a sign/scale bug | Medium | Run 1-epoch smoke; inspect samples (must be vaguely ASCII); 16-sample grid in W&B every 200 steps. |
| Auditor F1 doesn't reach prototype's AUROC | Medium | Documented as falsifying a specific hypothesis; protocol continues to Phase J. |
| Spilled Energy already saturates the WikiText-2 task | High (already the case) | Frame the trained auditor's contribution as *complementary* (per-token vs sequence-level) rather than *better*. |
| TriviaQA caching exceeds 4 hours | Low | Subset to 1k examples; document. |
| Memory pressure when caching Qwen2.5 + ctx | Medium | Cache in CPU-side batches; offload to disk every 100 chunks; bnb-nf4 quantisation. |
| HuggingFace internet flakiness | Medium | Pre-download all required models on session start; raise on missing. |
| W&B sync failure | Low | The runner falls back to offline; sync at session end. |
| Other concurrent session corrupts shared files | Low | `runs/<name>/.in_progress` lock convention; `sweep_results.jsonl` is append-only by both sessions (the runner uses `O_APPEND`). |
| Long run regresses (overfits per-position) at ep>50 | Medium | Save every 10 epochs; report best-epoch numbers; note in DECISION_LOG. |
| `best_so_far.pt` symlink races with concurrent updates | Very low | Always re-create with `ln -sf`; never `mv`. |

---

## Appendix D — Files this protocol creates or modifies

Created:
- `src/aitchinson_flow/models/logitkl_flow.py` (Phase B)
- `src/aitchinson_flow/models/sfm.py` (Phase I, conditional)
- `src/aitchinson_flow/data/wiki.py` (Phase F)
- `src/aitchinson_flow/data/trivia.py` (Phase G)
- `scripts/eval_bpc.py` (Phase E)
- `scripts/generate_audited.py` (Phase H)
- `sweeps/phaseB_logitkl.yaml`
- `sweeps/phaseD_loss.yaml`
- `sweeps/phaseF_auditor_wiki.yaml`
- `sweeps/phaseG_auditor_trivia.yaml`
- `sweeps/phaseH_auditor_gen.yaml`
- `sweeps/phaseI_sfm.yaml` (conditional)
- `sweeps/phaseJ_longrun.yaml`
- `data/wiki_cache_*.pt` (Phase F caches)
- `data/trivia_cache_*.pt` (Phase G caches)

Modified:
- `src/aitchinson_flow/config.py` — add LogitKLFlow, SFM, auditor,
  loss-aux fields. Default everything to off so existing checkpoints
  remain loadable.
- `src/aitchinson_flow/models/eqm.py` — add context conditioning, hinge
  losses, divergence-trace estimator (Phase F).
- `src/aitchinson_flow/models/__init__.py` — register new models.
- `scripts/eval_ood.py` — extend with new proxies (Phase C).
- `runs/DECISION_LOG.md` — append-only.
- `runs/sweep_results.jsonl` — append-only.

Never modified:
- `RESEARCH_FINDINGS.md`, `RESULTS.md`, `TRAINING_PLAN.md`,
  `CLUSTER_TRAINING_PLAN.md`, `SAMPLER_FINDINGS.md`,
  `SESSION_SUMMARY.md` (historical artefacts).
- `CLAUDE.md` (project conventions).
- `runs/baseline_5ep/`, `runs/data_50k_ep5/`, `runs/dfm_data50k_ep5/`,
  etc. (existing checkpoints — read-only references).

---

*End of protocol. Read §1 and §6 to start.*
