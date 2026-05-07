# Training Protocol v2 — UQ pipeline + continuous-FM validity

> **Lead document for executing claude session(s) on the post-Phase-H
> capstone work.** Specifies *what* to train, *in what order*, *with
> what configs*, *what to watch*, and *when to stop or pivot*.
> Companion files:
>
> - [`CAPSTONE_PLAN.md`](CAPSTONE_PLAN.md) — *why* each phase is on the
>   list. Read once at the start of a new session for strategic
>   framing.
> - [`REPORT.md`](REPORT.md) — synthesis of what the previous session
>   established. Reference for any "but didn't we already know X?"
>   question.
> - [`runs/DECISION_LOG.md`](runs/DECISION_LOG.md) — append-only diary
>   of every cell. **Read this first at the start of every session.**
>   Existing topmost `## NEXT SESSION` block describes where the prior
>   session paused.
>
> **Authority.** This protocol *supersedes*
> [`TRAINING_PROTOCOL.md`](TRAINING_PROTOCOL.md) (the original v1).
> Phases A–J of v1 are completed or deferred; the prior session's
> entries in `DECISION_LOG.md` document the outcomes. New phases here
> are labelled **K–P** (continuing v1's letter sequence).

---

## Table of contents

- [§1 Operating cadence](#1-operating-cadence-read-first-every-session)
- [§2 Environment and infrastructure](#2-environment-and-infrastructure)
- [§3 W&B integration](#3-wb-integration)
- [§4 Decision-log conventions](#4-decision-log-conventions)
- [§5 Phase plan](#5-phase-plan)
  - [Phase K — HaluEval-QA UQ baseline](#phase-k--halueval-qa-uq-baseline)
  - [Phase L — TruthfulQA UQ extension](#phase-l--truthfulqa-uq-extension)
  - [Phase M — Per-token localisation on hallucinations](#phase-m--per-token-localisation-on-hallucinations)
  - [Phase N — SFM √p continuous FM (priority 2a)](#phase-n--sfm-p-continuous-fm)
  - [Phase O — BPE-scale continuous FM (priority 2b, optional)](#phase-o--bpe-scale-continuous-fm-optional)
  - [Phase P — Two-model integration](#phase-p--two-model-integration)
- [§6 Parallelism opportunities](#6-parallelism-opportunities)
- [§7 Compute budget tracking](#7-compute-budget-tracking)
- [§8 Failure handling and resume](#8-failure-handling-and-resume)
- [§9 Termination criteria](#9-termination-criteria)
- [§10 Final deliverables](#10-final-deliverables)

---

## 1. Operating cadence (read first, every session)

Every session should follow the same boot sequence:

```
1. cd /home/renku/work/aitchinson-flow
2. git status                              # see uncommitted state
3. tail -200 runs/DECISION_LOG.md          # what happened last
4. cat CAPSTONE_PLAN.md | head -80         # strategic framing
5. cat runs/sweep_results.jsonl | wc -l    # total cells run to date
6. ls runs/ | grep -E '^(aud_|lkflow|hal_|trivia_|sfm_)' | sort
7. wandb status                            # confirm W&B authenticated
```

Then **always**:

- Identify the *next phase* from §5 by reading the most recent
  `## NEXT SESSION` block in `runs/DECISION_LOG.md`.
- Verify the phase's pre-conditions (listed per phase below).
- Skip cells where `runs/<name>/eval.json` already exists (sweep
  runner is idempotent by design).
- After **each cell completes**, *immediately* append a 5-line entry
  to `runs/DECISION_LOG.md` (template in §4) — *do this before the
  next cell* so a session interruption never loses a result-summary.
- Sync to W&B continuously (the runner already does per-step +
  per-epoch logging when `--wandb` is set).
- Update `runs/best_*.pt` symlinks whenever the headline metric for
  that track improves (`best_uq.pt` for the UQ track,
  `best_continuous.pt` for SFM, etc.).

**When in doubt, do NOT skip the decision-log entry.** The log is the
mechanism by which a future session knows what's done.

**Action policy.** Run-don't-ask: this is a long-horizon protocol; do
not pause for user confirmation between phases. Pause only if (a) a
phase's pre-condition fails (escalate with a `## NEXT SESSION` entry
describing the problem), (b) wall-clock budget is exhausted (§7), or
(c) a termination criterion is met (§9).

**Status entry-point as of v2 first run.** No phases here have run
yet. Start at **Phase K** (HaluEval-QA cache + UQ baseline).

---

## 2. Environment and infrastructure

### Hardware

- **A100 with 20 GB MIG slice.** Memory budget assumed throughout; do
  *not* enable Flash Attention (lacks second-order autograd).
- Host has many CPU cores; `num_workers=8` for `DataLoader` is the
  default. RAM is plentiful.
- The MIG slice is single-tenant. **Only one heavy training job at a
  time on the GPU.** Parallelism (§6) means *GPU train + CPU
  postprocessing*, not *GPU + GPU*.

### Software

- `uv` for package management; lockfile is `uv.lock`. Do not
  `pip install`; use `uv add` if a dependency is missing.
- Python 3.13. PyTorch already pinned.
- Key libraries already installed: `transformers 5.5.4`,
  `gpytorch 1.15.2`, `sklearn 1.8.0`, `scipy 1.17.1`. Use them.
- `bitsandbytes` is **not** installed; do not require nf4
  quantisation. GPT-2 small (124M) and GPT-2 medium (350M) fit
  comfortably in fp32; Qwen2.5-1.5B fits at fp16.
- `scripts/run_sweep.py` is the entry point for *every* training
  cell. Results land in `runs/sweep_results.jsonl`.

### Filesystem layout

```
runs/                           # canonical results store
├── DECISION_LOG.md             # append-only diary
├── sweep_results.jsonl         # one row per completed cell
├── best_*.pt                   # symlinks to current-best ckpt per track
├── <run-name>/
│   ├── config.json             # full Config dump
│   ├── history.jsonl           # per-epoch metrics
│   ├── eval.json               # standard eval scorecard
│   ├── auditor_eval.json       # if applicable
│   ├── uq_eval.json            # Phase K-onwards UQ eval
│   └── epoch_final.pt          # state-dict only, no optimizer
sweeps/                         # YAML specs, one per phase
scripts/                        # orchestration and eval
src/aitchinson_flow/            # library code
data/                           # local LM caches
data/hallueval_cache_<lm>.pt    # produced by Phase K cache step
data/truthfulqa_cache_<lm>.pt   # produced by Phase L cache step
```

### Pre-flight checks

Run once at the first new session:

```bash
# 1. W&B authenticated.
wandb status 2>&1 | head -3

# 2. GPU is the same A100 slice.
nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1

# 3. Python deps.
python -c "import gpytorch, transformers, torch; print('ok')"

# 4. The previous session's headline checkpoints still exist.
ls runs/aud_gpt2_ctx/epoch_final.pt          # Phase F1 winner
ls runs/dfm_data50k_ep5_v2/epoch_final.pt    # DFM at parity
ls data/wiki_cache_gpt2.pt                   # Phase F cache (ref only)

# 5. New deps for hallucination datasets.
python -c "from datasets import load_dataset; ds = load_dataset('truthful_qa', 'multiple_choice', split='validation'); print(len(ds))"
# Expect ~817 rows; first run will download the dataset.
```

If any of these fail, fix or document in a `## NEXT SESSION` block
before proceeding.

---

## 3. W&B integration

### Project structure

- **Project**: `eqm-text8` (continue using the same project as the
  previous session for traceability).
- **Group**: one group per phase, e.g. `phase-K-hallueval`,
  `phase-N-sfm`, `phase-P-pipeline`. The runner auto-derives this as
  `sweep:<spec_stem>` (so a sweep filename of `phaseK_hallueval.yaml`
  produces group `sweep:phaseK_hallueval`).
- **Tags**: `["sweep", phase-tag, model-tag, "v2"]` (the runner adds
  the first two; add `v2` manually via `--wandb-tags v2` if exposed,
  or skip).
- **Run name**: defaults to the YAML cell's `name` field; do not
  override.

### What to log

The runner already logs per-step (train loss, gradient norms),
per-epoch (val loss, applicable AUROCs), and end-of-run (full eval
JSON). For new UQ phases, **always** log:

- Per-epoch AUROC + ECE on the val split.
- Calibration reliability data (if available; can be a histogram in
  the W&B summary).
- For SFM (Phase N): per-epoch KL_uni / KL_bi / KL_tri probe (the
  same machinery as the original protocol's text8 KL probe).

### Naming protocol

| Phase | Group | Cell-name pattern |
|---|---|---|
| Phase K | `phase-K-hallueval` | `hal_<lm>_<setting>` (e.g. `hal_gpt2_qa`) |
| Phase L | `phase-L-truthfulqa` | `tqa_<lm>_<setting>` |
| Phase M | `phase-M-localise` | `loc_<dataset>_<method>` |
| Phase N | `phase-N-sfm` | `sfm_data50k_ep5`, `sfm_data50k_ep25` |
| Phase O | `phase-O-bpe` | `eqm_bpe_K64`, `lkflow_bpe_K64` |
| Phase P | `phase-P-pipeline` | `pipe_<gen-track>_<val-track>` |

Use **kebab-case** for groups and **snake_case** for run names.

---

## 4. Decision-log conventions

Every cell — train OR eval — gets one entry, appended to
`runs/DECISION_LOG.md`. Format (template):

```
## [<UTC timestamp>] <run-name>
- Hypothesis: <one sentence — what would this run prove or kill?>
- Result: <metric line — see below>
- Decision: <continue|branch to phase X|abandon and explain>
- Next: <run-name of next experiment>
- W&B: <run URL or sweep URL>
```

Result-line conventions:

- **Generation cells** (Phase N, O):
  `KL_uni=… KL_bi=… KL_tri=… H_ratio=… BPC=… [extras]`
- **UQ cells** (Phase K, L):
  `AUROC=… ECE=… AUROC_at_corrupt=… AUROC_at_uncorrupt=… [vs SE baseline AUROC=…]`
- **Localisation cells** (Phase M):
  `Tok_AUROC=… Local_gap=AUROC_at_target − AUROC_at_distractor=…`
- **Pipeline cells** (Phase P):
  `combined_AUROC=… combined_ECE=… ΔAUROC vs best_track_alone=…`

For sweeps with multiple post-hoc evals (e.g. UQ at multiple
calibration temperatures), group them under one entry.

**Phase summary.** At the end of each phase, append a 1-paragraph
"Phase X summary" block summarising the headline finding and what
changed in the recommended-next-phase priorities. The next session
reads these summaries first.

**Never reformat the existing log entries** — only append.

---

## 5. Phase plan

The phases below are ordered by dependency and leverage. Each header
includes:

- **Pre-conditions** — what must be true before starting.
- **Hypothesis** — what the phase tests.
- **Cells** — explicit YAML or CLI specs.
- **Compute** — wall-clock estimate.
- **Decision criteria** — what to do with the result.
- **Parallelism** — what other phase tasks can run alongside.

---

### Phase K — HaluEval-QA UQ baseline

**Pre-conditions**: pre-flight checks pass; HuggingFace dataset
`pminervini/HaluEval` is accessible.

**Hypothesis**: the SVGP / BLR-Laplace / linear-probe UQ pipeline that
hit AUROC = 0.99, ECE = 10⁻⁴ on synthetic span-corrupted WikiText-2
generalises to real ChatGPT-style hallucinations on HaluEval-QA. The
target is AUROC ≥ 0.85 with ECE ≤ 0.10 on held-out questions.

**Code changes required**:

1. New `src/aitchinson_flow/data/hallueval.py`:
   - `load_hallueval_qa(split, max_n)` — fetches the QA subset; returns
     `[{question, right_answer, hallucinated_answer, knowledge}, ...]`.
   - For each row, build two prompts:
     - `prompt_correct = f"Question: {question}\\nAnswer: {right_answer}"`
     - `prompt_hallucinated = f"Question: {question}\\nAnswer: {hallucinated_answer}"`
   - Optionally include `knowledge` in the prompt; document the
     choice in a config.

2. New `scripts/cache_hallueval.py`, modelled on
   `scripts/cache_wiki.py`:
   - Tokenises each prompt with the LM's tokenizer.
   - Runs the LM forward to get full per-position `(logits, hidden)`.
   - Identifies `answer_span` (positions inside the answer; everything
     before is the question/prompt, which is shared).
   - Saves a cache schema:
     ```
     prompts            : list[str] (n_rows × 2)         — for inspection only
     prompt_lens        : (n_rows × 2,) long             — boundary between prompt and answer
     full_ids           : (n_rows × 2, L) long           — padded BPE ids (L = max_seq_len)
     attn_mask          : (n_rows × 2, L) bool
     hidden_states      : (n_rows × 2, L, H) float       — last-hidden state
     SE_pos             : (n_rows × 2, L) float          — per-position LM NLL
     answer_mask        : (n_rows × 2, L) bool           — True at answer-span positions
     label              : (n_rows × 2,) bool             — True if hallucinated
     ```
   - Default LM: `gpt2` (124M; fast caching). Optional: `gpt2-medium`
     (350M; richer features) and `Qwen/Qwen2.5-1.5B` (fp16).

3. New `scripts/eval_uq.py`, generalised version of
   `scripts/phaseF_uq.py`:
   - Loads any cache file with the schema above.
   - Trains linear probe / BLR-Laplace / ensemble / SVGP on
     answer-span hidden states (positives = hallucinated answer
     positions, negatives = correct answer positions for the same
     question).
   - Reports per-method AUROC, ECE, calibration plot.
   - Mean-pools per-token features into a per-sequence prediction
     (since the headline task is "is *this answer* hallucinated?",
     not "is *this token* hallucinated?").

**Cells** (`sweeps/phaseK_hallueval.yaml`):

```yaml
# Phase K — UQ on HaluEval-QA. The cache is built once via
# scripts/cache_hallueval.py; sweep cells differ in LM and pooling.
- name: hal_gpt2_qa_meanpool
  overrides:
    auditor.enabled: true
    auditor.cache_path: data/hallueval_cache_gpt2.pt
    auditor.batch_size: 16
    training.model_name: EqM     # placeholder; eval uses scripts/eval_uq.py
    training.epochs: 0           # no training, eval-only
    text8_dataset.K: 0
    text8_dataset.L: 0
- name: hal_gpt2_qa_lasttoken
  overrides:
    auditor.enabled: true
    # ... same plus pooling=last
```

For the eval-only nature of these cells, you can also run them
directly without the sweep YAML — see "Run pattern" below.

**Compute**:

- Cache (once per LM): 30 min (gpt2), 1.5 hr (Qwen2.5-1.5B at fp16).
- Train SVGP + BLR + linear probe + Mahalanobis: 5 min total per
  setting (you'll have ~6 settings: 2 pool methods × 3 LMs).
- Eval: 5 min.

**Run pattern**:

```bash
# 1. Cache. Default n=10000 questions; reduce if budget is tight.
python scripts/cache_hallueval.py \
  --lm gpt2 --max-n 10000 --L 256 \
  --out data/hallueval_cache_gpt2.pt --lm-batch-size 8

# 2. UQ eval. The script trains all four methods and reports.
python scripts/eval_uq.py \
  --cache data/hallueval_cache_gpt2.pt \
  --train-frac 0.8 --pool meanpool \
  --out runs/hal_gpt2_qa_meanpool/uq_eval.json

# 3. Plot.
python scripts/plot_uq_calibration.py \
  --json runs/hal_gpt2_qa_meanpool/uq_eval.json \
  --out runs/hal_gpt2_qa_meanpool/uq_calibration.png
```

**Decision criteria**:

- **Pass**: SVGP per-question AUROC ≥ 0.85 AND ECE ≤ 0.10 on at least
  one LM. **Continue to Phase L** to test on the harder TruthfulQA.
- **Partial**: AUROC ∈ [0.65, 0.85] OR ECE > 0.10. Try the larger LM
  (Qwen2.5-1.5B) before pivoting. If still below 0.85, **continue to
  Phase M (per-token localisation)** as the headline angle.
- **Fail**: SVGP AUROC < 0.65 even with Qwen2.5. Document the negative;
  pivot to **Phase M** with a reduced ambition.

**Parallelism**: cache producer is GPU-heavy (LM forward); UQ training
is CPU-light. While Phase K cache is producing, *no other GPU work*.
After cache is done, Phase K eval is fast.

---

### Phase L — TruthfulQA UQ extension

**Pre-conditions**: Phase K landed (any verdict). Cache producer is
generalisable (i.e. `scripts/cache_hallueval.py` accepts the
TruthfulQA format with minor adapter).

**Hypothesis**: TruthfulQA is harder than HaluEval-QA — its
"hallucinations" are common misconceptions / plausible-but-wrong
answers, which are subtler than ChatGPT-style hallucinations. AUROC on
TruthfulQA should be in the 0.55–0.80 range; if it's much higher,
something's leaking; if it's at chance, the pipeline failed to
generalise.

**Code changes required**:

1. Adapt `scripts/cache_hallueval.py` (or copy → `cache_truthfulqa.py`)
   to handle the TruthfulQA format:
   - Each row has `mc1_targets` with 1 correct answer and several
     wrong choices (`labels[i] = 1` for correct).
   - For each question, build `(question, correct, incorrect)`
     pairs by drawing one correct and one incorrect answer.
   - Same cache schema as Phase K.

2. Re-use `scripts/eval_uq.py` unchanged.

**Cells**:

```bash
python scripts/cache_truthfulqa.py \
  --lm gpt2 --max-n 800 --L 200 \
  --out data/truthfulqa_cache_gpt2.pt --lm-batch-size 8

python scripts/eval_uq.py \
  --cache data/truthfulqa_cache_gpt2.pt \
  --train-frac 0.8 --pool meanpool \
  --out runs/tqa_gpt2_qa_meanpool/uq_eval.json
```

**Compute**:

- Cache: 5 min (TruthfulQA is small, ~800 rows).
- UQ eval: 2 min.
- Total: ~10 min per LM.

**Decision criteria**:

- **Pass**: SVGP AUROC ≥ 0.70 on TruthfulQA. The pipeline transfers to
  hard hallucinations.
- **Acceptable**: AUROC ∈ [0.55, 0.70]. Document; this is the
  expected range for a small-LM SVGP baseline on a hard benchmark.
- **Documented negative**: AUROC < 0.55. The pipeline doesn't
  generalise to subtle factual errors. Frame as "synthetic + ChatGPT-
  style hallucinations are detectable; common-misconception
  hallucinations are not, at GPT-2 scale".

**Parallelism**: while caching, can run Phase M on the existing
HaluEval cache.

---

### Phase M — Per-token localisation on hallucinations

**Pre-conditions**: Phase K cache exists.

**Hypothesis**: Spilled Energy (LM's per-token NLL) genuinely
localises the hallucinated tokens within an answer span, while
`h_LLM`-based methods cascade-flag the whole answer. Phase F's
`runs/phaseF_distance.png` showed this on synthetic span corruption;
Phase M tests it on real hallucinations.

**Code changes required**:

1. Adapt `scripts/phaseF_distance_audit.py` to operate on the HaluEval
   cache. The "corrupted positions" are now answer tokens; the
   question is: *within an answer*, which tokens have the highest
   per-token discriminator score, and does that align with the actual
   factually-wrong tokens?

2. Manual annotation of a small subset (≤ 50 questions) where you
   identify the specific *wrong* token(s) in the hallucinated answer
   (e.g., "Paris is the capital of Germany" — the wrong token is
   `Germany`, position depends on tokenisation). This gives a
   token-level ground truth.

3. New `scripts/localise_uq.py`:
   - Feeds answer-span tokens through SE, linear probe, SVGP.
   - Reports top-k recovery: at what position rank does the actual
     wrong token fall?
   - Reports per-method "localisation score": the rank of the
     ground-truth wrong token among answer-span tokens, divided by
     answer length. Lower is better.

**Cells**:

```bash
# Run on annotated subset.
python scripts/localise_uq.py \
  --cache data/hallueval_cache_gpt2.pt \
  --annotations data/hallueval_annotations.json \
  --methods se,linear_probe,svgp \
  --out runs/loc_hallueval_gpt2/loc_eval.json
```

**Compute**: 5 min (eval only).

**Decision criteria**:

- **Pass**: SE produces top-3 rank ≤ 25% of answer length on average,
  while linear probe / SVGP do not. **The locality claim transfers.**
- **Negative**: SE doesn't localise on real hallucinations either.
  Document; the "real-hallucination per-token attribution is hard"
  finding is itself reportable.
- **Surprising**: SVGP localises better than SE (unlikely given
  cascade physics, but worth investigating).

**Parallelism**: can run while Phase L caches.

---

### Phase N — SFM √p continuous FM

**Pre-conditions**: existing text8 datamodule and parity-compute
platform (`data_50k_ep5`) work. Reference paper:
[arXiv:2405.16441](https://arxiv.org/abs/2405.16441).

**Hypothesis** (`CAPSTONE_PLAN.md` Priority 2a): the regime-level
negative on continuous-on-simplex generation at K=27 was tested on
EqM, FMonCLR, LogitKLFlow — all linear-interpolation-style methods.
SFM uses a *spherical* geometry via `q = √p` reparameterisation; the
Fisher–Rao metric on the simplex pulls back to Euclidean on the
sphere. This is the *one* published continuous method that closes
most of the EqM-DFM gap on text8 (BPC 1.39 vs SEDD 1.32). If our
re-implementation hits ≤ 0.30 KL_bi, geometry was the missing
ingredient. If it lands in the 1.4 cluster, the regime-level negative
is irrefutable.

**Code changes required**:

1. New `src/aitchinson_flow/models/sfm.py`:
   - `SFMOnSimplex(nn.Module)` with the same training-step / sample
     interface as `EquilibriumFlowMatching`.
   - Reparameterisation: `q = √(label_smoothed_onehot)` with `q ∈
     S^{K−1}_+` (positive orthant of the unit sphere). Forward
     consumes `q ∈ R^K` (norm-1 per position, all-positive).
   - Conditional probability path follows Eq. 13 of the paper:
     spherical interpolation `q_t = sin((1−t)θ)/sin(θ) · q_0 +
     sin(tθ)/sin(θ) · q_1` where `θ = arccos(⟨q_0, q_1⟩)`.
   - Tangent-vector velocity head: project `proj(h, K) − q ⟨proj, q⟩`
     to keep the velocity tangent to `S^{K−1}` at `q`.
   - Sampler: Riemannian Euler ODE on the sphere. Step
     `q_{t+h} = exp_q(h · v)` where `exp_q(v) = cos(\|v\|) q +
     sin(\|v\|) v / \|v\|` (the sphere's exponential map).

2. New `src/aitchinson_flow/config.py` section `SFMConfig`:
   ```python
   @dataclass
   class SFMConfig:
       label_smoothing: float = 1e-3   # avoid q_i = 0 exactly
       diffeo_eps: float = 0.01
       sample_nfe: int = 64
       sampler: str = "riemannian_euler"  # "riemannian_euler" | "tangent_euler"
   ```

3. Register in `src/aitchinson_flow/models/__init__.py`:
   `@register("SFM")`.

4. Update `scripts/eval_full.py` to handle SFM the way it handles
   FMonCLR (top-1 argmax decode of the sphere-coord softmax).

**Cells** (`sweeps/phaseN_sfm.yaml`):

```yaml
- name: sfm_data50k_ep5
  overrides:
    training.model_name: SFM
    training.epochs: 5
    text8_dataset.max_train_windows: 50000
    transformer.d_model: 1024
    transformer.num_layers: 8
    sfm.diffeo_eps: 0.01
    sfm.sample_nfe: 64

# Long-run only if the 5-epoch result is encouraging (KL_bi ≤ 1.0).
- name: sfm_data50k_ep25
  overrides:
    training.model_name: SFM
    training.epochs: 25
    text8_dataset.max_train_windows: 50000
    transformer.d_model: 1024
    transformer.num_layers: 8
    sfm.diffeo_eps: 0.01
    sfm.sample_nfe: 128
```

**Compute**:

- 5-ep cell: ~75 min (matches EqM/DFM at parity).
- 25-ep cell: ~6 hr (only run if 5-ep is encouraging).

**Run pattern**:

```bash
# 1. Smoke test before the full run.
python scripts/run_sweep.py \
  --sweep sweeps/phaseN_sfm.yaml \
  --only sfm_data50k_ep5 \
  --override training.epochs=1 \
  --override text8_dataset.max_train_windows=1000 \
  --n 16 --steps 32

# 2. The full run with W&B.
python scripts/run_sweep.py \
  --sweep sweeps/phaseN_sfm.yaml \
  --n 256 --steps 200 \
  --wandb --wandb-project eqm-text8
```

**Decision criteria**:

- **Strong positive**: `sfm_data50k_ep5` KL_bi ≤ 0.30. Continuous FM
  with the right geometry beats EqM/FMonCLR by 5×. Schedule
  `sfm_data50k_ep25` to push toward published 1.39 BPC. **Major
  capstone contribution.**
- **Partial positive**: KL_bi ∈ [0.30, 0.80]. Geometry helps but
  doesn't fully close the gap to DFM. Document; possibly extend to
  25 ep.
- **Negative**: KL_bi ≥ 1.0. The regime-level continuous-on-simplex
  negative is confirmed at the published-method level. Strong
  writeup negative; **skip Phase O**.

**Parallelism**: while SFM trains, can run Phase M and Phase L evals
on CPU (caches and small probes).

---

### Phase O — BPE-scale continuous FM (optional)

**Pre-conditions**: Phase N landed at KL_bi ≥ 1.0 (i.e., regime-level
negative confirmed). Otherwise skip.

**Hypothesis**: the K=27 char-level negative might not generalise to
BPE-scale vocabulary. Train EqM on the existing
`data/wiki_cache_gpt2.pt` features (top-K=64 BPE simplex). If it
generates coherent text under the LM-vocab projection, the
continuous-vs-discrete trade-off is vocab-dependent.

**Code changes required**:

1. New training mode in EqM that consumes the wiki cache directly
   (skip the text8 datamodule). Re-use `WikiAuditorDataModule` from
   Phase F with `lambda_E_hinge = 0` (turn off the auditor branch;
   we want pure FM regression now).
2. New eval `scripts/eval_continuous_bpe.py`: sample sequences,
   re-feed through GPT-2, measure NLL.

**Cells**:

```bash
python scripts/run_sweep.py \
  --sweep sweeps/phaseO_bpe_continuous.yaml \
  --n 64 --steps 64 \
  --wandb --wandb-project eqm-text8
```

**Compute**: ~30 min training + 5 min eval = ~35 min per cell.

**Decision criteria**:

- **Pass**: generated NLL ≤ 6.0 (vs Phase H's 8.89 — much better).
  BPE-scale continuous FM works.
- **Fail**: NLL ≥ 7.5. The negative is regime-AND-vocab-level.

**Parallelism**: can run alongside Phase L cache producer.

---

### Phase P — Two-model integration

**Pre-conditions**: Phase K passed (UQ AUROC ≥ 0.65 on at least one
hallucination dataset) AND Phase N landed (any verdict). Phase L
optional but recommended for the integration's evaluation set.

**Hypothesis**: combining a strong validity-track generator's
likelihood (DFM at parity, or SFM if Phase N passed) with the
correctness-track UQ classifier's calibrated probability gives a
2-feature predictor whose AUROC and calibration exceed either
component alone.

**Code changes required**:

1. New `scripts/integrate_pipeline.py`:
   - Loads validity-track generator checkpoint
     (`runs/dfm_data50k_ep5_v2/epoch_final.pt` for DFM at parity, or
     `runs/sfm_data50k_ep5/epoch_final.pt` if Phase N passed).
   - Loads correctness-track UQ models trained in Phase K (SVGP +
     BLR + linear probe).
   - For each (q, a) pair from the held-out HaluEval split:
     - Compute generator NLL on the answer span.
     - Compute UQ classifier prob from cached `h_LLM`.
     - Combine via (a) max-rule, (b) sum-rule, (c) trained
       2-feature logistic regression.
   - Report combined AUROC + ECE; compare to each track alone.

2. Localisation extension:
   - Per-token: combine SE (locality-clean) with the SVGP per-token
     score. Show that SE-flagged positions correspond to wrong
     tokens; SVGP-flagged positions correspond to "questionable
     contexts".
   - Decoded examples grid: display the q+a with two coloured
     overlays — generator NLL per token, validator score per token.

**Cells**:

```bash
python scripts/integrate_pipeline.py \
  --gen-ckpt runs/dfm_data50k_ep5_v2/epoch_final.pt \
  --uq-cache data/hallueval_cache_gpt2.pt \
  --uq-method svgp \
  --eval-dataset hallueval_qa \
  --combine-rule learned_lr \
  --out runs/pipe_dfm_svgp/integration.json

python scripts/plot_pipeline.py \
  --json runs/pipe_dfm_svgp/integration.json \
  --out runs/pipe_dfm_svgp/integration.png
```

**Compute**: 10 min eval per combination.

**Decision criteria**:

- **Strong positive**: combined AUROC > each track alone by ≥ 0.02
  AND combined ECE ≤ 0.08. Two-model design adds value.
- **Acceptable**: combined ≈ best-track-alone. The components are
  correlated; the combination is defensible as a "complementary
  diagnostics" architecture even without extra separability.
- **Negative**: combined < best-track-alone. Document; the
  decomposition argument stands theoretically but doesn't compound
  empirically.

**Parallelism**: eval-only, no GPU pressure once caches and models
are in place.

---

## 6. Parallelism opportunities

| GPU task | CPU/light task | Notes |
|---|---|---|
| Phase K cache producer (LM forward) | nothing else on GPU | LM dominates; ~30 min |
| Phase N SFM training | Phase K UQ eval (CPU) + Phase M localisation (CPU/light GPU) | UQ training is fast |
| Phase L cache producer | Phase M localisation eval | TruthfulQA cache is tiny |
| Phase P integration eval | nothing simultaneous needed | ~10 min |

**Rule of thumb**: only one heavy GPU job at a time. UQ classifier
training (SVGP, BLR) is light enough to interleave between GPU runs.

---

## 7. Compute budget tracking

Maintain a running tally in `DECISION_LOG.md`. Approximate budget per
phase:

| Phase | Compute | Cumulative |
|---|---:|---:|
| K (HaluEval cache + UQ eval) | ~1.5 hr | 1.5 |
| L (TruthfulQA cache + UQ eval) | ~30 min | 2.0 |
| M (localisation eval) | ~30 min | 2.5 |
| N (SFM 5 ep + post-hoc evals) | ~2 hr | 4.5 |
| N (SFM 25 ep, conditional) | ~6 hr | 10.5 |
| O (BPE continuous, conditional) | ~2 hr | 12.5 |
| P (integration eval) | ~30 min | 13.0 |

**~13 GPU-hours** for the whole protocol with no contingencies; plan
~25 GPU-hours including failure recovery, reruns, and the conditional
SFM-25-ep / Phase O extensions.

The cluster runs 24/7, so wall-clock isn't a binding constraint. The
binding constraint is **avoiding bugs that burn long cells.** Always
run a 1-epoch smoke test on new code (override `training.epochs=1`,
`text8_dataset.max_train_windows=1000` for SFM; use `--max-n 100`
for the cache producer) before launching the real cell.

---

## 8. Failure handling and resume

### When a cell crashes mid-training

1. Read `runs/<name>/history.jsonl` and the W&B page for the partial
   run.
2. Determine cause:
   - OOM: reduce `B`, retry.
   - NaN loss: most likely σ/lr too high. Document and back off the
     relevant coefficient by 2× before retrying.
   - HuggingFace download error: usually transient; retry once.
   - SFM-specific: numerical instability on the sphere
     (`acos(⟨q,q'⟩)` overflows for nearly-parallel q,q'). Add an
     `eps=1e-6` clamp on the inner product.
   - Anything else: append a stack-trace excerpt to the
     DECISION_LOG entry and decide retry vs skip.
3. Remove `runs/<name>/.in_progress` (if used) before retrying.

### When a phase's pre-condition fails

Document in `DECISION_LOG.md`, skip to the next eligible phase, and
continue. **Do not block on a single phase.** Examples:

- Phase K cache fails because of a HuggingFace credentials issue:
  document, try TruthfulQA (Phase L) instead since it's
  smaller; resolve credentials offline.
- Phase N SFM fails to train (NaN losses): document, try
  Phase O (BPE continuous) as the alternative; come back to N
  with a fixed `eps`.

### When the protocol needs to be paused

If the user interrupts or budget approaches exhaustion:

1. Wait for the current cell to finish (or for a clean epoch
   boundary).
2. **Append** a `## NEXT SESSION` block at the top of
   `DECISION_LOG.md` (above any existing such block) describing:
   - Which phase you were in.
   - What's done, what's running, what's next.
   - Any anomalies the next session must investigate.
3. Sync W&B (`wandb sync runs/*/wandb/`).
4. Do not commit anything to git unless explicitly requested.

---

## 9. Termination criteria

Stop the protocol when **any** of these holds:

1. **Capstone goals achieved** (the strong positive ending):
   - Phase K landed at AUROC ≥ 0.85 AND ECE ≤ 0.10 on HaluEval-QA.
   - Phase N landed (any verdict; either SFM works → positive
     contribution, or SFM doesn't → continuous-FM negative
     confirmed at the published-method level).
   - Phase P landed at combined AUROC ≥ each-track-alone.
   - Final summary block in `DECISION_LOG.md`; ready for writeup.

2. **All Phase K–P landed** (with mixed verdicts): final summary
   describing the writeup arc the results support.

3. **Cell-level retry budget exceeded**: any single cell retried 3×
   with non-trivially-different configs and still failed. Escalate
   to a `## NEXT SESSION` block; do not loop.

4. **25 GPU-hours consumed without progress** on UQ AUROC or
   continuous-FM KL_bi metrics in the last 5 hours of compute.
   Final summary; the protocol has explored its hypothesis space.

When stopping voluntarily, prepare the deliverables in §10.

---

## 10. Final deliverables

When the protocol terminates (criterion 1 or 2), produce:

1. **`runs/DECISION_LOG.md` final summary block.** Includes:
   - Headline UQ numbers (AUROC + ECE per dataset and method).
   - Headline continuous-FM number (Phase N best KL_bi).
   - Phase P combined-system numbers vs each track alone.
   - Calibration plots and reliability diagrams.
2. **Updated `REPORT.md`** (or new `REPORT_v2.md`) with sections on:
   - Real-hallucination UQ results.
   - Continuous-FM at the published-method level (SFM).
   - Two-model pipeline integration.
3. **W&B Reports** — at least one consolidated report per phase, link
   in `DECISION_LOG.md`.
4. **Final-best symlinks**:
   - `runs/best_uq.pt` → highest-AUROC UQ model from Phase K/L.
   - `runs/best_continuous.pt` → SFM checkpoint if Phase N
     passed; else stays unset.
   - `runs/best_pipeline.json` → settings of the best Phase P
     combined predictor.
5. **Final figure set**:
   - `runs/phaseK_uq_compare.png` (UQ AUROC + ECE bars).
   - `runs/phaseM_localisation.png` (per-token attribution).
   - `runs/phaseN_sfm_kl.png` (SFM training curves vs DFM).
   - `runs/phaseP_pipeline.png` (combined system reliability).

The user reads `DECISION_LOG.md` and the W&B Reports first; everything
else is optional and recoverable from those.

---

## Appendix A — Quick-start commands for the executing session

Boot:

```bash
cd /home/renku/work/aitchinson-flow
tail -200 runs/DECISION_LOG.md
ls runs/ | grep -E '^(hal_|tqa_|loc_|sfm_|pipe_)' | sort -r | head -20
```

Phase K (the canonical first run after this protocol lands):

```bash
# 1. Cache HaluEval-QA features.
python scripts/cache_hallueval.py --lm gpt2 --max-n 10000 --L 256 \
  --out data/hallueval_cache_gpt2.pt --lm-batch-size 8

# 2. UQ eval.
python scripts/eval_uq.py \
  --cache data/hallueval_cache_gpt2.pt \
  --train-frac 0.8 --pool meanpool \
  --out runs/hal_gpt2_qa_meanpool/uq_eval.json

# 3. Plot.
python scripts/plot_uq_calibration.py \
  --json runs/hal_gpt2_qa_meanpool/uq_eval.json \
  --out runs/hal_gpt2_qa_meanpool/uq_calibration.png

# 4. Decision-log entry. Use the Decision Log template (§4).
```

Decision-log append (template):

```bash
cat >> runs/DECISION_LOG.md <<'EOF'

## [<UTC timestamp>] <run-name>
- Hypothesis: <one line>
- Result: AUROC=… ECE=… vs SE baseline AUROC=…
- Decision: <continue|branch|abandon>
- Next: <next-cell>
- W&B: <url>
EOF
```

---

## Appendix B — Phase-decision tree

```
START
  │
  ▼
Phase K (HaluEval-QA UQ) — run first
  │
  ├─ K passed (AUROC ≥ 0.85, ECE ≤ 0.10) ──┐
  │                                        │
  └─ K partial / failed                    │
       │                                   │
       ▼                                   │
   try larger LM (Qwen2.5)                 │
       │                                   │
       ├─ now passes ──┐                   │
       └─ still fails ─┤                   │
                       ▼                   ▼
                  Phase M (localisation) ←─┤
                       │                   │
                       ▼                   │
                  Phase L (TruthfulQA UQ)  │
                       │                   │
                       └─────────┬─────────┘
                                 ▼
                        Phase N (SFM)
                                 │
                                 ├─ N passed (KL_bi ≤ 0.30) ──┐
                                 │                            │
                                 └─ N failed (KL_bi ≥ 1.0) ─→ Phase O (optional)
                                                              │
                                                              ▼
                                                     Phase P (integration) ←─┘
                                                              │
                                                              ▼
                                                            END
```

---

## Appendix C — Risk registry

| Risk | Likelihood | Mitigation |
|---|---|---|
| HaluEval format incompatible with our cache schema | Low | Adapt the loader; fields are documented at the dataset's HF page |
| Qwen2.5-1.5B caching exceeds 20 GB MIG slice | Medium | Use fp16 (`torch_dtype=torch.float16`); cache one chunk at a time |
| SVGP training is too slow at full HaluEval n=10000 (cubic in n_inducing) | Low | Already capped at 64 inducing points in `phaseF_uq.py`; no change needed |
| SFM's spherical interpolation has NaN at antipodal points | Medium | Clamp `acos` argument to `[-1+eps, 1-eps]`; add unit test |
| TruthfulQA labels are subtle and the SVGP can't learn them | Medium-high | Documented as a partial result; the negative is informative |
| Phase P combined classifier overfits the 2-feature head | Low | Use closed-form ridge LR with strong prior; cross-validate on held-out |
| `bitsandbytes` not available, larger LMs hit memory | Already mitigated | Use fp16; document if it ever becomes binding |
| Existing cache (`data/wiki_cache_gpt2.pt`) gets overwritten | Low | Use distinct filenames per dataset; never reuse `wiki_cache_gpt2.pt` |
| Different sessions modify `DECISION_LOG.md` simultaneously | Very low | The session is solo-driven; lock file `runs/<phase>.start` if needed |
| `runs/sweep_results.jsonl` row format drifts | Low | All rows produced by `run_sweep.py` follow the same schema |

---

## Appendix D — Files this protocol creates or modifies

### Created

- `src/aitchinson_flow/data/hallueval.py` (Phase K)
- `src/aitchinson_flow/models/sfm.py` (Phase N)
- `src/aitchinson_flow/config.py` — new `SFMConfig` dataclass
- `scripts/cache_hallueval.py` (Phase K)
- `scripts/cache_truthfulqa.py` (Phase L)
- `scripts/eval_uq.py` (Phase K, generalised from `phaseF_uq.py`)
- `scripts/localise_uq.py` (Phase M)
- `scripts/eval_continuous_bpe.py` (Phase O, optional)
- `scripts/integrate_pipeline.py` (Phase P)
- `scripts/plot_uq_calibration.py` (Phase K)
- `scripts/plot_pipeline.py` (Phase P)
- `sweeps/phaseK_hallueval.yaml`
- `sweeps/phaseN_sfm.yaml`
- `sweeps/phaseO_bpe_continuous.yaml` (conditional)
- `data/hallueval_cache_<lm>.pt` (Phase K cache)
- `data/truthfulqa_cache_<lm>.pt` (Phase L cache)
- `data/hallueval_annotations.json` (Phase M, manual annotation file)

### Modified

- `src/aitchinson_flow/models/__init__.py` — register SFM
- `scripts/eval_full.py` — add SFM sample-decoding path
- `runs/DECISION_LOG.md` — append-only
- `runs/sweep_results.jsonl` — append-only

### Never modified

- `RESEARCH_FINDINGS.md`, `RESULTS.md`, `TRAINING_PLAN.md`,
  `CLUSTER_TRAINING_PLAN.md`, `SAMPLER_FINDINGS.md`,
  `SESSION_SUMMARY.md`, `TRAINING_PROTOCOL.md`, `REPORT.md`,
  `CAPSTONE_PLAN.md` (this protocol's companion files).
- `CLAUDE.md` (project conventions).
- `runs/{baseline_5ep,data_50k_ep5,dfm_data50k_ep5_v2,
  fmclr_data50k_ep5_v2,lkflow_data50k_ep5,aud_*}/` (existing
  checkpoints — read-only references).

---

*End of protocol. Read `CAPSTONE_PLAN.md` first for strategy, then
start at Phase K.*
