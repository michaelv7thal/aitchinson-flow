# Plan: add Fisher Flow Matching as a ninth benchmark arm

Written 2026-08-04 from the capstone-paper session. Target: a fresh Claude Code session
working in this repo (`~/projects/aitchinson-flow`).

## 0. Goal and definition of done

Add **Fisher Flow Matching** (Davis et al., NeurIPS 2024, arXiv:2405.14664, bib key
`davis2024fisherfm`) as a ninth arm of the budget-matched generation benchmark, so the
paper's coverage of the Fisher-Rao family is complete.

**This is a completeness exercise, not a search for a better model.** Dirichlet Flow
Matching is the selected generator and stays selected. The expected outcome is a ninth
row that lands in the unigram-collapse group next to SFM (`KL_bi` 1.494,
`Δ@.50 = -0.018`). If the new arm beats Dirichlet FM at the matched budget, stop and
report that before doing anything else, because it invalidates a selection claim the
paper already makes.

Done means:

1. `runs/sflm_bench_a100_20g_L256/FisherFM/{epoch_final.pt,train_meta.json,eval_all.json,recovery.json}` exist.
2. The numbers listed in §9 are reported back for `chapters/results.tex`.
3. The module docstring records exactly which recipe was implemented and which paper
   choices were ambiguous (§1).

**Do not** touch the eight existing arms' artifacts, and **do not** enable TF32, bf16, or
`autocast` anywhere. The whole benchmark trains in fp32 with
`float32_matmul_precision=highest`; changing precision for one arm destroys
comparability with the other eight.

## 1. Read the paper first (blocking step)

Do not write code before this. `models/sfm.py` implements **Cheng et al. 2024**
(arXiv:2405.16441, Statistical Flow Matching). Davis et al. is a *different* paper that
shares the same underlying geometry: the simplex mapped to the positive orthant of
`S^{K-1}` by `π: μ ↦ √μ`, under which Fisher-Rao becomes the round-sphere metric. So most
of what distinguishes the two arms is *not* geometry.

Answer these six questions from the paper and write the answers into the new module's
docstring:

1. **Source distribution.** Cheng uses `μ₀ ~ Dir(1,…,1)` mapped through `π`. What does
   Davis use?
2. **Target.** Cheng puts `x₁` exactly at the sphere vertex `e_c = √(one-hot)`. Does
   Davis smooth the target into the interior, and if so how?
3. **Coupling.** This is the delta I expect to matter most: my recollection is that
   Fisher-Flow's headline contribution over a plain geodesic interpolant is a
   **Riemannian / optimal-transport coupling** between source and target that straightens
   paths. Confirm whether it is a minibatch OT plan (OT-CFM style), a closed form on the
   sphere, or neither. **Treat that recollection as a hypothesis, not a fact.**
4. **Velocity parameterisation and loss.** Cheng regresses a tangent-projected network
   output against the constant-speed geodesic velocity under MSE. What is Davis's target
   and is there any time weighting?
5. **Sampler.** Exponential-map Euler on the sphere, projected Euler, or something else,
   and at what NFE.
6. **Reported text8 number**, if the paper has one, plus which metric it is.

If any of these cannot be pinned down confidently from the paper, **do not invent it**.
Implement the closest defensible reading and record the ambiguity verbatim in the
docstring, because the paper has to state what was actually implemented rather than
claiming to reproduce Fisher-Flow.

## 2. What already exists and must be reused

`src/aitchinson_flow/models/sfm.py` is the template. It already contains, as
module-level helpers acting on the last axis:

| helper | what it does |
|---|---|
| `_simplex_to_sphere(mu)` | `π: μ ↦ √μ` |
| `_sphere_to_simplex(x)` | `π⁻¹: x ↦ x²`, renormalised |
| `_normalize(x)` | project onto `S^{K-1}` |
| `_geodesic(p, q)` | great-circle distance |
| `_log_map(p, q)`, `_exp_map(p, v)` | sphere log/exp |
| `_project_tangent(p, v)` | project a raw `R^K` output into `T_p S` |

Import these from `aitchinson_flow.models.sfm` rather than copying them. Two eval scripts
already import them under aliases (`scripts/eval_generation.py:56-57`), so they are
effectively public.

The arm also reuses `DFMBackbone` + `DFMHead` from `transformer_backbone.py`, exactly as
SFM does. That is what makes the comparison compute-matched: same trunk, same learned
positional embedding, same sinusoidal time conditioning through `t_proj`.

## 3. Files to touch

| file | change |
|---|---|
| `src/aitchinson_flow/config.py` | new `@dataclass FisherFMConfig` (mirror `SFMConfig` at line 691: `sample_nfe`, `t_eps`, plus whatever §1 turns up, e.g. an OT-coupling flag), and a `fisher_fm: FisherFMConfig` field on `Config` next to `sfm:` |
| `src/aitchinson_flow/models/fisher_fm.py` | **new**: `class FisherFlowMatching(nn.Module)` + `@register("FisherFM")` builder |
| `src/aitchinson_flow/models/__init__.py` | import `FisherFlowMatching` and add to `__all__` |
| `scripts/train_for_sflm_bench.py` | `ARM_TO_MODEL["FisherFM"] = "FisherFM"` with a comment naming the paper, plus an `elif name == "FisherFM":` branch in `_model_cfg` (defaults are the recipe unless §1 says otherwise) |
| `scripts/eval_generation.py` | add to `GEN_ARMS` (line 74) and `_IDENTITY_PATH_BPC` (line 90); extend the `arm in ("DirichletFM", "SFM")` test in `_generate_ids` (line 171); extend the `elif arm == "SFM"` masked-inpaint branch (line 347) and the `if arm == "SFM"` partial-path Δ branch (line 419) |
| `scripts/eval_all.py` | line 108: the `cfg.training.model_name != "SFM"` guard must also exclude `FisherFM` |
| `scripts/recovery_check.py` | line 243 `is_sfm` predicate and the `elif is_sfm:` partial-path branch (line 362) |
| `scripts/aggregate_results.py` | optional: line 51 is the published-frontier table; add a Fisher-Flow row only if §1 question 6 yields a real text8 number |

**Refactor rather than scatter.** Those `== "SFM"` literals appear in four scripts. Replace
each with a module-level `_FISHER_SPHERE_ARMS = frozenset({"SFM", "FisherFM"})` and test
membership. It is the same diff size and it stops the next arm from missing a site.

## 4. API contract (this is what keeps the diff small)

Make the new class's public surface **byte-identical to `StatisticalFlowMatching`**:

```python
def velocity(self, x, t) -> Tensor            # (B,L,K) tangent at x
def training_step(self, batch, step) -> LossDict   # reads batch["token_ids"]
def eval_step(self, batch) -> LossDict
def decode_to_logprobs(self, x) -> Tensor
def decode_to_token_ids(self, x) -> Tensor
def _sphere_target(self, token_ids) -> Tensor      # used by recovery_check.py
@torch.no_grad()
def sample(self, B, L, *, nfe=None, max_steps=None,
           x_init=None, t_start=None, **_) -> Tensor   # returns token IDS, not latents
@torch.no_grad()
def bpd(self, token_ids, *, max_steps=None) -> Tensor  # diagnostic only, never a bound
```

`sample()` returning ids and honouring `x_init` / `t_start` is what lets every dispatch
site above be a one-string change. `batch["token_ids"]` is the only batch key the arm
needs; the collate also emits `x` (CLR features) but this arm ignores it.

The loss dict must use `TRAINING_LOSS_KEY` for the backward-able entry, like every other
model.

## 5. Known wart to mirror deliberately, not fix

SFM's **training** source is `Dir(1,…,1) → √·`, which lies in the positive orthant. Its
**recovery** init is `_normalize(torch.randn_like(x1))` (`eval_generation.py:352` and
`:428`, `recovery_check.py:379`), a uniform point on the *full* sphere, which generally leaves
the orthant and is therefore off the training source distribution.

Mirror SFM exactly, wart included, so the `Δ@.50` column compares like with like. If you
think the wart is worth fixing, fix it for **both** arms and re-run SFM's recovery, and
say so in the report; a half-fix silently mixes two protocols in one table column. It may
also be part of why SFM's `Δ@.50` is negative, but that is a hypothesis, not a finding,
and it is out of scope here.

## 6. Pre-flight checks before spending two hours of GPU

Write these as a throwaway script under `/tmp`, not as repo tests (there is no test
suite):

1. `_exp_map(x, _log_map(x, y)) ≈ y` for random orthant points, and `‖x‖ = 1` after each.
2. The interpolant at `t ∈ {0, 0.5, 1-t_eps}` stays on the sphere and, if §1 says the
   path should stay in the positive orthant, stays non-negative.
3. `velocity(x, t)` is tangent: `(x * v).sum(-1) ≈ 0`.
4. `decode_to_token_ids(_sphere_target(ids)) == ids`.
5. If a coupling is implemented: with the coupling disabled the loss must reproduce the
   plain-geodesic (SFM-shaped) objective, so the coupling is provably the only delta.

Then a smoke run:

```bash
.venv/bin/python scripts/train_for_sflm_bench.py \
    --scale a100_20g_L256 --only FisherFM --epochs 1 --max-train-windows 400
```

Check that the loss decreases, then delete `runs/sflm_bench_a100_20g_L256/FisherFM/`
before the real run (the trainer skips arms that already have `epoch_final.pt`; `--force`
overrides).

## 7. Run commands

Matched budget = the `a100_20g_L256` scale: `d_model=1024`, 10 layers, 16 heads, batch 8,
`L=256`, 10 epochs × 10,000 windows = 12,500 steps. Nothing needs overriding.

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=runs/sflm_bench_a100_20g_L256

# 1. train (~2 h, see §8)
.venv/bin/python scripts/train_for_sflm_bench.py \
    --scale a100_20g_L256 --only FisherFM --epochs 10 --seed 42

# 2. generation scorecard -> $R/FisherFM/eval_all.json
.venv/bin/python scripts/eval_all.py \
    --ckpt $R/FisherFM/epoch_final.pt --split test --n 256 --steps 200

# 3. recovery ladder -> $R/FisherFM/recovery.json  (--out is REQUIRED or nothing is written)
.venv/bin/python scripts/recovery_check.py \
    --ckpt $R/FisherFM/epoch_final.pt --alphas 0.1,0.3,0.5,0.7,1.0 \
    --n 256 --steps 200 --out $R/FisherFM/recovery.json
```

`scripts/eval_generation.py --scale a100_20g_L256 --no-auto-train --only FisherFM --n 256
--steps 200 --out <path>` is the alternative harness and also exercises the infill path;
run it too if you want the `recover`/`infill` dicts, but the paper's two columns come
from `eval_all.json` and `recovery.json`.

## 8. Expected wall-clock and memory (measured on the target machine)

Measured on 2026-08-04 on the laptop this will run on (NVIDIA RTX PRO 1000 Blackwell,
7.57 GiB, 20 SMs), by building the arm at this exact scale and timing real
forward/backward/AdamW steps on synthetic batches:

| arm | step time | throughput | 12,500 steps | peak VRAM allocated |
|---|---|---|---|---|
| SFM (closest proxy) | 520 ms | 1.92 it/s | **1.81 h** | 2.96 GiB |
| FM on CLR | 546 ms | 1.83 it/s | 1.90 h | 2.93 GiB |

Reference: the same FM-on-CLR arm took **1:30:26** on the A100 20 GB MIG slice, so this
machine is only 1.27× slower, because fp32 leaves both GPUs' tensor cores idle.

Sampling: 200 sampler steps for a batch of 8 at `L=256` took 33 s (0.55 GiB), so 256
samples is ~18 min at batch 8 and less if you raise the sampling batch. The recovery
ladder is 5 α values plus an unconditional pass, so budget 1 to 1.5 h. **End to end
roughly 4 h unattended for the one arm.**

Memory caveat: the `SCALES` comment for this scale claims "~11 GB peak on the 20 GB
slice". That figure was almost certainly calibrated on the EqM family, whose
`create_graph=True` double-backward roughly doubles the graph. A single-backward
geodesic-FM arm measured at 3 GiB here. If the real run with the data loader attached does
OOM, the ladder is already built in: `--grad-checkpointing`, then a smaller `--batch`
(which changes the budget, so report it if you use it).

## 9. What to report back for the paper

The paper session needs exactly these, plus the docstring from §1:

- From `eval_all.json`: `KL_uni`, `KL_bi`, `KL_tri`, `H_ratio`, `bpc`,
  `generation_metric_valid`, `collapsed`.
- From `recovery.json`: the `delta` of the `alpha = 0.5` row. That is the paper's
  `Δ_{@.50}` column (verified: SFM's `-0.0178` is the table's `-0.018`).
- One or two generated samples, for the qualitative read.
- Whether `bpc` is a real bound. It will not be: the arm has no exact likelihood, so it
  belongs in `_IDENTITY_PATH_BPC` and the paper prints `---`.
- The recipe decisions and ambiguities from §1, in prose.

Framing constraint for the write-up: this is **our budget-matched reimplementation of
Fisher Flow Matching**, never "Fisher-Flow's performance". The paper already uses that
framing for the Cheng arm at `chapters/conclusion.tex:69`.

For reference, the arm count is stated in six places in the paper and all of them move
from eight to nine: `frontmatter/abstract.tex:12`, `chapters/introduction.tex:72`,
`chapters/conclusion.tex:15`, `chapters/results.tex:17`, the `tab:gen` caption
(`chapters/results.tex:47`), and the "four remaining transport arms" sentence plus its
`KL_uni ≤ 0.019` / `KL_bi ≥ 1.378` bounds (`chapters/results.tex:41`). The new row also
needs a bullet in `chapters/methods.tex:122-128` and a line in the `app:config` table.
That editing is the paper session's job, not yours.

## 10. Non-goals

- No OOD or repair work with this arm. The detector and the healing procedure ride on the
  frozen Dirichlet FM backbone and stay there.
- No exact CNF likelihood / peer-comparable BPC.
- No hyperparameter search. One seed (42), one budget, defaults from the paper.
- No changes to the eight existing arms, their configs, or their artifacts.

## Result (added 2026-08-20)

**Executed 2026-08-04/05. Done.** The arm trained and became the ninth row of
`tab:gen`: KL_uni 0.006, KL_bi 1.418, KL_tri 5.151, H_ratio 0.996, Δ@.50 −0.000
— landing in the unigram-collapse group beside Statistical FM's 1.494, as §0
predicted, with Dirichlet FM's selection claim intact. The §1 recipe answers
are recorded in the paper's `tab:config`, "Fisher FM" row. The arm count was
updated to nine in all six locations listed in §9.
Artifact: `runs/sflm_bench_a100_20g_L256/FisherFM/` (the paper reads this, the
**smoothed** target). A `FisherFM_nosmooth/` control also exists at KL_bi 1.578
— the ablation of the §1-question-2 ambiguity — and is not reported in the paper.
