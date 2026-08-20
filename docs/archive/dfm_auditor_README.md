# DFM auditor — file index


> **ARCHIVED (2026-08-20). Retired research line; ten of the files this index maps no longer exist.** This indexed the GPT-2 / HaluEval "DFM auditor" iteration of 2026-05. That line was retired (see `CLAUDE.md`), it contributed nothing to the capstone paper, and every training, caching and eval script listed below was deleted by commit `7474f16` ("Prune retired auditor/UQ/phase scripts"). Still present: `src/aitchinson_flow/models/dirichlet_fm_auditor.py`, `src/aitchinson_flow/data/hallueval_dfm.py` (its config fields are documented inline in `config.py`), and the `runs/dfm_auditor_*` result files. Kept as the record of what was tried. Do not follow any path in it without checking that the path exists.

Pointers to all the code and docs added in this iteration. The runbook
(`docs/cluster_runbook_dfm_auditor.md`) is the operational entry point;
this file is a map of what's where for code review or debugging.

## Models

| file | what's in it |
|---|---|
| `src/aitchinson_flow/models/dirichlet_fm.py` | Stark et al. 2024 Dirichlet FM for unconditional text8 generation (existing). |
| `src/aitchinson_flow/models/dirichlet_fm_auditor.py` | **Auditor variant.** Three context modes (`off` / `hidden_only` / `product_concat`), closed-form mixture-of-Dirichlets EBM, conditional sampler, optional dual-head joint training (Architecture B). |

Key methods on `DirichletFMAuditor`:

* `forward(x_t, t, h_ctx)` — denoiser slot logits.
* `forward_features(x_t, t, h_ctx)` — encoder output (for SVGP / Arch A).
* `energy_at_lm_distribution(batch, t)` — closed-form unsupervised UQ.
* `halluc_score_at_lm(batch, t)` — supervised UQ (Architecture B only).
* `sample_conditional(h_ctx, nfe)` — Phase-H-style generation.
* `training_step(batch, step)` — joint slot-CE + halluc-BCE when
  `cfg.dirichlet_fm.joint_halluc=True`.

## Data

| file | what's in it |
|---|---|
| `src/aitchinson_flow/data/hallueval_dfm.py` | Paired-cache datamodule: pairs `(hidden, topk)` caches, propagates row labels to per-position, pair-level train/val split. |

## Caching scripts

| file | what's in it |
|---|---|
| `scripts/cache_hallueval.py` | Original GPT-2-only hidden-state cache (existing). |
| `scripts/cache_hallueval_topk.py` | Original GPT-2-only top-K + paper-ΔE cache (existing). |
| `scripts/cache_hallueval_llama.py` | **Unified script.** Single LM forward pass produces both caches; supports any HF causal LM (GPT-2, Llama-1B/3B/7B, Mistral-7B); fp16 / bf16 / 4-bit precision. |

## Training scripts

| file | what's in it |
|---|---|
| `scripts/run_dirichlet_fm_auditor.py` | Slot-only DFM auditor (the original, unsupervised path). |
| `scripts/run_dirichlet_fm_auditor_archB.py` | **Architecture B.** Joint dual-head (slot CE + halluc BCE), auto pos-weight, supervised UQ. |
| `scripts/run_dirichlet_fm_auditor_smoke.py` | (existing) text8 smoke. |

## Evaluation scripts

| file | what's in it |
|---|---|
| `scripts/run_dfm_auditor_archA.py` | **Architecture A.** Post-hoc supervised SVGP on a frozen DFM encoder; reports row + per-tok AUROC + ECE + locality split. |
| `scripts/run_dfm_auditor_archA_baseline.py` | Raw-h_LLM SVGP control (no DFM encoder), used to set the realistic supervised ceiling at this dataset size. |
| `scripts/eval_dfm_auditor_generation.py` | Phase-H NLL eval — conditional generation, decoded text grid, GPT-2 NLL scoring. |
| `scripts/compile_dfm_auditor_results.py` | Aggregates `summary.json` files across runs into a markdown table. |

## Results / docs

| file | what's in it |
|---|---|
| `runs/dfm_auditor_v1/results.md` | First-pass (d=192/L=4) result. AUROC 0.78. |
| `runs/dfm_auditor_d512_results.md` | Three-way mode ablation at d=512/L=6 + Phase-H generation samples. |
| `runs/dfm_auditor_archA_results.md` | Architecture A findings + Architecture B target metrics. |
| `docs/cluster_runbook_dfm_auditor.md` | **Operational guide** — pull, cache, train, eval. |

## Config additions (`src/aitchinson_flow/config.py`)

`DirichletFMConfig` gained:

* `context_features: str = "off"` — `"off" | "hidden_only" | "product_concat"`.
* `ctx_hidden: int = 768` — raw LM hidden dim.
* `ctx_proj_dim: int = 64` — projection size before concat.
* `energy_t: float = 4.0` — Dirichlet path time at which the EBM is evaluated.
* `joint_halluc: bool = False` — Architecture B flag.
* `lambda_slot / lambda_halluc / halluc_pos_weight` — joint loss knobs.

`HalluevalDFMAuditorConfig` (new dataclass):

* `enabled, topk_cache_path, hidden_cache_path, batch_size, train_frac, max_rows`.

## Headline results so far (GPT-2 small backbone, 4000-row HaluEval-QA)

| signal | row AUROC | per-tok ans | per-tok non-ans | source |
|---|---:|---:|---:|---|
| top-K entropy | 0.54 | n/a | n/a | zero-train |
| paper ΔE | 0.71 | 0.50 | 0.51 | zero-train |
| Hilbert-FM trajectory best (Phase Q) | 0.58 | 0.55 | n/a | unsupervised |
| **DFM EBM** (this work, unsupervised) | **0.81** | n/a | n/a | denoiser CE on clean rows |
| Architecture A — SVGP on slot-encoder | 0.85 | 0.71 | **0.53** ← clean locality | supervised post-hoc |
| Raw h_LLM SVGP, per-pos pool | 0.90 | 0.51 | 0.50 | supervised |
| **Architecture B ep 3** (early-stop, transformer) | 0.93 | 0.87 | 0.78 | joint slot+halluc |
| **Architecture B ep 5** (best row, transformer) | **0.985** | 0.97 | 0.98 (cascade lost) | joint slot+halluc |
| Architecture B + SVGP head (transformer) | 0.982 | 0.96 | 0.98 | best calibration (ECE 0.010) |
| **Architecture B + MLP backbone** ★ | **0.978** | 0.91 | **0.55** ← locality preserved | half the params; cascade-clean |
| Phase K SVGP @ 20000 rows (ceiling) | 0.996 | n/a | n/a | supervised, full data |

★ The MLP backbone — weight-shared per-position FFN with no
cross-positional attention — preserves locality by construction. Set
with `--backbone mlp` on either runner. **This is the architecturally
correct choice for per-position UQ**: 43-AUROC improvement on
non-answer locality (cascade contamination eliminated) at only −0.7
AUROC on row-level discrimination.

Generation NLL (Phase H protocol, n=32 held-out clean prompts, GPT-2 scored):

| | NLL |
|---|---:|
| EqM Phase H reference | 8.89 (FAIL) |
| **DFM `off`** | **5.70** (borderline PASS, 5.5 floor) |
| **DFM `hidden_only`** | 6.06 (PARTIAL) |
| **DFM `product_concat`** | 6.23 (PARTIAL) |
| Random-slot baseline | 7.4 |
| Clean reference | 3.20 |

## Outstanding cluster runs

The runbook (§2a-c) targets these:

1. **Llama-3.2-1B Architecture B** (full 10000 pairs). Target: row AUROC > 0.842 (EigenTrack-Llama-1B SOTA).
2. **Llama-2-7B Architecture B** (full 10000 pairs, 4-bit). Target: row AUROC > 0.894 (EigenTrack-Llama-7B SOTA) — ambitious.
3. Three-mode ablation on Llama-1B to confirm the GPT-2 finding (`off` is best generator, `hidden_only` is best UQ).
4. Phase-H generation eval on the best Llama-Architecture-B checkpoint. Target: NLL ≤ 5.5 (PASS).
