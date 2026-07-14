# RESULTS_README — objective → script → artifact → manifest map

The reproducibility map for the capstone L=256 cloud run. Pairs each **objective**
and **experiment** (`capstone_experiment_runbook.md`) with the **canonical script**,
its **output artifact**, and the **manifest `exp_id`**. Start from
`capstone_experiment_runbook.md` (the execution plan) and `EVAL_ASSESSMENT.md`
(which eval means what); this file is the index + the resume contract.

## Standard eval protocol (peer comparability)
- Standard split (last 5M = test); context **L=256**.
- **BPC only via a real bound**: `DFM.elbo_bpc` (`n_mc=8`) — the D3PM uniform-process
  variational NLL upper bound. Honest peer: **D3PM-uniform ≈1.61**; SEDD 1.32 /
  D3PM-absorb 1.45 / MDLM ≤1.38 / SFM 1.39 are absorbing/score methods → reference,
  not head-to-head.
- Identity-path arms {EqM, EqM_OneHot, EqMLatent, SFLM} and any non-finite or
  `<0.5` text8 BPC emit **`—`** (`generation_metric_valid=False`). DirichletFM's
  high-t denoiser number (~0.36) is demoted by the same guard.
- Headline per objective: **(1)** KL_bi/KL_tri + H_ratio + BPC-or-`—`; **(2)** Δ@.50;
  **(3)** corruption-ladder AUROC on the **shuffle** axis (anchored to `gpt2_baseline`).

## Harness (built; use, don't rebuild)
| Component | Script | What it gives |
|---|---|---|
| X1 unified eval | `scripts/eval_all.py` | `evaluate(ckpt, …)` → KL_uni/bi/tri, H_ratio, per_pos_entropy, BPC-or-`null`, `generation_metric_valid`, `collapsed`, samples |
| X2 manifest | `scripts/manifest.py` | `results/manifest.jsonl` append/lookup/`is_done`/load_all; CLI `--list`/`--get` |
| X2 runner | `scripts/run_experiment.py` | `run_experiment.py <exp_id> [--force]`: skip-if-done, capture git/gpu/wall/mem, retry-once, write record |
| X3 map | `RESULTS_README.md` | this file |
| **Obj-3 OOD bench** | `scripts/run_bench_ood.py` → `scripts/bench_aggregate.py` | 6-detector corruption-ladder sweep (NLL/BLR/BLR_ADV/BLR_FI/BGMM/GPT2-SE) + latent-split + plausible → `bench_ood/RESULTS.md` + figs; self-contained `bench_ood/manifest.json` (git SHA + `git_dirty` + `repro.patch` + `new_scripts/` + per-arm CLI) |
| **Obj-3 healing bench** | `scripts/run_bench_heal.py` (`--aggregate-only` to re-consolidate) | localize→inpaint across localizers × {replace, falseinfo} + insulin rows → `bench_heal/RESULTS.md`; operating point = least-damaging (max net/corrupt); deployable selection = cal-set F0.5 |

> **Obj-3 note.** These two benches are a *dedicated, self-contained* harness (own
> `manifest.json` + repro patch), not wired into `results/manifest.jsonl`. Write-up:
> **`RESULTS.md` §Objective 3**. Shared utils: `scripts/_bench_common.py`
> (`finalize_manifest`, `DETECTOR_KEYS`, `heal_style_examples`,
> `make_adversarial_negatives`).

## Experiment registry

| exp_id | Obj | Script | Artifact | Notes |
|---|---|---|---|---|
| **E1** | 1 | `scripts/train_for_sflm_bench.py --scale a100_20g_L256 --seeds 42,43,44 [--full-split]` → `scripts/eval_all.py` | `runs/sflm_bench_a100_20g_L256/<arm>[/seed<s>]/epoch_final.pt`, `eval_all.json` | all families @ L=256, 3 seeds; **DFM** reports `elbo_bpc`; identity-path `—` |
| **E1b** | 1 | `train_for_sflm_bench.py` at L∈{40,128,256} | per-L run dirs | length-effect sweep (P1) |
| **E2a** | mech | `scripts/ablate_training_signal.py` | `ablate_training_signal.json` | 4 targets × 2 recipes; KL_uni/KL_bi/Δ@.50/collapsed |
| **E2b** | mech | `scripts/ablate_sampler.py --ckpt <EqM>` | `ablate_sampler.json` | NAG/Euler(use_grad T/F)/SDE on a fixed EqM ckpt; annealed-Langevin scoped to the DSM arm |
| **E2c** | mech | (folded into `eval_all.py`) | — | collapse quantification |
| **E2d** | mech | `scripts/probe_field_geometry.py --ckpt <EqM/FMonCLR>` | `field_geometry.json` | curl fraction + cos(g,g*) vs γ (P1) |
| **E2e** | mech | `scripts/ablate_hyperparams.py --mode eps\|t` | `ablate_hyperparams_*.json` | smoothing-ε → generation KL; OOD feature-t → shuffle AUROC (P1) |
| **E3a** | 2 | `scripts/recovery_check.py --ckpt <ckpt>` | recovery json | Δ@α (headline Δ@.50); supports DFM/DirichletFM + EqM-family |
| **E3b** | 2 | `scripts/eval_healing.py --ckpt <ckpt>` | healing json/fig | recovery vs corruption rate |
| **E4a** | 3 | `scripts/fit_dfm_svgp_hinge.py --ckpt <DirichletFMSvgp>` → `scripts/sweep_dfm_svgp_corruption.py` | `model_with_svgp_hinge.pt`, `svgp_corruption_sweep.json` | needs a **DirichletFMSvgp** Stage-1 (e.g. `runs/dfm_svgp_L256/epoch_final.pt`), NOT the DFM arm |
| **E4b** | 3 | `scripts/eval_ood.py --ckpt <EqM/EqMLatent/SFLMEBM>` | ood json | native-energy AUROC + sign_inverted (encode-dispatch fixed) |
| **E4c** | 3 | `scripts/eval_ood.py` (DFM denoiser proxy) | ood json | DFM denoiser-NLL OOD |
| **E4d/E4e** | 3 | `scripts/eval_ood_baselines.py --ckpt <DFM>` | `ood_baselines.json` | DFM-ELBO-as-density baseline + constant-char/valid-perm controls |
| **E4f** | 3 | `scripts/ablate_hinge_vs_fm.py --dfm-ckpt <…>` | `ablate_hinge_vs_fm.json` | A FM+hinge / B random+hinge / C end-to-end / D FM+probe; replace→shuffle transfer |
| **E4g** | 3 | `scripts/bench_sflm_ebm.py --ref-lm gpt2` | `bench.json` (`gpt2_baseline` row) | external spilled-energy anchor |
| **E5a** | 1 | `scripts/train_logit_kl_flow.py --scale a100_20g_L256` | `runs/logitkl_*/epoch_final.pt` | LogitKLFlow (model `logitkl_flow.py`); P2 upside |
| **E1b** | 1 | `train_for_sflm_bench.py --scale a100_20g_L256 --length {40,128}` | `runs/…/<arm>_L<n>/` | length sweep; `--length` overrides scale L, dir gets `_L<n>` suffix (P1) |
| **E5b** | 1 | `train_for_sflm_bench.py --only SFLM` (existing hyperspherical arm) | run dir | Fisher-Rao/SFM contingency; dedicated √p model unbuilt (gated on E5a), P2 |
| **E6a** | 3 | `scripts/cache_llm_features.py --model Qwen/Qwen2.5-1.5B --4bit` | `runs/llm_cache/…` memmaps | cache LLM features (P2) |
| **E6b** | 3 | `scripts/train_eqm_auditor.py --cache <E6a out>` | `eqm_auditor_parity.json` | EqM-energy auditor vs spilled-energy parity (tok/seq AUROC), P2 |
| **E7** | all | `scripts/aggregate_results.py` | `results/RESULTS.md`, `results/figs/` | 4 tables + 4 figs from `results/manifest.jsonl`, every cell with provenance |

## How to resume (idempotent)
- `run_experiment.py <exp_id>` **skips** anything with a `done` record in
  `results/manifest.jsonl` (use `--force` to re-run).
- `train_for_sflm_bench.py` **skips** finished arms (`epoch_final.pt` present)
  unless `--force`; an interrupted multi-arm/seed chain resumes cleanly.
- **Early stopping**: `--early-stop-patience N` stops after N val-evals with no
  improvement and restores the best checkpoint as `epoch_final.pt` (implies
  `--val-eval`); `--max-hours 36` is the wall-clock cap.
- On a CUDA OOM or the MIG-NVML allocator assert, an arm escalates the
  **memory-fallback ladder** (grad-checkpointing → halve batch → quarter →
  L=128, recorded in `train_meta.json`) and, if still failing, writes
  `FAILED.json` and the queue **continues** (never aborts).
- `--full-split` trains on the entire `afmck/text8` split with lazy CLR features
  (memory scales with batch); `--seeds 42,43,44` for the 3-seed headline (seed 42
  → canonical dir, others → `<arm>/seed<seed>/`).

Every script exposes `--smoke` (tiny CPU sanity, no large download) — run it
before any full run of a new/modified script (the §1 "smoke first" policy).
