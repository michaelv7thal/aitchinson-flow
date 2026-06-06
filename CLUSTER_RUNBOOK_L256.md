# Cluster runbook — L=256 GPU workstream (for a fresh Claude session)

**You are a new Claude Code session running on the 20 GB compute cluster** (the
A100 MIG / equivalent). This is the **GPU-bound part** of the capstone eval work;
everything non-GPU (the assessment doc, code fixes, repo cleanup) was already done
on an 8 GB laptop box and is on this branch. Your job: produce the **L=256,
peer-length** numbers and fold them back into `EVAL_ASSESSMENT.md`.

Read `EVAL_ASSESSMENT.md` first — it defines the three objectives and what each
number means. This runbook is purely the execution recipe.

## Why L=256
Published text8 BPC (SEDD 1.32 / D3PM 1.45 / MDLM ≤1.38 / SFM 1.39; frontier
1.32–1.47) is reported at **context length 256**. Our other runs are L=40 (short,
not comparable). The first L=256 bench **crashed** (CUDA/NVML allocator assert in
the per-position readout). That crash is **already fixed** on this branch — see below.

## What is already fixed on this branch (so it "just runs")
- `models/sflm_ebm.py::position_uncertainty` now **chunks over the batch**
  (numerically identical to full-batch; verified) → bounds peak memory at L=256.
- `scripts/bench_sflm_ebm.py` wraps every per-arm readout in `_safe_cuda`: an OOM
  on one arm **logs + nulls that readout** instead of aborting the whole bench
  (you no longer lose `bench.json` to one crash); it also `empty_cache()`s between
  arms, **flags identity-path BPC** as `—` (`generation_metric_valid=False`),
  **falls back** to an existing SVGP checkpoint when L matches, and writes
  `_meta.skipped_arms`.

## 0. Environment
```bash
cd <repo root>
git fetch && git checkout capstone-project && git pull   # this branch
uv sync                                                    # project venv (uv.lock)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True    # reduce fragmentation
nvidia-smi --query-gpu=memory.total --format=csv           # confirm ~20 GB
```
All commands below are `uv run python ...` (the package lives under `src/`; `uv run`
puts it on the path). The text8 + GPT-2 weights download from HuggingFace on first
use (set `HF_TOKEN` to avoid rate limits).

## 1. (Optional but recommended) narrow the data-subset gap
The L256 scale trains on only `max_train_windows=10_000` (≈2.5M chars vs the 90M
published protocol). Raise it as far as the 20 GB MIG / your time budget allows so
BPC is less penalized by data starvation. Edit `scripts/train_for_sflm_bench.py`,
`SCALES["a100_20g_L256"]["max_train_windows"]` (e.g. 10_000 → 100_000), and note
the value you used in the report.

## 2. Train the Stage-1 arms at L=256
`--scale a100_20g_L256` = d768/8L/12H/B8/L256. Each arm writes
`runs/sflm_bench_a100_20g_L256/<ARM>/epoch_final.pt`. Budget **hours per arm**
(the first run was ~3.8 h for 10 epochs of one arm). Use a real epoch count
(≥20; 50 matches the L40 runs). Run them sequentially (or as background jobs):
```bash
uv run python scripts/train_for_sflm_bench.py --scale a100_20g_L256 --only DFM        --epochs 15
uv run python scripts/train_for_sflm_bench.py --scale a100_20g_L256 --only SFLMEBM     --epochs 15
uv run python scripts/train_for_sflm_bench.py --scale a100_20g_L256 --only SFLMEBM_FM  --epochs 15
# optional extra generators for the OOD/gen tables:
uv run python scripts/train_for_sflm_bench.py --scale a100_20g_L256 --only DirichletFM --epochs 15
```
**DFM is the priority** (it's the only arm with a peer-comparable BPC). SFLMEBM /
SFLMEBM_FM give the native-energy OOD rows (the "EBM fails on shuffle" evidence).

## 3. Fit the hinge SVGP detector at L=256 (Objective 3 "hinge works")
⚠️ **`fit_dfm_svgp_hinge.py` needs a `DirichletFMSvgp` Stage-1 checkpoint**, NOT
the §2 `DFM` arm — `train_for_sflm_bench`'s `DFM` is `models/dfm.py`, a different
model with no `fit_svgp_hinge`/pooler/SVGP. The SVGP wraps a `DirichletFM`
(`self.dfm`). So first train a `DirichletFMSvgp` Stage-1 at L256 via a sweep,
mirroring `DFM_SVGP_FINDINGS.md`:

```bash
# (a) Make an L256 Stage-1 sweep: copy sweeps/_dfm_svgp_pure50_lr3e4.yaml →
#     sweeps/_dfm_svgp_L256.yaml and set, in its overrides:
#       training.L: 256   text8_dataset.L: 256   training.B: 8
#       transformer.d_model: 768  transformer.num_layers: 8  transformer.nhead: 12
#       text8_dataset.max_train_windows: <as large as fits>   name: dfm_svgp_L256
uv run python scripts/run_sweep.py --sweep sweeps/_dfm_svgp_L256.yaml --runs-root runs
# (b) Stage-2 hinge SVGP on the frozen Stage-1, then the severity sweep:
uv run python scripts/fit_dfm_svgp_hinge.py \
    --ckpt runs/dfm_svgp_L256/epoch_final.pt \
    --out-dir runs/sflm_bench_a100_20g_L256/DFM_SVGP \
    --n-epochs 5 --lr 1e-3 --eval-n 500
uv run python scripts/sweep_dfm_svgp_corruption.py \
    --ckpt runs/sflm_bench_a100_20g_L256/DFM_SVGP/model_with_svgp_hinge.pt \
    --n 500
```
The sweep's `svgp_corruption_sweep.json` (`auroc_prob_as_ood` per scheme×rate) is
the **primary hinge-OOD evidence** — independent of bench checkpoint discovery.
Placing the checkpoint under `.../DFM_SVGP/` also lets the bench populate rows 1g/1h.

## 3b. Ablation — does FM-first matter, or is a direct hinge comparable? (optional, ~10–20 min)

Tests whether the two-stage **FM-then-hinge** design earns its keep or whether a
direct hinge would match. Four arms share the same linear energy-head hinge
readout and are **all trained on replace-only negatives**, then evaluated on the
**shuffle (order) axis** — i.e. the replace→shuffle *transfer*:
**A** FM+hinge · **B** random-backbone+hinge · **C** end-to-end hinge (no FM) · **D** FM+linear-probe.

```bash
uv run python scripts/ablate_hinge_vs_fm.py \
    --dfm-ckpt runs/dfm_svgp_L256/epoch_final.pt \
    --n-train 4000 --n-eval 1000 --epochs 3 \
    --out runs/sflm_bench_a100_20g_L256/ablate_hinge_vs_fm.json
```
Uses the §3 `DirichletFMSvgp` Stage-1 (a plain `DirichletFM` ckpt also works — the
driver remaps keys and **asserts the backbone actually loaded**, so arm A can't be
silently random). Read the printed `shuffle@1.0` row:
- **A ≈ D ≫ B** ⇒ the *FM representation* does the work, not the hinge (a plain probe on FM features matches the hinge; a hinge on a random backbone fails). **This is the case FOR FM-first.**
- **C high on replace but low on shuffle** ⇒ a direct end-to-end hinge overfits the trained corruption and does **not** transfer to order corruption.
- If instead **C ≈ A on shuffle**, FM-first is *not* buying OOD generalization — then only the generator/recovery (Obj 1/2) justify the two-stage; say so honestly.

Local mechanics check (no GPU/HF, synthetic tokens): `uv run python scripts/ablate_hinge_vs_fm.py --smoke`.
Fold the `shuffle@1.0` row into `EVAL_ASSESSMENT.md` §Obj3 as the FM-vs-direct-hinge result.

## 4. OOD bench at L=256
```bash
uv run python scripts/bench_sflm_ebm.py --scale a100_20g_L256 --n 128 --ref-lm gpt2
# (writes runs/sflm_bench_a100_20g_L256/bench.json)
```
`--n 128` (not 256) for headroom; raise it if memory allows. The crash-fix makes a
single-arm OOM non-fatal, but smaller n is cheaper. Read the printed table:
- (0) DFM **BPC** is the peer-comparable number; identity-path arms show `—`.
- (1a/1b/1c) `gpt2_baseline` SE AUROC = the external anchor (esp. **shuffle**).
- (1d/1d') EBM energy AUROC — expect **chance on shuffle** (the headline failure).
- (1g/1h) SVGP AUROC if the Stage-2 checkpoint was found.

## 5. Generation + BPC eval at L=256
```bash
uv run python scripts/eval_generation.py --scale a100_20g_L256 --n 256 --bpc-mc 8 --no-auto-train
```
Gives KL_uni/bi/tri, H_ratio, samples, and the MC-ELBO BPC per arm. This is the
Objective-1 peer-comparability row.

## 6. Verify
- `runs/sflm_bench_a100_20g_L256/bench.json` exists; DFM `bpc` is **finite** and
  `generation_metric_valid:true`; `_meta.skipped_arms` lists any arm that didn't train.
- `gpt2_baseline` shuffle AUROC ≫ 0.5; SFLMEBM energy shuffle ≈ 0.5 (expected).
- `svgp_corruption_sweep.json` shuffle `auroc_prob_as_ood` ≫ 0.5 (hinge works).

## 7. Report back (fold into `EVAL_ASSESSMENT.md`)
Fill the **pending L=256 cells**:
1. Peer-comparability table: **DFM BPC@L256** (from §5/§4) vs the published
   frontier (1.32–1.47), **with the `max_train_windows` value** you used and the
   data-subset caveat. State whether L256 improved KL_bi/BPC vs L40 (L40 DFM bench
   BPC ≈ 3.0, KL_bi 0.45 at the lr3e4 setting).
2. Objective-3 table: replace the L40 shuffle numbers with L256 (gpt2_baseline,
   SFLMEBM energy, hinge-SVGP) if you want the headline at standard length.
3. Note any arm that OOM'd even with the fixes (it'll be in `_meta.skipped_arms`).

## Failure modes & knobs
- **Still OOM:** lower `--n` (64, 32); lower the L256 `batch` in `SCALES`; the
  bench will null just the failing readout and continue (check stderr for
  `[... ] CUDA error — skipping readout`).
- **HF download/rate-limit:** set `HF_TOKEN`; text8 is `afmck/text8`, ref LM `gpt2`.
- **Training too slow:** drop `--epochs` (≥20 still meaningful), but say so in the report.
- **DFM_SVGP not in bench:** ensure the Stage-2 checkpoint is at
  `runs/sflm_bench_a100_20g_L256/DFM_SVGP/model_with_svgp_hinge.pt`; otherwise rely
  on the standalone `svgp_corruption_sweep.json` (§3) for the hinge numbers.

Total budget: dominated by §2 training (hours/arm). Prioritize **DFM** (BPC) and
**SFLMEBM** (energy OOD) if time-limited; DirichletFM and SFLMEBM_FM are secondary.
