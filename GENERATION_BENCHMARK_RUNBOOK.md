# Generation Benchmark Runbook — for a remote Claude Code session

**Goal.** Produce one **apples-to-apples generation benchmark** of the **7 model
families** on **KL_uni, KL_bi, KL_tri, generation entropy (H_gen / H_ratio), and
BPC (where valid)** — at **L=40** and **L=256**, every arm trained at the **same
budget** and evaluated by the **same code**. This fixes the prior table, which
mixed 5/20/50-epoch arms, missing arms, and an EBM-vs-normal mixup.

**Plus the recovery sweep** (Objective 2): for each arm, **Δ@α = token_acc −
token_acc_perturbed** across perturbation levels **α ∈ {0.1, 0.3, 0.5, 0.7, 1.0}**
— the self-healing metric (headline **Δ@.50**), reported alongside generation.

**Scope.** Only the 7 below. The energy-based-model framings (`SFLMEBM`,
`SFLMEBM_FM`) are **NOT** part of generation — `EqM` is the only energy-framed
model kept. (Those belong to the OOD bench, a separate objective.)

## The 7 models → arm → class → BPC
| # | model | `--only` arm | class | BPC |
|---|---|---|---|---|
| 1 | EqM one-hot+CLR | `EqM_OneHot` | EquilibriumFlowMatching (dirichlet_sampling=False) | — (identity-path) |
| 2 | EqM Dirichlet-thickened | `EqM` | EquilibriumFlowMatching (dirichlet_sampling=True) | — (identity-path) |
| 3 | **VAE+EqM** | `EqMAE` | EquilibriumFlowMatchingAE (frozen VAE latent) | — (identity-path) |
| 4 | Discrete FM | `DFM` | DiscreteFlowMatching (uniform path) | **real** (`elbo_bpc`; peer D3PM-uniform≈1.61) |
| 5 | Dirichlet FM | `DirichletFM` | DirichletFlowMatching | — (high-t NLL, <0.5 artifact) |
| 6 | SFLM | `SFLM` | SFLM (hyperspherical) | — (identity-path) |
| 7 | Standard FM (Lipman) | `FMonCLR` | FMonCLR (linear FM on CLR) | — (identity-path) |

> **Only `DFM` has a peer-comparable BPC** (a real variational bound). All other
> arms print `—` (the eval sets `generation_metric_valid=False`); never paste
> their PPL/BPC into a SEDD/D3PM table.
>
> **Naming caution:** the `DFM` arm is **Discrete** FM (uniform path). It is a
> *different model* from **Dirichlet** FM (`DirichletFM`). Older docs that say
> "best DFM KL_bi 0.45" actually mean DirichletFM (run `dfm_svgp_pure50_lr3e4`).
> `EqMLatent` is a *learned-embedding* EqM — **not** a VAE and **not** one of the
> 7; it is excluded from this benchmark (the VAE+EqM model is `EqMAE`).

## What was just wired (so this "just runs")
`FMonCLR` and `EqMAE` are now arms in `train_for_sflm_bench.py`
(`ARM_TO_MODEL`) and in `eval_generation.py:GEN_ARMS` (= exactly the 7);
`SFLMEBM`/`SFLMEBM_FM`/`EqMLatent` are dropped from generation. `eval_all.py`
now records `epoch`/`sample_steps`/`n_samples`/`split` and the correct arm label
(via `--model-kind` or the run-dir name). `train_for_sflm_bench.py` has
`--ae-ckpt` for the EqMAE 2-stage. **These are uncommitted local changes — make
sure they are on the branch you deploy.**

## 0. Environment
```bash
cd <repo root>
git fetch && git checkout capstone-project && git pull
uv sync
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SC=a100_20g_L256            # L=256 scale (d1024/10L/16H/B8); use "local" for L=40
EP=50                       # MATCHED epoch budget for ALL 7 arms (do not vary)
DATA="--max-train-windows 100000"   # MATCHED data for ALL 7 (or use --full-split)
```
All commands are `uv run python ...`. text8 + GPT-2 weights download from HF on
first use.

## 1. Stage A — train the VAE for `EqMAE` (the only 2-stage arm)
`EqMAE` is EqM over a **frozen** pretrained VAE latent, so train the VAE first
(defaults d256/2L/z64 — `_model_cfg` reads the dims back from this checkpoint):
```bash
uv run python scripts/train_autoencoder.py --mode vae \
    --out runs/vae_${SC} --seq-len 256 --epochs ${EP} \
    --windows 100000 --d-latent 64 --d-model 256 --num-layers 2 --nhead 4
# -> runs/vae_${SC}/epoch_final.pt
```
(For the L=40 run use `--seq-len 40` and `--out runs/vae_local`.)

## 2. Stage B — train all 7 at the SAME budget
**`--force` is mandatory** or the idempotent skip will reuse the old
under-trained (5-epoch) checkpoints. Run sequentially (or background per arm):
```bash
# 6 single-stage arms:
uv run python scripts/train_for_sflm_bench.py --scale ${SC} --force --epochs ${EP} ${DATA} \
    --only EqM_OneHot,EqM,DFM,DirichletFM,SFLM,FMonCLR

# VAE+EqM (needs the Stage-A checkpoint):
uv run python scripts/train_for_sflm_bench.py --scale ${SC} --force --epochs ${EP} ${DATA} \
    --only EqMAE --ae-ckpt runs/vae_${SC}/epoch_final.pt
```
Each writes `runs/sflm_bench_${SC}/<arm>/epoch_final.pt` + `train_meta.json`.
The memory-fallback ladder auto-handles OOM/NVML (grad-ckpt → halve batch →
L=128 → `FAILED.json` + continue); check `train_meta.json` for any fallback.
**Budget:** the second-order arms (EqM, EqM_OneHot, EqMAE) are hours/arm at
L256/50ep — if time-bound, drop `EP` to ≥20 but keep it **identical** across all
7 (matched budget is the whole point); say so in the report.

## 3. Stage C — evaluate ALL 7 with ONE eval path (matched n/steps/split)
Use `eval_all.py` for every arm (same `--n`, `--steps`, `--split`); pass
`--model-kind` so the two EqM arms stay distinguishable:
```bash
for ARM in EqM_OneHot EqM EqMAE DFM DirichletFM SFLM FMonCLR; do
  uv run python scripts/eval_all.py \
      --ckpt runs/sflm_bench_${SC}/${ARM}/epoch_final.pt \
      --model-kind ${ARM} --split test --n 256 --steps 200 --bpc-mc 8 \
      --out runs/sflm_bench_${SC}/${ARM}/eval_all.json
done
```
Each `eval_all.json` carries `KL_uni/KL_bi/KL_tri, H_gen, H_gt, H_ratio,
per_pos_entropy, bpc` (None unless DFM), `generation_metric_valid`, `collapsed`,
plus the new provenance (`epoch, sample_steps, n_samples, split`).

## 3b. Stage C2 — recovery sweep (Δ@α across perturbation levels)
For each arm, sweep the perturbation level α and report **Δ = token_acc −
token_acc_perturbed** (recovery does *work* iff Δ > 0). `recovery_check.py`
handles all 7 model families (it maps α per family — Gaussian latent/CLR
perturbation for EqM/EqM_OneHot/EqMAE/SFLM/FMonCLR; keep-prob κ=1−α corruption
for DFM; Dirichlet-path start-time for DirichletFM — see the `sigma_perturb`
column for the actual magnitude):
```bash
for ARM in EqM_OneHot EqM EqMAE DFM DirichletFM SFLM FMonCLR; do
  uv run python scripts/recovery_check.py \
      --ckpt runs/sflm_bench_${SC}/${ARM}/epoch_final.pt \
      --alphas 0.1,0.3,0.5,0.7,1.0 --n 256 --steps 200 \
      --out runs/sflm_bench_${SC}/${ARM}/recovery.json
done
```
Each `recovery.json` has one `recovery` row per α with `delta`, `token_acc`,
`token_acc_perturbed`, `sigma_perturb`, `KL_bi`. **Headline = Δ@.50**;
also keep the full Δ-vs-α curve. (Δ@α is comparable *within* a family;
across families compare at matched `sigma_perturb` / corruption fraction, not
raw α.) Expected (EVAL_ASSESSMENT): compositional/EqM Δ@.50 ≈ +0.06, a
deterministic-CLR ref ≈ +0.00.

## 4. Stage D — aggregate into the benchmark table
```bash
uv run python - <<'PY'
import json, glob, os
SC=os.environ.get("SC","a100_20g_L256")
def g(v,p=3): return f"{v:.{p}f}" if isinstance(v,(int,float)) else "—"
def delta_at(d, a=0.5):
    for r in d.get("rows",[]):
        if r.get("mode")=="recovery" and abs((r.get("alpha") or -9)-a)<1e-6: return r.get("delta")
    return None
hdr=f"{'model':12s}{'ep':>4}{'KL_uni':>9}{'KL_bi':>8}{'KL_tri':>8}{'H_ratio':>8}{'bpc':>8}{'Δ@.50':>8}  collapsed"
print(hdr); print("-"*len(hdr))
for f in sorted(glob.glob(f"runs/sflm_bench_{SC}/*/eval_all.json")):
    e=json.load(open(f)); d=os.path.dirname(f)
    rec=os.path.join(d,"recovery.json")
    d50=delta_at(json.load(open(rec))) if os.path.exists(rec) else None
    print(f"{e['model_name']:12s}{str(e.get('epoch','?')):>4}{g(e['KL_uni'],4):>9}{g(e['KL_bi']):>8}"
          f"{g(e['KL_tri']):>8}{g(e['H_ratio']):>8}{g(e['bpc']) if e['generation_metric_valid'] else '—':>8}"
          f"{g(d50):>8}  {e['collapsed']}")
PY
```
Repeat §1–4 with `SC=local`, `--scale local`, `--seq-len 40` for the **L=40**
table, then put the two side by side.

## 5. Verify / done
- All 7 `eval_all.json` exist with **the same `epoch`, `sample_steps=200`,
  `n_samples=256`, `split=test`** (matched — confirm in the JSON; mismatched
  budgets invalidate the comparison).
- `DFM` has a finite `bpc` with `generation_metric_valid=true`; the other 6 show
  `bpc=null` / `—`.
- No arm sits at a 5-epoch budget (the prior bug). Any arm with a `FAILED.json`
  or a `train_meta.json` `length_fallback` is footnoted.
- Every arm has a `recovery.json` with 5 α rows (0.1–1.0) and a `Δ@.50`; the
  Δ-vs-α curve and headline Δ@.50 are reported next to generation.
- Expected pattern to confirm or refute: at L=40 several arms generate well
  (Discrete FM was best ≈0.15, SFLM ≈0.42, Dirichlet FM ≈0.45); at L=256 the
  open question is whether they collapse (length effect) or just needed this
  matched budget — which this run answers.

## Gotchas
- **`--force`** every train command (else 5-epoch ckpts are reused).
- **`--model-kind`** every eval (else `EqM_OneHot` mislabels as `EqM`).
- **EqMAE** needs the Stage-A VAE checkpoint trained at the **same L**; its AE
  dims are read from that checkpoint, so don't change `--d-model/--d-latent`
  between Stage A and B.
- **BPC** is only meaningful for `DFM`; everything else is `—` by design.
- `EqMLatent` is intentionally **not** in the 7 (add `EqMLatent` to `--only` and
  `GEN_ARMS` only if you want it as an optional 8th "learned-embedding EqM").
