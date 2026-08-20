# Training plan — DSM vs EqM, three-cell capstone experiment


> **ARCHIVED / SUPERSEDED (banner added 2026-08-20).** Execution plan from 2026-05-13 for the three-cell DSM-against-EqM comparison, on the frozen d=1024 autoencoder latent. The published version of this experiment is `tab:ablation2x2` in the paper, and it crosses the training target against the *endpoint recipe* at development scale L=40 (`runs/dsmx_clr_*`, `runs/compu_*`), not the latent cells planned here. Its verdict: DSM escapes the unigram collapse and still does not generate. Two stale pointers: the `POSITIONING.md` link in the header does not resolve from this directory (the file is now at `docs/archive/POSITIONING.md`), and `runs/ae_d256_l2_z64/ae/epoch_final.pt` no longer exists.

This document is the execution-side counterpart of
[`POSITIONING.md`](POSITIONING.md). It is intended to be handed to a
fresh cloud Claude Code session with no prior context. Read it
top-to-bottom; the steps are ordered for sequential execution.

For project background and earlier-phase context, see the existing
`TRAINING_PLAN.md`; this file is scoped specifically to the DSM-vs-EqM
comparison.

## What this experiment is

Three model cells, identical backbone / AE / data / seed, trained on
text8 with a frozen pretrained autoencoder:

| cell        | model name  | training signal       | score parameterisation         | sampler                |
|-------------|-------------|-----------------------|--------------------------------|------------------------|
| `eqm_fm`    | `EqMAE`     | FM regression (linear)| conservative-gradient bilinear | NAG-GD on ∇⟨x, f⟩      |
| `score_dsm` | `ScoreDSM`  | ε-prediction DSM      | direct (free vector field)     | annealed Langevin      |
| `eqm_dsm`   | `EqMDSM`    | ε-prediction DSM      | conservative-gradient bilinear | annealed Langevin      |

The empirical questions are (A) energy-gradient vs direct-score, (B)
FM vs DSM training signal, (C) recovery / healing benchmark behaviour.
Background in `POSITIONING.md`; failure-mode theory in
`NOTE_WHY_EBM_INIT_STUCK.md`.

Compute budget: ~3-4 h on a single A100 for the three single-seed cells.
Triplicate seeds (seed43 / seed44 cells) add ~6-8 h.

## Repository state you start from

Branch: `capstone-project`. Working tree is configured for this
experiment. The pieces already in place (already committed, do not
re-create):

```
src/aitchinson_flow/
  config.py                          # DSMConfig added
  models/
    __init__.py                      # ScoreDSM / EqMDSM registered
    score_dsm.py                     # direct-score DSM
    eqm_dsm.py                       # energy-gradient DSM
    eqm_ae.py                        # EqMAE — the FM baseline (existing)
  sampling/
    annealed_langevin.py             # NCSN-style sampler
    sde.py                           # existing Langevin used by EqM-SDE branch

sweeps/
  dsm_vs_eqm.yaml                    # 3-cell + smoke + triplicate sweep

scripts/
  run_sweep.py                       # existing orchestrator
  recovery_check.py                  # existing recovery harness — works for the
                                     # new models out of the box (uses model.sample
                                     # and model.encode, both implemented).
  evaluate_eqm.py                    # existing eval-plots script
  train_autoencoder.py               # existing AE pretrain script
```

Required pretrained AE checkpoints:
```
runs/ae_d256_l2_z64/ae/epoch_final.pt          # d=256 AE — for smoke test
runs/ae_d1024_l8_z128_v3/ae/epoch_final.pt     # d=1024 AE — for main cells
```

If either is missing, regenerate (see §3) before running the sweep.

## §1. Environment setup

```bash
# Python 3.13 + uv lockfile.
uv sync

# Confirm CUDA + torch.
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# Confirm the new models are registered.
uv run python -c "
import aitchinson_flow.models as m
assert 'ScoreDSM' in m.REGISTRY and 'EqMDSM' in m.REGISTRY, list(m.REGISTRY)
print('registry OK:', sorted(m.REGISTRY))
"
```

Optional WandB:
```bash
export WANDB_API_KEY=<key>
# Or set cfg.wandb.mode = "offline" in the sweep YAML.
```

## §2. Smoke test (~3 min) — verify the new code paths run end-to-end

The first sweep cell `dsm_smoke` exercises:
- DSM training step on the d=256 AE,
- σ-conditioning routed through `_LatentBackbone`'s γ path,
- annealed Langevin sampling,
- eval + recovery_check pipeline.

```bash
uv run python scripts/run_sweep.py \
  --sweep sweeps/dsm_vs_eqm.yaml \
  --only dsm_smoke \
  --runs-root runs/dsm_vs_eqm
```

Expected: a ~3 min training run that writes
`runs/dsm_vs_eqm/dsm_smoke/{config.json, history.jsonl, eval.json,
epoch_final.pt}` and no Python exceptions.

If this fails: read the traceback, fix at source (not by skipping
hooks), re-run. Common pitfalls and fixes:

- **`TypeError: forward() got unexpected keyword argument 'gamma'`** —
  the new models pass `cond` as the second positional arg to
  `_LatentBackbone.forward`. The backbone signature already accepts a
  positional second arg (it just calls it `gamma`). If you see this,
  you have a stale import; restart Python.
- **`AE load: missing=N unexpected=M`** — non-fatal; the warning is
  expected when loading the AE into a DSM wrapper. Counts should be
  < 5; if much larger, the AE config in the sweep does not match the
  checkpoint's config.
- **`cfg.dsm.ae_ckpt_path must be set`** — the smoke cell sets
  `dsm.ae_ckpt_path: runs/ae_d256_l2_z64/ae/epoch_final.pt`. Confirm
  that file exists, or regenerate the d=256 AE per §3.

## §3. AE pretraining (skip if both AE checkpoints already exist)

```bash
# d=1024 AE — 5 epochs on 10k windows. ~30-40 min on an A100.
uv run python scripts/train_autoencoder.py \
  --out runs/ae_d1024_l8_z128_v3/ae \
  --d-model 1024 --num-layers 8 --nhead 8 --d-latent 128 \
  --denoising-schedule relative_uniform --denoising-sigma 0.5 \
  --latent-l2 0.001 \
  --epochs 5 --windows 10000 --batch-size 64 --lr 0.0003 \
  --seed 42 --mode ae \
  --seq-len 128 --variable-length --L-min 40 --L-max 128
```

```bash
# d=256 AE (for the smoke test). ~5 min on an A100.
uv run python scripts/train_autoencoder.py \
  --out runs/ae_d256_l2_z64/ae \
  --d-model 256 --num-layers 2 --nhead 4 --d-latent 64 \
  --denoising-schedule relative_uniform --denoising-sigma 0.5 \
  --latent-l2 0.001 \
  --epochs 5 --windows 10000 --batch-size 64 --lr 0.0003 \
  --seed 42 --mode ae \
  --seq-len 40
```

Sanity-check AE quality before proceeding:
```bash
uv run python -c "
import json
for p in ('runs/ae_d1024_l8_z128_v3/ae', 'runs/ae_d256_l2_z64/ae'):
    h = [json.loads(l) for l in open(f'{p}/history.jsonl')]
    last = h[-1]
    assert last['tok_acc'] > 0.98, (p, last)
    print(p, {k: last[k] for k in ('loss', 'ce', 'tok_acc', 'z_norm')})
"
```

## §4. The three main cells

Run after smoke succeeds. Each cell is ~45-60 min on an A100.

```bash
# eqm_fm — FM regression baseline. Note this sweep cell uses
# eqm.gamma_power=0.5 — the CORRECTED value from the v3 post-mortem
# (the original v3 used 1.0 and exhibited the §3 flat-field collapse;
# gamma_power=0.5 upweights γ≈1 and is the recipe that works on d=256).
# We intentionally compare DSM against the *fixed* EqM-FM, not the
# broken one — the broken one's phenotype is already documented in
# runs/ae_d1024_l8_z128_v3/.
uv run python scripts/run_sweep.py \
  --sweep sweeps/dsm_vs_eqm.yaml \
  --only eqm_fm \
  --runs-root runs/dsm_vs_eqm

# score_dsm — direct-score DSM.
uv run python scripts/run_sweep.py \
  --sweep sweeps/dsm_vs_eqm.yaml \
  --only score_dsm \
  --runs-root runs/dsm_vs_eqm

# eqm_dsm — energy-gradient DSM. Slowest of the three because of
# double-backward through the backbone (same constraint as EqMAE
# training).
uv run python scripts/run_sweep.py \
  --sweep sweeps/dsm_vs_eqm.yaml \
  --only eqm_dsm \
  --runs-root runs/dsm_vs_eqm
```

These can run in parallel on three separate GPUs if available. The
sweep orchestrator is single-process; one cell per invocation is fine.

After each cell:
```
runs/dsm_vs_eqm/<cell>/
  config.json
  history.jsonl
  eval.json          # unigram/bigram/trigram KL, H_ratio, samples
  epoch_final.pt
```

## §5. Recovery / healing evaluation

After all three main cells finish:

```bash
for cell in eqm_fm score_dsm eqm_dsm; do
  uv run python scripts/recovery_check.py \
    --ckpt runs/dsm_vs_eqm/$cell/epoch_final.pt \
    --alphas 0.10,0.15,0.20,0.30,0.40,0.50,0.70,1.00 \
    --n 256 --steps 200 \
    --out runs/dsm_vs_eqm/$cell/recovery.json
done
```

For the DSM cells, `--steps 200` is ignored by the model's own sampler
(annealed Langevin uses `cfg.dsm.n_sigma * cfg.dsm.steps_per_sigma`,
= 128 NFE for the default). The `--steps` flag is honoured by the
EqMAE cell.

`recovery_check.py` writes a JSON with rows for `score_reference`,
`unconditional`, and one `recovery` row per α. Inspect for the no-op
signature documented in `NOTE_WHY_EBM_INIT_STUCK.md`: `rc_sample0 ==
perturbed_sample0` AND `token_acc == token_acc_perturbed`. Expected on
`eqm_fm` if the §3 collapse reproduces; *not* expected on either DSM
cell.

## §6. Headline table

Generate the comparison table once all three cells have eval.json +
recovery.json:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

root = Path("runs/dsm_vs_eqm")
rows = []
for cell in ("eqm_fm", "score_dsm", "eqm_dsm"):
    e = json.loads((root / cell / "eval.json").read_text())
    r = json.loads((root / cell / "recovery.json").read_text())
    uncond = next(x for x in r["rows"] if x["mode"] == "unconditional")
    recov = {x["alpha"]: x for x in r["rows"] if x["mode"] == "recovery"}
    row = {
        "cell": cell,
        "KL_uni": e["unigram_kl"],
        "KL_bi": e["bigram_kl"],
        "H_ratio": e["H_ratio"],
        "uncond_KL_bi": uncond["KL_bi"],
    }
    for a in (0.15, 0.30, 0.50):
        if a in recov:
            row[f"acc@{a}"] = recov[a]["token_acc"]
            row[f"delta@{a}"] = recov[a]["token_acc"] - recov[a]["token_acc_perturbed"]
    rows.append(row)

cols = ["cell", "KL_uni", "KL_bi", "H_ratio", "uncond_KL_bi",
        "acc@0.15", "delta@0.15", "acc@0.3", "delta@0.3", "acc@0.5", "delta@0.5"]
print("\t".join(cols))
for r in rows:
    vals = [f"{r.get(c):.4f}" if isinstance(r.get(c), float) else str(r.get(c, "-")) for c in cols]
    print("\t".join(vals))
PY
```

Save to `runs/dsm_vs_eqm/headline.tsv` for the writeup.

## §7. Triplicate-seed extension (optional, run after §6 is clean)

```bash
for cell in eqm_fm_seed43 score_dsm_seed43 eqm_dsm_seed43; do
  uv run python scripts/run_sweep.py \
    --sweep sweeps/dsm_vs_eqm.yaml \
    --only $cell \
    --runs-root runs/dsm_vs_eqm
done

# Then duplicate the seed cells in the YAML with `training.seed: 44`
# and run those for a three-seed estimate.
```

## §8. Diagnostics — what to inspect when something is off

### EqM-FM phenotype.
With `gamma_power=0.5` (the corrected value), expected behaviour is
similar to the d=256 partial-success baseline:
- KL_uni < 0.03, KL_bi ≈ 1.2–1.6
- Recovery Δ@0.50 > 0.05
- *Not* the no-op signature.

If you see the no-op signature on the corrected EqM-FM, that is itself
informative — the §3 collapse persists even with gamma_power=0.5 at
this larger scale (50k windows + L=128). Document and continue.

### ScoreDSM should show real basins.
Expected:
- KL_uni < 0.05, KL_bi < 1.0 (likely lower than EqM-FM)
- Recovery Δ@0.5 > +0.05
- Unconditional samples have recognisable text fragments

If recovery is no-op, the annealed Langevin σ-schedule is mis-tuned:
- Check `dsm.sigma_max ≈ embed_norm` (z_norm in
  `runs/ae_d1024_l8_z128_v3/ae/latent_stats.json` ≈ 9.7).
- Check `dsm.sigma_min` ≈ 0.05 (NCSN convention: ~1% of σ_max).

### EqMDSM should match ScoreDSM qualitatively.
If EqMDSM lags by > 2× on DSM loss:
- Try `dsm.loss_weighting: "snr+1"` instead of `"constant"`. The
  Salimans-Ho intuition: energy parameterisations need σ-dependent
  rescaling to train cleanly.
- Try the σ²-scaled energy variant: edit `eqm_dsm.py` to use
  `E = ⟨x, f⟩ / σ²` (matches variance-preserving DSM weighting).

### NVML / curvature OOM during recovery_check.
Known issue from the v3 logs — `score_curvature` does double-backward
and trips the CUDA allocator on d=1024. The current `recovery_check.py`
wraps the curvature call in try/except; non-fatal. For deterministic
recovery numbers under load: `--n 64` instead of `--n 256`.

### Annealed Langevin chain diverging (NaN energies / samples).
Lower `dsm.sampler_eps` by 10× (`1e-6` or `1e-7`). NCSN-style chains
are sensitive to the base step size; the geometric scaling with
$(\sigma_i/\sigma_\min)^2$ amplifies any over-step at large σ.

### Aux CE dominating early.
`dsm.lambda_ce` defaults to 0.5. If `history.jsonl` shows the CE term
much larger than the DSM term in epoch 1, lower to 0.1 for a warmup
phase and anneal up. The CE is the multi-modal carving term —
important for basins, but if it dominates the DSM signal you collapse
to a decoder-classifier rather than a score model.

## §9. What to report

Four artefacts for the writeup:

1. **Headline table** from §6 (KL ladders + recovery Δ at α ∈ {0.15, 0.3, 0.5}).
2. **α-sweep recovery plot** — token accuracy vs α, three lines (one per cell).
3. **Energy-vs-distance plot** for the EqM cells — sample
   `score_energy(x)` for x = (1-t)·z_clean + t·z_perturbed as t sweeps
   [0, 1] at fixed α. EqM-FM should show a flat curve; EqMDSM should
   show a clear basin profile. ScoreDSM's `score_energy` returns NaN
   by design — there is no scalar energy to plot, which is itself the
   point of axis (A).
4. **Samples figure** — 8 unconditional samples per cell, side-by-side.

The fourth is qualitative; the first three are the load-bearing
empirical claims.

## §10. Outputs and committing

All sweep artefacts land under `runs/dsm_vs_eqm/`. Suggested commit:

```bash
git add POSITIONING.md TRAINING_PLAN_DSM_VS_EQM.md sweeps/dsm_vs_eqm.yaml
git add src/aitchinson_flow/models/score_dsm.py src/aitchinson_flow/models/eqm_dsm.py
git add src/aitchinson_flow/sampling/annealed_langevin.py
git add src/aitchinson_flow/config.py src/aitchinson_flow/models/__init__.py
# Eval artefacts are small and worth committing; checkpoints (epoch_final.pt
# at 50–200 MB each) probably belong in git-lfs or .gitignore.
git add runs/dsm_vs_eqm/*/eval.json runs/dsm_vs_eqm/*/recovery.json runs/dsm_vs_eqm/*/history.jsonl runs/dsm_vs_eqm/headline.tsv
git commit -m "DSM vs EqM 3-cell experiment: models, sampler, sweep, results"
```

## §11. If you have time after the headline experiment

Two cheap follow-ups that strengthen the contribution without scope creep:

- **σ-schedule ablation** for EqMDSM: try σ_max ∈ {2.0, 5.0, 10.0}. The
  trained model's basin geometry depends on the highest σ seen during
  training; Karras 2022 reports image-domain optima at σ_max ≈ 80
  (relative to a unit-variance prior). Add 2-3 cells to the YAML.
- **Energy-gradient parameterisation with σ-dependent scaling.** Replace
  the bilinear $E = \langle x, f\rangle$ with $E = \langle x, f\rangle / \sigma^2$ —
  matches variance-preserving DSM weighting, may improve EqMDSM ↔
  ScoreDSM parity. One-line change in `_grad_energy` /
  `energy_per_sample`.

Anything beyond these two should wait until the headline table is in
hand and points at a specific follow-up question.
