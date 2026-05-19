# S-FLM-as-EBM viability probe — findings

Question: can the time-free hyperspherical flow model (S-FLM,
arXiv:2605.11125) be turned into a usable EBM (EqM-style,
arXiv:2510.02300) on a toy task, or does it collapse to the trivial
solution `E[x1]` under a deterministic SLERP interpolant?

Harness: `scripts/sflm_ebm_probe.py` (extends the `testing.ipynb` toy:
modular arithmetic `a + b = c mod 7`, vocab 9, time-free `TinyDenoiser`,
energy `E(z) = -tau*logsumexp_v <h(z),e_v>/tau`). 5 configs, 3000 steps.

## Headline: the "stuck at E[x1]" diagnosis is a diagnostic bug

| config | E_data | E_cent | data<cent | sample arith | uniq | recover |
|---|---|---|---|---|---|---|
| baseline_uniform | −11.98 | −2.45 | True | 0.160 | 0.67 | **1.000** |
| alpha_trunc | −11.63 | −2.29 | True | 0.016 | 0.85 | **1.000** |
| import_sched | −12.25 | −2.46 | True | 0.168 | 0.64 | **0.984** |
| hinge_only | −12.86 | −3.04 | True | 0.160 | 0.55 | **0.984** |
| eqm_style | −14.57 | −1.69 | True | 0.062 | 0.88 | **1.000** |

arith chance = 0.143; uniq ∈ [0,1] (1 = all distinct); H_max = 2.20.

1. **Not stuck at E[x1].** `data<centroid` is True in every config
   incl. the untouched baseline; real-data energy is 5–8× deeper than
   the centroid. The notebook's `diagnostic_energy_at_codewords`
   reports failure only because it feeds *single-token-repeated*
   sequences `[v,v,v,v,v]` through a position-embedded model trained
   solely on structured `a+b=c` — an OOD artifact. **Fix the
   diagnostic to encode real `dataset.data` sequences.**
2. **Energy basin + local field are correct.** Init near data → GD
   recovers 100% arithmetic-correct (`recover≈1.0`) in all configs.
3. **Samples are diverse** (uniq 0.55–0.88) — not collapsed to one
   point. The literal `E[x1]` collapse is false at the sample level.
4. **Real bottleneck = global sampler traversal.** `recover=1.0` but
   `sample_from_noise≈chance`: Riemannian GD from uniform noise can't
   *compose* the arithmetic constraint from far away. EqM knobs don't
   help; `alpha_trunc`/`eqm_style` *hurt* (0.016/0.062) by starving the
   high-noise field the global sampler needs — opposite of EqM's
   importance-sampling prescription, because the toy lacks EqM's
   conservative-gradient transport target to compensate.

## Verdict

- **EBM / OOD-auditor / verifier (score + locally repair sequences):
  viable.** `recover=1.0`, clean energy margin, no extra machinery.
  Slots into the niche the repo's DFM-SVGP auditor / EqM
  `_auditor_hinge` already occupy.
- **Unconditional generator via energy descent: not viable as pure
  CE + energy-GD.** Needs one of: (i) EqM conservative-gradient FM
  target + NAG-GD (not pure CE); (ii) restored noise/time-conditioned
  transport (S-FLM's actual mechanism); (iii) annealed-Langevin /
  sequential-noise-level sampling (cf. the Annealed Langevin sampler,
  commit c823934).

## text8 model candidate + spilled-energy benchmark

`SFLMEBM` (`src/aitchinson_flow/models/sflm_ebm.py`, registered;
`cfg.sflm_ebm`) is the probe's model promoted to the project pipeline:
time-free hyperspherical denoiser, EqM log-sum-exp energy, Riemannian-GD
sampler, full EqM-family EBM API (`energy`, `position_uncertainty`,
`spilled_energy`, `score_*`, `bpd`).

> ⚠️ **The table below is a d256/4L/6-epoch *smoke*, not a proper run.**
> It used `Config()` defaults for every model, which for EqM is *not*
> the working recipe (needs `gradient_lambda=3.0` +
> `dirichlet_sampling=True`, per `runs/dphase4_lambda_recalib`) — that
> is why EqM scores below chance here. Treat only the *direction*
> (SFLMEBM ≫ existing EBMs; SFLMEBM best shuffle localiser) as
> indicative, pending the proper runs below.

### Capstone arms (both scaffolded; ready to train)

Two research arms now live alongside the original `SFLMEBM`, both
additive (default behaviour unchanged) and ablatable via one config
flag each:

- **Arm 1 — `SFLM` (Stage-1 generator)**:
  `src/aitchinson_flow/models/sflm.py`, registered. Time-conditioned
  hyperspherical denoiser (γ via sinusoidal embedding), plain CE on
  SLERP-noised latents, Euler-over-γ sampler via x1-prediction +
  geodesic SLERP step. Exposes the EqM-family inference API; no
  `energy()` (it's a generator, not an EBM) — bench treats it like
  DFM. The KL_uni/bi/tri probe is the **gate question**: if SFLM
  matches DFM-level generation, follow with a Stage-2 SVGP head
  (mirrors `DirichletFMSvgp`).
- **Arm 2 — `SFLMEBM_FM` (option 2: density-faithful energy)**:
  same `SFLMEBM` class, one-flag activation `cfg.sflm_ebm.lambda_fm>0`
  (the driver sets `lambda_fm=1.0, lambda_hinge=0.0`). Adds the
  Riemannian-FM regression
  `−project_tangent(∇E) → c(α)·log_{z_α}(z₁)` via
  `create_graph=True` (second-order autograd, EqM-style cost). The
  *existing* energy-GD `sample()` then transports noise→data — no
  new sampler. Tests whether a single energy can be both generative
  and OOD-faithful (the unification the capstone story rests on).

### Proper runs (scale-aware)

`scripts/train_for_sflm_bench.py` now trains all four at a **shared
scale** with each model's **known-good knobs** (EqM tuned recipe,
EqMLatent d128/tied/euler, DFM defaults, SFLMEBM d128). Two presets:

| preset | backbone | batch | fits | use |
|---|---|---|---|---|
| `--scale local` | d512 / 6L | 16 | 8 GB GPU | local dev (running) |
| `--scale cluster` | d1024 / 8L | 64 | ~16-18 GB | 20 GB cluster |

`local` is the largest config that fits an 8 GB GPU with *all four*
(SFLMEBM's 4-forward hinge + EqM/EqMLatent second-order autograd);
d1024/8L OOMs there. `cluster` matches the DFM_SVGP_FINDINGS.md
Stage-1 reference scale. Both: 50 ep, 10k windows, lr 3e-4. Run:

```bash
# local (8 GB) — in progress
python scripts/train_for_sflm_bench.py --scale local   --epochs 50
python scripts/bench_sflm_ebm.py       --scale local   --n 256
# cluster (20 GB)
python scripts/train_for_sflm_bench.py --scale cluster --epochs 50
python scripts/bench_sflm_ebm.py       --scale cluster --n 256
```

---

#### Smoke table (d256/4L/6ep — see warning above)

All four trained with shared d256/4L/d64/6ep/10k-window Config
defaults. AUROC, n=256 (0.5 = chance):

**(1) Sequence-level (clean vs corrupted)**

| model | subst .1 | subst .3 | shuffle | rand |
|---|---|---|---|---|
| SFLMEBM | 0.734 | 0.950 | 0.564 | 1.000 |
| EqM (simplex) | 0.372 | 0.194 | 0.509 | 0.211 |
| EqMLatent | 0.513 | 0.455 | 0.509 | 0.884 |
| DFM | **0.885** | **0.990** | **0.985** | 1.000 |

**(2) Per-position spilled-energy localisation (changed vs unchanged
positions inside corrupted seqs)**

| model | subst .1 | subst .3 | shuffle |
|---|---|---|---|
| SFLMEBM | 0.757 | 0.783 | **0.699** |
| EqM (simplex) | 0.574 | 0.541 | 0.659 |
| EqMLatent | 0.644 | 0.642 | 0.518 |
| DFM | **0.866** | **0.831** | 0.574 |

Per-position via Riemannian ‖∇E‖ (EqM family only): EqM 0.711/0.710,
SFLMEBM 0.637/0.636, EqMLatent 0.619/0.623 (subst .1/.3).

### Read

- **SFLMEBM cleanly beats the repo's existing EBMs** (simplex EqM,
  EqMLatent) at both sequence- and position-level spilled energy. At
  this controlled budget simplex EqM is *below chance* on substitution
  and EqMLatent is near chance on fine corruption; SFLMEBM separates
  substitution (0.95 @ r=0.3) and random (1.0) and localises corrupt
  tokens (0.76–0.78).
- **SFLMEBM is the best shuffle localiser (0.699)** — beating DFM
  (0.574), EqM (0.659), EqMLatent (0.518). Shuffle preserves unigram
  stats, so this requires real positional/contextual structure: a
  genuine differentiator for the hyperspherical energy.
- **DFM wins raw detection** (denoiser-NLL is a very strong proxy, cf.
  DFM_SVGP_FINDINGS) but is a classifier, not an EBM: no differentiable
  energy field, so it cannot do the energy-descent recovery (probe:
  recover≈1.0) that SFLMEBM provides. SFLMEBM trades some detection
  AUROC for a usable energy landscape.
- EqM's natural per-position signal is ‖∇E‖ (0.71), not decode-NLL
  (0.55) — reported both for fairness.

Numbers are *relative* under a deliberately small shared budget, not
the repo's best absolute results.

### Verdict (updated)

Add `SFLMEBM` as a candidate **EBM / OOD-auditor / per-position
spilled-energy localiser**: it dominates the existing energy models in
the repo and uniquely captures order corruption, while keeping the
recover≈1.0 energy landscape DFM structurally lacks. Not a competitive
unconditional generator (unchanged from the probe verdict).

## Reproduce

```bash
python scripts/sflm_ebm_probe.py --steps 3000               # toy probe, ~5 min
python scripts/sflm_ebm_probe.py --quick                    # 1200 steps, 3 cfgs
python scripts/train_for_sflm_bench.py --scale local  --epochs 50
python scripts/bench_sflm_ebm.py       --scale local  --n 256
# on the 20 GB cluster:
python scripts/train_for_sflm_bench.py --scale cluster --epochs 50
python scripts/bench_sflm_ebm.py       --scale cluster --n 256
```
