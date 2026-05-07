# Cluster training plan — capstone EqM continuation

> Handoff document for the cluster Claude session. The local session added
> code (W1 Euler sampler, W2 non-factorised bigram head, W3 OOD eval, W4
> FMonCLR baseline) but **did not run any experiments**. This file describes
> what to run, in what order, and how to interpret the results. Compute
> target: **A100 with a 20 GB MIG slice** (matches the previous session's
> environment per `runs/DECISION_LOG.md`, where 5 ep × 50k windows ≈ 14 min
> at d=1024/8L).

## TL;DR for the cluster session

1. **You have one new code drop.** Do not modify the model code — the local
   session validated the smoke tests. Only run the sweeps below and write
   results.
2. **Run W1+W3+W4 in parallel** (sweeps are independent). Total ~5 hr at
   the A100/20GB throughput.
3. **Then decide on W2/W5** based on W1's outcome (kill criteria below).
4. **Update `runs/DECISION_LOG.md` after each cell** with a short
   hypothesis/result/decision entry — the previous session established that
   convention.
5. **Never commit destructive git operations** without confirmation; you
   are working on the `capstone-project` branch and the user has uncommitted
   in-progress work (see `git status`).

## What's already in the repo (added by the local session)

| File | Purpose |
|---|---|
| `src/aitchinson_flow/config.py` | `EqM` extended with `sampler`, `euler_nfe`, `euler_use_grad`, `sample_sigma_init`, `lambda_bigram_joint`. All defaults preserve previous behaviour. |
| `src/aitchinson_flow/models/eqm.py` | New `BigramHead` class. `EquilibriumFlowMatching` gets an optional `bigram_head` (only when `lambda_bigram_joint > 0`), `sample_euler()`, `energy()` scalar method. `sample()` dispatches on `cfg.eqm.sampler`. |
| `src/aitchinson_flow/models/fm_clr.py` | New `FMonCLR` model — direct flow matching on CLR features (no conservative-grad indirection). Registered as `"FMonCLR"`. |
| `src/aitchinson_flow/models/__init__.py` | `FMonCLR` registered. |
| `scripts/eval_full.py` | New `--override key=value` CLI (e.g. `--override eqm.sampler=euler`). Lets one checkpoint be re-evaluated under multiple sampler configs. |
| `scripts/eval_ood.py` | New canonical OOD scorecard (W3). Outputs `<run_dir>/ood_eval.{json,png}`. |
| `sweeps/phase10_euler.yaml` | W1 + W4 — three retrained baselines (EqM, DFM, FMonCLR) at the `data_50k_ep5` platform. |
| `sweeps/phase11_bigram.yaml` | W2 — bigram-head sweep, two cells. |
| `sweeps/phase12_fmclr.yaml` | W4 — single-cell FMonCLR with explicit Euler sampler config (alternative to the consolidated phase10 cell). |
| `sweeps/phase13_longrun.yaml` | W5 — 50-epoch long run on the winning architecture. |

**Smoke tests run locally:** EqM with `sampler="nag"`, `sampler="euler"`,
`sampler="euler"+euler_use_grad=True`, FMonCLR train+sample+energy,
BigramHead train step. All passed. The pipeline (`fit()` + `_unigram_kl_probe()`)
ran 1 epoch on 1k windows in 17s — extrapolating to 50k×5ep gives ~70 min
on this small-VRAM machine; on the A100 MIG expect closer to 14 min as
documented in `RESULTS.md`.

## Compute budget

| Operation | Wall-clock at A100/20GB MIG | Source |
|---|---:|---|
| 5 ep × 10k windows, d=1024/8L | ~3 min | DECISION_LOG `baseline_5ep` |
| 5 ep × 50k windows, d=1024/8L | ~14 min | DECISION_LOG `data_50k_ep5` |
| 25 ep × 10k windows, d=1024/8L | ~14 min | DECISION_LOG `ep25_default` |
| 25 ep × 50k windows, d=1024/8L | ~70 min | extrapolation |
| 50 ep × 50k windows, d=1024/8L | ~140 min | extrapolation |
| `eval_full.py` (256 samples × 200 NAG/Euler steps) | ~3 min | DECISION_LOG |
| `eval_ood.py` (256 samples × 11 corruption cells) | ~5 min | est. (similar order to eval_full) |

**Total budget for the priority experiments** (W1+W3+W4 all running, W5
optional): ~6 hours. **W5 long run** adds ~2.5 hours overnight if it
graduates from W1's verdict.

The previous session noted plan §3 numbers were 5× optimistic at their
MIG slice — re-budget against the measured 14 min / 5 ep above.

## Workstreams & sweep specs

The four workstreams (W1 sampler swap, W2 bigram head, W3 OOD eval, W4
FMonCLR baseline) are independent at the code level but have ordering
constraints below.

### W1 — Euler-γ sampler (RESULTS.md §1)

**Hypothesis**: EqM's NAG-GD sampler is the bottleneck. Swapping in an
Euler integrator over γ that uses raw `f(x;γ)` should close the gap to
DFM (currently 9.4× on KL_bi at parity compute).

**Code locations**: `src/aitchinson_flow/models/eqm.py` —
`EqM.sample_euler()`, `EqM.sample()` dispatcher. Configurable via
`cfg.eqm.sampler ∈ {"nag", "euler"}` and `cfg.eqm.euler_nfe`.

**Sign convention**: `_eqm_loss` line 60 sets
`u_tgt = c(γ) · (x0 − x1)` so `f` (and `∇⟨x,f⟩`) regresses to noise minus
data. The Euler step toward data is therefore `x ← x − h·v`. Sanity check
this against the actual sample quality before declaring W1 succeeded —
if Euler samples look uniformly random, flip the sign of `h` (`x + h·v`)
and re-evaluate. The local smoke test confirmed shape and finite values
but did not score sample quality.

**How to run**:

```bash
# 1. Train the three baselines (EqM-nag, DFM, FMonCLR-euler).
python scripts/run_sweep.py --sweep sweeps/phase10_euler.yaml \
    --n 256 --steps 200

# 2. Post-hoc swap EqM's sampler to Euler at multiple NFEs.
for nfe in 32 64 128 200; do
    python scripts/eval_full.py \
        --ckpt runs/eqm_data50k_ep5_v2/epoch_final.pt \
        --override eqm.sampler=euler \
        --override eqm.euler_nfe=$nfe \
        --steps $nfe \
        --out runs/eqm_data50k_ep5_v2/eval_euler_nfe${nfe}.json
done

# 3. Optional — sigma_init sweep on the best NFE.
for sigma in 0.05 0.1 0.3; do
    python scripts/eval_full.py \
        --ckpt runs/eqm_data50k_ep5_v2/epoch_final.pt \
        --override eqm.sampler=euler \
        --override eqm.euler_nfe=128 \
        --override eqm.sample_sigma_init=$sigma \
        --steps 128 \
        --out runs/eqm_data50k_ep5_v2/eval_euler_sigma${sigma}.json
done

# 4. Conservative-grad ablation (keeps Euler integrator, swaps raw f for ∇⟨x,f⟩).
python scripts/eval_full.py \
    --ckpt runs/eqm_data50k_ep5_v2/epoch_final.pt \
    --override eqm.sampler=euler \
    --override eqm.euler_use_grad=true \
    --override eqm.euler_nfe=128 \
    --steps 128 \
    --out runs/eqm_data50k_ep5_v2/eval_euler_usegrad.json
```

**Kill criterion**: if best Euler result on the EqM checkpoint does not
beat the NAG baseline (KL_bi=1.38) by ≥30% (i.e. KL_bi ≤ 0.97), the
sampler isn't the lever. Demote and rely on W3+W4 for the writeup.

**Expected runtime**: ~70 min training + ~30 min for the 6+ post-hoc evals.

### W2 — Non-factorised bigram head (RESULTS.md §2)

**Hypothesis**: emit K² joint logits per adjacent position pair from the
encoder hiddens, supervise with NLL on observed digrams. Phase 5
established that the *factorised* bigram (`log p(a)+log p(b)`) is a
re-weighted unigram CE — a real lever needs joint logits.

**Code locations**: `src/aitchinson_flow/models/eqm.py` — `BigramHead`
class; `_eqm_loss` activates it when
`cfg.eqm.lambda_bigram_joint > 0.0`. Distinct from the dead factorised
`lambda_bigram` which is preserved for reproducibility of the negative.

**How to run** (only after W1 lands and reveals the better platform):

```bash
# Update phase11_bigram.yaml's overrides to match W1's winner before running.
# E.g. add `eqm.sampler: euler` and `eqm.euler_nfe: 128` to both cells.
python scripts/run_sweep.py --sweep sweeps/phase11_bigram.yaml \
    --n 256 --steps 200
```

**Kill criterion**: if `lambda_bigram_joint=0.3` raises KL_uni above 0.10
(Phase 5 distortion cap) without dropping KL_bi by ≥20%, the term doesn't
compose with conservative-grad training. Drop and treat the trained
BigramHead as a *post-hoc* bigram-coherence scorer for W3's OOD harness:

```python
# Pseudo: pred_bg_logits = model.bigram_head(model.backbone(x_clean))
# Compute -log p(observed_digram) under pred_bg_logits as an additional
# OOD statistic in eval_ood.py.
```

**Expected runtime**: 2 × ~70 min = ~2.5 hr.

### W3 — EBM-specific OOD eval harness (RESULTS.md plan §B2)

**Hypothesis**: the EqM energy field separates clean text8 from
substitution-corrupted, shuffled, and i.i.d.-uniform sequences with
ROC-AUC > 0.65 (sanity floor 0.85 vs uniform random). DFM, having no
trained energy, gets a fairness proxy
`-log p_{1|t≈1}(x|x)` from its denoiser logits.

**Code locations**: `scripts/eval_ood.py`. Reuses
`data/corruption.py:{corrupt_token_ids,partially_shuffle_token_ids}` and
`token_ids_to_features`. Output schema is one JSON per checkpoint with
per-(stat, corruption) means/medians/stds and per-contrast ROC-AUC.

**How to run** (independent of W1 — only needs trained checkpoints):

```bash
# Score every meaningful checkpoint for the writeup table.
for run in eqm_data50k_ep5_v2 dfm_data50k_ep5_v2 fmclr_data50k_ep5_v2; do
    python scripts/eval_ood.py \
        --ckpt runs/$run/epoch_final.pt \
        --n 256 \
        --out runs/$run/ood_eval.json
done
```

**Kill criterion**: if EqM's clean-vs-substitution ROC-AUC < 0.65, fall
back to per-position `position_uncertainty` ("calibrated *positional*
uncertainty"). The healing demo qualitative result already shows
position-level signal; W3 just needs to land it as a quantitative table.
Run this fallback by re-scoring with the `U_pos_mean` / `U_pos_max`
stats already in the JSON output.

**Expected runtime**: ~5 min per checkpoint; ~15 min total.

### W4 — FMonCLR third continuous baseline (RESULTS.md plan §C)

**Hypothesis**: a non-conservative continuous-on-simplex model
triangulates the diagnosis. If FMonCLR ≈ DFM in KL_bi, the
conservative-grad indirection is the specific culprit. If FMonCLR ≈ EqM,
the issue is continuous-on-simplex more broadly.

**Code locations**:
`src/aitchinson_flow/models/fm_clr.py:FMonCLR`. Reuses EqM's
`TransformerBackbone`, `VelocityHead`, `_c_gamma` schedule, and
`decode_to_logprobs`. Trains direct MSE on `f(x_γ;γ)` against
`c(γ)·(x0−x1)` — no autograd-grad, no `⟨x,f⟩` energy. Samples via
Euler-γ (same loop as `EqM.sample_euler`, no NAG path).

**Note**: FMonCLR forces `time_conditioning="add"` in `__init__` if it's
left as `"off"` — the FM regression *needs* γ as a path index. The
sweep YAMLs set this explicitly to be safe.

**How to run**: included in `sweeps/phase10_euler.yaml`'s
`fmclr_data50k_ep5_v2` cell. Eval is automatic via
`run_sweep.py → evaluate_checkpoint`.

**Expected runtime**: ~70 min training + included in the W3 OOD batch.

### W5 — Phase 9 long run (RESULTS.md §3, conditional)

**Pre-condition**: only run if W1 closed the EqM-DFM gap to within 2× of
DFM's KL_bi. No point spending overnight on a configuration that hasn't
beaten parity-compute baselines.

**Code locations**: `sweeps/phase13_longrun.yaml`. Update its
`overrides` to match W1's winner before running. Saves every 10 epochs.

**Kill criterion** (during the run): if the 25-ep checkpoint regresses
on KL_bi vs the 10-ep checkpoint (the same overfitting signal as
`data_50k_ep10`), stop early and report the 10-ep number.

**Expected runtime**: ~140 min.

## Recommended ordering

```
parallel:
  W1 train  (eqm_data50k_ep5_v2, dfm_data50k_ep5_v2, fmclr_data50k_ep5_v2)  [~3.5 hr]
  W3 code   already in repo — runs after W1 train completes

# W3 OOD scorecard ← W1's checkpoints                                        [~15 min]
# W1 post-hoc Euler eval (NFE sweep, sigma sweep, use_grad ablation)         [~30 min]

# DECISION POINT: did Euler help (KL_bi drop by ≥30%)?
#   yes → schedule W5 long run on the Euler config                           [~2.5 hr overnight]
#   no  → demote W1 narrative; rely on W3 + W4 for the positive results

# W2 (run conditional on bandwidth):
#   if W1 winner exists and writeup needs another lever → run phase11        [~2.5 hr]
#   else: keep BigramHead unused; reuse it post-hoc as an OOD coherence
#         scorer in eval_ood.py
```

## Reading the results

Each cell writes:
- `runs/<name>/config.json` — full Config dump (use to verify the cell ran with the intended overrides)
- `runs/<name>/history.jsonl` — per-epoch metrics (train+val loss, in-loop unigram/bigram/trigram KL probe)
- `runs/<name>/eval.json` — `eval_full.py` scorecard (256 samples × 200 steps): KL_uni/bi/tri, H_ratio, |∇E|gen/gt, 8 decode strings
- `runs/<name>/ood_eval.{json,png}` — `eval_ood.py` scorecard (W3)
- `runs/<name>/epoch_final.pt` — final-only checkpoint (no optimizer state)

Cross-cell aggregation: `runs/sweep_results.jsonl` (one record per cell,
appended). Re-run `eval_full.py` post-hoc with `--override` flags as
shown above to write extra `eval_<variant>.json` files alongside.

### Headline table (for the writeup)

After all cells are done, build a single table:

| Run | KL_uni | KL_bi | KL_tri | H_ratio | OOD-AUC clean-vs-rand | OOD-AUC clean-vs-subst_0.5 | OOD-AUC clean-vs-shuffle_0.5 |
|---|---|---|---|---|---|---|---|
| eqm_data50k_ep5_v2 (NAG)             | … | … | … | … | … | … | … |
| eqm_data50k_ep5_v2 (Euler nfe=128)   | … | … | … | … | n/a (same ckpt) | n/a | n/a |
| fmclr_data50k_ep5_v2 (Euler)         | … | … | … | … | … | … | … |
| dfm_data50k_ep5_v2 (Euler-on-tokens) | … | … | … | … | … | … | … |
| (cond.) bigram_joint_l03_ep5         | … | … | … | … | … | … | … |
| (cond.) longrun_winner_50ep          | … | … | … | … | … | … | … |

The "DFM" OOD column uses the `-log p_{1|t≈1}(x|x)` proxy that
`eval_ood.py` computes when the model has no `.energy()` method.

## Environment notes

- **GPU memory**: at the default `B=64`, `d_model=1024`, 8 layers, peak
  memory is ~3 GB at training time (probed locally). The 20 GB MIG slice
  has 6× headroom — you can safely run two cells concurrently if the MIG
  slice exposes parallelism, or push `B` to 256 if you want faster
  throughput per epoch (but then re-budget against the 14-min reference).
- **Attention backend**: `TransformerBackbone` runs under
  `sdpa_kernel(SDPBackend.MATH)` because the conservative-grad branch
  uses `create_graph=True` and FlashAttention doesn't support second-order
  autograd. Don't switch backends. `DFMBackbone` and FMonCLR's backbone
  (which is `TransformerBackbone`) inherit this; FMonCLR doesn't *need*
  MATH (no second-order autograd) but uses it because it shares the
  module — that's a minor perf cost (≤ 10%), acceptable.
- **HuggingFace**: text8 loads via `afmck/text8`. The local environment
  fetched it without a token; if the cluster lacks internet, set
  `HF_TOKEN` and pre-download or vendor the dataset. The cache lives at
  `cfg.text8_dataset.cache_dir` (defaults to HF's default, typically
  `~/.cache/huggingface`).
- **W&B logging**: optional, off by default. Enable per cell with
  `--wandb` on `run_sweep.py`. Project is `eqm-text8`; the previous
  session used it. The local session did not enable it.

## Risk registry / failure modes

| If you see... | Likely cause | Action |
|---|---|---|
| Euler samples look like uniform CLR noise across γ | Sign error: `x + h·v` instead of `x − h·v`. The plan's pseudocode in `RESULTS.md` line 416 has the wrong sign — trace against `eqm.py` line 60. | Flip the sign in `EqM.sample_euler()` and re-evaluate. |
| `fmclr_data50k_ep5_v2` produces NaNs | The MSE loss on raw velocity is sensitive when σ is large at γ≈0. | Reduce `eqm.source_sigma` to 0.05, or clip the velocity output (`v.clamp(-10, 10)`). |
| BigramHead spike in `KL_uni` (above 0.10) | Joint-head NLL pulls the linear decoder's per-position margins around. This is the documented Phase-5 distortion mode reappearing under a new head. | Drop `lambda_bigram_joint`; reuse the trained head as a post-hoc OOD bigram scorer. |
| OOD eval ROC-AUC near 0.5 on shuffle | Expected if EqM's per-position aux CE is the only joint anchor — the energy field doesn't see joint structure. | Document as a *predicted* failure mode. The fix (a non-factorised bigram head, W2) is itself a hypothesis. |
| OOD eval ROC-AUC near 0.5 on substitution | The energy field is degenerate (e.g., the FM target's c(γ=1)=0 trap if `sample_gamma` defaulted to 1). | Verify `cfg.eqm.sample_gamma=0.5`; the local default is 0.5, but if checkpoint config saved `sample_gamma=1.0` for any reason, override at eval time. |

## Update conventions for `runs/DECISION_LOG.md`

Append entries in the format the previous session used:

```
## [<UTC>] <run-name>
- Hypothesis: <single sentence>
- Result: KL_uni=… KL_bi=… KL_tri=… H_ratio=… OOD-AUC(clean-vs-subst_0.5)=…
- Decision: <continue|branch to phase X|abandon and explain>
- Next: <run-name>
```

For W1 post-hoc evals, group under one entry (e.g. `eqm_data50k_ep5_v2 W1
sampler sweep`) rather than one entry per NFE — the sweep is a single
hypothesis with multiple data points.

## Writeup arc (for the user)

The local session settled on this three-part structure:

1. **Setup & negatives-as-controls**: text8 K=27/L=40, four documented
   architectural negatives from the previous session (epoch scaling,
   factorised n-gram NLL, γ-conditioning, backbone scaling) localise the
   bottleneck. Honest, load-bearing for the methods section.

2. **Closing the gap with a sampler swap (W1 + W4 + W5)**: parity-compute
   KL_bi for {EqM-NAG, EqM-Euler, FMonCLR-Euler, DFM}. Even one
   continuous-on-simplex model crossing the 0.50 KL_bi target carries
   this section. W4 specifically lets the writeup attribute outcomes to
   the conservative-grad indirection (if FMonCLR ≈ DFM) vs the simplex
   geometry (if FMonCLR ≈ EqM).

3. **The differentiator (W3)**: ROC curves for clean-vs-corrupt sequence
   scoring. EqM's energy field is what DFM's denoiser doesn't have. *Even
   if W1/W4 fail, this section can carry the writeup on its own* —
   because it measures an existing property of an existing checkpoint,
   the failure modes are bounded.

If W1 succeeds: lead with §2 ("the sampler matters more than the
representation") and use §3 as the unique-value-add.
If W1 fails: lead with §3 ("the energy field is uniquely useful even
when sample quality lags") and use §2 as honest negative.
Either path produces a publishable writeup; the local session designed
the experiments so the user gets at least one positive result.

## Files NOT to touch

- `runs/DECISION_LOG.md` formatting from the previous session (append
  only, don't reformat).
- `RESULTS.md` (it's the previous session's writeup; only update if you
  add a new top-level section for the current session).
- `TRAINING_PLAN.md`, `SESSION_SUMMARY.md` — historical artefacts.
- The dead config knobs (`eqm.lambda_vol`, `eqm.lambda_mse`, `eqm.alpha`)
  — leave them in place; removing would break checkpoint loading.

## After all cells run

Drop a "Phase 14 summary" at the bottom of `runs/DECISION_LOG.md` mirroring
the previous session's "Final summary (this session)" block. Include the
headline table from §"Reading the results" above and one paragraph about
which workstream produced the writeup-headline result. The user will read
this summary first.
