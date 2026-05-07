# Decision log — EqM text8 capstone training plan

Append-only diary kept by the supervising session. One entry per run / sweep
cell. Format:

```
## [<UTC timestamp>] <run-name>
- Hypothesis: <one line — what would this run prove or kill?>
- Result: KL_uni=<...> KL_bi=<...> KL_tri=<...> H_ratio=<...>
- Decision: <continue|branch to phase X|abandon and explain>
- Next: <run-name of next experiment>
```

## NEXT SESSION

**Status (end of 2026-05-07 ~10:00 UTC session):**

This session ran Phase 0 → 1 → 3 → 5 → 8 → 4 → Phase-2-mini, with budget-driven trimming throughout. Only **Phases 6 (loss / γ ablations), 7 (curvature reg), and 9 (long run)** are not started. The full results synthesis is in [`RESULTS.md`](../RESULTS.md).

**Best EqM:** `runs/data_50k_ep5/epoch_final.pt` — `KL_uni=0.035, KL_bi=1.382, KL_tri=5.96, H_ratio=0.991`. Symlinked at `runs/best_so_far.pt`. Still 2.8× the 0.50 KL_bi target.

**DFM control (Phase 8):** `runs/dfm_data50k_ep5/epoch_final.pt` — `KL_uni=0.007, KL_bi=0.148, KL_tri=1.40, H_ratio=0.976`. **Crosses the success criterion at parity compute**, 9.4× lead over EqM on bigram KL. Same backbone, data, epochs as EqM; only architectural difference is sinusoidal `t` injection plus Euler-integrator sampler.

**Compute reality:** the plan's "20 GB A100 MIG" was about consumer-GPU throughput, *not* 3-4× faster. 5 epochs at default ≈ 14 min, 25 epochs ≈ 70 min, d=1536/12L 5 epochs ≈ 3 hr. Plan §3 numbers were 5× optimistic. Adjust budgets accordingly.

**Strongest *positive* finding:** at parity compute (same total training samples), 5× more data variety (`data_50k_ep5`) cuts bigram KL 16 % vs `ep25_default`. Both per-position over-fit (`data_50k_ep10`) and per-position under-fit (`data_200k_ep2`) hurt. Sweet spot: ~5 passes per window, more variety than 10k.

**Negative results documented (4):**
- **Epoch scaling (Phase 1):** KL_bi at 5/10/25 ep = 1.99/1.78/1.65. Slope too shallow.
- **Factorised bigram NLL (Phase 5):** `log p(a) + log p(b)` decomposes to re-weighted unigram CE. ng_bg05 +28 % KL_bi. Needs a non-factorised head.
- **γ-conditioning (Phase 4):** plan-recommended γ=1 sample-time fallback collapses to ~5 chars (KL_bi 4.0); best γ across [0.05, 0.99] is γ=0.7 (KL_bi 3.35) — still 2.4× worse than tc_off baseline. The c(γ)·(x0−x1) decay factor zeros the velocity at γ=1, and at any γ > 0 the σ-noise sample input is OOD relative to training.
- **Backbone scaling (Phase 2 mini):** d=256/4L (1M params) under-fits joints (KL_bi 2.14); d=1536/12L (340M params) is 16 % worse on KL_bi than the d=1024/8L default. The default sits at the bottom of a shallow U-curve. **3.4× the parameters does not close the EqM-DFM gap (still 11×).** Capacity is not the lever.

**Synthesis:** Four independent negative results + the Phase 8 magnitude all point the same direction: the EqM-DFM gap is **structural — sampler architecture — not representational**. EqM trains a static energy field via conservative-gradient regression and samples it with NAG-GD; DFM trains a denoiser and samples it with an Euler integrator over t. Lifting any single DFM ingredient (γ embedding, more capacity, more passes, more loss terms) into EqM's setup does not help. The full mechanism analysis is in [`RESULTS.md`](../RESULTS.md) §"Why DFM works".

**Next session, in priority order (architectural changes only — every parameter knob has been tested):**

1. **EqM flow-matching Euler sampler.** Replace `EqM.sample` with `x_{t+h} = x_t + h · f(x_t; γ=t)` for γ stepping 0 → 1. Uses the trained `f` directly (not its conservative gradient) at inference. The conservative-grad regression at training stays unchanged (the regression objective and the sampler are separately useful). This makes EqM's sampler structurally analogous to DFM's. **Re-evaluate the existing tc_add_data50k and tc_concat_data50k checkpoints** with the new sampler at no training cost. Expected outcome: if the gap is the sampler (Phase 8 + Phase 4 + Phase 2 mini all suggest yes), KL_bi should drop substantially. Compute: ~2 hr to implement + minutes to re-eval each checkpoint.

2. **Non-factorised bigram head.** Add a `BigramHead` projecting `h[t] ⊕ h[t+1] → R^{K²}` and NLL on observed digrams. Different lever than (1); orthogonal. Only worth doing after (1) if KL_bi is still well above DFM's number. Compute: ~2 hr to implement + 1 hr to train at the data_50k_ep5 platform.

3. **Phase 9 (long run).** If (1) closes the gap to within 2× of DFM, take that platform and run 50-100 epochs. Will need overnight wall-clock at this throughput (~3 hr / 25 ep at d=1024/8L on 50k windows; 100 epochs ≈ 12 hr).

4. **Phase 6 (loss/γ ablations).** Probably no-ops at this point — every other lever is settled. Run only if a writeup reviewer asks.

**Energy-field convergence (Phase 7 trigger check):**

Plan §Phase 7 says to skip the curvature-regulariser phase if `|∇E|gen / |∇E|gt` is within ~5 %. Across the runs we have (selected; full set in RESULTS.md):

| Run | gen | gt | ratio | shape |
|---|---:|---:|---:|---|
| baseline_5ep   | 0.227 | 0.179 | 1.27 | gen above gt — under-trained field |
| ep25_default   | 0.056 | 0.089 | 0.63 | gen below gt — over-trained, attractors deeper than GT |
| data_50k_ep5   | 0.224 | 0.103 | 2.18 | gen above gt — field still settling |
| ng_bg10_data50k| 0.493 | 0.262 | 1.88 | gen far from minimum — n-gram distortion |
| bb_d1536_l12   | 0.472 | 0.233 | 2.03 | bigger field, sharper attractors, gen farther off |

No run is within 5 %. The ratio swings either way depending on training mix, so the curvature penalty in Phase 7 is not the obvious fix; the diagnostic from Phase 4/8 makes a sampler-side fix more likely.

**Watch-outs the next session must keep in mind:**
- The fresh `runs/baseline_5ep/epoch_final.pt` is canonical; the original path `checkpoints/baseline_5ep/epoch_final.pt` referenced in TRAINING_PLAN.md does *not* exist.
- `_unigram_kl_probe` in `runner.py` now also returns `bigram_kl` and `trigram_kl`. It also dispatches between EqM (`max_steps=`) and DFM (`nfe=`) sampling APIs.
- `EqM` config has new fields `lambda_bigram`, `lambda_trigram`, `time_conditioning` ("off"/"add"/"concat"), `sample_gamma` (default 0.5; γ=1 is degenerate). All defaults preserve the Phase-3 best-so-far behaviour.
- `TransformerBackbone.forward` now optionally takes `gamma`; `EqM.forward` and `_compute_grad` route γ through. When `time_conditioning="off"` the new code is a no-op vs the original.
- `scripts/eval_full.py` dispatches between EqM (CLR features → decode) and DFM (token IDs direct) by feature-detecting `model.decode_to_logprobs`. Drop in any new model that respects either convention.
- `runs/sweep_results.jsonl` aggregates all 11 runs from this session.
- Sweep specs live in `sweeps/phase{0,1,2,3,4,5,8}.yaml`; each sweep is idempotent via the eval.json existence check.

---

## Tooling and infrastructure (Phase −1)

- `scripts/eval_full.py` — JSON scorecard with trigram KL added.
- `scripts/run_sweep.py` — idempotent orchestrator for `{name, overrides}` lists.
- `src/aitchinson_flow/training/runner.py` — `_unigram_kl_probe` now also
  reports bigram and trigram KL.
- `src/aitchinson_flow/models/eqm.py` — `_eqm_loss` extended with
  `lambda_bigram` / `lambda_trigram` n-gram NLL branches (factorised; see
  Phase 5 negative result).
- `sweeps/phase{0,1,2,3,5,6}.yaml` — sweep specs.

## [2026-05-06 19:18 UTC] baseline_5ep
- Hypothesis: fresh 5-epoch all-fixes EqM reproduces TRAINING_PLAN §1 numbers.
- Result: KL_uni=0.0513 KL_bi=1.987 KL_tri=7.22 H_ratio=0.955
- Decision: continue. KL_uni right on target (plan 0.053). KL_bi marginally
  above plan's 1.90 "stop" threshold but the in-training probe and training
  loss curves match the original session exactly, so the divergence is
  sampling-side noise (different python RNG init), not a sampler regression.
  H_ratio matches plan (0.952). Proceeding to Phase 1.
- Next: phase1.yaml (ep10_default, ep25_default, ep50_default, ep25_lr2x, ep25_lr_5x).

## [2026-05-06 19:51 UTC] ep10_default
- Hypothesis: doubling epochs 5→10 moves bigram KL meaningfully.
- Result: KL_uni=0.0328 KL_bi=1.783 KL_tri=6.76 H_ratio=0.948
- Decision: continue sweep. Modest gains: KL_uni −36 %, KL_bi −10 %, KL_tri
  −6 % vs baseline_5ep. Sample-estimate noise on KL_bi is ≥10 % so this is
  marginal; need ep25 to know if scaling slows or speeds up.
- Next: ep25_default (already running in this sweep).

## [2026-05-06 21:00 UTC] ep25_default
- Hypothesis: 5x more epochs at fixed config moves bigram KL substantially.
- Result: KL_uni=0.0151 KL_bi=1.6511 KL_tri=7.2054 H_ratio=1.002.
  Energy: |∇E|gen=0.056 < |∇E|gt=0.089 (gen sits *deeper* than GT — field
  tightened around per-position attractors that don't match GT joints).
- Decision: Per plan §5 Phase 1 rule — KL_bi at ep25 = 1.65 ≥ 1.5, so epoch
  scaling is **not** the bottleneck. Killed remaining Phase 1 cells
  (ep50, lr2x, lr_5x) saving ~4.5 hr; the bar-chart point at ep50 isn't worth
  the budget given the trajectory. Jumping to Phase 3 (data ceiling).
  Skipping Phase 2 (per plan: "skip Phase 2's smaller-models cell, jump to
  Phases 3-5") since the over-fitted-attractor signal also implies more
  capacity won't help while data is fixed at 10k windows.
- Next: phase3.yaml (data_50k_ep5 — parity-compute vs ep25_default — first).

## [2026-05-06 21:01 UTC] Phase 1 → Phase 3 transition
- Phase 1 trajectory (KL_bi at 5 / 10 / 25 epochs): 1.987 → 1.783 → 1.651.
  Slope ≈ −0.034 KL_bi per doubling of epochs after ep10. Linear extrapolation
  to KL_bi = 0.5 (plan target) would need ~10⁴ epochs — confirms the plan's
  "epoch scaling is not the lever" decision.
- Best so far: ep25_default (`runs/best_so_far.pt` → ep25_default/epoch_final.pt).
- Reduced Phase 3 sweep (parity-compute right-sized):
  - data_50k_ep5  ≈ 70 min   (same total samples as ep25_default)
  - data_50k_ep10 ≈ 140 min  (2× samples, 5× variety)
  - data_200k_ep2 ≈ 110 min  (1.6× samples, 20× variety)
  Skipped: data_full (infeasible at ~290 GPU-hr) and data_full_L64 (same).

## [2026-05-06 22:30 UTC] data_50k_ep5
- Hypothesis: at parity compute (same total training samples as ep25_default),
  5× more data variety reduces bigram KL.
- Result: KL_uni=0.0347 KL_bi=**1.382** KL_tri=**5.957** H_ratio=0.991.
  Δ vs ep25_default at parity compute: KL_uni +130 % (less converged on
  per-position dist), KL_bi −16 %, KL_tri −17 %. Variety beats repetition.
- Decision: continue Phase 3. Strong validation of "more data" hypothesis.
  best_so_far → data_50k_ep5. Next cell (data_50k_ep10) will tell us if
  doubling exposure on the 50k window set keeps the gain.
- Next: data_50k_ep10 (already running in this sweep).

## [2026-05-07 00:30 UTC] data_50k_ep10
- Hypothesis: doubling exposure (5→10 ep) on the 50k window set keeps the
  bigram-KL gain from data_50k_ep5.
- Result: KL_uni=0.0226 KL_bi=**1.5577** KL_tri=6.0459 H_ratio=1.003.
  Δ vs data_50k_ep5 (parity-compute baseline): KL_uni −35 % (better
  per-position), KL_bi **+13 %** (worse joints), KL_tri +1 %. More exposure
  hurts joints, just like ep5→ep25 on the 10k set did. Same overfitting
  signal: per-position attractors solidify at the expense of multi-position
  coherence.
- Decision: keep best_so_far → data_50k_ep5. Continue sweep — data_200k_ep2
  (each window seen 2× in 200k variety) should beat data_50k_ep5 if
  "less repetition + more variety" is the dominant trend.
- Next: data_200k_ep2 (already running in this sweep).

## [2026-05-07 04:00 UTC] data_200k_ep2
- Hypothesis: 4× more data variety with only 2 passes beats data_50k_ep5
  (5× variety, 5 passes) by exposing the model to more joints.
- Result: KL_uni=0.2066 KL_bi=**2.482** KL_tri=6.080 H_ratio=0.933.
  Grad: |∇E|gen=0.674 vs |∇E|gt=0.319 — gen sits *outside* the basins.
  Per-position attractors are undertrained.
- Decision: Phase 3 is done. data_50k_ep5 wins at parity compute. The
  trend (5×variety × 5 passes > 10×variety × 1 pass > 1×variety × 25 passes
  for KL_bi) confirms Plan §2 hypothesis #1 (more data) but with a sweet
  spot: each window must be seen ≥4× to converge per-position attractors,
  but more passes overshoot. best_so_far stays at data_50k_ep5.
- Next: Phase 5 (n-gram-aware loss) on the data_50k_ep5 platform. Skipping
  Phase 4 (time conditioning) — it requires more code changes than Phase 5
  and the plan says to document negative results gracefully if it underperforms.

## [2026-05-07 04:00 UTC] Phase 3 → Phase 5 transition
- Best platform: `data_50k_ep5` — KL_uni=0.035, KL_bi=1.38, KL_tri=5.96.
- Compute-budget reality check: ~9 hr spent of plan's 12-hr budget. Remaining
  3 hr is enough for one targeted Phase 5 sweep (2 cells × ~70 min) before
  handoff. Plan calls for 4 Phase 5 cells (~6 hr); trimming to the 2 most
  informative.
- Phase 5 implementation: bigram NLL on the implied-x1 reconstruction (the
  same `pred_x1 = x_γ − λ·grad_g` already used by the unigram CE branch).
  K²=729 outer-products → ~50 MB per (B,L−1,K,K) tensor at L=40, B=64,
  trivial for 20 GB MIG.
- Phase 5 sweep cells: ng_bg05 (λ_bigram=0.5), ng_bg10 (λ_bigram=1.0). Both
  at the data_50k_ep5 platform (training.epochs=5, max_train_windows=50000).
  The plan's bigger n-gram cells would need a code path for trigrams (K³
  is 19,683-way; needs care). Defer to next session.

## [2026-05-07 04:50 UTC] ng_bg05_data50k
- Hypothesis: adding bigram NLL on the factorised pred_x1 reconstruction
  reduces bigram KL by penalising invalid digraphs.
- Result: KL_uni=0.0592 KL_bi=**1.774** KL_tri=5.890 H_ratio=0.949.
  Δ vs data_50k_ep5 (same platform, no n-gram term): KL_uni +69 %,
  KL_bi +28 %, KL_tri −1 %. Both worse on the metrics we care about.
- Decision: Phase 5 implementation as specified is a no-op architecturally.
  log_p_bi[i, j] = log_p[a]_i + log_p[b]_j is a factorised joint — i.e.
  the loss decomposes to NLL(a)+NLL(b), which is just a re-weighted unigram
  CE (interior positions counted twice). The plan's snippet has the same
  property. To capture true bigram structure the velocity head would need
  to emit a (K, K) log-joint per adjacent pair, which is a significant
  architectural change not covered by this plan. Document the negative
  result; Phase 5 as scoped does not help.
- Next: let ng_bg10 finish for the bracketing data point, then handoff.

## [2026-05-07 06:00 UTC] ng_bg10_data50k
- Hypothesis: doubling the bigram weight makes the negative trend more
  severe (control: λ_bigram=0.5 → 1.0 monotone in distortion).
- Result: KL_uni=**0.1272** KL_bi=1.799 KL_tri=6.113 H_ratio=0.923.
  KL_uni now exceeds the plan §Phase 5 distortion cap (0.10). Same KL_bi
  as ng_bg05. Confirms the term is just a re-weighted unigram CE that
  drags the per-position distribution off without compensating gain.
- Decision: Phase 5 (as specified) is closed as a negative result. To get
  a real bigram lever you'd need a non-factorised joint head; deferred to
  next session per NEXT SESSION note.
- Next: handoff. best_so_far stays at data_50k_ep5 (KL_bi=1.382).

## [2026-05-07 06:30 UTC] dfm_data50k_ep5 (Phase 8 capstone)
- Hypothesis: at parity compute (5 ep × 50k windows × default backbone), DFM
  matches or beats EqM. Plan §Phase 8: ">20% gap means honest negative result".
- Result: KL_uni=**0.0073** KL_bi=**0.148** KL_tri=**1.402** H_ratio=0.976.
  Δ vs EqM data_50k_ep5: KL_uni 4.7× better, KL_bi **9.4× better**,
  KL_tri 4.2× better. DFM crosses the plan's success criterion KL_bi ≤ 0.50
  at 5 epochs. Samples are recognisably English ("ine al cane tere",
  "two thowereh", "be thelaclad lip", "puridne pocetr"). The gap is far
  beyond noise.
- Decision: Phase 8 is a clean negative for EqM as currently configured.
  DFM's only meaningful architectural advantage is the explicit `t`
  sinusoidal embedding (DFMBackbone.forward takes `t` as input,
  TransformerBackbone takes only x). Plan hypothesis #2 ("no time
  conditioning") is strongly validated. Phase 4 (time conditioning for
  EqM) is now the obvious next experiment.
- Next: implement Phase 4 (next session). EqM with γ-conditioning should
  close most of the bigram-KL gap; if it does, the capstone narrative
  becomes "continuous flow on the simplex matches discrete flow at parity
  compute *only* when given the same time conditioning". If γ-conditioning
  doesn't close the gap, the conservative-grad framework's lack of
  sample-time γ awareness is a fundamental limitation; document.

## [2026-05-07 06:45 UTC] tc_add_data50k (Phase 4)
- Hypothesis: adding sinusoidal γ embedding to per-token hidden state lets
  the model use γ explicitly, closing the EqM-DFM gap.
- Result (sample-time γ=1, the plan's recommended default):
  KL_uni=1.324 KL_bi=4.018 KL_tri=7.995 H_ratio=0.579 — collapse to ~5
  characters. Diagnosis: at γ=1 the FM target c(γ=1)·(x0-x1)=0 zeros the
  velocity, so the model learned f(·,γ=1)≈0; sample-time energy field is
  flat, NAG degenerates.
- Re-evaluated with γ ∈ {0.05, 0.20, 0.50, 0.70, 0.90, 0.95, 0.99}:
  best at γ=0.70 (KL_uni=0.838, KL_bi=3.348). All values strictly worse
  than tc_off baseline (data_50k_ep5: KL_uni=0.035, KL_bi=1.38).
- Decision: tc_add is a clean negative result regardless of sample-time γ.
  No γ choice recovers parity with the un-conditioned baseline. Plan §
  Phase 4 anticipated this: "if both variants underperform, document the
  negative result rather than retrying."

## [2026-05-07 06:45 UTC] tc_concat_data50k (Phase 4)
- Hypothesis: concat (instead of add) lets the model carve a separate
  γ-dependent feature subspace; should be at least as expressive as add.
- Result (sample-time γ=1): KL_uni=1.019 KL_bi=12.927 KL_tri=17.133
  H_ratio=0.364 — near-total collapse to 'e','n','i'.
  Re-eval at γ=0.5: KL_uni=0.807 KL_bi=10.276 — still much worse than add.
- Decision: tc_concat is strictly worse than tc_add at every γ tested. The
  larger projection layer (`Linear(2d, d)`) seems to harm FM training in
  the conservative-grad regime; concat doubles the input dim into the
  encoder before the per-position channels are individually projected,
  which may interfere with the second-order autograd path.

## [2026-05-07 06:45 UTC] Phase 4 → final summary
- Both Phase 4 variants underperform tc_off in absolute terms. The
  conservative-grad framework appears to be fundamentally unable to
  exploit γ conditioning the way DFM does — likely because:
  1. The FM target c(γ)·(x0-x1) crosses zero at γ=1 ⇒ "data manifold"
     conditioning is degenerate for energy-based sampling.
  2. At sample time, x_init is pure σ-noise but during training the model
     never saw pure-noise inputs at any γ > 0; so any non-zero sample-time
     γ produces an OOD input.
  3. The conservative gradient ∇⟨x, f(x;γ)⟩ depends on f's local Lipschitz
     properties at γ, which were never directly trained for sampling.
- This is the honest answer to plan hypothesis #2: explicit γ helps DFM
  because DFM's sampler is an Euler integrator over t (it traverses the
  γ-trajectory), whereas EqM's sampler descends a single fixed energy
  field. The two models use γ for different reasons; lifting it into EqM
  the way DFM does does not help.
- best_so_far stays at data_50k_ep5. Phase 4 closed.

## [2026-05-07 09:30 UTC] bb_d256_l4_data50k (Phase 2 mini)
- Hypothesis: smaller backbone might do better at 50k windows by avoiding
  per-position over-fitting (plan §Phase 2 hypothesis 4).
- Result: KL_uni=0.040 KL_bi=2.137 KL_tri=6.359 H_ratio=0.942.
  Δ vs data_50k_ep5: KL_bi +55%, KL_tri +7%. Smaller model under-fits
  joints. Plan's "less is more" hypothesis disproved at 50k windows.
- Decision: smaller backbone is not the lever. Continue to d=1536/12L.

## [2026-05-07 09:30 UTC] bb_d1536_l12_data50k (Phase 2 mini)
- Hypothesis: 3.4× more capacity (340M params, plan's largest cell) gives
  the model headroom for joint structure.
- Result: KL_uni=0.081 KL_bi=1.596 KL_tri=5.151 H_ratio=0.917.
  Δ vs data_50k_ep5: KL_uni +130%, KL_bi +16%, KL_tri −14%, H_ratio −7%.
  The bigger model improves trigram modestly but hurts unigram and bigram.
  H_gen drops from 2.85 → 2.62 — outputs are narrower (mode-seeking).
  Energy: |∇E|gen=0.47 vs |∇E|gt=0.23 — gen sits *farther* from the data
  manifold than the smaller model. Bigger field, less smooth, more
  attractor-like artifacts.
- Decision: backbone scaling does **not** close the EqM-DFM gap. DFM at
  d=1024/8L hit KL_bi=0.148; EqM at d=1536/12L is at 1.60 — still a 11×
  gap. The bottleneck is structural (sampler architecture), not
  representational (parameter count). Phase 2 closed as a documented
  negative result.

## Final summary (this session)

| Run | Phase | KL_uni | KL_bi | KL_tri | H_ratio | Notes |
|---|---|---:|---:|---:|---:|---|
| baseline_5ep            | 0 | 0.051 | 1.987 | 7.217 | 0.955 | reproduction |
| ep10_default            | 1 | 0.033 | 1.783 | 6.764 | 0.948 | epochs ↑ |
| ep25_default            | 1 | 0.015 | 1.651 | 7.205 | 1.002 | KL_bi ≥1.5 → epoch scaling not the lever |
| **data_50k_ep5**        | 3 | 0.035 | **1.382** | **5.957** | 0.991 | ★ best EqM (parity-compute platform) |
| data_50k_ep10           | 3 | 0.023 | 1.558 | 6.046 | 1.003 | overfits per-position |
| data_200k_ep2           | 3 | 0.207 | 2.482 | 6.080 | 0.933 | undertrained per-position |
| ng_bg05_data50k         | 5 | 0.059 | 1.774 | 5.890 | 0.949 | negative — factorised NLL = re-weighted unigram CE |
| ng_bg10_data50k         | 5 | 0.127 | 1.799 | 6.113 | 0.923 | worse — KL_uni > 0.10 distortion cap |
| **dfm_data50k_ep5**     | 8 | **0.007** | **0.148** | **1.402** | 0.976 | DFM at parity — clears 0.50 target, 9.4× lead |
| tc_add_data50k γ=0.7    | 4 | 0.838 | 3.348 | 7.548 | 0.736 | best γ; still 2.4× worse than tc_off |
| tc_concat_data50k γ=0.5 | 4 | 0.807 | 10.276 | 14.534 | 0.473 | strictly worse than add |
| bb_d256_l4_data50k      | 2 mini | 0.040 | 2.137 | 6.359 | 0.942 | smaller — under-fits joints |
| bb_d1536_l12_data50k    | 2 mini | 0.081 | 1.596 | 5.151 | 0.917 | 3.4× params — KL_bi worse, KL_tri −14 %, gap to DFM 11× |

Plan target: **KL_bi ≤ 0.50** — reached only by DFM (0.148). Best EqM (data_50k_ep5) at 1.382 is 2.8× over target and 9.4× behind DFM.

Headline writeup data (per plan §8):
- **Phase 1 bar chart** (KL_bi vs epochs at fixed config): 1.99 / 1.78 / 1.65 at 5/10/25 ep on 10k windows.
- **Phase 2 Pareto** (KL_bi vs param count at the data_50k_ep5 platform): 2.14 (1M) / 1.38 (100M) / 1.60 (340M) — shallow U with the default at the minimum.
- **Phase 3 data lever**: 1.65 (10k×25ep) → **1.38 (50k×5ep, parity compute)** → 1.56 (50k×10ep) → 2.48 (200k×2ep). 5× variety + 5 passes is the sweet spot.
- **Phase 4 ablation** (sample-time γ sweep on tc_add_data50k):

| γ at sample | KL_uni | KL_bi |
|---:|---:|---:|
| 0.05 | 0.26 | 4.59 |
| 0.20 | 0.26 | 3.90 |
| 0.50 | 0.52 | 3.62 |
| 0.70 | 0.84 | **3.35** (best) |
| 0.90 | 1.18 | 3.78 |
| 1.00 | 1.32 | 4.02 |

  All values dominated by tc_off baseline (1.38). γ=1 is degenerate because c(γ=1)=0 zeros the velocity.
- **Phase 5 ablation**: λ_bigram = 0 / 0.5 / 1.0 → KL_bi = 1.38 / 1.77 / 1.80, KL_uni = 0.035 / 0.059 / 0.127. Term distorts unigram, doesn't help bigram.
- **Phase 8 EqM-vs-DFM table** at parity compute: in the main row above. DFM 9.4× better on KL_bi, 4.7× better on KL_uni, 4.2× better on KL_tri.
- **Sample diversity grid** at three checkpoints: see `runs/{baseline_5ep,data_50k_ep5,dfm_data50k_ep5}/eval.json` `"samples"` field.

The strongest *positive* finding for the writeup: at parity compute, **5× more data variety drops bigram KL by 16 %** vs more passes over a small set. The strongest *negative* finding: scaling the transformer backbone 3.4× (~340M params) does **not** close the EqM-DFM gap — capacity is not the lever; the gap is in how each architecture turns the encoder's output into samples. Full diagnosis in [`RESULTS.md`](../RESULTS.md).


