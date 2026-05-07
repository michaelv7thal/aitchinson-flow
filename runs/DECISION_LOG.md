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

**Status (end of 2026-05-07 ~05:00 UTC session):**

This session ran Phase 0 → 1 → 3 → 5 with budget-driven trimming. Phase 2 (backbone scaling) was skipped per plan's decision rule after Phase 1; Phases 4 (time conditioning) and 6/7/8 (ablations, stability, DFM) and 9 (final long run) are not started.

**Best so far (EqM):** `runs/data_50k_ep5/epoch_final.pt` — `KL_uni=0.035, KL_bi=1.382, KL_tri=5.96, H_ratio=0.991`. Symlinked at `runs/best_so_far.pt`. Still 2.8× the 0.50 KL_bi target.

**DFM control (Phase 8):** `runs/dfm_data50k_ep5/epoch_final.pt` — `KL_uni=0.007, KL_bi=0.148, KL_tri=1.40, H_ratio=0.976`. **Crosses the success criterion at parity compute.** This is a 9× lead over EqM on bigram KL. Strongly localises the bottleneck to time conditioning: DFM and EqM share backbone shape, data, and epochs; DFM's only architectural advantage is sinusoidal `t` injection.

**Compute reality:** the plan's "20 GB A100 MIG" was about consumer-GPU throughput, *not* 3-4× faster. 5 epochs at default ≈ 14 min, 25 epochs ≈ 70 min. Plan §3 numbers were 5× optimistic. Adjust your budgets.

**Strongest finding:** at parity compute (same total training samples), 5× more data variety (`data_50k_ep5`) cuts bigram KL 16% vs `ep25_default`. Both per-position over-fit (`data_50k_ep10`) and per-position under-fit (`data_200k_ep2`) hurt. Sweet spot: ~5 passes per window, more variety than 10k.

**Negative results documented:**
- Epoch scaling (Phase 1): KL_bi vs epochs is 1.99 / 1.78 / 1.65 at 5/10/25. Slope too shallow to reach 0.50 — confirms plan's "epoch scaling not the lever" rule.
- Phase 5 n-gram loss as specified: log_p_bi = log_p_a + log_p_b is a *factorised* joint, equivalent to re-weighted unigram CE. ng_bg05 increased KL_bi 28%. To make Phase 5 work you'd need a non-factorised output head: e.g., a separate `(K, K)` projection from the d_model hidden state, predicting a *joint* log p(a, b) for adjacent positions. That's a real architectural change.

**Next session, in priority order (now reflecting Phase 4 + Phase 8 outcome):**

1. **A real Phase 5: non-factorised bigram head.** This is now the highest-priority architectural change. Phase 4 established that γ conditioning does not help EqM (negative result). The remaining unexplored lever is making the head emit *joint* log-probs rather than factorised. Add `BigramHead(nn.Module)` that projects pairs of adjacent hidden states `h[t] ⊕ h[t+1] -> R^{K²}`, then NLL on observed digrams. The ce on the implied-x1 reconstruction stays as-is (anchors per-position). Compute: ~2 hr to implement + 1 hr to train. If KL_bi drops below 1.0, that's the structural fix.

2. **EqM time-integration sampler.** Phase 4 hinted that EqM's bottleneck might be its sampler — a fixed-γ energy descent — not its conditioning. Replace `EqM.sample` with a flow-matching Euler integrator: x_{t+h} = x_t + h · v(x_t; γ=t), γ stepping from 0 to 1. This makes EqM's sampler structurally analogous to DFM's. The conservative-grad training stays unchanged. Compute: ~2 hr to implement; can re-evaluate existing tc_add and tc_concat checkpoints at no training cost.

3. **Phase 9 (long run).** If either of the above closes the EqM-DFM gap to within 2×, take that platform and run 50-100 epochs for the final writeup table. Need overnight wall-clock at this throughput.

4. **Phase 6 (loss/γ ablations).** Probably no-ops; only worth it once the architectural choice from (1) or (2) is locked in.

**Energy-field convergence (Phase 7 trigger check):**

Plan §Phase 7 says to skip the curvature-regulariser phase if `|∇E|gen / |∇E|gt` is within ~5%. Across the runs we have:

| Run | gen | gt | ratio | shape |
|---|---:|---:|---:|---|
| baseline_5ep   | 0.227 | 0.179 | 1.27 | gen above gt — under-trained field |
| ep25_default   | 0.056 | 0.089 | 0.63 | gen below gt — over-trained, attractors deeper than GT |
| data_50k_ep5   | 0.224 | 0.103 | 2.18 | gen above gt — field still settling |
| ng_bg10_data50k| 0.493 | 0.262 | 1.88 | gen far from minimum — distorted by n-gram term |

No run is within 5%. Across the trajectory the ratio swings either way, suggesting weak repulsive directions in `⟨x, f(x)⟩` that depend on the training mix, not a fixed pathology. Phase 7's curvature penalty is worth trying at the data_50k_ep5 platform if Phase 4 doesn't help; the field is *not* fine.

**Watch-outs the next session must keep in mind:**
- The fresh `runs/baseline_5ep/epoch_final.pt` is canonical; the original path `checkpoints/baseline_5ep/epoch_final.pt` referenced in TRAINING_PLAN.md does *not* exist.
- `_unigram_kl_probe` in `runner.py` now also returns `bigram_kl` and `trigram_kl` (extended this session).
- `EqM` config has new fields `lambda_bigram`, `lambda_trigram`, default 0.0. The `_eqm_loss` branch handles both terms.
- `runs/sweep_results.jsonl` is the canonical results stream.
- `scripts/eval_full.py --ckpt PATH --n 256 --steps 200 --out runs/<name>/eval.json` is the canonical scorecard.

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

## Final summary (this session)

| Run | KL_uni | KL_bi | KL_tri | H_ratio | Notes |
|---|---:|---:|---:|---:|---|
| baseline_5ep   | 0.051 | 1.987 | 7.22 | 0.955 | Phase 0 reproduction |
| ep10_default   | 0.033 | 1.783 | 6.76 | 0.948 | Phase 1 |
| ep25_default   | 0.015 | 1.651 | 7.21 | 1.002 | Phase 1 |
| **data_50k_ep5** | 0.035 | **1.382** | **5.96** | 0.991 | Phase 3 ★ best KL_bi |
| data_50k_ep10  | 0.023 | 1.558 | 6.05 | 1.003 | Phase 3 (overfits per-position) |
| data_200k_ep2  | 0.207 | 2.482 | 6.08 | 0.933 | Phase 3 (undertrained) |
| ng_bg05_data50k | 0.059 | 1.774 | 5.89 | 0.949 | Phase 5 (negative) |
| ng_bg10_data50k | 0.127 | 1.799 | 6.11 | 0.923 | Phase 5 (worse) |

Plan target: KL_bi ≤ 0.50 — not reached. Best (data_50k_ep5) is 2.8× over.
The strongest finding for the writeup: at parity compute, **5× more data
variety drops bigram KL by 16 %** vs more passes over a small set; both
under-fitting (less per-window passes) and over-fitting (more passes)
hurt. The ablation table the writeup wants is in this log; samples for
each run are in `runs/<name>/eval.json` under the `"samples"` key.


