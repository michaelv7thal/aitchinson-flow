# Evaluation assessment — which evals to report, and what they show


> **STALE (banner 2026-08-20). Retained for its protocol reasoning and run triage; its per-objective VERDICTS are superseded by the paper and three of them are now wrong.**
>
> This file was last substantively true on 2026-07-15, before the full-corpus Dirichlet Flow Matching run and before the final-checkpoint benchmark re-run. Against the finished paper:
>
> - **Obj 1 is not a negative result.** The claim below (§Objective 1) that "no model produces English words" and that the failure is "structural, not just scale" was true of the 10k-window budget only. The full-corpus model generates mostly word-like English with local syntax at KL_bi = 0.13 (paper `tab:gen` lower block).
> - **Obj 2 fails only for Equilibrium Matching.** Transport recovery is a headline success for the selected model: Δ@.50 = +0.282 (paper `tab:recovery`). Read every "recovery fails" statement below as scoped to the EqM arms.
> - **Obj 3's detector recommendation is inverted.** The strongest localizer is the training-free denoiser NLL (word-max 0.981 on replace), not a fitted head; the Bayesian-linear energy head is second at 0.930, and SVGP is not in the paper at all.
> - **False information is a triage signal, not a word-level fix.** The 2026-07-12 banner's claim that "sequence-triage-only was an artifact of measuring per-character" does not survive: the paper reads the same ~0.81 as the triage regime (per-word F1 ≤ 0.435, `tab:ood-prf`).
> - **No peer-comparable BPC was ever produced.** The peer-comparability section below describes a deliverable the paper does not cash; `bpd()` is marked "diagnostic only" throughout (paper `tab:config`).
> - **All detector/healing numbers below are epoch_best-era** (`bench_ood/`, `bench_heal/`). The paper reads `bench_ood_final/` and `bench_heal_final/`. Tell them apart by shuffle: 0.963 is old, 0.967 is current.
>
> Current sources: the paper's chapters 4–5, `bench_ood_final/RESULTS.md`, and the benchmark manifests in `bench_ood_final/` and `bench_heal_final/`.

Capstone: continuous flow-matching on the simplex for character-level **text8**
(K=27). This document picks the **meaningful eval per objective**, states **what
works / what fails and WHY** (linked to theory), makes the benchmarking
**comparable to published peer papers**, and flags metrics that are misleading.
It is the eval reference for the capstone report.

> **CORRECTION (2026-06-21).** Every "**conditional recovery works**" claim in this
> document (the Obj-2 TL;DR row, the "local works, global fails" throughline, and
> §Objective 2) is **superseded**: EqM/SFLMEBM energy descent is a **no-op** that
> returns the corrupted input (Δ@α≈0); the **+0.06 Δ@.50** is a marginal mid-α bump,
> **not** recovery, and is not competitive with DirichletFM. Recovery and generation
> are the **same** descent failure. The measured numbers below are retained as the
> historical record; the *verdicts* are corrected here and in
> **`NOTE_EQUILIBRIUM_FAILURE_CLASS.md`** (§0, §B).

> **UPDATE (2026-07-12) — Obj-3 superseded by the detector benchmark.** The
> §Objective-3 material below (hinge-**SVGP** on frozen features) is the historical
> OOD story. The current, reproducible Objective-3 results are the **detector +
> healing benchmark** in **`RESULTS.md` §Objective 3** (`bench_ood/` + `bench_heal/`,
> each with a `manifest.json` + `repro.patch`). Headlines (**final numbers, full re-run
> 2026-07-14**): training-free **NLL** is the best localizer, and wins at the *fair* unit —
> word-level **0.981 replace / 0.963 shuffle** vs GPT-2's 0.738/0.746; **false-info DOES
> localize to the word** (0.813 — the old "sequence-triage-only" was an artifact of
> measuring per-character), but **GPT-2's NLL beats us there** (0.853), an honest loss on
> the axis where world knowledge pays; **plausible** (model-fluent) errors are invisible to
> the model's own unsupervised detectors (NLL *inverts* to 0.264) and need **supervision** —
> a specialist gets 0.904 (~0.79 frequency-controlled), and one **sign-agnostic rank-8
> quadratic** head trained on the union covers *all* corruptions at once (0.727/0.683),
> though it pays ~0.19 vs the specialist; **healing** recovers geometric corruption (+0.420
> net, +0.374 on the real insulin article) but never false-info (net-negative at every
> operating point). SVGP does **not** beat the linear head (saturates; variance collapses
> via concentration-of-measure).
>
> **UPDATE (2026-07-13) — the GPT-2 baseline was wrong twice.** It computed a per-token
> **NLL** and called it *spilled energy* (the real thing, Minut et al. ICLR 2026, is the
> CROSS-step `logsumexp(logits[i]) − logits[i-1][x_i]`), and it attributed BPE scores down
> onto characters by uniform spreading. **Every "GPT-2 spilled-energy" number in this
> file — including 0.847 and the table in §Obj3 — is superseded.** The corrected
> baseline now runs as two arms (`gpt2_se` = real ΔE, `gpt2_nll` = the fair comparator),
> and each model is scored in its native unit (**FM → char · GPT-2 → BPE · both → word**).
> See `CLAUDE.md` §"Spilled energy", `RESULTS.md` §"The external-LM baseline, corrected",
> and `SESSION_OOD_HEALING.md` Finding 5. Two results worth carrying forward: real ΔE
> **cannot localize** (it straddles two decoding steps — seq ~1.0, BPE ~0.48 ≈ chance),
> and **GPT-2_NLL is genuinely strong on false-info** (word 0.853), plausibly beating our
> detectors on that axis.

Model families & sources: **EqM** (Equilibrium Matching, Wang & Du 2025,
arXiv:2510.02300 — conservative gradient of the bilinear energy `E=⟨x,f(x)⟩`),
**DFM / DirichletFM** (Dirichlet Flow Matching, Stark et al. 2024,
arXiv:2402.05841), **SFLM / SFLMEBM** (hyperspherical flow, arXiv:2605.11125;
the EBM variant adds the EqM log-sum-exp energy), **SVGP** (post-hoc Sparse
Variational GP OOD head, hinge-trained on frozen features). Training targets /
samplers: Flow Matching (Lipman et al. 2022), DSM (Vincent 2011), annealed
Langevin (Song & Ermon 2019).

---

## TL;DR — verdicts

| Objective | Verdict | Headline eval | Headline number |
|---|---|---|---|
| 1 Unconditional generation | **Mixed** — works at L=40 for several FM variants, collapses at L=256 except Dirichlet FM | n-gram **KL_bi/KL_tri**, **H_ratio**, samples; **Discrete-FM BPC** for peer comparison | **L=40**: Discrete FM **KL_bi 0.148** (best), SFLM 0.42, **Dirichlet FM 0.45**; **L=256**: only Dirichlet FM survives (**KL_bi 0.31**), Discrete FM/SFLM collapse. NB "DFM 0.45" in older text = **Dirichlet**FMSvgp, not the Discrete-FM arm (0.148). |
| 2 Conditional recovery | **Fails (no-op)** — descent returns the corrupted input; marginal non-competitive bump only at mid-α | recovery sweep **Δ@α = acc − acc(argmax)** | compositional **Δ@.50 = +0.06** (path-center only, ≈0 elsewhere) vs deterministic **+0.00**; not competitive with DirichletFM — see `NOTE_EQUILIBRIUM_FAILURE_CLASS.md` §B |
| 3 OOD detection | **EBM energy fails on order / is stuck; hinge-SVGP works** | corruption-ladder **AUROC on the shuffle axis** vs the `gpt2_baseline` | EBM energy shuffle ≈ **0.50** (chance); hinge-SVGP shuffle **0.93**; GPT-2 ref **0.87→1.0** |

The throughline (**corrected** — see `NOTE_EQUILIBRIUM_FAILURE_CLASS.md` §B):
**evaluation survives, iteration fails.** Both unconditional generation *and*
conditional recovery iterate the conservative gradient to *move* a point, and both
fail — the field is supported only on a thin interior shell, so descent no-ops or
collapses. Only *evaluation* survives: energy-based OOD on *order* still fails
(native energy is stuck), but a hinge head on **frozen features** works (no
descent). The earlier "local works, global fails" reading is superseded — recovery
is not a local success, it is the same descent failure measured at small α.

---

## Peer-comparable benchmarking (text8 protocol)

To sit next to published numbers, three things must hold; status in this repo:

1. **Standard split.** ✓ `afmck/text8` HF splits (90M train / 5M val / **last 5M
   test**); the bench evaluates on `dm.splits.test` (`data/hf_text_loader.py`).
2. **Standard context length.** Published text8 diffusion-LM BPC is reported at
   **L=256**. The repo's main runs are L=40 (short). The `a100_20g_L256` scale
   (**d1024/10L/16H/B8**, L=256) exists and is the W3 cluster deliverable.
3. **BPC = a real bound, not a recovery artifact.** Published BPC is a data
   NLL/ELBO (bits/char to compress the test set). In this repo **only DFM**
   computes a comparable bound — `DFM.elbo_bpc` (`models/dfm.py`, n_mc=8), a
   genuine **D3PM uniform-process variational NLL upper bound** (the per-step KL
   between the closed-form true posterior and the model posterior). Its honest
   peer is **D3PM-uniform ≈1.61** (this model uses a *uniform* source); the
   absorbing/score methods (SEDD 1.32, D3PM-absorb 1.45, MDLM ≤1.38) are a
   reference, not a head-to-head. `DFM.bpd()` returns this bound; the old
   uniform-t training CE (≈3.0, **not** a likelihood) is kept as
   `DFM.denoiser_ce_bpc`. **EqM / EqM_OneHot / EqMLatent / SFLM cannot** give a
   bound: their `bpd()` is an (essentially identity) recovery path → **BPC≈0 /
   PPL≈1.0**, unrelated to data likelihood. **Do not report it.** A `<0.5`-BPC
   sanity guard plus the identity-path set make the bench/eval print `—`
   (`generation_metric_valid=False`) — this also correctly demotes DirichletFM's
   high-t denoiser number (~0.36).

**Published text8 BPC frontier** (lower = better; from `RESEARCH_FINDINGS.md`):

| Model | BPC | Class |
|---|---|---|
| Plaid 1B (Gulrajani & Hashimoto 2023) | 1.12 | continuous diffusion |
| AR Transformer (Al-Rfou 2018), T12 / T64 | 1.18 / 1.13 | autoregressive |
| **SEDD-absorb** (Lou 2024) | **1.32** | discrete diffusion |
| MDLM (Sahoo 2024) | ≤1.38 | masked diffusion |
| **SFM / Fisher-Rao** (Cheng 2024, arXiv:2405.16441) | **1.39** | simplex/sphere FM |
| MultiFlow / BFN / ARDM | 1.41 / 1.41 / 1.43 | discrete |
| **D3PM-absorb** (Austin 2021) | **1.45** | discrete diffusion |
| SEDD-uniform / D3PM-uniform / LinearFM | 1.47 / 1.61 / 1.65 | discrete/continuous |

Realistic discrete-diffusion frontier: **1.32–1.47 BPC**.

> **Honesty caveat (state in the report).** Our runs train on
> `max_train_windows≈10k` (≈2.5M chars at L=256) — ~40× less data than the 90M
> published protocol — and at a ~50M-param "paper-small" tier on a 20 GB MIG. So
> our BPC is **protocol-comparable but data-/scale-subset**: expect it *above*
> the frontier. The current L=40 DFM bench BPC ≈ **3.0** reflects exactly this
> (short context + 10k windows + undertrained). **W3 produces the L=256 number**
> (with `max_train_windows` raised as far as the MIG/time allows); whether it
> approaches the frontier or merely matches the standard length, report it
> honestly against the table above.

---

## Objective 1 — Unconditional generation → **negative result**

**Report:** n-gram **KL_bi / KL_tri** vs corpus (KL_uni is too easy), **H_ratio =
H_gen/H_gt** (≈1.0 ideal), **qualitative samples**, and the peer-comparable **DFM
BPC** (above). Source: per-run `eval.json` via `scripts/eval_generation.py`.

| Run (model) | KL_uni | KL_bi | KL_tri | H_ratio | sample |
|---|---|---|---|---|---|
| **Discrete FM** `dfm_data50k_ep5` (DiscreteFlowMatching, L40) | 0.0073 | **0.148** | — | 0.98 | char-soup (best low-order) |
| **Dirichlet FM** `dfm_svgp_pure50_lr3e4` (DirichletFMSvgp, L40) | 0.0074 | **0.45** | 2.54 | 1.02 | "and ancmnau and is is read caw or yeria" |
| SFLM `sflm_bench_cluster` (L40) | 0.015 | 0.42 | — | 0.96 | char-soup |
| Dirichlet FM `dfm_svgp_L256` (DirichletFMSvgp, **L256**, 20ep) | 0.009 | **0.31** | 1.93 | 1.02 | char-soup (only L256 survivor) |
| best continuous EqM (`latent_cluster_d256_L128`) | — | ≈0.92 | — | 0.96 | char-soup |

> **NB — naming:** the "DFM" arm is **Discrete** FM (uniform path, real
> `elbo_bpc`); the historical "best DFM 0.45" is **Dirichlet** FM
> (`DirichletFMSvgp`). They are different models. At L=40 Discrete FM is the best
> generator (0.148); it collapses at L=256 (length effect). See
> `GENERATION_BENCHMARK_RUNBOOK.md` for the matched 7-model re-run.

**WHY it fails (theory).** Two compounding mechanisms, both documented:
- **Training-time collapse** (`NOTE_WHY_EBM_INIT_STUCK.md`): the FM regression's
  Bayes-optimum at the no-information limit is the **constant unigram-mean field**
  `−c(γ)μ₁` (§2); at lazy init the bilinear `⟨x,f⟩` energy is **quadratic with a
  single basin** (§3, §5), and there is no force to create per-token basins. The
  failure is specific to the **FM / flow-map target class** (conditioned on the
  noiseless interpolant); **DSM** (conditioned on a noisy view) provably escapes
  it (§7.5) — confirmed by `dsm_clr_ablation` (FM-det-CLR KL_uni **0.042** vs
  DSM-det-CLR **0.630**).
- **Sampling-time failure** (`NOTE_WHY_UNCONDITIONAL_FAILS.md`): the field is
  trained only near the data manifold (γ>0.5); unconditional sampling starts in
  untrained noise space, so NAG-GD rolls into a spurious basin. This is the
  **local/global asymmetry** that also explains why recovery (Obj 2) works.

→ Conclusion: no model produces English words under the 20 GB constraint; even the
best matches only low-order n-gram statistics. **Structural, not just scale.**

**Drop:** §0 PPL/BPC for the identity-path arms (artifact; now shown as `—`).

---

## Objective 2 — Conditional recovery → **negative (no-op; superseded)**

> **CORRECTED (2026-06-21).** Recovery does **not** work: EqM/SFLMEBM descent is a
> no-op that returns the corrupted input (Δ@α≈0). The **+0.06 Δ@.50** below is a
> marginal bump confined to the path center (α∈[0.5,0.7]) and is **not competitive**
> with DirichletFM. The table and analysis are retained as the historical record;
> read with `NOTE_EQUILIBRIUM_FAILURE_CLASS.md` §B. The "starts inside a real data
> basin → local problem" explanation below is wrong — the per-token basins are
> sub-resolution spikes and a perturbed init lands on a flat shoulder (∇E≈0).

**Report (headline):** the recovery sweep from `scripts/recovery_check.py` mode
(B): perturb an encoded held-out window by α·‖x₁‖, run the sampler, measure
token accuracy and especially **Δ@α = token_acc − token_acc_perturbed** (the work
the sampler does *beyond* argmax of the noisy input). Headline = **Δ@.50** + the
per-α curve. Complement: `eval_healing.py` (recovery_at_corrupt / _uncorrupt).

| Condition | Δ@.50 | acc@.10 | acc@.30 | acc@.50 |
|---|---|---|---|---|
| Compositional MSE | **+0.060 ±0.000** | 1.00 | 0.90 | 0.58 |
| Compositional Hilbert | +0.054 ±0.003 | 1.00 | 0.91 | 0.58 |
| Deterministic-CLR (ref) | **+0.000** | 1.00 | 0.89 | 0.52 |

**WHY recovery works where generation fails:** recovery starts *inside* a real
data basin (γ≈1 region is well-trained), so it is a **local** problem
(`NOTE_WHY_UNCONDITIONAL_FAILS.md` §"why recovery works"). **WHY the deterministic
field is stuck (Δ=0):** the single-basin §3 energy + a deterministic source give a
no-op sampler; **Dirichlet "thickening" of x₁ + source stochasticity** breaks the
symmetry and yields real recovery (`PROPOSAL_COMPOSITIONAL_EQM.md`). MSE ≈ Hilbert
(within seed noise).

**Don't lead with KL here:** the deterministic refs look *better* on KL_uni (they
overfit the unigram marginal) while doing **zero** recovery work — KL is
misleading for this recipe; **Δ is the right metric**
(`compositional_eqm_test_summary.md`).

---

## Objective 3 — OOD detection → **EBM energy fails on order / is stuck; hinge-SVGP works**

**Method.** Anchor every per-arm OOD readout to the **`gpt2_baseline`** row (the
external GPT-2 spilled-energy reference the bench computes once and that per-arm
scores "should match or beat"). **Split the corruption ladder:**
- **substitution** changes the unigram histogram → detectable by marginals; AUROC
  saturates to ~1.0 for almost anything. **Easy; do not headline.**
- **shuffle / histogram-preserving** keeps unigram stats fixed → requires modeling
  **sequence order**. **This is the discriminating axis — headline it.**
- **false-info / word-swap** (`falseinfo` scheme, `data/corruption.py`) replaces whole
  words with a *different real, same-length* word — lexically valid, semantically wrong.
  Preserves local char n-grams, so the **denoiser-NLL localizer barely moves** (heal
  Track C: recall 0.20, net/corrupt −0.142; `RESULTS_HEAL_POC_INSULIN.md`). This is the
  **contextual/semantic axis** — the hard case NLL cannot reach. A **one-class Gaussian
  mixture over frozen DirichletFM contextual features** (`scripts/ood_gmm_perpos.py`,
  detector `PerPosGMM`; healing localizer `make_gmm_localizer`) is the candidate detector:
  it scores tokens by negative log-likelihood under the ID feature mixture, so an
  off-manifold wrong-word-in-context can register even where char-NLL stays low. The
  `bayeslin` / `nll` / `gmm` OOD sweeps now all carry the `falseinfo` scheme for a
  head-to-head AUROC comparison; the GMM detector sweeps `--t-evals` (lower t = more
  context-dependent, less token-dominated features) since at the default `t_eval=4.5`
  the deterministic-mean feature is token-dominated and a valid swap may land *on* the ID
  manifold. **A GMM that also misses `falseinfo` across all t is itself a clean result**
  ("feature-density does not fact-check either"). Numbers pending the L256 cluster run.

Grounded (current `sflm_bench_a100_20g_gridsweep_v2/bench.json`, GPT-2 ref, L=40,
+ `dfm_svgp_pure50_lr3e4/svgp_corruption_sweep.json`):

> ⚠️ The "GPT-2 spilled-energy" row below is **a per-token NLL, not spilled energy**, and
> its per-char numbers used uniform BPE→char spreading. Superseded — see the 2026-07-13
> update at the top of this file. (Its *sequence*-level numbers happen to be unaffected by
> the attribution bug, but the row is still mislabelled.)

| Detector | subst (r=.1→1.0) | **shuffle (r=.1→1.0)** | rand |
|---|---|---|---|
| **GPT-2 spilled-energy (external baseline)** *(mislabelled: is NLL)* | 0.97 → 1.0 | **0.87 → 1.0** | 1.0 |
| SFLMEBM native energy | 0.71 → 1.0 | **0.49–0.50 (chance)** | 1.0 |
| EqM native energy | 0.54 → 0.80 | ≈0.50 (chance) | 0.79 |
| EqMLatent native energy | 0.54 → 0.94 | ≈0.50 (chance) | 0.93 |
| SFLMEBM_FM native energy | **0.38 → 0.0015 (sign-inverted)** | ≈0.50 | 0.004 |
| **Hinge-SVGP on frozen DFM** | 0.70 → **0.997** | **0.62 → 0.93** | — |

Per-position localization (`pospair`): GPT-2 SE subst 0.86→0.54 / shuffle
0.86→0.53; EqM ‖∇E‖ subst **0.38 (below chance)**; EqMLatent ‖∇E‖ ≈0.75 (best
EBM, subst only). SVGP variance is **uninformative** (AUROC(std)=0.50,
concentration of measure — `DFM_SVGP_FINDINGS.md`); the signal is entirely in the
trained mean/prob, so don't claim "uncertainty quantification".

**WHY (theory).** The native conservative-gradient **energy captures per-token
marginals, not order** — exactly the §3 single-basin / unigram-attractor energy,
which is permutation-blind, so shuffle is at chance. Two pathologies confirm the
*stuck/trivial-solution* reading: SFLMEBM_FM's energy is **sign-inverted** (learned
the wrong convention), and EqM's ‖∇E‖ is **below chance**. The discriminative
**hinge** (train `E_invalid > E_clean` against synthetic negatives) supplies the
symmetry-break the energy lacks → SVGP on frozen DFM features recovers the order
axis (shuffle 0.93) that every EBM energy misses (`SFLM_EBM_FINDINGS.md`: the EBM
is a viable **verifier/auditor**, not a generator).

→ Conclusion: **switch the OOD readout from native energy to hinge/SVGP.**

**Does the FM stage matter, or would a direct hinge be comparable?** The FM
stage's job is the *representation* + the generator/recovery landscape — not OOD
accuracy per se. Repo evidence cuts both ways: Phase F showed a linear probe on a
good representation matched a trained head (the *representation* carries the
signal), and the SVGP — trained on replace-only negatives — still detects
**shuffle (0.93)**, a generalization it can only inherit from the FM
representation (a corruption-specific discriminative hinge would likely overfit
the replace cue). So FM-first is justified by **cross-corruption generalization +
unification**, not by a higher number on the trained corruption. The decisive
test is the 4-arm ablation `scripts/ablate_hinge_vs_fm.py` (FM+hinge vs
random+hinge vs end-to-end-hinge vs FM+linear-probe), run at L256 per
`CLUSTER_RUNBOOK_L256.md` §3b; prediction: A≈D≫B on shuffle, C fails the
replace→shuffle transfer.

**Caveats / what to drop:**
- The bench reports **no per-arm SE** (removed to kill an identity-path artifact),
  so the **DFM-denoiser-NLL OOD signal is not in the bench**; the DFM/DirichletFM/
  SFLM generator rows are empty. Use the GPT-2 baseline as the external proxy + the
  hinge-SVGP number, or run `scripts/eval_ood.py` for the DFM denoiser-NLL.
- **SVGP rows (1g/1h) are empty in the committed bench** (arms were skipped — no
  L256 checkpoint). The bench now (a) falls back to the existing
  `dfm_svgp_pure50_lr3e4` SVGP checkpoint at matching L, and (b) records
  `_meta.skipped_arms`; the hinge numbers above come from the standalone
  `svgp_corruption_sweep.json`. W3 fits a fresh DFM_SVGP at L256.
- The GPT-2/WikiText **"auditor" line (Phase F/H) is dead** — the trained EqM
  auditor adds nothing beyond a linear probe / spilled energy and is
  cascade-contaminated (`DECISION_LOG.md`). Do **not** present it as a positive.

---

## Theory → evidence (the WHY, one table)

| Claim | Source § | Mechanism | Evidence |
|---|---|---|---|
| Unigram-mean Bayes collapse | EBM§2 | FM optimum at no-info γ is constant `−c(γ)μ₁` | KL_uni collapse signature |
| Conservative-grad single basin | EBM§3,§5 | bilinear `⟨x,f⟩` ⇒ quadratic energy, one critical point | EqM energy at chance on shuffle; ‖∇E‖ below chance |
| FM/flow-map vs DSM target | EBM§7.5 | FM targets noiseless interpolant (degenerate); DSM targets noisy view (proper) | `dsm_clr_ablation`: FM-det KL_uni **0.042** vs DSM-det **0.630** |
| Local vs global | UNC§1–2 | field trained near manifold only; uncond starts in untrained noise | recovery ≈100%@α.2 vs KL_bi≫1 uncond |
| NAG vs Langevin | EBM§9 | deterministic → biggest basin; Langevin ∝e^{−E/T} explores by depth | SDE ≫ NAG (AE report, up to 47×) |

EBM = `NOTE_WHY_EBM_INIT_STUCK.md`; UNC = `NOTE_WHY_UNCONDITIONAL_FAILS.md`.

---

## Canonical eval per objective (what to run/report going forward)

| Objective | Canonical eval | Headline metric |
|---|---|---|
| 1 Unconditional | `scripts/eval_generation.py` (`eval.json`) + DFM `bpd()` MC-ELBO @L256 | KL_bi/KL_tri, H_ratio, **BPC vs published** |
| 2 Recovery | `scripts/recovery_check.py` (+ `eval_healing.py`) | **Δ@.50**, per-α curve |
| 3 OOD | `scripts/bench_sflm_ebm.py` (energy vs `gpt2_baseline`) + `svgp_corruption_sweep.json` | **shuffle-axis AUROC**: energy (fails) vs SVGP (works) vs GPT-2 |

Everything else in `scripts/` (phase-specific runners, plotters, the GPT-2 auditor
line) is legacy — see the cleanup (`scripts/legacy/`).

---

## Run triage (which runs hold the evidence)

| Tier | Runs |
|---|---|
| **Obj1** | `dfm_svgp_pure50_lr3e4`, `dfm_svgp_pure50`*, `latent_cluster_d256_L128_data20k_ep10`, `eqm_data50k_ep5_v2` |
| **Obj2** | `comp_{mse,hilbert}_seed{42,43,44}`, `comp_ref_det_{mse,hilbert}` |
| **Obj3** | `sflm_bench_a100_20g_gridsweep_v2`*, `dfm_svgp_pure50_lr3e4` (svgp sweep), `dfm_svgp_{scaled,poc}`, `dsm_clr_ablation` |
| **Failed/incomplete** | `sflm_bench_a100_20g_L256` (crash, no JSON; kept as 20 GB-limit evidence until W3 re-runs) |
| **Archive (legacy)** | GPT-2 auditor (`aud_gpt2_*`, `dfm_auditor_*`, `hilbert_uq_*`, `hal_gpt2_*`); early phases (`dphase*`, `hilbert_*`, `dsm_vs_eqm_poc`); ablations (`data_*`, `tc_*`, `bb_*`, `latent_d*`); smoke/poc (`*_smoke`, `*_poc`, `capstone/`); alt-continuous (`fmclr_*`, `lkflow_*`); `ng_*`, `baseline_*`, `ep*_default` |

(* = user's focus runs.)

---

## Reproduce

```bash
# Obj1 generation (KL/H_ratio/samples)
python scripts/eval_generation.py --ckpt runs/dfm_svgp_pure50_lr3e4/epoch_final.pt
# Obj2 recovery sweep (Δ@α)
python scripts/recovery_check.py --ckpt <eqm_ckpt> --alphas 0.1,0.3,0.5,1.0 --n 256
# Obj3 OOD bench (energy vs gpt2_baseline) — L=40 here, L=256 on the cluster (see CLUSTER_RUNBOOK_L256.md)
python scripts/bench_sflm_ebm.py --scale a100_20g --n 256 --ref-lm gpt2
```

The L=256, peer-length benchmark + a fresh DFM_SVGP hinge fit run on the 20 GB
cluster — see **`CLUSTER_RUNBOOK_L256.md`**. Until then the L=256 BPC cell in the
peer-comparability table is pending.
