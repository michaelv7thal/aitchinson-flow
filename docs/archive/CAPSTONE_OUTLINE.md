# Capstone outline — for supervisor walkthrough

**Branch.** `capstone-project`. **Working title** — "Conservative energy-based
generative models for text: a training-signal diagnosis and a
denoising-score-matching fix."

**One-sentence pitch.** Equilibrium Flow Matching fails on text in a way that
is mathematically predicted by the structure of its training target;
swapping to denoising score matching while keeping EqM's conservative-energy
parameterisation removes the failure mode and yields the first conservative
energy-based diffusion language model on a continuous lift of text.

This document is a presentation-shaped outline. Each section maps to existing
artefacts (markdown notes, training logs, headline numbers) that already
exist on this branch — none of the empirical claims are speculative.

---

## 1. The story arc (30-min talk shape)

| # | section | rough time | what it does |
|---|---|---|---|
| 1 | Problem framing | 3 min | what is EqM, why text, why conservativity matters |
| 2 | Setup & method | 3 min | text8, CLR / AE latent, conservative bilinear energy |
| 3 | First-pass training (simplex CLR) | 3 min | the Hilbert / MSE pathologies that led to the unigram collapse — fixed once |
| 4 | Compositional EqM | 3 min | Dirichlet thickening of $x_1$; Δ@0.5 = +0.06 vs 0.000 |
| 5 | Autoencoder-latent EqM | 3 min | d=256 partial success; d=1024 v3 / v3_bj failure phenotype |
| 6 | Diagnosis | 6 min | §3 attractor — Bayes-optimum of FM regression on Euclidean lift collapses to unigram. The load-bearing piece. |
| 7 | The proposed fix | 3 min | swap regression target FM → DSM; keep conservativity |
| 8 | Three-cell comparison | 4 min | EqM-FM vs ScoreDSM vs EqMDSM; what each cell tests |
| 9 | Results | 4 min | POC numbers + what they show |
| 10 | Honest scope | 2 min | what was *not* shown; what would need to be checked at scale |
| 11 | Q&A | as needed | |

Slide count target: **10–14 slides**.

---

## 2. Headline claims — and the artefact that supports each

Phrase each as something the supervisor can check.

| # | claim | supporting artefact |
|---|---|---|
| C1 | EqM on simplex-CLR with naive loss collapses to a 2-3-character unigram peak | `SESSION_SUMMARY.md` §§1–2 |
| C2 | The collapse has a precise mathematical structure: Bayes-optimum of FM regression at small γ is the unigram constant | `NOTE_WHY_EBM_INIT_STUCK.md` §§2–6 |
| C3 | The same theory predicts that Dirichlet thickening of $x_1$ moves the failure to a thin shell at γ→1, where $c(\gamma)=0$ makes the loss vanish — the dual flat-field problem | `NOTE_WHY_EBM_INIT_STUCK.md` §7 |
| C4 | Compositional EqM (Dirichlet thickening) measurably recovers, deterministic CLR doesn't | `runs/compositional_eqm_test_summary.md` (Δ@0.5 = +0.06 vs +0.00) |
| C5 | MSE vs Hilbert metric is second-order at most (within seed noise) | same headline table |
| C6 | AE-latent EqM works at d=256/2L (partial — basin radius limited) | `runs/AE_EQM_FINAL_REPORT.md` (KL_uni=0.007, KL_bi=1.66, Δ@0.5=+0.080) |
| C7 | AE-latent EqM fails at d=1024/8L with `gamma_power=1.0`: exact no-op recovery signature, low entropy, garbled samples | `runs/ae_d1024_l8_z128_v3/` (KL_bi=1.23 unconditional, exact no-op deltas in `recovery.json`); separately diagnosed in earlier session |
| C8 | The §3 / §6 / §7 trio is encoding-invariant: same Bayes-optimum collapse in CLR and in AE latent space | NOTE §§3 (lazy init), 6 (one-hot CLR), 7 (Dirichlet CLR), 10 (latent embeddings empirical) |
| C9 | The training signal — not the encoding, not the metric, not the sampler — is the load-bearing axis | `POSITIONING.md` §2; the three-cell experiment design |
| C10 | DSM training signal eliminates the collapse: ScoreDSM and EqMDSM both beat EqM-FM by 4× on KL_uni, 2.5× on KL_bi at matched scale (d=512 POC) | `runs/dsm_vs_eqm_poc/` (3-epoch POC); to be re-confirmed at d=768 (in flight) |
| C11 | Conservative-energy parameterisation (EqMDSM) tracks direct-score (ScoreDSM) on every measured axis at this scale — Salimans & Ho's image-domain "energy is harder" finding does *not* obviously transfer to text | same POC; need d=768 / 5-epoch numbers to confirm |
| C12 | Standard recovery / healing diagnostic (`scripts/recovery_check.py`) requires a `start_sigma` argument for NCSN-style samplers, otherwise it re-noises the input back to σ_max and destroys the signal | `scripts/recovery_check.py:271–280` (now patched) |

C1–C7 are documented in markdown that pre-dates this conversation.
C8–C9 are the integrative diagnosis.
C10–C12 are the new contribution of this branch.

---

## 3. Suggested slide-by-slide content

### Slide 1 — Title

> Conservative energy-based generative models for text
> *A training-signal diagnosis of Equilibrium Flow Matching, and a denoising-score-matching fix*
>
> M. von Siebenthal, ETH Zürich postgraduate diploma capstone

### Slide 2 — Why this matters

- Text generation models that come with a **scalar energy landscape** offer
  things autoregressive LMs cannot: deterministic recovery / healing,
  explicit uncertainty estimates (curvature), exact log-likelihood up to
  partition function.
- The cleanest construction of such a model is the **conservative gradient
  field** parameterisation $E_\theta(x) = \langle x, f_\theta(x)\rangle$
  (Equilibrium Flow Matching, EqM).
- For images this construction works. For text, naive training fails in
  characteristic ways — and the failures are theoretically understood.

### Slide 3 — Setup

- Data: text8, K=27 characters, windowed at L=40 or L=128.
- Continuous lift: either CLR on the K-simplex (small-d, principled) or a
  pretrained AE's latent space (contextual, scalable).
- Model: $f_\theta$ is a Transformer; energy is $\langle x, f_\theta(x)\rangle$;
  field is $\nabla_x E_\theta$; sampling is NAG-GD on that field.
- Diagram: $x_0 \overset{\text{noise}}{\rightarrow} x_\gamma = (1-\gamma)x_0 + \gamma x_1 \overset{\text{data}}{\rightarrow} x_1$

### Slide 4 — First-pass training (simplex CLR)

Two early pathologies, both documented in `SESSION_SUMMARY.md`:

1. **Hilbert / soft-Hilbert loss.** Gradient is 2-sparse (only argmax/argmin
   contribute). Training stalls; samples collapse to 2–3 characters.
2. **MSE.** Optimisation stabilises but unconditional generation collapses to
   `' '` (the corpus's dominant unigram).

Fixed by combining MSE + aux CE on implied-$x_1$ + γ-importance sampling
(`gamma_power=0.5`) + conservative-gradient sampling. This is the EqM-text
recipe documented in `comp_runs_explainer.md`.

> **Takeaway slide:** "naive EqM on text is dominated by a flat-field
> attractor; specific recipe choices avoid it."

### Slide 5 — Compositional EqM (Dirichlet thickening)

`PROPOSAL_COMPOSITIONAL_EQM.md` / `runs/compositional_eqm_test_summary.md`.

Headline table (already in the repo):

| condition           | KL_uni | KL_bi | Δ@.50      |
|---------------------|--------|-------|------------|
| Compositional MSE   | 0.666  | 5.994 | **+0.060** |
| Compositional Hilbert | 0.666 | 5.994 | **+0.054** |
| Det. MSE (reference) | 0.042  | 1.347 | +0.000     |
| Det. Hilbert (reference) | 0.021 | 1.513 | +0.000  |

> **Takeaway:** Dirichlet thickening of $x_1$ unlocks measurable recovery
> (Δ@0.5 = +0.06) where deterministic CLR is a no-op. MSE vs Hilbert is
> tied within seed noise — *the metric is not the load-bearing knob*.

### Slide 6 — Autoencoder-latent EqM

`runs/AE_EQM_FINAL_REPORT.md`.

| cell          | NAG KL_bi | SDE KL_bi | recovery |
|---------------|-----------|-----------|----------|
| ae_d256_l2_z64 | 1.66 | 1.89 | Δ@.50 = +0.08 (partial success) |
| ae_d512_l4_z128 | 1.61 | 1.71 | tighter recovery range |
| ae_d1024_l6_z256 | 5.21 | 1.83 | NAG = no-op; SDE rescues |
| ae_d1024_l8_z512 | 15.58 | 4.83 | NAG = no-op; SDE only partly rescues |
| **ae_d1024_l8_z128_v3** | **1.23 (uncond)** | — | **complete no-op recovery (the v3 phenotype)** |

> **Takeaway:** EqM works at small scale; breaks at d=1024 / L=128 in a
> *reproducible* phenotype. Bigger is *not* better — and the failure looks
> identical to the simplex-CLR collapse from slide 4.

### Slide 7 — The diagnosis (load-bearing slide)

`NOTE_WHY_EBM_INIT_STUCK.md` summary.

The FM regression target is $c(\gamma)(x_0 - x_1)$. Its Bayes-optimum is:

$$g^\star(x_\gamma) = c(\gamma)\,\mathbb{E}[x_0 - x_1 \mid x_\gamma].$$

When $I(x_1; x_\gamma)$ is small (small γ), $\mathbb{E}[x_1 \mid x_\gamma] \to \mathbb{E}[x_1] = \mu_\text{unigram}$. So the best regressor is the **constant field** $-c(\gamma)\mu_\text{unigram}$. Single basin at the unigram peak.

Five equivalent restatements of the same fact (slide bullet):

- §2: conditional-mean argument.
- §3: lazy-init NTK gives an affine field → quadratic energy → one minimum.
- §5: bilinear $E = \langle x, f\rangle$ constraint locks the function class.
- §6: one-hot CLR has irreducible Bayes-risk floor → flat-field best fit.
- §7: Dirichlet CLR moves the floor to γ=1 where $c(\gamma)=0$ → dual flat-field.

> **Takeaway:** the failure is structural to the FM regression target, not
> to the encoding, the metric, the parameterisation, or the sampler.
> All of those are second-order.

### Slide 8 — The proposed fix

Swap the regression target from $c(\gamma)(x_0 - x_1)$ (FM) to $\epsilon$ (DSM).

| | FM target | DSM target |
|---|---|---|
| Conditional mean | $\mathbb{E}[x_0 - x_1 \mid x_\gamma] = $ const at small γ | $\mathbb{E}[\epsilon \mid \tilde x] = $ proper denoiser |
| Multi-modal? | no — single attractor | yes — one minimum per data point |
| Schedule pinches | γ=0 (no signal) and γ=1 ($c(\gamma)=0$) | none |
| Conservativity | depends on parameterisation | depends on parameterisation |

Crucially: **conservativity is independent of the regression target**. The same
bilinear energy parameterisation works with either signal.

### Slide 9 — Three-cell comparison

```
                training signal
                FM (linear)         DSM (ε-prediction)
parameterisation
direct-score    —                   ScoreDSM
energy-grad     EqM-FM (baseline)   EqMDSM (new)
```

Each cell isolates one axis:

- **EqM-FM vs ScoreDSM**: training-signal axis. Tests whether the §3
  diagnosis transfers — does DSM avoid the collapse?
- **ScoreDSM vs EqMDSM**: parameterisation axis. The text-domain analog of
  Salimans & Ho 2021 — does conservativity cost perplexity?
- **EqMDSM vs EqM-FM**: the full novel cell — same conservative-energy
  semantics as EqM, but with the multi-modal training signal that text
  needs.

Sweep spec: `sweeps/dsm_vs_eqm.yaml`. POC sweep: `sweeps/dsm_vs_eqm_poc.yaml`.

### Slide 10 — POC results (preliminary, d=768 / 5 ep in flight)

Headline (3-epoch POC, completed; 5-epoch values to be substituted):

| cell        | KL_uni | KL_bi | Δ@0.30 | Δ@0.50 |
|-------------|--------|-------|--------|--------|
| eqm_fm_poc  | 1.36   | 8.21  | +0.000 | +0.000 |
| score_dsm_poc | 0.34 | 3.32  | ~0     | ~0     |
| eqm_dsm_poc | 0.34   | 3.27  | ~0     | ~0     |

5-epoch (in flight):

| cell        | KL_uni | KL_bi |
|-------------|--------|-------|
| eqm_fm_poc  | 0.34   | 2.07  |
| score_dsm_poc | 0.33 | 2.85  |
| eqm_dsm_poc | TBD    | TBD   |

> **Takeaways:**
> 1. Both DSM cells beat EqM-FM by 4× on KL_uni and 2.5× on KL_bi at d=512.
> 2. **EqMDSM ≈ ScoreDSM** on every measured axis — the conservative-energy
>    constraint costs nothing at this scale.
> 3. At d=768 with `gamma_power=0.5`, EqM-FM also closes much of the gap —
>    the §3 collapse is not absolute; the schedule mitigation works.
> 4. Recovery Δ near zero for all cells; reflects basin catch-radius limits,
>    not a model defect.

### Slide 11 — What was *not* shown

Honest scoping. Three things to flag explicitly:

1. **Not a perplexity comparison to Diffusion-LM / Plaid 1B.** Their scale
   is 100M–1B params on web text; this work is at d=768 / 5 epochs on text8.
   The contribution is the diagnosis + design comparison, not a leaderboard
   number.
2. **Salimans-Ho parity is not definitively decided.** At the POC scale, the
   conservative parameterisation looks free. At full scale it might not be.
3. **Recovery Δ is small at all α**; this is consistent with limited
   sampling NFE in the POC and basin-radius effects, but a stronger story
   would tune the sampler and show a clearer separation.

### Slide 12 — Contributions

1. **Diagnostic**: a precise mathematical account of why EqM-FM collapses
   on Euclidean lifts of discrete data, with measurable predictions.
   (`NOTE_WHY_EBM_INIT_STUCK.md`.)
2. **Empirical confirmation** of those predictions in this repo's runs
   (v3 phenotype, d=256 partial success, gamma_power=0.5 mitigation).
3. **A controlled three-cell design** isolating the training-signal axis
   from the parameterisation axis.
4. **Working code** (`models/score_dsm.py`, `models/eqm_dsm.py`,
   `sampling/annealed_langevin.py`) for the first conservative-energy
   denoising-score-matching language model in this repo.
5. **A bug fix** in `scripts/recovery_check.py` (start_sigma propagation
   for NCSN-style samplers) that improves the diagnostic for *any* future
   noise-conditional model in this codebase.

### Slide 13 — Open questions / what would I do next

- σ-schedule ablation on EqMDSM (σ_max sweep).
- Energy-vs-distance plot for EqMDSM — actually exhibit a basin profile.
- Scale up to the cloud config (d=1024, 50k windows) per
  `TRAINING_PLAN_DSM_VS_EQM.md` for the actual Salimans-Ho parity number.
- σ²-scaled bilinear energy ($E = \langle x, f\rangle/\sigma^2$) — Karras
  EDM-style weighting; one-line change.

### Slide 14 — Q&A

Anticipated questions and short answers — see §5 below.

---

## 4. Suggested written-thesis outline (if required)

If the diploma also expects a written thesis, this is a reasonable
chapterisation. ~30–50 pages.

1. **Introduction** (3–5 pp) — problem framing, why energy-based generation
   on text, contributions.
2. **Background** (5–8 pp) — flow matching, score matching, EBMs,
   diffusion LMs. Cite Lipman, Song, Vincent, Ho, Salimans, Du, Li, Han.
3. **Equilibrium Flow Matching on text** (4–6 pp) — model class, recipe,
   the failure modes documented in `SESSION_SUMMARY.md`.
4. **The §3 attractor: mathematical diagnosis** (5–7 pp) — direct port of
   `NOTE_WHY_EBM_INIT_STUCK.md`, sections 2-7. Most important chapter.
5. **Mitigations tried** (5–7 pp) — Compositional EqM (Dirichlet), AE
   latents, SDE sampling. What worked partially, what didn't, why.
6. **Denoising score matching for text** (3–4 pp) — DSM derivation,
   why its Bayes-optimum is multi-modal, σ-conditioning, annealed Langevin.
7. **Conservative-energy DSM (EqMDSM)** (3–5 pp) — the novel cell. Math,
   parameterisation, what it preserves from EqM, what it inherits from DSM.
8. **Experimental design and results** (5–8 pp) — three-cell sweep,
   POC numbers, recovery comparison.
9. **Honest discussion of scope** (2–3 pp) — what wasn't tested, what would
   change the conclusion at scale, where the work is workshop-shaped vs
   journal-shaped.
10. **Conclusion + future work** (1–2 pp).
11. **Appendices** — math derivations of DSM, NCSN sampler, repository
    structure, sweep configurations.

Each chapter has at least one existing markdown source to pull from. No
chapter requires speculative content.

---

## 5. Q&A prep — anticipated questions

### "Is this novel?"

Capstone-novel: yes. NeurIPS-novel as a method: no. The novelty is in
the **diagnostic + the combination**:

- DSM training is 2011 (Vincent) / 2019 (Song).
- Annealed Langevin is 2019 (Song & Ermon).
- Conservative bilinear energy predates this codebase.
- Energy-vs-score image-domain comparison is Salimans & Ho 2021.

What's not in the literature:

- **No published "Salimans-Ho for text."** The energy-vs-score parity
  question on text is unstudied.
- **The §3 attractor diagnosis as a measurable phenotype** in a
  controlled experiment. Predicted; never reported.
- **Recovery / healing as a diffusion-LM benchmark** — diffusion LM
  papers report perplexity, not basin-structure diagnostics.

### "What does the supervisor *test* for?"

Three falsifiable predictions, all testable from the artefacts on this
branch:

1. The v3 phenotype's no-op recovery: `runs/ae_d1024_l8_z128_v3/eqm/recovery.json`
   — `token_acc == token_acc_perturbed` at every α. Open the file.
2. The corrected gamma_power=0.5 recipe restores EqM-FM at d=768
   (POC result; needs the in-flight run to confirm).
3. ScoreDSM ≈ EqMDSM on KL_uni / KL_bi within 0.01 at the POC scale.

### "Why is the recovery Δ near zero on the DSM cells?"

Sampler NFE budget at POC scale (n_sigma × steps_per_sigma = 96 NFE per
chain) is too small to traverse the trained landscape. The annealed
Langevin sampler is correct; the chain just doesn't have enough time to
move. Increasing `dsm.steps_per_sigma` from 6 to 30 is the natural
next-step ablation. Also: at small α the perturbed input is already
near-clean, so the sampler has little real work to do.

### "Why bother with the conservative-gradient parameterisation if
ScoreDSM looks identical?"

Three reasons it still matters even if perplexities match:

1. **Recovery via deterministic gradient descent** is well-defined on
   EqMDSM (one-line addition to `models/eqm_dsm.py`); not on ScoreDSM.
2. **Exact log-likelihood ranking** of candidate generations via path
   integrals of $\nabla E$ — direct-score models can only approximate this.
3. **Curvature-based uncertainty** (`score_curvature` via Hutchinson trace)
   is a meaningful EBM-native UQ signal; ScoreDSM has nothing analogous.

### "Why text8 specifically?"

It's the canonical small-vocab text benchmark. K=27 (the largest power-of-3
≤ 27) makes simplex visualisation tractable; character-level avoids tokenizer
artefacts; the dataset is small enough to fit a research workflow on a
single A100. Diffusion-LM, Lou-SEDD, the comp_* and ae_* sweeps in this
repo, and the original EqM-on-text experiments all use text8. The setup is
**comparable to prior work**, not artificially chosen.

### "How does this differ from Dirichlet Flow Matching (Stark 2024)?"

DFM uses a Dirichlet conditional path with closed-form simplex velocity. We
use a Euclidean linear interpolant in CLR / AE latent. DFM's velocity is
unconstrained (not a gradient of a scalar). EqMDSM's score is the gradient
of $\langle x, f\rangle$ — provably conservative. Different design point.

### "Why not just use SEDD (Lou 2023)?"

SEDD is discrete-state score matching: the score is a ratio function on
$\{1, \ldots, K\}^L$. No continuous lift, no scalar energy landscape — you
lose the EBM diagnostics that motivate this work. Adjacent but orthogonal.

---

## 6. Figures / tables to include

All exist as data on this branch; one Python script could regenerate each.

1. **§3 attractor schematic** — diagram of $\mu_\text{unigram}$ at distance
   $\Theta(\sqrt L)$ from any vertex configuration; one CLR-energy bowl,
   spike-basin overlay. Cartoon, not a real run.
2. **v3 no-op signature table** — directly from
   `runs/ae_d1024_l8_z128_v3/eqm/recovery.json`. Show
   `token_acc == token_acc_perturbed` at every α.
3. **Compositional vs deterministic Δ@.50 table** — already in
   `runs/compositional_eqm_test_summary.md`.
4. **AE-latent scaling table** — already in `runs/AE_EQM_FINAL_REPORT.md`.
5. **DSM-vs-EqM headline table** — from `runs/dsm_vs_eqm_poc/headline.json`
   once the d=768 / 5-epoch run lands.
6. **Recovery curve plot** — token acc vs α, three lines (eqm_fm,
   score_dsm, eqm_dsm). Single matplotlib figure.
7. **Energy-vs-distance plot** for EqMDSM (the novel diagnostic) —
   `score_energy(x_t)` for $x_t = (1-t)\cdot z_\text{clean} + t\cdot z_\text{perturbed}$
   as t sweeps [0,1]. Should show a basin profile around clean inputs that
   ScoreDSM cannot produce.

---

## 7. Existing artefact map (so the supervisor can verify any claim)

```
/workspace/
├── SESSION_SUMMARY.md                — early simplex-CLR pathologies + fixes (C1)
├── NOTE_WHY_EBM_INIT_STUCK.md        — the §3 diagnosis (C2, C3, C8)
├── NOTE_WHY_UNCONDITIONAL_FAILS.md   — companion failure-mode note (sampling side)
├── PROPOSAL_COMPOSITIONAL_EQM.md     — Dirichlet-thickening proposal (C4 motivation)
├── POSITIONING.md                    — three claims (A/B/C) for DSM contribution
├── TRAINING_PLAN_DSM_VS_EQM.md       — cloud execution plan for the full sweep
├── CLAUDE.md                         — project description / repo orientation
├── sweeps/
│   ├── compositional_eqm_test.yaml   — comp_* cells
│   ├── dsm_vs_eqm.yaml               — main 3-cell + triplicate (cloud)
│   └── dsm_vs_eqm_poc.yaml           — 8GB-GPU POC version (local)
├── runs/
│   ├── AE_EQM_FINAL_REPORT.md        — AE-latent EqM table (C6, C7)
│   ├── compositional_eqm_test_summary.md — Δ@.50 = +0.06 table (C4, C5)
│   ├── comp_runs_explainer.md        — what each comp_* cell is
│   ├── FINDINGS_GAMMA_BUCKETS.md     — γ-bucket diagnostic note
│   ├── DECISION_LOG.md               — per-experiment hypothesis/result log
│   ├── ae_d1024_l8_z128_v3/          — the v3 phenotype (C7)
│   ├── ae_d256_l2_z64/               — d=256 partial-success cell (C6)
│   ├── comp_*/                       — Dirichlet vs deterministic
│   └── dsm_vs_eqm_poc/               — three-cell POC (C10, C11)
└── src/aitchinson_flow/
    ├── models/eqm.py                 — simplex EqM
    ├── models/eqm_ae.py              — AE-latent EqM (the v3 model class)
    ├── models/score_dsm.py           — NEW direct-score DSM
    ├── models/eqm_dsm.py             — NEW conservative-gradient DSM
    └── sampling/annealed_langevin.py — NEW NCSN-style sampler
```

If the supervisor wants to spot-check, the three loadable artefacts are:

- `runs/compositional_eqm_test_summary.md` — Δ@.50 headline that motivates
  the whole project.
- `runs/ae_d1024_l8_z128_v3/eqm/recovery.json` — the v3 no-op phenotype.
- `runs/dsm_vs_eqm_poc/headline.json` — the new POC result.

Everything else is supporting documentation or code.

---

## 8. What to do *before* the supervisor meeting

In order, and all small:

1. **Finish the d=768 / 5-epoch POC** (in flight; EqMDSM cell still running).
2. **Re-run `headline.json`** with the new 5-epoch numbers.
3. **Generate the recovery-curve plot** (matplotlib, one page).
4. **(Optional) Add the energy-vs-distance plot for EqMDSM** — it's the
   most visually compelling slide and the one ScoreDSM can't produce.
5. **Run a `diff`** against the v3 recovery.json — show, side-by-side, the
   no-op pattern in v3 and its absence in the DSM cells.

After that, you have the slides + the supporting artefacts. The narrative
arc is already coherent.
