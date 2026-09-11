# Research Findings — Continuous flow matching on the simplex for valid-text generation


> **STALE — 2026-05-08. Research synthesis and forward plan, written before any of the paper's results existed.** Its own header block already records three of its predictions coming back negative; this banner records the rest.
>
> **Superseded:** §2.3 and §1.1 present the auxiliary CE on the implied `x1` as the fix that "stops mode collapse". The paper concludes the opposite — masked to γ ≥ 0.5 the anchor is solved before the first epoch ends because it sits where the label is already visible, so it supplies no per-token information (paper appendix §Training-time collapse). Do not carry the "ablation-paper-worthy fixes" framing forward.
>
> **Never cashed:** §5 item 5 calls expressing our own models in BPC "non-negotiable for writeup defensibility". The final paper reports **no BPC for any of our models**; the only BPC it cites is Statistical Flow Matching's published 1.39, used to bound the scope of our budget-matched ordering. The frontier table in §1.2 / §2.1 remains the repo's record of that comparison.
>
> **Never run:** everything in §3.D, §3.E and §5 items 3, 6, 7, 8 — the LLM-auditor track, TriviaQA semantic hallucination, the SEP / MARS / Min-K%++ comparisons, and experiments E1–E3 / F1–F3. This line was retired.
>
> **Still current:** §1.3 and §4.1 on the Hilbert metric. The 2-sparse subgradient argument is the reason the paper uses the LSE-smoothed variation seminorm, and this document is the only place the derivation is written out.
>
> Naming: "DFM" throughout this document means **Discrete** Flow Matching. Dirichlet Flow Matching, the paper's selected generator, appears only as cited literature.

> **Scope.** Independent research synthesis prepared while a separate Claude
> Code session executes the W1–W5 sweeps from `CLUSTER_TRAINING_PLAN.md`.
> No code or configs are changed by this document. Goal: identify viable
> approaches for embedding discrete tokens into a continuous space (CLR/ILR
> being the existing choice), train flow matching such that **the generated
> *words* are valid English** even if the *sentences* aren't, and use the
> implicit energy from EqM as a *uniqueness signal* DFM cannot match.
>
> Audience: the supervisor of the next cluster session and the writeup author.
> Authority of citations: every claim flagged with an arXiv ID was checked
> against the abstract; numbers cited in BPC/PPL columns come from the
> referenced tables.

> **Update — 2026-05-07 (W1+W3+W4 results landed).** Several *expected
> positives* in this file came back negative; the recommendations below
> are revised in place but the highlights are:
>
> - **W1 negative.** The Euler-γ sampler swap on `eqm_data50k_ep5_v2`
>   does *not* close the EqM-DFM gap. Best Euler-on-raw-`f` result is
>   KL_bi = 1.682 (NFE=200), vs NAG = 1.382 and DFM = 0.148. **The
>   trained EqM field encodes its data-pull in the conservative gradient
>   `∇⟨x, f(x)⟩`, not in `f` itself**: Euler with `use_grad=True`
>   reproduces NAG (1.391); Euler on raw `f` plateaus at 1.68 regardless
>   of NFE. The §1.5 / §3 framing of "Euler-γ on raw `f` as the surgical
>   fix" is wrong — the sampler is not the lever for EqM. This raises,
>   not lowers, the priority of Approach A (Logit-KL Flow) as the
>   forward path.
>
> - **W4 informative negative.** FMonCLR (Euler, no conservative-grad
>   indirection) at the same platform reaches KL_bi = 1.559 — close to
>   EqM-NAG (1.382), far from DFM (0.148). **Removing the conservative-
>   grad indirection does not help.** The continuous-on-simplex regime
>   itself is the bottleneck — Stark's mechanism (§1.4) confirmed at
>   K=27 character-level.
>
> - **W3 partially positive, partially invalidated.** EqM's *sequence-
>   level* energy `E(x) = ⟨x, f(x)⟩` is uninformative on substitution
>   and shuffle contrasts (AUC ≈ 0.50). The unique value-add narrows to
>   **per-position uncertainty** `U_pos_mean`: clean-vs-subst_0.5
>   AUC = 0.987, clean-vs-rand AUC = 1.000, but clean-vs-shuffle_0.5
>   AUC = 0.525 — shuffle remains unsolved. **DFM's denoiser proxy
>   `−log p_{1|t≈1}(x|x)` dominates every contrast** (rand 1.00,
>   subst 1.00, shuffle 0.99), invalidating the §3 claim that "DFM
>   structurally cannot produce a sequence-level scalar score." DFM
>   does so, and better than EqM on text8.
>
> The §3.D / §3.E LLM-auditor framings are *unaffected* — they don't
> rely on text8 sequence-level OOD claims — and remain the highest-
> leverage forward path alongside Approach A. Revised honest writeup
> arc in §6.

## 0. TL;DR — five concrete decisions

1. **Treat W4 (`FMonCLR` — naive linear FM on CLR) as a *predicted* negative
   result, not as a candidate winner.** Stark et al. (Dirichlet FM, 2024)
   proved that linear flow matching on the simplex has an *increasingly
   discontinuous marginal field as K grows*; KL-Flow's text8-class
   experiments report `linear-on-simplex PPL = 1344` vs their fixed version
   at `PPL = 41` — a 30× gap. Continue running it because it's a needed
   control, but **set the bar for "Approach A" by something else**.
   *(Confirmed 2026-05-07: W4 KL_bi = 1.559, ≈ EqM-NAG (1.382), not
   ≈ DFM (0.148). The bottleneck is the continuous-on-simplex regime,
   not the conservative-grad indirection — removing it does not help.)*
2. **The publishable upgrade beyond FMonCLR is *Logit-KL Flow Matching*
   (arXiv 2411.16821).** It is the only published continuous-on-simplex-ish
   method I could verify *beats DFM* at non-AR text generation
   (TinyStories: 19.0 vs 20.8 PPL; FineFineWeb: ≥27% lower PPL than DFM at
   matched NFE). Mechanically it is FMonCLR plus two changes: (a) regress
   *clean logits* not velocity, (b) deterministic-then-stochastic hybrid
   sampler. **If anything from this document is to graduate into next
   session's experiments, it is this method as a fifth baseline.**
3. **The energy-based OOD harness (W3) survives only as a *per-position*
   claim, not a sequence-level one.** *Original framing (now invalidated):
   EqM's `E(x) = ⟨x, f(x)⟩` gives a sequence-level scalar score that DFM
   structurally cannot produce.* **Result 2026-05-07:** EqM's E_seq is at
   chance on substitution (AUC=0.51) and shuffle (AUC=0.50). DFM's
   denoiser proxy `−log p_{1|t≈1}(x|x)` does produce a sequence-level
   score and dominates every contrast (rand 1.00, subst 1.00, shuffle
   0.99). The surviving differentiator is the *per-position* signal:
   EqM's `U_pos_mean` reaches subst_0.5 AUC = 0.987 and rand AUC = 1.000,
   but shuffle still fails (0.525). **Forward path (still publishable):**
   reframe the claim as "per-position localisation under a continuous
   energy" — useful for OOD *and* the auditor framings (§3.D / §3.E),
   *not* as a beats-DFM-on-text8 result. Add the divergence-uncertainty
   trace ([2605.00941](https://arxiv.org/abs/2605.00941)) to make the
   per-position claim apply to FMonCLR / Logit-KL too — a fair
   cross-method probe rather than an EqM-only assertion.
4. **The most ambitious framing — *FM-as-auditor on top of a small open
   LLM* — is publishable and structurally novel** (see §3.D below). The
   exact combination "conservative-gradient FM training of a residual
   energy `E(x | LLM_features)`, deployed as per-token uncertainty score
   on UQ benchmarks (TriviaQA / WikiBio-GPT3 / NQ)" is **not in the
   published literature** as of May 2026. Components exist (residual EBMs:
   [Bakhtin 2020](https://arxiv.org/abs/2004.10188), [EDLM
   2024](https://arxiv.org/abs/2410.21357); FM-as-energy unification:
   [Energy Matching 2025](https://arxiv.org/abs/2504.10612); single-hidden-
   state UQ probes: [SEP 2024](https://arxiv.org/abs/2406.15927)) — the
   combination does not. The recommended *first* experiment is the
   self-distillation variant **E1: DFM-teacher, EqM-auditor on text8** —
   which costs almost nothing because it reuses every piece of existing
   infrastructure.

5. **A *related project* has already implemented and benchmarked much of
   Approach D using a Sparse Variational GP (SVGP) auditor — see §3.E.**
   On WikiText-2, the GP + product kernel + Qwen2.5-1.5B hits
   **Seq AUROC = 0.999, Tok AUROC = 0.996**. This re-prioritises the EqM
   capstone: **don't redo the GP — replace it.** The novel claim is "the
   conservative-gradient flow-matching energy is a drop-in replacement
   for the SVGP auditor head, with no inducing-point overhead and the
   property that the same network is also a generator (which the GP
   prototype's auditor demonstrably is not — its BayesianGenerator hits
   text8 BPC ≈ 7)." Two open empirical gaps remain in the prototype that
   EqM has a credible shot at: **TriviaQA semantic hallucination** (best
   so far is **Spilled Energy AUROC = 0.82**;
   [arXiv:2412.10770](https://arxiv.org/abs/2412.10770), ICLR 2026 — *the
   forced baseline*) where the GP underperforms (0.45–0.55), and
   **auditor-driven valid-text generation** (the prototype can't do this
   with the GP energy at all).

The honest writeup arc is **"continuous flow on the simplex matches DFM on
n-gram fidelity *only* with the right geometry/sampler (positive: Logit-KL
Flow as the new baseline; negative: FMonCLR-CLR is what Stark predicts —
*confirmed 2026-05-07 at KL_bi=1.559*), and the EqM energy field provides
*per-position* uncertainty localisation that complements (rather than beats)
DFM's denoiser proxy on OOD (W3 — AUC=0.987 subst per-position; sequence-
level dominated by DFM), extending naturally to a per-token LLM auditor
that — unlike the existing SVGP prototype — is *also* a generator (§3.E)."**
The LLM-auditor extension ships independently of how big a KL_bi gap
remains, and is *not* damaged by the W3 sequence-level negative because
the auditor framing was always per-token.

---

## 1. Theoretical foundation — three metrics, one simplex

The probability simplex `Δ_{K-1}` (with K = 27 for text8 character-level)
admits at least three natural metrics, each implying a different generative
model. Pick wrong and the model fights its own geometry.

### 1.1 The Aitchison / CLR metric (what this codebase uses)

`CLR(p) = log p − mean(log p)` maps `Δ_{K-1}` isometrically to a hyperplane
`V_d ⊂ R^K` under the Aitchison inner product. The Aitchison inner product
is the "natural" *Riemannian* metric that makes the simplex flat and turns
ratios `log(p_i/p_j)` into Cartesian coordinates. The codebase trains MSE on
CLR features, which is L2 in this geometry.

**Failure mode** (`SESSION_SUMMARY.md` §2.2): MSE on CLR is *not* equivalent
to KL between the underlying simplex points. KL penalises errors on
high-probability coordinates much more strongly than Aitchison-L2 does. So
the FM regression converges to "slightly wrong on every coordinate" — which
is exactly the *marginal-mode* solution. The aux CE on the implied-`x_1`
reconstruction (added in the all-fixes config) re-introduces a KL-flavoured
gradient on the target coordinate, which is what stops mode collapse to
` ` (space).

### 1.2 The Fisher–Rao metric (SFM, FisherFlow)

Reparameterise `p` as `q = √p` on the positive orthant of `S^{K-1}`. Under
this `√`-map, the Fisher information becomes the Euclidean metric on the
sphere, so flow matching reduces to *Riemannian flow matching on a sphere*
(Cheng et al. 2024 ["Statistical FM"](https://arxiv.org/abs/2405.16441),
Davis et al. 2024 ["Fisher Flow"](https://arxiv.org/abs/2405.14664)). Both
papers report **direct text8 BPC numbers** in their Table 2:

| Method (text8, K=27) | BPC |
|---|---:|
| LinearFM (≈ FMonCLR-twin) | 1.65 |
| MultiFlow / D3PM-abs / BFN | 1.41–1.47 |
| **SFM (Fisher-Rao geodesics)** | **1.39** |
| **SEDD-absorb** | **1.32** |
| AR Transformer T12 (Al-Rfou 2018) | 1.18 |
| AR Transformer T64 | 1.13 |

The 0.26 BPC gap between LinearFM and SFM is *purely* the geometry change.
**This is the single most important benchmark table for the writeup**: our
own DFM and EqM should be expressed in BPC and overlaid on it.

### 1.3 The Hilbert projective metric

`ρ_H(p, q) = max_i log(p_i/q_i) − min_i log(p_i/q_i)` — equivalently, the
*variation norm* of CLR residuals. This is the metric on the
"Hilbert–Birkhoff projective space" (Birkhoff 1957; Bushell 1973;
[Nielsen & Sun 2017/2019](https://arxiv.org/abs/1704.00454)). Properties:

- **Projective / scale invariance**: `ρ_H(λp, μq) = ρ_H(p,q)`. In CLR
  coordinates, the metric ignores additive constants in the residual. Useful
  for *clustering* (which Nielsen–Sun apply); **misaligned with regression**,
  where the magnitude of `u_tgt = c(γ)·(x_0 − x_1)` carries the step-size
  signal needed for sampling.
- **Information monotone under coarse-graining** (a property KL has but
  Aitchison-L2 does not).
- **Polytopal balls / non-Riemannian**: balls are hexagons, not ellipsoids;
  geodesics are Euclidean line segments through the polytope.
- **2-sparse subgradient**: `∂ρ_H/∂r = e_{argmax(r)} − e_{argmin(r)}`. Out
  of 27 coordinates, only two get a gradient per token — disastrous as a
  primary signal, *fine* as a small auxiliary at λ ≈ 0.05.

### 1.4 Why does any of this predict DFM > FMonCLR?

Stark et al. proved (constructively, via the discontinuity of the marginal
vector field on `Δ_{K-1}` near vertices) that **the support of conditional
probability paths shrinks toward δ-functions as K grows**, making the
marginal field non-Lipschitz. The integrator then takes over: a naïve Euler
step at any meaningful step size lands off-manifold, propagates through the
next step, and so on. K=27 is well past the "discontinuity is observable"
threshold their experiments document on K=4 DNA.

This explains the 30× PPL gap reported in
[KL-Flow](https://arxiv.org/abs/2411.16821) Table 1 between their fix and
naive linear-on-simplex flow. **It also predicts that FMonCLR-with-Euler in
this codebase will reproduce the gap.** Running it as W4 is still useful —
it provides the controlled negative — but writing the prior expectation down
matters for the next session's interpretation.

### 1.5 What EqM is, expressed cleanly

EqM trains a velocity field `f: V_d → V_d` so that the *conservative gradient*
`g(x) = ∇_x ⟨x, f(x)⟩` regresses to the FM target `c(γ)·(x_0 − x_1)`. After
training, `E(x) = ⟨x, f(x)⟩` is a *non-normalised scalar potential* on `V_d`
whose gradient was steered toward the FM velocity. The minima of `E` are
where `g(x) = 0`, i.e. where the trained field expects `x ≈ x_1`.

Two consequences:

- **Sampling is energy descent**, not transport. NAG-GD on `E` has no notion
  of γ-trajectory; the field is *fixed*, the dynamics are second-order
  inertial. This is mathematically nice (deterministic, cheap, no
  schedule) but operationally wrong for FM regression objectives — the
  field is *trained* across many γ values but *sampled* without one
  (`RESULTS.md` §"Why EqM's setup makes sampling hard"). W1 was
  proposed as the surgical fix. **2026-05-07 update: W1 was negative.**
  Euler-γ on raw `f` plateaus at KL_bi ≈ 1.68 (independent of NFE);
  Euler with `use_grad=True` reproduces NAG (1.391); σ_init=0.3 also
  recovers parity but doesn't surpass. The trained EqM checkpoint
  encodes its data-pulling field in `∇⟨x, f(x)⟩`, not in `f`. The
  sampler is *not* the lever — neither integrator (NAG vs Euler) nor
  NFE moves the headline. The *field* is what matters.
- **The energy field doubles as an OOD score**. EqM gets this for free;
  DFM does not have anything analogous because its "denoiser" predicts
  per-position categorical logits, not a sequence-level scalar. This is
  the unique-value-add §3 below builds on.

---

## 2. Competitive landscape (concrete numbers + what to benchmark against)

### 2.1 Reproducible-on-text8 benchmark table

This is the table the writeup should reproduce *with our own DFM and EqM
checkpoints overlaid*. All numbers character-level text8 BPC unless noted.

| Method | text8 BPC | Source |
|---|---:|---|
| AR Transformer T64 (deep) | 1.13 | [Al-Rfou 2018](https://arxiv.org/abs/1808.04444) |
| AR Transformer T12 | 1.18 | Al-Rfou 2018 |
| Plaid 1B (continuous diffusion on learned embedding) | **1.12** | [Gulrajani & Hashimoto 2023](https://arxiv.org/abs/2305.18619) |
| SEDD-absorb | **1.32** | [Lou 2024](https://arxiv.org/abs/2310.16834) |
| MDLM (≤) | 1.38 | [Sahoo 2024](https://arxiv.org/abs/2406.07524) |
| **SFM** (Fisher-Rao geodesic FM) | **1.39** | [Cheng 2024](https://arxiv.org/abs/2405.16441) |
| MultiFlow | 1.41 | Cheng 2024 |
| BFN | 1.41 | Cheng 2024 |
| ARDM | 1.43 | [MDLM table](https://arxiv.org/abs/2406.04329) |
| D3PM-absorb | 1.45 | [Austin 2021](https://arxiv.org/abs/2107.03006) |
| SEDD-uniform | 1.47 | Lou 2024 |
| **LinearFM (≈ FMonCLR)** | **1.65** | Cheng 2024 |
| D3PM-uniform | 1.61 | Austin 2021 |

Two reading-points for the writeup:

- **The realistic discrete-diffusion frontier on text8 is 1.32–1.47 BPC.**
  AR is at 1.13. We are targeting "match the diffusion frontier", not "match
  AR".
- **DFM does not publish text8 BPC.** Our in-house DFM (`KL_bi = 0.148`)
  is the only DFM-vs-EqM apples-to-apples comparison. We should report
  *both* `KL_bi` and BPC against this table to be defensible.

### 2.2 The four prior attempts at continuous-on-simplex for text and what
they teach

| Paper | Geometry | Sampler | text8? | Verdict |
|---|---|---|---|---|
| [Floto 2023](https://arxiv.org/abs/2309.02530) "Diffusion on Probability Simplex" | softmax-of-OU | DDPM | no | small categorical / image-quant only |
| [Stark 2024](https://arxiv.org/abs/2402.05841) "Dirichlet FM" | Dirichlet-mixture probability paths on simplex | manifold ODE | no (DNA only) | proves linear FM on simplex is broken at large K |
| [Davis 2024](https://arxiv.org/abs/2405.14664) "Fisher Flow" | √p on sphere, Fisher-Rao geodesics | Riemannian ODE | no | beats Dirichlet on DNA, not text8 |
| [Cheng 2024](https://arxiv.org/abs/2405.16441) "SFM" | √p on sphere + diffeomorphism for stability | manifold ODE | **yes — 1.39 BPC** | the most relevant published result |
| [Sevriugov 2024](https://arxiv.org/abs/2411.16821) "Logit-KL Flow" | linear-in-logit (≈ KL geodesic) | hybrid det-then-stochastic | yes — beats DFM | **only verified method that beats DFM on non-AR text** |

### 2.3 What is (probably) novel in this codebase

Searching the literature: **no published method uses the
conservative-gradient / implicit-energy formulation** (`g = ∇⟨x, f(x)⟩`) on
CLR features for sequence data. EqM was published at
[arXiv:2510.02300](https://arxiv.org/abs/2510.02300) (Wang & Du 2025) for
images; this codebase appears to be the first text application. The
contribution territory is:

1. The CLR + conservative-gradient framing as a *generative* model on the
   simplex (already novel; the question is whether it's *competitive*).
2. The mode-collapse fixes documented in `SESSION_SUMMARY.md` (γ-importance,
   matched train/sample σ, aux CE on implied-x_1) — independently
   ablation-paper-worthy.
3. **Energy-based OOD on sequences** as a uniquely-EqM capability vs DFM —
   the cleanest novel contribution that *does not depend on closing the
   KL_bi gap*.

---

## 3. Three viable paths forward (in priority order for next session)

### Approach A — Logit-KL Flow Matching as a fifth baseline (highest leverage)

[arXiv:2411.16821](https://arxiv.org/abs/2411.16821). Mechanical recipe:

- **Representation**: token `i` ↦ logit vector `l_i ∈ R^K` (one-hot logits
  with magnitude γ_l, e.g. `γ_l · e_i`). Linear interpolation in logit
  space is the **KL geodesic** between softmax(l_0) and softmax(l_1) up to
  a constant offset. This is essentially CLR up to additive normalisation,
  but the FM target now lives in unconstrained `R^K`.
- **Loss**: regress *clean logits* `l_1` from `x_t`, not velocity:
  `L = E ‖v̂(x_t, t) − l_1‖²`. The optimal velocity is then
  `v̂*(x_t, t) = E[l_1 | x_t]` (posterior mean logit).
- **Sampler (the key innovation)**: deterministic ODE on `E[l_1 | x_t]`
  for `t < 0.28`; stochastic re-noising sampling for `t ≥ 0.28`. The
  hybrid is what fixes the support-shrinkage / discontinuity problem
  Stark identified.
- **Reported headline**: TinyStories PPL 19.0 (KL-Flow) vs 20.8 (DFM);
  FineFineWeb 51.5 vs 150.6 vs SEDD 70.8 at 1024 NFE.

**Why this is a higher-leverage baseline than FMonCLR**: Stark already
predicts FMonCLR will fail; KL-Flow already exists and beats DFM. Adding it
turns the writeup table from "we tried CLR, it lost; here's a sampler fix"
into "we tried CLR, it lost as predicted, but the fix has been published
and we reproduced it on text8 character-level — and our energy-OOD signal
extends *that* method too." This is a stronger story.

**Implementation cost** (notional, do not actually code without the cluster
session's signoff): ~1 day to add a `LogitKLFlow` model registered like
`FMonCLR`. The training loop is the existing FM regression with a different
target (`l_1` not `c(γ)(x_0 − x_1)`); the sampler is new but small (200
lines). Compute: same 70 min for 5 ep × 50k as FMonCLR.

### Approach B — Fisher-Rao reparameterisation (SFM-style)

[arXiv:2405.16441](https://arxiv.org/abs/2405.16441). If Approach A doesn't
pan out, SFM is the well-published alternative with a *direct* text8 BPC
number to beat (1.39). Implementation is heavier — needs the `√p`
reparameterisation, sphere geodesic interpolation, and Riemannian Euler
sampler — but the geometry change is what the literature says actually
fixes the problem at K=27.

A "compromise" experiment: keep the conservative-gradient framework, but
parameterise `f` to live in *spherical-tangent* coordinates instead of CLR.
This would test whether the EqM mode-collapse fixes transfer to a more
appropriate geometry. Probably 2 days of work. Lower priority than A.

### Approach C — Per-position uncertainty as the surviving contribution

**2026-05-07 status update.** The W3 harness was run on the
`eqm_data50k_ep5_v2` / `dfm_data50k_ep5_v2` / `fmclr_data50k_ep5_v2`
checkpoints. Headline:

| Model / stat | clean-vs-rand | clean-vs-subst_0.5 | clean-vs-shuffle_0.5 |
|---|---:|---:|---:|
| EqM `E_seq`                  | 0.16 (\|0.84\|) | 0.51   | 0.50 |
| **EqM `U_pos_mean`**         | **1.000**       | **0.987** | 0.525 |
| EqM `U_pos_max`              | 1.000           | 0.977  | 0.511 |
| **DFM `−log p_{1|t≈1}` proxy** | **1.000**     | **1.000** | **0.993** |
| FMonCLR `E_seq`              | 0.12 (\|0.88\|) | 0.35 (\|0.65\|) | 0.516 |

Two findings invalidate the original §3 framing:

1. **EqM's sequence-level energy is at chance on substitution and
   shuffle.** The trained `E(x) = ⟨x, f(x)⟩` does not separate clean
   from corrupted text on text8 except via the trivial uniform-noise
   contrast (and even there the sign is *inverted* — clean text has
   *higher* energy than uniform random, which scores |AUC|=0.84 only
   when you take absolute value).
2. **DFM's denoiser proxy `−log p_{1|t≈1}(x|x)` dominates every
   sequence-level contrast at AUC ≈ 1.00.** The §3 claim that "DFM
   structurally cannot produce a sequence-level scalar score" was
   wrong — the denoiser logits at γ ≈ 1 give a normalized log-likelihood
   that beats EqM on every text8 contrast.

What survives:

- **EqM's per-position uncertainty `U_pos_mean` reaches AUC = 0.987 on
  substitution_0.5** — within noise of DFM (1.00) on subst, but
  *failing* on shuffle (0.525 vs DFM 0.993). The "per-position
  localisation" claim narrows to *substitution-style* corruption.
- **Shuffle remains an open problem for both EqM and FMonCLR** (AUC
  ≈ 0.5). DFM solves it via its denoiser. The §3.5 "honesty
  ablation" (valid grammatical permutation) was the predicted hard
  contrast — confirmed.
- **The auditor framings (§3.D / §3.E) are unaffected** because they
  build on per-position scoring of LLM logits, not on text8 sequence
  scoring.

The forward path therefore narrows from "EqM uniquely scores OOD" to
"EqM provides a per-position localisation signal that *complements*
(rather than beats) DFM's sequence-level proxy on substitution-style
contrasts; on shuffle, neither EqM's energy nor FMonCLR's untrained
field is informative." Bake this into the writeup; do not claim more.

The energy field `E(x) = ⟨x, f(x)⟩` still exists for every EqM
checkpoint. The remaining W3 extensions below quantify the *per-
position* signal more rigorously and add proxies that apply to DFM
and Logit-KL (so the comparison becomes fair across methods).

What to compute (extending the existing `scripts/eval_ood.py`):

1. **Sequence-level energy** `E(x)` for clean text8 windows, substitution-
   corrupted at rates {0.1, 0.3, 0.5}, position-shuffled at the same rates,
   and uniform-noise sequences. Report ROC-AUC for each contrast.
2. **Per-position gradient norm** `‖∂E/∂x_ℓ‖`. Report (a) mean across
   positions per window as a second sequence-level score, and (b)
   per-position ROC-AUC: "this position was substituted" given a 50%
   Bernoulli mask. This is the *localisation* signal that DFM cannot give
   except via per-position softmax entropy.
3. **Multiscale stacking** (per [Mahmood 2020](https://arxiv.org/abs/2010.13132)):
   compute the per-position grad norm at γ ∈ {0.3, 0.5, 0.7, 0.9, 1.0} and
   stack into a feature vector. Score with a tiny logistic regression on a
   held-out clean/corrupt split.
4. **Two novel proxies the literature suggests** (both worth implementing
   even if (1)–(3) already work):

   - **NAG basin-drift indicator**: initialise NAG-GD at the test sequence
     and run K=5 sampling steps. `drift(x) = d_H(x, NAG^K(x))`. Clean
     sequences are near the field's fixed point; corrupted ones drift.
     Cheap (no second-order autograd).
   - **Divergence-uncertainty trace**:
     [arXiv:2605.00941](https://arxiv.org/abs/2605.00941) (May 2026) proves
     `tr(Cov(x_1 | x_t)) = c(t)·∇·v(x_t) + const` *for any FM velocity*.
     Estimate with Hutchinson trace (~16 random vectors). **No second-order
     autograd needed** — applicable to FMonCLR and Logit-KL too. This
     extends the OOD signal to non-energy methods, making it a fair
     cross-method probe.

5. **Honesty ablation**: score a *grammatically valid permutation*
   ("the cat sat on the mat" → "the mat sat on the cat") and check that
   `E` flags it. If it doesn't, the energy is just a fancy unigram-bigram
   classifier; if it does, the energy genuinely captures sequence-level
   coherence (Bakhtin et al.'s
   [residual EBM result](https://arxiv.org/abs/2004.11714) hits ~0.95
   AUROC on this kind of contrast).

**Expected numbers** (anchored against
[EqM CIFAR baseline](https://arxiv.org/abs/2510.02300) Table 6 and
[Bakhtin 2020](https://arxiv.org/abs/2004.11714)):

- Substitution at 50%: AUROC ≥ 0.85 is realistic.
- Uniform noise: AUROC near 1.0 (saturates easily).
- Position-shuffle: 0.65–0.80 (the hardest because unigram statistics are
  preserved). A result here ≥ 0.75 would *uniquely* validate that EqM's
  energy captures sequence-level structure DFM cannot.
- DFM proxy `−log p_{1|t≈1}(x|x)`: should under-perform EqM's
  sequence-level energy on shuffle, match it on substitution, both saturate
  on uniform noise. This is the figure that tells the writeup story.

### Approach D — FM-as-auditor on top of a small open LLM

This is the *re-framing* the user proposed: load a small open LLM (Qwen2.5-1.5B,
Phi-3-mini, Llama-3.2-1B class) inside the 20 GB MIG, get per-position
logits + final hidden states, and train an FM/EBM that takes both the LLM
output *and* the raw text as input. The FM model generates valid text as a
sanity check and (the actual goal) reports per-token / per-character /
per-sequence uncertainty scores indicating where the LLM was uncertain. The
trained energy `E(x | h_LLM, logits_LLM)` is the auditor.

#### D.1 Direct architectural precedents

The cleanest precedent is the **residual-EBM-on-frozen-LM line**:

- [Bakhtin et al. 2020 — *Residual EBMs for Text*](https://arxiv.org/abs/2004.10188)
  (JMLR 22:20-326, sometimes also cited as the companion ICLR paper
  [Deng et al. 2020](https://arxiv.org/abs/2004.11714)). Defines a joint
  sequence distribution `p(x) ∝ p_LM(x) · exp(−E_θ(x))` where `p_LM` is a
  *frozen* AR Transformer LM and `E_θ` is an unnormalised scalar field.
  `E_θ` is a BERT/RoBERTa encoder + scalar head, trained by **noise
  contrastive estimation** with `p_LM` as the noise distribution
  (positives: real text; negatives: LM samples; logistic loss). Reports
  reliable human-vs-machine discrimination and importance-resampling
  perplexity gains. **This is the structural template** — the only
  architectural change EqM needs is to *replace the NCE binary classifier
  with a conservative-gradient FM regression target*.

- [Xu et al. 2024 — *Energy-Based Diffusion Language Models* (EDLM, ICLR 2025)](https://arxiv.org/abs/2410.21357).
  The same recipe with a *diffusion / non-AR LM teacher*. Confirms the
  pattern generalises beyond AR teachers — relevant if the user wants to
  use the in-house DFM as the teacher instead of an open-source LLM.

- [Balcerak et al. 2025 — *Energy Matching*](https://arxiv.org/abs/2504.10612).
  Recent unification showing FM training and explicit energy
  parameterisation can co-exist on the same network. Makes "EqM provides
  `E(x)` for free" a formally established claim in the FM literature, not
  a folk theorem.

On the *deployment* side, the closest analogue is the **outcome / process
reward model** literature:
[Cobbe et al. 2021 (GSM8K verifier)](https://arxiv.org/abs/2110.14168)
and especially [Lightman et al. 2023 — *Let's Verify Step by Step*](https://arxiv.org/abs/2305.20050).
PRM800K is an existence proof that "small scalar verifier on top of a
large generator" is productive; the user's setup is the same architectural
pattern with FM training instead of supervised step-labelling.

A subtle but important note: **speculative decoding**
([Leviathan 2023](https://arxiv.org/abs/2211.17192)) goes the *opposite*
direction (small drafter, large verifier). The user's setup — *small FM
verifying a frozen LLM* — appears to be genuinely under-explored. No
paper I found does exactly this with FM-trained energies.

#### D.2 Conditional FM with LM features — what's published, what's open

- [Tae et al. 2025 — TESS-2](https://arxiv.org/abs/2502.13917): Mistral-7B
  AR backbone, continued pretraining with a diffusion CE head. The
  architectural pattern "AR hidden states fed into a non-AR generative
  head" is exactly what an FM auditor would reuse.
- [Plaid 1B](https://arxiv.org/abs/2305.18619): likelihood-based diffusion
  on continuous embeddings. Closest to CLR-flow.
- [Discrete Flow Matching](https://arxiv.org/abs/2407.15595) defines DFM
  with an *arbitrary* source distribution; using an LM's marginals as the
  source is a one-line modification — to my knowledge, unpublished.
- [FlowRL (2025)](https://arxiv.org/abs/2509.15207): uses FM to match an
  LLM's policy to a target reward distribution. RL alignment, not
  auditing — but proves FM machinery composes with LLM features.

**None of these treat LLM logits as a *feature* fed into the FM model for
the express purpose of producing an uncertainty score.** That gap is the
contribution.

#### D.3 The UQ benchmarks the auditor must compete against

| Method | Inputs | Score | Year |
|---|---|---|---|
| [Semantic Uncertainty / Entropy](https://arxiv.org/abs/2302.09664) (Kuhn, Gal, Farquhar) | K LLM samples, NLI clustering | entropy over clusters | 2023, Nature 2024 |
| [Semantic Entropy Probes (SEP)](https://arxiv.org/abs/2406.15927) | single hidden state → linear probe | predicted semantic entropy | 2024 |
| [MARS](https://arxiv.org/abs/2402.11756) (Bakman et al.) | per-token importance × prob | weighted score | 2024 |
| [Kadavath et al. — *LMs Mostly Know What They Know*](https://arxiv.org/abs/2207.05221) | LLM self-evaluation | P(True), P(IK) | 2022 |
| [SelfCheckGPT](https://arxiv.org/abs/2303.08896) | K samples, BERTScore consistency | mean disagreement | 2023 |
| [Min-K%-Prob](https://arxiv.org/abs/2310.16789) / [Min-K%++](https://arxiv.org/abs/2404.02936) | per-token min-K logprob | mean of bottom K% | 2023–24 |
| [Semantic Energy](https://arxiv.org/abs/2508.14496) | logit aggregation | energy score | 2025 |

The publishable bar: **beat SEP and Min-K%++ on AUROC for hallucination
detection on TriviaQA closed-book + NQ + WikiBio-GPT3, with comparable
ECE.** SEP is the natural comparator because it also takes a single hidden
state as input — the FM auditor is a strictly more expressive model on
the same signal.

#### D.4 Practical recipe for this codebase

**Memory budget.** Qwen2.5-1.5B / Llama-3.2-1B in fp16: ~3 GB weights +
1–2 GB activations at L=512, B=8. In bf16 you have ~15 GB headroom for
the FM model, optimizer state, gradients. With **4-bit (bnb-nf4 or AWQ)
quantisation** of the LLM, footprint drops to ~1 GB. **The right default
is to cache once**: precompute (token_ids, top-k logits per position,
last-hidden-state per position) for the training corpus, store as
memory-mapped tensors, and during FM training the LLM is not loaded at
all. This trivially fits the 20 GB MIG.

**Tokenizer choice — three options ranked.**

1. **(Recommended for E1 below.) In-house DFM as teacher, K=27 char-level.**
   No tokenizer mismatch, no new infrastructure. Lets you isolate the
   architectural question (does FM-trained energy beat NCE-trained energy,
   does conditioning on a teacher's hidden states help over the unconditional
   EqM?) without LLM logistics. **Cost: ≈ 0**.
2. **Open LLM teacher (Qwen2.5-1.5B / Phi-3-mini / Llama-3.2-1B) on
   BPE-tokenised WikiText-103 / FineWeb-edu-10M.** Requires switching FM
   from CLR (K=27) to a *logit-space* parameterisation (K = 32k) — the
   simplex at K=32k is degenerate; treat outputs as `R^K` with softmax at
   decode. Effectively this is *Logit-KL Flow Matching (Approach A)
   conditioned on LLM features* — a clean composition of the two
   approaches. **Cost: 2–3 days infra + ≈ 1 day caching the teacher.**
3. **Byte-level via [ByT5 tokenizer](https://arxiv.org/abs/2105.13626),
   K=257.** Most novel, most painful. Use a small ByT5/CharBERT
   teacher. Keeps the simplex framing while removing the
   tokenizer-mismatch problem. **Cost: 1 week dev + 2× compute over BPE.**

**Architecture for feeding LLM features into the velocity head.** The
pattern from EDLM and TESS-2: at each position `t`, the FM transformer
block receives `[x_t (CLR or logit) ; project(h_LLM_t) ; project(top_k_logits_t)]`
concatenated and projected to `d_model`. **`h_LLM` must be detached** so
the autograd path through `∇_x ⟨x, f(x)⟩` doesn't leak into the frozen
teacher. Cross-attention from FM tokens to LLM hidden states is more
expressive but pays an attention cost — only worth it if the FM and LLM
use different tokenisations. The conservative-gradient formulation is
unaffected.

#### D.5 Three experiments, in order of cost

**(E1) EqM-as-residual-EBM-on-DFM, char-level text8 — ≈ 1 day total.**
- Teacher: frozen `runs/dfm_data50k_ep5_v2/epoch_final.pt`.
- Student: existing EqM with backbone modified to consume DFM's logits
  + last hidden state.
- Loss: existing FM regression of `∇_x ⟨x, f(x)⟩` toward `c(γ)·(x_0 − x_1)`,
  with the input to the encoder now augmented with detached DFM features.
- Eval: per-position `E(x)` ROC-AUC on synthetic corruptions (random
  char-swap, n-gram repeats, valid-permutation control from §3.C) vs.
  DFM's own per-position `−log p(x_t | x)`.
- *Why this is the right first step*: it derisks the *architectural*
  claim ("conditioning the FM energy on a teacher's features improves
  per-position uncertainty over both the unconditional EqM and the
  teacher's own NLL") without taking on tokenizer / quantisation logistics.

**(E2) EqM-on-Qwen-1.5B, BPE WikiText / FineWeb — ≈ 1 week total.**
- Teacher: Qwen2.5-1.5B-Instruct in 4-bit, cached logits + last hidden state
  on a 100M-token subset.
- Student: Logit-KL-Flow-style FM in `R^K` (K = top-8k tokens; map OOV
  to UNK). Conservative-gradient or direct velocity — both worth ablating.
- Eval: AUROC for hallucination detection on TriviaQA + NQ + WikiBio-GPT3
  vs. SEP, MARS, SelfCheckGPT, Min-K%++. Report AUARC and ECE alongside.
- *Conditional on E1 showing a clear margin.*

**(E3) Process-level scoring on PRM800K / GSM8K — ≈ 2 weeks total.**
- Per-step EqM energy as a process reward signal vs. Lightman et al.'s
  PRM. Highest-impact framing if E1 + E2 both work.

#### D.6 Honest assessment of D vs A

| | Approach A (Logit-KL Flow) | Approach D (LLM-auditor) |
|---|---|---|
| Floor cost | 1 day | 1 week+ |
| Ceiling | matches DFM (verified) on PPL | beats SEP/Min-K%++ on UQ (unverified) |
| Novelty | reproduces published work on text8 | combination is unpublished |
| Cleanness as a methods paper | high (single model, one ablation table) | medium (multiple moving pieces) |
| Risk | known-good recipe | hidden states + logits may already saturate UQ; FM bump may be 1–2 AUROC points (= noise) |
| Existing-infra reuse | ~70% | ~95% (E1), ~30% (E2) |

**Recommendation:** treat A and D as *parallel tracks*, not competitors.
Run E1 (≈ 1 day) *before* deciding whether E2 is worth the engineering;
E1 reuses existing infra and tells you whether the conditional-FM energy
even helps in principle. If E1 shows a margin ≥ 5 AUROC points over
unconditional EqM and ≥ 3 over DFM's own NLL, scale up to E2. If E1
doesn't show a margin, fall back to A (Logit-KL Flow) as the headline
contribution and present E1 as a negative result with mechanism.

#### D.7 The novel claim, written out

If E1 + E2 work, the publishable claim is:

> *"FM-trained residual energies are a strict generalisation of NCE-trained
> residual EBMs (Bakhtin 2020) for LLM auditing: (a) they get per-token
> gradients of `E` for free via autograd (NCE doesn't), (b) they produce a
> calibrated continuous score, not a binary discriminator, (c) the same
> network is also a generator, so the auditor can be sanity-checked by
> sampling from it. We show on TriviaQA / NQ / WikiBio-GPT3 that an
> EqM-trained residual energy on top of Qwen2.5-1.5B beats Semantic
> Entropy Probes and Min-K%++ on hallucination AUROC."*

The auditor-is-also-generator point (c) is the **key differentiator** from
SEP / MARS / Min-K%++ / Semantic Energy, none of which can generate. It
is also the link back to W3 / Approach C: the same trained `E` that scores
LLM tokens is the same `E` that scores text8 corruptions. One model, two
deployments.

### Approach E — EqM-augmented per-token auditor (extending the existing GP/SVGP prototype)

A *related project* — `discrete_hilbert` — has already implemented and benchmarked
much of Approach D using a **Sparse Variational GP (SVGP)** as the auditor head
on top of frozen GPT-2 / Qwen2.5-1.5B logits. This re-prioritises what the EqM
capstone should attempt: instead of starting from scratch, the high-leverage
move is to **replace the GP auditor with the EqM conservative-gradient flow-
matching energy**, while reusing the prototype's training and healing
infrastructure.

#### E.1 What the prototype already established

The `discrete_hilbert` codebase implements `PerTokenBayesianAuditor` —
essentially Approach D §D.4 architecture with a *product GP kernel* over LM
logits and hidden states:

```
K((z, h), (z', h')) = K_logit(z, z') · K_context(h, h')
```

where `z = pos_proj(backbone(log_x))` are per-token logit-CLR latents and
`h = ctx_proj(GPT-2 last hidden)` are detached LM context. The product of two
PSD kernels is PSD, so all SVGP math applies. Training combines flow-matching
regression with a **contrastive hinge** on (valid, corrupted) pairs:

- `mean_loss = E_valid² + relu(margin_energy − E_invalid)`
- `var_loss  = Var_valid  + relu(margin_var    − Var_invalid)`

After training, `per_token_ood(log_x, ctx) → (energy B×L, variance B×L)` is
the deployment-time score; **GP epistemic variance** is the calibrated
uncertainty signal (high variance ↔ unseen distribution).

**Verified results** (worth reading as the bar to beat):

| Setup | Seq AUROC | Tok AUROC |
|---|---:|---:|
| GPT-2 logit-only | 0.965 | 0.961 |
| GPT-2 + product kernel (context) | 0.994 | 0.976 |
| Qwen2.5-1.5B logit-only | 0.981 | 0.980 |
| Qwen2.5-1.5B + product kernel | **0.999** | **0.996** |
| **Spilled Energy** ([ICLR 2026, arXiv 2412.10770](https://arxiv.org/abs/2412.10770)) | **0.998** | 0.54 |

Two hard-won empirical findings the EqM capstone must absorb:

1. **WikiText-2 *syntactic* corruption detection is essentially solved.**
   AUROC ≈ 0.99 is achievable both with the GP + product kernel *and* with
   Spilled Energy (single forward pass, **zero training cost**, ICLR 2026
   result). Any new method must either *match* this on sequence-level or
   add localisation that SE can't (SE per-token AUROC = 0.54 — i.e. SE
   does **not** localise).
2. **TriviaQA *semantic* hallucination is an open problem.**
   Best result is **SE on Qwen2.5 = 0.82 AUROC**; the trained GP auditor
   actively under-performs (0.45 logit-only — *worse than chance*; 0.55
   with context). The GP was trained on random-substitution corruption,
   which doesn't transfer to QA-confusor distribution shift. **This is
   where genuinely novel work is possible.**

The prototype also implements **vocabulary-constrained healers** (Langevin /
Gibbs / Langevin-Gibbs / GP-slice) that *repair* corrupted text by minimising
GP energy subject to the constraint that each token must be in GPT-2's top-K
at that position. Best healer result on a single 25%-corrupted L=64 chunk:
LM log-prob improves from −7.49 (corrupted) to −6.27 (Gibbs with SE∪GP union
mask) — a 1.22-nat recovery. The architecture composes cleanly with EqM
energies because the optimisation primitive is `−∇_x E(x)` — agnostic to
whether `E` is a GP mean or `⟨x, f(x)⟩`.

#### E.2 The novel EqM contribution on top of the prototype

The clean publishable claim — **not** in the prototype, **not** elsewhere in
the literature — is:

> *"Replace the SVGP auditor head with the conservative-gradient flow-matching
> energy `E(x) = ⟨x, f(x)⟩`. The same network is then both a generator
> (sample EqM-Euler over γ) and an auditor (per-token gradient norm
> `‖∂E/∂x_t‖`). The product-kernel structure transfers as a context-
> conditioning of the EqM backbone (concat or cross-attend `h_LLM` into the
> Transformer), and contrastive training transfers verbatim. Healing infra
> is reused with the EqM gradient as the descent direction."*

Concrete advantages over the GP prototype:

- **No SVGP overhead**: no inducing points, no Cholesky, no MPS-instability
  workarounds. The per-token gradient is one autograd call against `E(x)`.
- **Generative-and-discriminative is *literal*, not a property claim**:
  the same `E(x)` that scores tokens is the same field whose Euler-γ
  integration produces samples. The prototype's `BayesianGenerator`
  achieves text8 BPC ≈ 7 — the GP's posterior is too smooth to drive
  sampling. EqM's flow-matched energy is, by construction, sharper on
  the data manifold.
- **Calibration via the divergence-uncertainty trace**
  ([arXiv:2605.00941](https://arxiv.org/abs/2605.00941)) replaces GP
  posterior variance with a closed-form, trained-energy-compatible
  uncertainty: `tr(Cov(x_1 | x_t)) ∝ ∇·v(x_t)`. This *plus* `‖∂E/∂x_t‖`
  gives two complementary uncertainty signals from one energy.
- **Spilled Energy still composes**: SE is computed from *the LLM's* logits,
  independent of the auditor head. Stack EqM-energy + SE for a hybrid:
  SE for sequence-level screening (0.998), EqM-grad-norm for per-token
  localisation. The prototype's table shows this combination beats either
  alone.

#### E.3 What to keep from the prototype

Verbatim:

- **Product-kernel-style context conditioning** of the EqM backbone with LM
  hidden states. Implementation: concat `[x_clr ; project(h_LLM)]` before
  the encoder and **detach `h_LLM`** so autograd doesn't leak into the
  frozen teacher.
- **Contrastive training on (valid, corrupted) pairs** with energy and (in
  the EqM case) divergence-trace hinges:
  `L = flow_loss + λ_E · (E_valid² + relu(m_E − E_invalid)) +
       λ_div · (div_valid + relu(m_div − div_invalid))`.
  Margins from the prototype (`margin_energy = 2.0`, `margin_var = 0.8`)
  are reasonable starting points.
- **Healer infrastructure as-is**, with the GP-energy descent direction
  swapped for `−∇_x ⟨x, f(x;γ)⟩`. The vocabulary constraint (top-K from
  the LM at each position) and the Gibbs / Langevin / Langevin-within-
  Gibbs / GP-slice ensembling are agnostic to the energy definition.
- **Spilled-Energy as a forced baseline column** in every results table.
  No-train, single-forward-pass, AUROC 0.998 — there is no excuse to
  publish without comparing.
- **Engineering caveats** (already debugged):
  - Math SDPA backend for `create_graph=True`.
  - `torch.cdist` has no second-order backward — use matmul-based
    distances. *EqM already does this in the existing geometry.py;
    re-verify if any new modules are added.*
  - GPT-2 `logits[i]` predicts token at position `i+1` — off-by-one in
    healers and per-token alignment.
  - Re-run the LLM forward after each one-hot commit in Langevin-Gibbs.
  - Cast Qwen2.5 BFloat16 logits to float32 before softmax.

#### E.4 Three experiments, in priority order, that build on the prototype

**(F1) EqM-energy auditor on WikiText-2 — the parity check (~3 days dev,
~70 min train).** Train EqM on top-K GPT-2 / Qwen2.5 log-simplex sequences
with context-kernel-style hidden-state conditioning, contrastive
hinge loss, and the existing 25%-span-corruption pipeline. Report:
- Sequence AUROC and per-token AUROC vs the GP prototype (target: match
  ≥ 0.99 sequence; meet or exceed 0.97 per-token).
- Per-position uncertainty signals: `‖∂E/∂x_t‖`, divergence-trace,
  and *NAG basin-drift* (§3.C).
- Same healer ensemble; report LM log-prob recovery vs the GP healer.

This is the **derisk experiment**. If it doesn't match the GP, the EqM
energy doesn't have enough localisation precision — fall back to the
prototype.

**(F2) TriviaQA semantic hallucination (~2 weeks).** This is where the
prototype hits a wall (GP 0.45–0.55, SE 0.82). The thesis: contrastive
training on **semantic** confusor pairs (real answer vs distractor from
the same TriviaQA item) — *not* random-token corruption — should give
the EqM auditor a different inductive bias than the GP got. Compare:
- EqM-energy auditor trained on TriviaQA confusor pairs, evaluated on
  held-out TriviaQA.
- Same auditor evaluated on WikiText-2 corruption (cross-domain
  generalisation test).
- Spilled Energy as the unconditional baseline (still 0.82).

A result of EqM ≥ 0.85 on TriviaQA *and* ≥ 0.95 on WikiText-2 cross-
domain would be **the unique-value-add for the writeup** — the prototype
can't currently do both simultaneously.

**(F3) Auditor-driven generation (~1 week, conditional on F1).** Use the
trained EqM-Euler sampler to *generate* text under the energy that scores
LLM tokens. The healer infrastructure is the bridge: start from
σ-noise, run Euler-γ on the *same* energy that audits LLM outputs, project
to the LM's vocabulary at decode. This is the *literal* "auditor-is-also-
generator" claim from §D.7 — and one the GP prototype cannot make (the
GP-driven generator hits text8 BPC ≈ 7, dominated by the GP's smoothness
prior).

#### E.5 Honest assessment vs the prototype

| | Prototype (GP + product kernel) | F1 (EqM-energy) |
|---|---|---|
| WikiText-2 detection | **proven 0.999 / 0.996** | parity is the bar |
| Vocabulary-constrained healing | **proven, comparable to GPT-2 nucleus** | reuses infra |
| TriviaQA generalisation | **0.45–0.55 (under-performs SE)** | open question — F2 thesis |
| Auditor-also-generates text8 | broken (BPC ≈ 7) | **the unique EqM advantage** |
| Spilled Energy comparison | already integrated | must include |

**Recommendation:** F1 is required to validate that EqM as auditor doesn't
*regress* the prototype's results. F2 is the *novel-result* gamble — the
case where EqM's contrastive-trained energy could give different
inductive bias than GP variance does. F3 is the unique structural claim
EqM can make and the GP cannot.

If F1 passes parity but F2 is no better than SE, the writeup arc is
"EqM-energy is a drop-in replacement for the SVGP auditor with no
inducing-point overhead, and additionally generates text8 unconditionally
— the SVGP cannot." That alone is publishable.

---

## 4. Loss design — Hilbert and alternatives

The user explicitly asked whether *enhancing MSE with a Hilbert metric* is
viable. The honest answer:

### 4.1 Hilbert as a *primary* FM loss is structurally wrong

Three reasons (each from §1.3):

1. 2-sparse subgradient → catastrophic SNR for a transformer with
   `d_model · K` outputs.
2. Soft-Hilbert at `α` large enough to track the hard metric collapses LSE
   back to a hard max in the regime where CLR residuals are O(12); at `α`
   small enough for smoothness, it approximates `(2/α)·log K` plus a
   second-order shape term with no per-coordinate signal.
3. Projective invariance discards the magnitude of `u_tgt = c(γ)(x_0−x_1)`,
   which is the part the integrator needs.

This matches the historical record in `SESSION_SUMMARY.md` §2.1.

### 4.2 Hilbert as a *small auxiliary* with the all-fixes config: probably viable

The conjecture in `SESSION_SUMMARY.md` §5.4 — "with CE anchoring per-token
attractors and γ-importance pushing mass to γ ≈ 1, Hilbert's 2-sparse
subgradient may no longer be crippling" — holds *only* in the auxiliary
regime. At `λ_H ≈ 0.05`, Hilbert contributes ~5% of the gradient norm on
~7% of coordinates per step, which is the regulariser regime, not the
signal regime.

What it would *do*: enforce alignment of the argmax/argmin coordinates of
the residual, i.e. the coordinates that drive sampling outcomes. This is
the simplex analogue of the well-known "MSE + L∞" recipe — dense gradient
from MSE, worst-case control from the sup-norm.

### 4.3 Loss recipes worth trying, in priority order

Each line: `name — formula — λ — γ-mask — α-schedule (if any)`.

1. **MSE + CE + soft-Hilbert (primary aux candidate).**
   `L = MSE(grad_g, u_tgt) + λ_CE · CE(softmax(pred_x1), token_ids)·𝟙[γ≥0.5]
        + λ_H · soft_hilbert(grad_g, u_tgt)`
   - `λ_H = 0.05`, applied at all γ.
   - `α` annealed from 1.0 → 3.0 over `epochs/2` (start in the truly
     smooth regime, end where it tracks hard Hilbert closer).
2. **MSE + CE + Hilbert on implied-`x_1` (target-aligned variant).**
   `L = MSE + λ_CE · CE + λ_H · soft_hilbert(pred_x1, x1)·𝟙[γ≥0.5]`
   - `λ_H = 0.1`, `α = 2` fixed. Applies Hilbert to the *reconstruction*
     not the velocity; directly aligns argmax/argmin of pred and true
     simplex points.
3. **MSE + CE + cosine-on-CLR (control / cheaper alternative).**
   `λ_cos = 0.1`. Cosine gives projective-invariance flavour with a
   *dense* gradient. Useful for isolating "scale-invariant alignment"
   from "sup-norm pressure."
4. **MSE + CE + JS(softmax(pred_x1), one-hot)** at `λ_JS = 0.1`,
   γ ≥ 0.5. Symmetric KL surrogate; bounded; clean gradient. If this
   beats Hilbert, the win was "second info-geometric anchor on target,"
   not "sup-norm regularisation."
5. **MSE + CE + sliced-Wasserstein on softmax(pred_x1) vs one-hot.**
   Lower priority; expensive, unlikely to beat JS for one-hot targets.

### 4.4 Sanity test — a 10-minute decision

Train one epoch on a 50k subset with two arms:
- A: existing MSE+CE config (= `runs/eqm_data50k_ep5_v2/`).
- B: identical + recipe #1 above (`λ_H = 0.05`, soft-Hilbert α = 2).

Compare at epoch 1: (i) val FM MSE, (ii) sampled-256 unigram-KL vs corpus
(the canonical mode-collapse probe), (iii) per-coordinate mean |residual|
on the argmax coord of the target. If (i) within 5% of A *and* either (ii)
or (iii) drops materially, Hilbert-as-aux is helping — graduate to a full
sweep cell. If (i) regresses by >10%, kill it.

---

## 5. Recommended concrete experiments for the next-next session

Ordered by leverage. I am *not* asking the current cluster session to
deviate from `CLUSTER_TRAINING_PLAN.md`; this is for after W1–W5 settle.

| # | Experiment | Compute | Lever |
|---|---|---|---|
| 1 | **Logit-KL Flow Matching baseline** at `data_50k_ep5` platform | ~70 min train + 5 min eval | Approach A — closes the ~9× DFM-vs-EqM gap if the Stark-style support-shrinkage is the cause |
| 2 | **W3 OOD harness extension**: add divergence-uncertainty trace, NAG basin-drift, valid-permutation control | ~15 min eval per checkpoint, no training | Approach C — uniqueness signal vs DFM |
| 3 | **E1 — EqM-as-residual-EBM-on-DFM, text8** (LLM-auditor derisk) | ~70 min train + 30 min eval | Approach D §D.5 — derisks the auditor framing on existing infra |
| 4 | **Hilbert-aux sanity test** (recipe #1) | ~15 min training | Loss design — viable / kill decision in &lt; 1hr |
| 5 | **Express our DFM and EqM in BPC**, overlay on the SFM table | ~30 min | Writeup defensibility |
| 6 | **F1 — EqM-energy auditor on WikiText-2 (parity check vs GP/SE)** | ~3 days dev + 70 min train | Approach E §E.4 — derisks the EqM-as-auditor claim against the existing GP prototype's 0.999 sequence-AUROC and Spilled Energy's 0.998 |
| 7 | **F2 — TriviaQA semantic-hallucination auditor with confusor-pair contrastive training** | ~2 weeks | Approach E §E.4 — the *novel-result gamble*: prototype caps at SE 0.82; GP underperforms |
| 8 | **F3 — Auditor-driven generation: EqM-Euler sampler under the auditor energy + LM vocabulary constraint** | ~1 week (after F1) | Approach E §E.4 — the *unique* structural claim EqM can make and SVGP cannot (prototype's BayesianGenerator hits text8 BPC ≈ 7) |
| 9 | **SFM-style `√p` parameterisation** | ~3 days dev + 70 min train | Approach B — geometry contingency if Approach A doesn't close the gap |

(1)+(2)+(3) produce the strongest *self-contained* publishable arc in &lt; 2
days of cluster work. (6)+(8) produce the strongest *LLM-auditor*
publishable arc, building on the prototype rather than redoing it. (4) is
cheap insurance. (5) is non-negotiable for writeup defensibility. (7) is
the high-risk semantic-hallucination thesis. (9) is the geometry
contingency if Approach A fails.

---

## 6. Honest assessment for the writeup

Given the 12-hour cluster budget and W1–W5 already running, the *most
plausible* publishable claim is **not** "matches DFM on KL_bi" — that
requires changing geometry (Dirichlet/Fisher), not just the sampler. The
literature is consistent on this: Stark, Davis, Cheng, Sevriugov all
identified that flat-simplex linear FM has structural problems at K=27 and
*all four* pivoted to a different parameterisation.

**2026-05-07 revision.** The original two-pronged arc below assumed W1
would close most of the gap and W3 would deliver a unique sequence-
level OOD signal. Both assumptions are now falsified — see the §0a
update block. The revised arc:

1. **Mechanism paper / negative-result-with-diagnosis (strengthened).**
   *Five* independent negatives now triangulate the diagnosis: epoch
   scaling (P1), backbone scaling (P2), data lever (P3 ✱ best),
   γ-conditioning (P4), factorised bigram NLL (P5), the W1 sampler swap,
   and the W4 FMonCLR triangulation. Together they localise the gap to
   "continuous-on-simplex with a velocity field that the trained
   network represents as a *conservative gradient*" — *not* capacity,
   *not* epoch budget, *not* the integrator (NAG vs Euler is a wash via
   W1.use_grad), *not* time-conditioning, *not* removing the
   conservative-grad indirection (W4). The single remaining unexplored
   lever is *geometry* (Logit-KL / Fisher-Rao), exactly what Stark and
   Cheng predicted.
2. **EqM's per-position uncertainty as a *complementary* (not
   dominant) signal.** W3 quantifies clean-vs-substitution_0.5 at
   `U_pos_mean` AUC = 0.987, vs DFM's denoiser proxy at AUC = 1.000.
   Within-noise on substitution; *worse* on shuffle (0.525 vs 0.99);
   *parity* on uniform random (1.00 each). The publishable framing is
   "EqM's energy admits per-position localisation by autograd; DFM
   gets sequence-level scoring from its denoiser logits; the two
   methods produce *different* signals from the same underlying
   model." This is honest, defensible, and weaker than the original
   §3 framing — but it is what the data supports.
3. **The auditor framings (§3.D / §3.E) survive untouched** because
   they were always per-token. F1 (EqM-as-residual-EBM-on-DFM,
   text8 char-level) is now the *first* forward experiment that does
   not double-bet on results we already have. F2/F3 ride on F1.

**The three prongs are mutually independent**: the auditor claim does
not require beating DFM on KL_bi or on text8 OOD; the mechanism
diagnosis does not require an OOD positive; and the per-position
localisation does not require closing the KL gap. The worst-case
reviewer complaint is now "your text8 sequence-level OOD numbers
under-perform DFM" — true; mitigation is the per-position prong plus
the auditor extension where DFM and EqM both serve as feature
extractors for a separately-trained energy.

This is the writeup the existing experiments (with Approach A added,
and with the §3.D / §3.E auditor framings as the *forward* novel
contribution) can honestly support.

---

## 7. Open questions / risks the next session must keep in mind

- **W1 sign convention.** *Resolved 2026-05-07.* `CLUSTER_TRAINING_PLAN.md`
  notes the Euler step could be `x ← x − h·v` or `x ← x + h·v` depending
  on whether the velocity convention is data→noise or noise→data. The
  existing `_eqm_loss` has `u_tgt = c(γ)(x0 − x1)` so `f` regresses to
  *noise minus data*; step toward data is `x ← x − h·v`. The W1 sweep
  used the correct sign — the negative result is not a sign-flip artifact.
  See `runs/DECISION_LOG.md` 2026-05-07 17:25 UTC for the post-hoc sweep.
- **Energy at γ ≠ 1 may not be calibrated.** *Open.* The OOD harness
  should report AUROC at multiple γ (per Mahmood 2020) rather than a
  single number. The W3 results above are at γ ≈ 1; whether
  multiscale stacking helps the per-position signal on shuffle (the
  unsolved hard contrast) is the natural next ablation.
- **Second-order autograd cost.** `create_graph=True` through a 100M-param
  Transformer at L=40 is ≈10–15 GB extra memory. The divergence-uncertainty
  trace ([2605.00941](https://arxiv.org/abs/2605.00941)) is the
  production-grade fallback when memory budget tightens, and it
  generalises to FMonCLR / Logit-KL where there is no energy.
- **MLE-flow OOD failure mode.**
  [Kirichenko 2020](https://arxiv.org/abs/2006.08545) showed flows can
  rank simple OOD *higher* density than complex ID. EqM is not MLE-trained,
  but the same pathology — rewarding low-complexity sequences — should be
  probed by including a constant-character baseline (`"aaaa…"`,
  `"      "`) in the OOD harness.
- **The `c(γ=1) = 0` trap (already documented).** Setting γ=1 at sample
  time flattens the field. Phase 4's collapse confirmed this. Any new
  sampler must walk γ from 0 to 1, not park at 1.

---

## 8. References

### Continuous flow / diffusion on the simplex
- [Stark et al. 2024 — Dirichlet Flow Matching](https://arxiv.org/abs/2402.05841) — proves linear FM on simplex is broken at large K.
- [Davis et al. 2024 — Fisher Flow Matching](https://arxiv.org/abs/2405.14664) — Fisher-Rao geodesics on √p sphere.
- [Cheng et al. 2024 — Statistical Flow Matching (SFM)](https://arxiv.org/abs/2405.16441) — text8 1.39 BPC; benchmark we should overlay.
- [Floto et al. 2023 — Diffusion on the Probability Simplex](https://arxiv.org/abs/2309.02530) — small categorical only.
- [Sevriugov & Oseledets 2024 — Logit-KL Flow Matching](https://arxiv.org/abs/2411.16821) — beats DFM on TinyStories / FineFineWeb. **Approach A.**

### Discrete-token diffusion baselines
- [Gat et al. 2024 — Discrete Flow Matching (DFM)](https://arxiv.org/abs/2407.15595)
- [Lou et al. 2024 — SEDD (best paper ICML 2024)](https://arxiv.org/abs/2310.16834)
- [Sahoo et al. 2024 — MDLM](https://arxiv.org/abs/2406.07524)
- [Shi et al. 2024 — MD4 / GenMD4](https://arxiv.org/abs/2406.04329)
- [Austin et al. 2021 — D3PM](https://arxiv.org/abs/2107.03006)
- [Savinov et al. 2021 — SUNDAE](https://arxiv.org/abs/2112.06749)

### Logit / embedding-space diffusion for language
- [Li et al. 2022 — Diffusion-LM](https://arxiv.org/abs/2205.14217)
- [Han et al. 2022 — SSD-LM](https://arxiv.org/abs/2210.17432)
- [Mahabadi et al. 2023 — TESS](https://arxiv.org/abs/2305.08379)
- [Dieleman et al. 2022 — CDCD](https://arxiv.org/abs/2211.15089)
- [Gulrajani & Hashimoto 2023 — Plaid 1B (text8 1.12 BPC)](https://arxiv.org/abs/2305.18619)
- [Chen et al. 2022 — Bit Diffusion](https://arxiv.org/abs/2208.04202)
- [Al-Rfou et al. 2018 — AR Transformer T64 (text8 1.13 BPC)](https://arxiv.org/abs/1808.04444)

### Hilbert / Aitchison geometry on the simplex
- [Nielsen & Sun 2019 — Clustering in Hilbert simplex geometry](https://arxiv.org/abs/1704.00454)
- [Nielsen 2023 — Non-linear Embeddings in Hilbert Simplex Geometry](https://proceedings.mlr.press/v221/nielsen23a/nielsen23a.pdf)

### Energy-based / score-based OOD detection
- [Wang & Du 2025 — Equilibrium Matching (EqM)](https://arxiv.org/abs/2510.02300) — the codebase's parent paper.
- [Liu et al. 2020 — Energy-based OOD detection](https://arxiv.org/abs/2010.03759)
- [Grathwohl et al. 2019 — JEM](https://arxiv.org/abs/1912.03263)
- [Du & Mordatch 2019 — Implicit Generation with EBMs](https://arxiv.org/abs/1903.08689)
- [Mahmood et al. 2020 — Multiscale Score Matching for OOD](https://arxiv.org/abs/2010.13132)
- [Kirichenko et al. 2020 — Why Normalizing Flows Fail at OOD](https://arxiv.org/abs/2006.08545)
- [Bakhtin et al. 2020 — Residual EBMs for Text (JMLR)](https://arxiv.org/abs/2004.10188)
- [Deng et al. 2020 — Residual EBMs for Text Generation (ICLR)](https://arxiv.org/abs/2004.11714)
- [Xu et al. 2025 — Energy-Based Diffusion Language Models (ICLR 2025)](https://arxiv.org/abs/2410.21357)
- [Balcerak et al. 2025 — Energy Matching](https://arxiv.org/abs/2504.10612)
- ["Divergence is Uncertainty"](https://arxiv.org/abs/2605.00941) — closed-form posterior covariance for any FM velocity. Gradient-free OOD proxy.
- [Mitchell et al. 2023 — DetectGPT](https://arxiv.org/abs/2301.11305)

### LLM-auditor / verifier / process-reward models (Approach D)
- [Cobbe et al. 2021 — GSM8K verifier](https://arxiv.org/abs/2110.14168)
- [Lightman et al. 2023 — Let's Verify Step by Step (PRM800K)](https://arxiv.org/abs/2305.20050)
- [Leviathan et al. 2023 — Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Tae et al. 2025 — TESS-2](https://arxiv.org/abs/2502.13917)
- [Zhu et al. 2025 — FlowRL](https://arxiv.org/abs/2509.15207)
- [Translate Policy to Language: FM Rewards for LLM Explanations](https://arxiv.org/abs/2502.12530)
- [Xue et al. 2021 — ByT5 (byte-level tokenizer for char-FM)](https://arxiv.org/abs/2105.13626)

### LLM uncertainty-quantification benchmarks the auditor must beat
- [Kuhn et al. 2023 — Semantic Uncertainty / Entropy](https://arxiv.org/abs/2302.09664)
- [Kossen et al. 2024 — Semantic Entropy Probes (SEP)](https://arxiv.org/abs/2406.15927)
- [Bakman et al. 2024 — MARS](https://arxiv.org/abs/2402.11756)
- [Kadavath et al. 2022 — Language Models (Mostly) Know What They Know](https://arxiv.org/abs/2207.05221)
- [Manakul et al. 2023 — SelfCheckGPT](https://arxiv.org/abs/2303.08896)
- [Shi et al. 2023 — Min-K%-Prob](https://arxiv.org/abs/2310.16789)
- [Zhang et al. 2024 — Min-K%++](https://arxiv.org/abs/2404.02936)
- [Semantic Energy 2025](https://arxiv.org/abs/2508.14496)
- [Spilled Energy in LLMs (Minut, Dewidar, Masi — ICLR 2026)](https://arxiv.org/abs/2412.10770) — **the forced baseline**: zero-cost, single-forward-pass, 0.998 sequence AUROC on WikiText-2 corruption.

### Sparse-variational GP / Bayesian uncertainty (Approach E prototype)
- [Titsias 2009 — Variational Learning of Inducing Variables in Sparse GPs](https://proceedings.mlr.press/v5/titsias09a/titsias09a.pdf) — the SVGP prototype's foundation.
- [Hensman, Fusi, Lawrence 2013 — Gaussian Processes for Big Data](https://arxiv.org/abs/1309.6835) — stochastic variational extension.
- [Wilson & Adams 2013 — Spectral Mixture Kernels](https://arxiv.org/abs/1302.4245) — relevant for product-kernel design.

### Open code worth reading
- [SFM](https://github.com/ccr-cheng/statistical-flow-matching)
- [Dirichlet FM](https://github.com/HannesStark/dirichlet-flow-matching)
- [Fisher Flow](https://github.com/olsdavis/fisher-flow)
- [SEDD](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion)
- [MDLM](https://github.com/kuleshov-group/mdlm)
- [Plaid](https://github.com/igul222/plaid)
