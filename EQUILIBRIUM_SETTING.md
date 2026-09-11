# The equilibrium setting — what it is, where it comes from, and why it fails for text

**Purpose.** A self-contained presentation of the *equilibrium* setting for
generative modeling: its definition, its lineage in the literature (it is **not**
a coinage of the Equilibrium Matching paper), and a precise argument for why it
fails on categorical text. Intended as a background / "related settings" section
for the capstone. The mechanistic deep-dive (shell-supported field, no-op
recovery, the model-class taxonomy, SFM≠EBM) lives in
**`NOTE_EQUILIBRIUM_FAILURE_CLASS.md`**; the failure derivations and numbers are
in `NOTE_WHY_EBM_INIT_STUCK.md` and `NOTE_WHY_UNCONDITIONAL_FAILS.md`.

**Capstone scope note (updated 2026-08-20).** The capstone paper is written
(`../capstone-paper/`); its §The Negative Result and appendix `app:theory` are
the published form of this material, and this document's lineage argument
(Hopfield/Boltzmann/DEQ) was not carried into it — it survives only here.
Documenting *why* an approach fails is a valid outcome. The thesis here is therefore deliberately
negative-but-grounded: the equilibrium setting is a classical, well-studied
framework whose known limitations *predict* the text failure, and this project
*instantiates and localizes* that failure on character-level text8. We inherit
the theory and add the evidence.

---

## 1. What "the equilibrium setting" means

A generative model is in the **equilibrium setting** when its inference reaches a
**fixed point / stationary state of a single autonomous (time-free) operator**,
and the data distribution is identified with the **equilibria of an (implicit)
energy** `E(x)`:

- **Object:** a static scalar energy `E(x)` (often implicit), with data at its
  low-energy configurations.
- **Inference:** reach equilibrium — either *deterministically* by descending
  `∇E` to a fixed point (optimization-driven sampling), or *stochastically* by
  running MCMC to the stationary Gibbs/Boltzmann measure `p(x) ∝ e^{−E(x)}`.
- **No time index:** there is one field, queried repeatedly; generation is "settle
  to equilibrium," not "follow a schedule."

Contrast — the **transport (non-equilibrium) setting**: a *time-indexed* family
`v(x, t)` (or a CTMC rate matrix) pushes mass along a prescribed path from a
tractable source to the data over `t: 0→1`. There is no fixed point and no
stationary measure to sample; generation is finite-time transport. Flow matching,
diffusion/score models, and discrete/Dirichlet flow matching are all in this
setting.

> **The one bit that separates them:** does generation route through a
> **stationary object** (an energy's equilibrium) or along a **time-indexed
> transport path**? That bit predicts success on categorical data (§3).

Where Equilibrium Matching (EqM) sits: it is an **implicit energy-based model**
whose sampler is **deterministic descent to a fixed point** — i.e. senses (a)+(b)
of §2 below. Its descent sampler is in fact *weaker* than full Gibbs sampling: it
seeks a **mode**, not a draw from `e^{−E}`.

---

## 2. The lineage — this is not a coinage

The Equilibrium Matching paper itself locates the work in the established
energy-based tradition. Its subtitle is *"Generative Modeling with **Implicit
Energy-Based Models**,"* and the abstract frames EqM as offering *"a tighter
bridge between flow and energy-based models"* by replacing *"time-conditional
velocities with a unified equilibrium landscape"* (Wang & Du 2025). The *method*
and the *name* "Equilibrium Matching" are new; the **setting** is decades old.

"Equilibrium" carries three established, technically distinct meanings, and EqM
sits at their intersection:

| Sense | Inference = | Canonical lineage |
|---|---|---|
| **(a) Thermodynamic / Gibbs equilibrium** | sample the stationary `p ∝ e^{−E}` by MCMC | Boltzmann machines (Ackley, Hinton & Sejnowski 1985 — *named for the Boltzmann equilibrium distribution*); energy-based learning (LeCun et al. 2006); modern EBMs (Du & Mordatch 2019) |
| **(b) Attractor / associative-memory equilibrium** | descend `E` to an **attractor** (stored pattern) | Hopfield 1982; modern Hopfield networks (Ramsauer et al. 2020) |
| **(c) Fixed-point-of-an-autonomous-map equilibrium** | solve `z* = f(z*)` | Deep Equilibrium Models (Bai, Kolter & Koltun 2019); Equilibrium Propagation (Scellier & Bengio 2017) |

EqM is an *implicit EBM* (a) whose inference is *deterministic descent to a fixed
point* (b)/(c). Sense (b) is the one to keep in mind for **recovery**: cleaning a
corrupted input by descending an energy to the nearest stored pattern **is**
the Hopfield associative-memory task — and the classical Hopfield failure modes
(spurious attractors, capacity limits, "returns a blend / the corrupted input")
are exactly what we observe.

The transport setting was, historically, introduced *in explicit opposition* to
the equilibrium one: diffusion models were proposed as *"Deep Unsupervised
Learning using **Nonequilibrium** Thermodynamics"* (Sohl-Dickstein et al. 2015),
and annealed Langevin sampling (Song & Ermon 2019) exists **because** un-annealed
equilibrium sampling fails to mix. "Non-equilibrium / transport beats equilibrium
sampling" is, in other words, already received wisdom — there is now a formal
statistical-thermodynamics literature making the distinction precise for
generative models (Ambrogioni 2023; Sclocchi et al. 2024; and the position paper
"Beyond Equilibrium" 2025).

---

## 3. Why it fails for text

There is no single named theorem "equilibrium models don't work." Instead there
is a **convergence of established results**, all of which bite when the data is
**combinatorially-multimodal / "low-temperature" discrete** — which is exactly
what categorical text is.

### 3.1 What makes text the worst case

Character-level text8 (K=27) at window length L has support on `K^L ≈ 10^57`
near-vertex configurations of the probability simplex. As a Gibbs measure this is
a **low-temperature Potts-like model**: astronomically many sharp modes (valid
sequences), separated by large energy barriers, with a marginal (unigram) mean
`μ₁` that is a smooth interior point **no real sequence is ever near** (the
`Θ(√L)` distance is the quadrature Aitchison-norm aggregate over positions —
the per-position Hilbert distance is a constant independent of L; see the
paper's appendix §sqrt-L and `NOTE_WHY_EBM_INIT_STUCK.md` §4).

### 3.2 The four established results that apply

1. **Metastability / Eyring–Kramers.** The expected time to cross an energy
   barrier of height `ΔE` at temperature `T` scales as `exp(ΔE/T)`; the mixing
   time of local MCMC is exponential in the number and height of barriers between
   modes (Kramers 1940; Bovier & den Hollander 2015). ⇒ **equilibrium *sampling*
   of a many-mode Gibbs measure is intractable** by local dynamics.
2. **Low-temperature discrete sampling hardness.** Sampling Potts/Ising-type
   discrete Gibbs measures below criticality has provably slow Glauber mixing;
   MAP is NP-hard and the partition function `Z` is #P-hard. Text-as-categorical
   is a sharply peaked K-state model — squarely in the known-hard regime.
3. **Descent ≠ sampling, plus Hopfield pathologies.** Deterministic gradient
   descent returns MAP / local minima, not draws from `e^{−E}`. Hopfield theory
   already catalogues the failures: spurious attractors, capacity limits, and
   "returns a blend or the corrupted input." Our **unigram collapse** (generation)
   and **no-op recovery** (descent returns the corrupted string) are textbook
   attractor pathologies, not novel breakage.
4. **The non-equilibrium escape route is the established fix.** Diffusion was
   introduced as non-equilibrium thermodynamics (Sohl-Dickstein et al. 2015) and
   annealed Langevin (Song & Ermon 2019) was introduced *specifically* to overcome
   the mixing failure of single-temperature equilibrium sampling. Transport models
   never sample a stationary measure: a finite-time ODE/CTMC pushes mass along a
   prescribed path and **never touches the barriers**. This is why every working
   text model is in the transport setting (D3PM / discrete diffusion, SEDD,
   Dirichlet FM, SFM).

### 3.3 How it shows up in this project (pointer)

The general principles above instantiate as a single concrete picture: the trained
conservative field `∇⟨x, f(x)⟩` is **supported only on a thin interior shell** of
the noise→data interpolant (flat near data, unigram-tilted far from it), and the
per-token basins are sub-resolution spikes. Consequently:

- **Generation** (descend from noise) collapses to the unigram peak.
- **Recovery** (descend from a perturbed input) is a **no-op** — the descent
  applies its one rule (sharpen whichever token already dominates) where that
  rule does not apply, and an isotropic perturbation starts it outside the
  trained radius (paper §Energy landscape; the earlier "flat shoulder, ∇E ≈ 0"
  mechanism did not survive measurement — the near-data field is an aggressive
  sharpener). (This
  supersedes earlier "recovery works" claims; only a marginal, non-competitive
  bump survives at mid-perturbation α∈[0.5,0.7].)
- **Only *evaluation* survives** — OOD detection via a discriminative head on
  frozen features uses no descent. *Iteration fails, evaluation survives.*

Full derivation, numbers, and the model-class taxonomy:
**`NOTE_EQUILIBRIUM_FAILURE_CLASS.md`** (§B, §C); `NOTE_WHY_EBM_INIT_STUCK.md`;
`NOTE_WHY_UNCONDITIONAL_FAILS.md`.

---

## 4. The caveat that keeps the claim honest

The negative result is **not** "equilibrium models don't work." EqM is a
competitive *image* generator, and image EBMs work with enough compute, because
continuous image data is a connected, low-multimodality manifold whose marginal
mean sits near the typical set. The established hardness bites only at the
**intersection**:

> **equilibrium inference × combinatorially-multimodal (categorical) data.**

Claim generality for *that intersection* — which is well-grounded — never for
equilibrium models per se.

| | Continuous / unimodal-ish data | **Categorical / `K^L`-modal data** |
|---|---|---|
| **Equilibrium** (autonomous fixed point) | ✅ EqM-on-images; image EBMs (long Langevin) | ❌ EqM, SFLMEBM, deterministic-descent EBM, DEQ-generative, unannealed single-level score |
| **Transport** (time-indexed schedule) | ✅ flow matching; diffusion | ✅ D3PM / discrete diffusion, SEDD, Dirichlet FM, SFM |

---

## 5. One-paragraph summary

The "equilibrium setting" is a classical, named framework — Boltzmann-machine /
energy-based sampling (Gibbs equilibrium), Hopfield associative memory (attractor
equilibrium), and deep-equilibrium / equilibrium-propagation fixed points — and
Equilibrium Matching explicitly places itself in it ("implicit energy-based
models," "a tighter bridge between flow and energy-based models"). For categorical
text the setting fails not by accident but by a convergence of established results:
sampling a low-temperature, combinatorially-multimodal Gibbs measure is
exponentially slow (metastability) or hard (discrete sampling), deterministic
descent returns modes/attractors rather than samples (Hopfield pathologies), and
the field's own training signal supports only a thin interior shell. The fix is not
a better energy, sampler, or geometry — it is leaving the equilibrium setting for
time-indexed transport, which is precisely what diffusion was invented to do
(*non-equilibrium* thermodynamics) and what every working text generator does.

---

## Sources

*ArXiv IDs from memory are marked "(verify)"; confirm before formal citation.*

**Focal methods**
- Wang, R. & Du, Y. (2025). *Equilibrium Matching: Generative Modeling with Implicit Energy-Based Models.* [arXiv:2510.02300](https://arxiv.org/abs/2510.02300).
- Stark, H. et al. (2024). *Dirichlet Flow Matching with Applications to DNA Sequence Design.* [arXiv:2402.05841](https://arxiv.org/abs/2402.05841).
- Cheng, C., Li, J., Peng, J. & Liu, G. (2024). *Categorical Flow Matching on Statistical Manifolds* (Statistical Flow Matching). [arXiv:2405.16441](https://arxiv.org/abs/2405.16441).

**The equilibrium lineage (energy-based, attractor, fixed-point)**
- Hopfield, J. J. (1982). *Neural networks and physical systems with emergent collective computational abilities.* PNAS 79(8):2554–2558.
- Ackley, D., Hinton, G. & Sejnowski, T. (1985). *A Learning Algorithm for Boltzmann Machines.* Cognitive Science 9(1):147–169.
- LeCun, Y., Chopra, S., Hadsell, R., Ranzato, M. & Huang, F. (2006). *A Tutorial on Energy-Based Learning.* In *Predicting Structured Data*, MIT Press.
- Scellier, B. & Bengio, Y. (2017). *Equilibrium Propagation: Bridging the Gap between Energy-Based Models and Backpropagation.* Frontiers in Computational Neuroscience. [arXiv:1602.05179](https://arxiv.org/abs/1602.05179) (verify).
- Bai, S., Kolter, J. Z. & Koltun, V. (2019). *Deep Equilibrium Models.* NeurIPS. [arXiv:1909.01377](https://arxiv.org/abs/1909.01377) (verify).
- Du, Y. & Mordatch, I. (2019). *Implicit Generation and Modeling with Energy-Based Models.* NeurIPS. [arXiv:1903.08689](https://arxiv.org/abs/1903.08689) (verify).
- Ramsauer, H. et al. (2020). *Hopfield Networks is All You Need.* ICLR 2021. [arXiv:2008.02217](https://arxiv.org/abs/2008.02217) (verify).

**Sampling hardness / metastability (why equilibrium sampling is intractable here)**
- Kramers, H. A. (1940). *Brownian motion in a field of force and the diffusion model of chemical reactions.* Physica 7(4):284–304.
- Bovier, A. & den Hollander, F. (2015). *Metastability: A Potential-Theoretic Approach.* Springer.

**The non-equilibrium / transport counterpart (the established escape route)**
- Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N. & Ganguli, S. (2015). *Deep Unsupervised Learning using Nonequilibrium Thermodynamics.* ICML. [arXiv:1503.03585](https://arxiv.org/abs/1503.03585) (verify).
- Song, Y. & Ermon, S. (2019). *Generative Modeling by Estimating Gradients of the Data Distribution* (annealed Langevin). NeurIPS. [arXiv:1907.05600](https://arxiv.org/abs/1907.05600) (verify).
- Austin, J. et al. (2021). *Structured Denoising Diffusion Models in Discrete State-Spaces* (D3PM). NeurIPS. [arXiv:2107.03006](https://arxiv.org/abs/2107.03006) (verify).
- Lou, A., Meng, C. & Ermon, S. (2024). *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution* (SEDD). [arXiv:2310.16834](https://arxiv.org/abs/2310.16834) (verify).
- Lipman, Y. et al. (2022). *Flow Matching for Generative Modeling.* [arXiv:2210.02747](https://arxiv.org/abs/2210.02747) (verify).

**Statistical thermodynamics of generative models (modern equilibrium↔non-equilibrium formalizations)**
- [Nonequilibrium physics of generative diffusion models](https://arxiv.org/html/2405.11932v1) — arXiv:2405.11932.
- [The Statistical Thermodynamics of Generative Diffusion Models: Phase Transitions, Symmetry Breaking, and Critical Instability](https://arxiv.org/pdf/2310.17467) — arXiv:2310.17467.
- [Beyond Equilibrium: Non-Equilibrium Foundations Should Underpin Generative Processes in Complex Dynamical Systems](https://arxiv.org/html/2505.18621v1) — arXiv:2505.18621.

**Project documents**
- `NOTE_EQUILIBRIUM_FAILURE_CLASS.md` — SFM≠EBM, the corrected single failure mode, the model-class taxonomy.
- `NOTE_WHY_EBM_INIT_STUCK.md` — training-time collapse (degenerate Bayes-optimum, the shell argument).
- `NOTE_WHY_UNCONDITIONAL_FAILS.md` — sampling-time collapse (with the §0 recovery correction).
- `EVAL_ASSESSMENT.md` — objectives, evals, and corrected verdicts.
