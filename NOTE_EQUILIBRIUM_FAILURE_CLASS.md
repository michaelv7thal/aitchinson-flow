# Why the equilibrium / energy-based frame is structurally exclusive with categorical text generation

**Scope.** This note answers three linked questions that recur in the project and
in the capstone paper's "negative EqM / exclusivity" arm:

1. Can Statistical Flow Matching (SFM, arXiv:2405.16441) be placed in an
   equilibrium / energy-based frame, à la Equilibrium (Flow) Matching
   (EqM, Wang & Du 2025, arXiv:2510.02300)?
2. Is the EqM failure on text *generic* — does it apply to any energy-based
   model in the flow-matching setting?
3. **Which class of models does this failure mode apply to?** (the deliverable)

It consolidates and supersedes parts of `NOTE_WHY_EBM_INIT_STUCK.md`
(training-time collapse) and `NOTE_WHY_UNCONDITIONAL_FAILS.md` (sampling-time
collapse). The goal is *understanding the failure*, not repairing EqM — that
line is closed.

---

## 0. Status correction (read first)

`NOTE_WHY_UNCONDITIONAL_FAILS.md` (§"Why recovery works") and `CLAUDE.md`'s
Obj-2 framing claim that EqM/SFLMEBM **recover** cleanly even though they fail
to generate. **This is empirically false and is retracted here.**

- EqM and SFLMEBM also **fail recovery**: energy descent is a near-no-op and
  returns the *corrupted* string essentially unchanged (Δ@α ≈ 0).
- Dirichlet thickening buys only **marginal** recovery in a narrow band
  α ∈ [0.5, 0.7], and it is **not competitive** with standard DirichletFM
  (a transport model).

Consequence: there is **no "local problem that works."** Recovery and
generation are the **same** failure (see §B). The old local-works / global-fails
dichotomy is an artifact of measuring recovery only at small α.

---

## Part A — SFM cannot be placed in an equilibrium / EBM frame

### A.1 What each model is

| | SFM (arXiv:2405.16441) | EqM (arXiv:2510.02300) |
|---|---|---|
| Geometry | Fisher–Rao simplex ≅ positive sphere via π: μ ↦ √μ | flat CLR space `V_d` (sphere variant: SFLMEBM) |
| Path | constant-speed great-circle geodesic `x_t = exp_{x₀}(t·log_{x₀}x₁)` | linear interpolant `x_γ = (1−γ)x₀ + γx₁` |
| Field | **time-dependent** `v(x_t, t) ∈ T_x S`, regressed on geodesic velocity (MSE) | **autonomous** conservative gradient `∇_x⟨x, f(x)⟩` (Eq. 7 Dot-Product variant) |
| Inference | integrate ODE `ẋ = v(x,t)`, `t: 0→1` (transport) | NAG-GD descent to a fixed point of the energy (equilibrium) |
| Likelihood | **exact** CNF via the continuity equation (paper Eqs. 12–14) | implicit, unnormalized; no tractable `log p` |

(Implementations: `src/aitchinson_flow/models/sfm.py`,
`models/eqm.py`, `models/sflm_ebm.py`.)

### A.2 EqM imposes three requirements; SFM violates each independently

**(1) Autonomy.** EqM uses a single static field; "drop the time conditioning"
is *the EqM move* (`sflm_ebm.py` docstring). SFM's field is explicitly
`v(x_t, t)` and inference integrates `t: 0→1`. You can only make it autonomous
by discarding the time-indexed probability path — which is exactly the path the
continuity-equation likelihood (SFM's only deliverable) is computed along.
Autonomy is therefore not a free reframing; it deletes SFM's reason to exist.

**(2) Conservativity — the load-bearing obstruction.**
EqM requires the field to be a (Riemannian) gradient of a scalar.

- *Per data sample*, SFM's target **is** a gradient:
  `unit(log_x x₁) = −∇_x d(x, x₁)` (gradient of geodesic distance).
- *The marginal field SFM learns is generically not.* It regresses onto
  `u_t(x) = 𝔼_{x₁|x_t}[c · unit(log_x x₁)]` — a **posterior-weighted average of
  gradients of different potentials**, with *x-dependent* weights `p(x₁|x)`.
  Writing `u = ∇[∫ p φ] − ∫(∇p)φ`, it is a gradient iff the residual
  `∫(∇p)φ` is curl-free — and there is no reason for that on a curved manifold.

  In **flat Gaussian** FM the residual cancels: the marginal velocity is affine
  in `x` and the score `∇log p_t`, both gradients, so
  `u_t = ∇[½a|x|² + b·log p_t]`. *That Tweedie/score identity is exactly the
  loophole EqM exploits to make conservativity nearly free.* On the **Fisher–Rao
  sphere** there is no Tweedie identity (log/exp are nonlinear, curvature
  introduces holonomy), so the cancellation fails and the marginal field carries
  a genuine divergence-free (rotational) component. **EqM's central assumption is
  violated *more severely* on SFM's manifold than in EqM's own flat setting** —
  the flat-Gaussian conservativity is a non-generic coincidence that curvature
  destroys.

**(3) Equilibrium vs. finite transport.** EqM puts data at the *critical points*
of a static energy and *descends to equilibrium*. SFM's terminal law is a
finite-time transport pushforward read as `argmax(x²)`; it has no fixed-point
structure. Forcing SFM's field "to equilibrium" reproduces the EqM collapse
(§B), as `NOTE_WHY_UNCONDITIONAL_FAILS.md` predicts.

### A.3 Preempting "but every model has an energy"

Trivially `E = −log p` for any density. SFM even gives `−log p_θ` **exactly and
normalized** (its CNF likelihood), whereas EqM's `⟨x, f(x)⟩` is implicit and
unnormalized. The sharp statement:

> SFM has a *better-defined* energy than EqM, yet is *less* of an equilibrium
> model than EqM — because "having an energy" ≠ "training and sampling are
> gradient descent on a single static learned energy." SFM's dynamics are
> transport, not energy descent.

### A.4 You can graft an EBM onto SFM's geometry — but it is a different model

`sflm_ebm.py` is the constructive proof: take the *sibling* sphere flow (SFLM),
drop `t`, define `E(z) = −τ·logsumexp_v⟨h(z), e_v⟩/τ`, sample by Riemannian GD,
optionally regress the conservative Riemannian-gradient FM target
(`cfg.sflm_ebm.lambda_fm > 0`). The same recipe transfers verbatim to SFM's √μ
sphere ("SFM-EBM"). But that object is **not SFM**: it is lossy w.r.t. SFM's
exact likelihood, and it inherits the EqM failure (§B) — SFLMEBM is empirically a
**strong verifier, weak generator**.

**Verdict for the paper (trichotomy):**
1. SFM is a transport / CNF model, categorically not an equilibrium/EBM model
   (fails autonomy, conservativity, equilibrium; conservativity obstruction is
   *provably worse* on the Fisher–Rao sphere — no Tweedie cancellation).
2. An EBM can always be *grafted* onto SFM's geometry (the SFLMEBM construction),
   but it is a different, lossy model — not a reinterpretation of SFM.
3. That grafted SFM-EBM would inherit the text failure (§B) — geometry is not the
   lever.

---

## Part B — The failure mode (corrected): recovery and generation are one failure

### B.1 The field is supported only on a thin interior shell

The trained conservative field is null at *both* ends of the noise→data
interpolant and informative only in a middle slab:

| region | what the field is | mechanism |
|---|---|---|
| γ → 1 (near data) | `∇E ≈ 0` (flat) | `c(γ) → 0` trivializes the FM target (`EBM_INIT_STUCK` §7); field probes give `f(·,γ=1) ≈ 0` |
| γ → 0 (far / noise) | `−c(γ)μ₁` (unigram tilt) | degenerate Bayes-optimum at the no-information point (`EBM_INIT_STUCK` §2) |
| interior shell | usable gradient | the only slab with `I(x₁; x_γ)` large enough to carry per-token signal |

Overlay the **spike-basin** geometry: per-token wells are sub-resolution spikes
with tiny catch-radii, surrounded by flat shoulders.

### B.2 Why recovery is a no-op (and why the α-curve proves it)

A perturbed input `x₁ + α·noise` lands on a **flat shoulder outside any spike** →
`∇E ≈ 0` → descent does not move → the corrupted string comes back.

The decisive tell is the **shape** of the recovery-vs-α curve: it **peaks at the
path center (α ∈ [0.5, 0.7]) and is ~zero at small α.** A genuine "descend from a
nearby basin" story would peak at *small* α and decay outward. The observed
center-peaked curve is the signature that the near-data field is flat and the only
usable gradient is the interior shell — which a *mid-magnitude* perturbation
happens to sample. Dirichlet thickening widens the spikes just enough to give that
shell a faintly coherent gradient → the marginal, non-competitive bump.

Recovery (init near data) and generation (init at noise) are therefore the **same
operation** — descend a shell-supported autonomous field — and fail identically.

### B.3 The real dissociation: *evaluation* survives, *iteration* fails

| Use | Operation | Survives on text? |
|---|---|---|
| OOD / verification | **evaluate** energy or features at a point (1 forward pass + head) | ✅ |
| Recovery | **iterate** descent to move a point | ❌ no-op |
| Generation | **iterate** descent from noise | ❌ collapse |

The OOD win (`BayesLinHead`) reads **frozen DirichletFM features + a
discriminative head** — the native energy `∇⟨x,f⟩` is at chance on the shuffle
axis. So even Obj 3 does not use the equilibrium dynamics; it uses the backbone as
a feature extractor. **The distinctive equilibrium mechanism contributes nothing on
text; only the generic feed-forward backbone does.** That is the sharpest statement
of "EqM-as-a-model-class fails here."

---

## Part C — Which class of models this failure applies to

### C.1 Necessary structural ingredients

The failure is the **intersection** of two properties:

1. **Inference = iterate an autonomous (time/noise-index-free) operator to a fixed
   point** — gradient descent on an energy, Langevin-to-stationarity, denoiser
   fixed-point iteration, deep-equilibrium (DEQ) layers. *This is the load-bearing
   property.*
2. **Target = categorical / combinatorially-multimodal** — a `K^L ≈ 10⁵⁷` cloud of
   barrier-separated spike-modes; the marginal `μ₁` is maximally off-data
   (Hilbert distance `Θ(√L)` from every vertex).

### C.2 Each factor alone is survivable; only the intersection fails

| Inference type | Continuous / unimodal-ish data | **Categorical / `K^L`-modal data** |
|---|---|---|
| **Equilibrium** (autonomous fixed point) | ✅ EqM-on-images; image EBMs (long Langevin) | ❌ **EqM, SFLMEBM, deterministic-descent EBM, DEQ-generative, unannealed single-level score** |
| **Transport** (time-indexed schedule) | ✅ FM; diffusion | ✅ DirichletFM, SFM, D3PM / discrete-FM, SEDD, MDLM |

### C.3 What is load-bearing vs. incidental

- **NOT the "energy/EBM" property.** Conservativity is incidental — a
  *non*-conservative autonomous fixed-point iterator on categorical data fails the
  same way. The gate is **autonomy + fixed-point inference**, not "has an energy."
- **NOT the target's degeneracy.** A non-degenerate target (DSM) on a single noise
  level still fails — there is no schedule to transport along (Song–Ermon's
  original motivation for *annealing*). Conversely a *degenerate* target (flow-map
  CE) **works under transport** (DirichletFM uses exactly that). See the
  signal-class table in `EBM_INIT_STUCK` §7.5: target-degeneracy and the
  one-hot/categorical lift set the *severity / phenotype* (unigram collapse, no-op
  recovery), they do not *create or cure* the failure.
- **NOT the interpolation geometry.** Per-position-distribution paths still fail
  under equilibrium: SFLMEBM is "distribution-space sphere + descent" → weak
  generator. Geometry helps the variance floor and the *likelihood* (the point of
  the positive arm), not the equilibrium failure.

### C.4 The deep reason — stationary measure vs. transport

The one bit that predicts success on categorical data:

> Does generation route through a **stationary / Gibbs measure of an autonomous
> process**, or along a **time-indexed transport path**?

Text's law is a `K^L` barrier-separated point cloud. The stationary-measure route
is intractable on it — deterministic descent is mode-seeking (no-op / collapse),
and MCMC has mixing time exponential in the number of barriers. The transport
route is a finite-time ODE / CTMC that never touches the barriers. That asymmetry
is geometry-independent and is the structural reason transport beats equilibrium on
discrete/categorical data.

---

## Citable one-liner

> The EqM failure is generic to **equilibrium-inference generative models** — those
> whose sampling reaches a fixed point / stationary distribution of a single
> autonomous operator — applied to **combinatorially-multimodal (categorical)**
> data. It is independent of whether the operator is conservative (energy-based),
> of the interpolation geometry (flat CLR or Fisher–Rao sphere), and of the
> training target's degeneracy; those only set severity. The cure is not a better
> energy, sampler, or path — it is abandoning equilibrium inference for
> time-indexed transport, which is what every working text model (discrete-FM,
> SEDD, DirichletFM, SFM) does.

---

## References

**Papers.**
- SFM — *Categorical Flow Matching on Statistical Manifolds*, Cheng et al. 2024, arXiv:2405.16441.
- EqM — *Equilibrium (Flow) Matching*, Wang & Du 2025, arXiv:2510.02300 (Eq. 7 Dot-Product variant).
- Tweedie / score↔velocity identity (flat Gaussian conservativity); Song & Ermon 2019 (annealed Langevin / why single-level descent fails).

**Code.** `models/sfm.py`, `models/eqm.py`, `models/sflm_ebm.py`; OOD head `scripts/ood_bayes_linear.py`.

**Sibling notes.** `NOTE_WHY_EBM_INIT_STUCK.md` (training-time collapse; §7.5 signal-class table), `NOTE_WHY_UNCONDITIONAL_FAILS.md` (sampling-time collapse — **§"Why recovery works" superseded by §0 above**), `EVAL_ASSESSMENT.md` (objectives / verdicts).
