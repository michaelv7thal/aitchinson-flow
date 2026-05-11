# Why unconditional generation does not work for EqM (and EBMs in general)

A short note on a recurring failure mode in this project: even when an EqM
model recovers cleanly from a perturbed input, its unconditional samples
(start from N(0, σ²I), descend the energy) are gibberish. Why?

## TL;DR

The energy field is well-trained *near the data manifold* (where the FM
regression has signal) and essentially uninformative *far from it* (where
unconditional sampling starts). Gradient-descent-based "sampling" then
just rolls into whichever basin happens to lie under the noise init —
which is almost never a coherent text basin. This is not an EqM-specific
bug; it is a structural property of energy-based generative models that
sample by descending a learned energy.

## Where the energy is trained

EqM trains the conservative gradient `∇_x ⟨x, f(x)⟩` to match the FM
target `c(γ)·(x_0 − x_1)` at points

    x_γ = (1 − γ)·x_0 + γ·x_1,    γ ∈ [0, 1].

The training distribution over points is therefore concentrated on
*line segments between noise and data*. Crucially:

* **At γ ≈ 1**: the model learns `∇E(x_1) ≈ 0` (data is an energy minimum).
* **At γ ∈ (0, 1)**: the model learns gradients pointing roughly noise → data
  along the interpolant.
* **At γ ≈ 0**: `c(γ)·(x_0 − x_1)` has expected value `−x_1/λ` (since
  `E[x_0] = 0`); the model is being told "from random noise, point
  toward the *average* x_1." This is a unigram-mode-collapse signal —
  it doesn't carry per-token basin structure.

The energy at *random points in noise space* — i.e., points the
unconditional sampler actually starts from — is supervised only weakly,
through the γ ≈ 0 endpoint of the FM path. The bulk of useful gradient
information lives at γ > 0.5, around the data manifold.

## What the unconditional sampler sees

NAG-GD sampling starts at `x_0 = σ·N(0, I)`, evaluates `∇E(x_0)`,
and descends. Two things conspire against it:

1. **The gradient at noise init is uninformative.** It's been trained to
   point toward "the average data point," not toward the nearest valid
   sequence. The sampler walks toward the unigram peak (or, with γ
   conditioning, drifts according to whatever `f(x; γ=γ_sample)` produces
   — typically a flat field).
2. **Spurious local minima dominate the noise space.** The energy field
   has many basins. The trained ones (around real x_1) are sharp and
   narrow; everything else is unconstrained and may form spurious
   minima with larger basin volumes than the real ones. NAG-GD rolls
   into the *nearest* basin, which is almost always spurious.

Result: unconditional samples land in basins, but they aren't the right
basins — they're whatever local minima the random init was closest to.

## Why recovery works even though unconditional doesn't

This is the key asymmetry the recovery diagnostic exposes:

* **Recovery starts at `x_0 = x_1 + α·N(0, I)`** — already inside a real
  data basin. The energy field around `x_1` *is* well-trained (γ ≈ 1
  region). NAG-GD has a real gradient to follow and lands back near
  `x_1`. Recovery is essentially solving a *local* problem.
* **Unconditional sampling is a global problem.** It demands the energy
  field to be informative *everywhere*, including in regions that were
  never visited during training. There is no FM training signal in
  those regions, so the field is whatever the network's inductive bias
  produces — generally not what you want.

A useful mental picture: EqM is trained to be a *valid-sequence
attractor* in a small neighborhood of the data manifold. Outside that
neighborhood, it is best understood as untrained.

## The same problem in EBMs more broadly

For a generic EBM with energy `E(x)`, unconditional samples are
supposed to come from the Gibbs distribution `p(x) ∝ exp(−E(x))`, drawn
by MCMC (Langevin, HMC). In high dimensions, two things go wrong:

* **Mixing time grows exponentially with the number of well-separated
  basins.** Real datasets have many modes; an MCMC chain spends most of
  its time in one basin and almost never crosses the energy barriers
  between modes. Single-chain unconditional sampling biases heavily
  toward the basin nearest the chain's init.
* **The partition function `Z` is never computed.** EBMs define `E`
  only up to a constant; without a tractable normaliser, you cannot
  compare absolute basin masses. A spurious basin with low energy and
  large volume looks indistinguishable from a real basin, both to the
  model and to the sampler.

Image EBMs work around this with long Langevin chains (1000+ steps),
persistent contrastive divergence (replay buffers), and sometimes
explicit short-run MCMC with clipping. None of these scale gracefully
to text-shaped data, where the energy landscape is closer to a
combinatorial basin structure than a smooth manifold.

EqM with NAG-GD is a *very weak* substitute for MCMC: it does
deterministic gradient descent and stops at the first stationary point.
That makes it excellent at recovery (one descent into the nearest
basin) and terrible at unconditional sampling (no chain mixing, no
partition-function-aware acceptance, no exploration).

## The text-specific failure: spike basins

For character-level text in CLR features, each token's `x_1` looks
like `[..., +12, ..., 0, ..., 0]` after CLR. Basins around vertices
are *spikes* — extremely deep, extremely narrow, and well-separated by
huge log-ratio gaps. A unigram-trained energy field collapses to one
or two dominant spikes; unconditional samples are sequences of the
high-frequency tokens (`e`, `t`, space) repeated.

The compositional / Dirichlet-thickening recipe widens the spikes
into smoother basins, which restores some recovery capability (Δ@α=0.5
≈ +0.06 vs +0.00 for deterministic). But unconditional sampling still
fails because *spike-vs-smooth-basin is a local property* near the
data manifold; far from the manifold, the field is no better
constrained than before.

## What does work: latent EBMs with a decoder

The right way to do unconditional generation with an EBM is to give
the EBM a tractable latent space and a separate decoder:

* Pretrain a contextual autoencoder so the latent space is
  approximately Gaussian (or VAE-style explicitly).
* Train EqM on the latents — same recipe, different geometry.
* For unconditional sampling: draw `z ~ N(0, I)`, optionally do a few
  NAG-GD steps to land near a latent basin, then decode `z` through
  the AE decoder.

The decoder absorbs the slack: it produces valid text from any
*reasonable* latent point, regardless of whether the EqM's basin
structure has perfectly localised that point. The EBM's job is reduced
to "land somewhere in the data-supported region of latent space" — a
much easier global problem than "land at a specific spiky vertex of
the simplex."

This is the AE-EqM direction the project is exploring. Early evidence
(`runs/ae_d256_l2_z64`, KL_uni 0.006 vs comp_'s 0.66, ~100× lower)
suggests it materially improves unconditional sample quality even
before the decoder takes over the rest of the slack.

## Possible remedies

Ordered roughly by expected impact for *this* project, with the cheapest
sampler-only fixes first and the architectural ones last.

### 1. Switch to a stochastic sampler (Langevin / SDE)

What it is: replace NAG-GD with Langevin dynamics

    x ← x − η·∇E(x) + √(2η·T)·ξ,   ξ ~ N(0, I)

(or annealed Langevin: schedule the temperature `T` from large to small,
running a few Langevin steps at each level). The noise lets the chain
*cross* between basins instead of getting stuck in the first one.

Why it helps unconditional generation: NAG-GD is a deterministic
"find the nearest local minimum" procedure — incapable of exploring
multiple basins from a single init. Langevin introduces stochasticity
that allows the chain to escape spurious minima and find lower-energy
basins. Annealed Langevin (Song & Ermon 2019) is the standard fix in
the image-EBM literature.

What's already in the repo: `src/aitchinson_flow/sampling/sde.py`
implements an SDE sampler (Langevin noise on the conservative gradient,
walks γ from 0 → 1 with noise scale α). The simplex EqM dispatches to
it via `cfg.eqm.sampler="sde"`. **EqMAE does not yet wire it up — only
NAG and Euler are dispatched.** Adding the dispatch is ~20 lines.

Cheapest fix to try; expected to help unconditional KL/H_ratio more
than recovery (which already works without mixing). Not a silver
bullet — Langevin in high-dim text-feature space still has long mixing
times, but it's strictly better than NAG-GD.

### 2. Best-of-N with multiple parallel chains

What it is: instead of one NAG-GD chain, run N chains from independent
noise inits, score each final point by `−E(x)`, return the best (or a
mixture). Trivially parallel.

Why it helps: even without sampler improvements, this approximates
"land in the best of N nearby basins" which catches more of the data
distribution than a single chain. With N=64 and existing NAG, this is
essentially free at our batch sizes.

Risks: still local — N independent chains all biased toward whichever
spurious basins are dense. Doesn't fix the underlying gradient field.

### 3. Persistent contrastive divergence (PCD) at training time

What it is: maintain a buffer of "model samples" across training
steps. At each FM training batch, also evaluate the energy on a few
buffer samples and treat them as negative examples (their energy
should rise). Replenish the buffer with fresh samples occasionally.

Why it helps: the FM regression only constrains the energy along the
noise-to-data path. PCD adds explicit pressure to *raise* the energy
at points the model itself currently considers low (i.e. spurious
basins). This carves out the spurious modes that unconditional
sampling falls into.

Cost: doubles training compute; requires careful buffer management;
known stability issues. The classic image-EBM training trick. Worth
trying if (1) and (2) don't move the needle.

### 4. VAE-style training of the autoencoder (the big one)

What it is: instead of a deterministic AE, train a VAE: encoder
produces `(μ(x), logσ(x))` per position, latent is `z = μ + σ·ε`
sampled per step, loss adds `β·KL(q(z|x) ‖ N(0, I))` to the CE.

Why it helps unconditional generation:
* The KL prior **forces the marginal latent distribution `q(z) ≈ N(0, I)`**.
* That makes unconditional sampling almost trivial: draw `z ~ N(0, I)`,
  decode through the AE → valid text. **No EBM sampling needed for the
  prior.**
* The EBM then only has to refine: add a few NAG-GD steps to land on
  a basin if needed.

This is the Stable-Diffusion-style decomposition: Gaussian latent
prior + (optional) score-based refinement + denoising decoder. For our
problem it directly targets the global-vs-local asymmetry — the
Gaussian prior IS the global solution.

Trade-off: tuning β is the standard VAE balancing act (too high → posterior
collapse, latent uninformative; too low → no Gaussianisation).
β-annealing or free-bits typically required.

### 5. Diffusion-style multi-noise-level training

What it is: train the energy / score field at *all* noise levels, not
just along the FM interpolant. Concretely: sample `t ~ U(0, T)`,
compute `x_t = √(ᾱ_t)·x_1 + √(1 − ᾱ_t)·ε`, and supervise the score
`∇log p_t(x_t)` to match the noise residual (denoising score
matching). Sample by reverse-SDE / DDPM step.

Why it helps: this is exactly the regime where the energy lives
*everywhere* in noise space (because every noise level is supervised),
not just on the FM interpolant. Reverse-diffusion sampling
deterministically transports noise → data through a learned score
field that's well-defined at every noise level. Unconditional
generation is the literal training objective.

Cost: this is a different model family (score-based diffusion, not
EqM). EqM's "implicit energy = ⟨x, f(x)⟩" parameterisation can be
adapted, but you lose some of the explicit-energy / OOD-detection
properties EqM was chosen for. Worth considering if unconditional
generation is the primary goal and the OOD use is secondary.

### 6. GFlowNets (discrete, different paradigm)

What it is: train a flow that samples discrete sequences proportional
to a reward (here: `exp(−E(x))`). Sequential construction step-by-step.

Why it helps: GFlowNets are designed for unconditional sampling from
discrete energy-defined distributions, with explicit credit assignment
across modes. They sidestep the MCMC mixing problem.

Cost: requires re-architecting the model around the GFlowNet
trajectory-balance objective. Substantial change, mature literature.
Listed for completeness; would not be my first choice for this
project.

### Recommended order for this project (formal)

This is the project's official path forward for unconditional generation.
All four steps are queued to run sequentially after the current AE-scaling
sweep finishes. Compute is unmetered, so each step is fully evaluated
before deciding whether the next one is still warranted.

Given where we are (AE-EqM working for recovery, unconditional still
weak):

1. **Wire SDE sampler into EqMAE** (1 hour, minimal risk). Re-evaluate
   unconditional KL_uni/KL_bi against the NAG-GD numbers. If KL drops
   substantially, the EBM has more structure than the deterministic
   sampler was finding.
2. **Best-of-N with NAG-GD or SDE** (1 hour). Cheap secondary check on
   whether single-chain bias is the dominant issue.
3. **VAE-train the AE** (~half a day to implement, then another sweep
   like the current one). The biggest single architectural improvement
   available, and aligned with the "valid text from random latent" goal.
4. **PCD or diffusion-style training** if (1)–(3) leave a gap. These
   are larger commitments and should follow from the data the first
   three produce, not be undertaken speculatively.

## Summary

* EqM (and EBMs generally) train the energy where the data is, and
  leave it unconstrained where the noise is.
* Recovery is a local problem and works as long as the data-side
  basins are well-shaped.
* Unconditional sampling is a global problem and fails because the
  energy at noise init points has no useful supervision.
* MCMC mitigates this in principle, but not in practice for the
  dimension and modality of text-shaped data.
* The fix is structural: separate the "find a coordinate in the
  data-supported region" problem (EBM in latent space, easy) from the
  "produce valid text from a coordinate" problem (decoder, easy
  separately) — i.e. latent EBMs with a denoising decoder.
* Sampler choice matters too: stochastic samplers (Langevin / SDE,
  annealed Langevin) beat NAG-GD for any unconditional use case
  because mixing across basins is essential. The SDE sampler is
  already implemented in this repo for the simplex EqM and just needs
  to be wired into EqMAE.
