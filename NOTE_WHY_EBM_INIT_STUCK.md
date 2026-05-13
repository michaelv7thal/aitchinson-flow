# Why energy-field learning gets stuck on text from random init

Sibling note to `NOTE_WHY_UNCONDITIONAL_FAILS.md`. That note treats the
*sampling-time* failure (where chains end up); this note treats the
*training-time* failure (why the field has nothing useful to descend on in
the first place).

The two failures are different mechanisms with the same surface symptom
(unigram collapse). Diagnosing the right one matters: training-time
failures cannot be fixed by changing the sampler.

---

## 1. Setup

Per position, the state lives in
$V_d = \{u \in \mathbb{R}^K : \sum_k u_k = 0\}$, $d = K-1 = 26$.
Per window, $\mathcal{X} = V_d^L$ has dimension $L(K-1) = 1040$.
Source $x_0 \sim \mathcal{N}(0, \sigma^2 P)$ ($P$ the projector onto $V_d$).
Data $x_1$ is supported near the $K^L = 27^{40} \approx 10^{57}$ vertex
configurations of the simplex (label-smoothed one-hots).

EqM parameterises a velocity head $f_\theta(x)$ and trains the
**conservative gradient of a bilinear energy** (`eqm.py:240–246`):

$$E_\theta(x) := \langle x, f_\theta(x)\rangle, \qquad
g_\theta(x) := \nabla_x E_\theta(x) = f_\theta(x) + J_{f_\theta}(x)^{\!\top} x.$$

FM regression on the data-to-noise direction:

$$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{\gamma,x_0,x_1}\,
\bigl\| g_\theta(x_\gamma) - c(\gamma)(x_0 - x_1)\bigr\|^2,
\quad x_\gamma=(1-\gamma)x_0+\gamma x_1,$$

with $c(\gamma)=(1-\gamma)/\lambda$ in the default `decay_strategy="linear"`
path (`eqm.py:_c_gamma`).

---

## 2. The Bayes-optimum of $\mathcal{L}_{\text{FM}}$ is the unigram mean-flow

Point-wise, the unique minimiser of MSE is the conditional expectation:

$$g^\star(x_\gamma) \;=\; c(\gamma)\,\mathbb{E}\bigl[x_0 - x_1 \,\big|\, x_\gamma\bigr].$$

When the mutual information $I(x_1;x_\gamma)$ is small (small $\gamma$, or
any $\gamma$ before the model can read structure):

$$\mathbb{E}[x_0-x_1 \mid x_\gamma] \;\longrightarrow\; \mathbb{E}[x_0]-\mathbb{E}[x_1] \;=\; -\mu_1, \qquad \mu_1 := \mathbb{E}[x_1].$$

Because $x_1$ is built from CLR-encoded one-hots,
$\mu_{1,k} = \log p_{\text{uni}}(k) - \overline{\log p_{\text{uni}}}$ —
$\mu_1$ is **the centred log-unigram vector**.

The trivial Bayes-optimum is the constant field
$g^\star(x_\gamma)\approx -c(\gamma)\,\mu_1$, whose potential

$$E^\star(x) \;=\; -c(\gamma)\,\langle\mu_1, x\rangle$$

is **a single tilted plane with one minimum at $x=\mu_1$** — the unigram
peak.

---

## 3. Random init *already sits on this attractor*

Under NTK / lazy init, $f_\theta$ is approximately a Gaussian random
field; locally

$$f_\theta(x) \approx A_0 x + b_0 \;\Rightarrow\; E_\theta(x)\approx x^{\!\top} A_0 x + b_0^{\!\top} x \;\Rightarrow\; g_\theta(x) \approx (A_0+A_0^{\!\top})x + b_0,$$

a **quadratic energy with exactly one critical point**
$x_\star = -(A_0+A_0^{\!\top})^{-1}b_0$.

MSE restricted to this affine class is **convex in $(M,b) := (A_0+A_0^{\!\top}, b_0)$**,
and its OLS solution is exactly the projection of $g^\star$ onto affine
fields:

$$M^\star = \mathrm{Cov}(x_\gamma)^{-1}\mathrm{Cov}(x_\gamma,\,c(\gamma)(x_0-x_1)),\quad b^\star = \mathbb{E}[c(\gamma)(x_0-x_1)] - M^\star\mathbb{E}[x_\gamma].$$

With $\mathbb{E}[x_0]=0$, $\mathbb{E}[x_1]=\mu_1$:

$$\mathbb{E}[x_\gamma]=\tfrac{1}{2}\mu_1,\quad \mathbb{E}[c(\gamma)(x_0-x_1)]\propto -\mu_1.$$

Both the offset and the dominant covariance direction are **aligned
with $\mu_1$**. Gradient flow on $\mathcal{L}_{\text{FM}}$ from random
init is therefore a *monotone contraction onto a single-basin energy
whose minimum is the unigram peak*.

There is **no force in $\mathcal{L}_{\text{FM}}$ that can spontaneously
create extra basins** — the loss is quadratic in $g_\theta$, the
parametrisation is locally affine, and an affine field has a single
zero by linear algebra. Symmetry-breaking would require either a
non-convex loss in $g_\theta$, or a parametrisation rich enough that
the linearisation is non-affine — but during the early "lazy" phase
that's where you are.

---

## 4. Why this is **catastrophic specifically for text**

For continuous data (Gaussian-like images) $\mu_1$ usually sits near
or inside the typical set, so converging to it is benign — you start
near the right place.

For text the support of $x_1$ is $K^L$ delta-like vertices, while
$\mu_1$ is **a smooth interior point of the simplex no data window has
ever been near**. The Hilbert distance from $\mu_1$ to the nearest
vertex configuration is $\Theta(\sqrt{L})$. The trivial attractor is
*maximally far* from any valid sample.

Worse, the inverse problem $x_1 \mid x_\gamma$ is informationally
vacuous until $\gamma$ is close to 1:

$$I(x_1;x_\gamma) \;\lesssim\; \frac{\gamma^2}{2\sigma^2(1-\gamma)^2}\,\|x_1-\mu_1\|^2,$$

so the gradient signal that distinguishes per-token modes only switches
on in a thin slab near $\gamma=1$. The `gamma_power=0.5` importance
sampling (`eqm.py:200`) tries to upweight that slab — but at random
init the model has no per-token structure for the slab to teach yet,
so the upweighted signal is also dominated by $-\mu_1$.

---

## 5. How this plays out specifically for **Equilibrium Flow Matching**

Three EqM-specific aggravations turn the generic failure above into
the exact phenotype reported in `SESSION_SUMMARY.md`:

**5.1 The bilinear constraint $E_\theta(x)=\langle x,f_\theta(x)\rangle$.**
A general MLP can parameterise any scalar potential. EqM does not — it
constrains $E$ to the bilinear form, and the realisable energies are
exactly those for which $\nabla E$ can be written as
$f + J_f^\top x$. At lazy init $f\approx A_0 x + b_0$ gives
$E\approx x^\top A_0 x + b_0^\top x$, a quadratic. **The
realisable class at init is exactly the single-basin class.** There is
no slow, high-frequency "carving" happening underneath — the function
class itself is quadratic until features escape lazy.

**5.2 The $c(\gamma)$ schedule pinches signal at both endpoints.**
With $c(\gamma)=(1-\gamma)/\lambda$:

| $\gamma$ | $c(\gamma)$ | target $c(\gamma)(x_0-x_1)$ | what's there to learn |
|---|---|---|---|
| 0 | $1/\lambda$ | $(x_0-x_1)/\lambda$, but $x_\gamma=x_0$ is independent of $x_1$ | nothing — model must guess $\mathbb{E}[x_1]=\mu_1$ |
| ½ | $1/(2\lambda)$ | mid-strength signal, $x_\gamma$ informative about $x_1$ | this is where structure lives |
| 1 | 0 | **0** regardless of $x_1$ | model can output anything — gradient is zero |

So **the only useful $\gamma$-slab is the interior**, and at random
init the model can't read $x_\gamma$ well enough to use it. Both
endpoints actively pull toward triviality.

**5.3 Sample-time uses the *same* conservative gradient (`eqm.py:474`).**
NAG-GD descends $g_\theta$ at $\gamma=\texttt{sample\_gamma}$. If
training settled into the single-basin $-c(\gamma)\mu_1$ attractor,
inference descends that exact field — every chain converges to $\mu_1$
regardless of initialisation. There is no sampling-side fix because
the field has no other minima to find.

---

## 6. Why the standard **one-hot → CLR** path produces an essentially flat field

This is the variance-floor argument from `PROPOSAL_COMPOSITIONAL_EQM.md §2`.

Conditional on $\gamma$, the FM regression has Bayes risk

$$\inf_g \mathbb{E}\|g - c(\gamma)(x_0-x_1)\|^2 \;=\; c(\gamma)^2 \cdot \mathrm{Tr}\,\mathrm{Cov}(x_1 \mid x_\gamma).$$

For deterministic one-hot CLR, $x_1$ is a function of the token ID
$y$ alone, so $\mathrm{Cov}(x_1 \mid x_\gamma) = \mathrm{Cov}(x_1 \mid y)\cdot p(y\mid x_\gamma)$-mixture
which **does not vanish at any $\gamma<1$**: there are $K^L$ candidate
$x_1$'s, each contributes an irreducible inter-mode variance
$\sim \mathrm{Var}(\mu_1)$.

Translating "Bayes risk floor at every $\gamma$" into geometry:

- The optimal field $g^\star$ has *finite norm* everywhere, but its
  variance across the data conditional is bounded *below*.
- For an under-parameterised affine class (which is what lazy init
  gives you, §3), the best affine fit absorbs that variance by going
  **flat**: $g_\theta(x_\gamma) \approx -c(\gamma)\,\mu_1$.
- A flat $g$ corresponds to a *linear* energy
  $E_\theta(x) = -c(\gamma)\langle\mu_1,x\rangle$ — i.e. **no basins,
  no curvature, no per-token structure**. This is what we observe at
  early epochs and what trapped runs never escape.

So "the field is flat" is not a metaphor: under the FM-on-CLR
objective with one-hot data, the optimal regressor in the realisable
class **is literally a constant vector field**, and the corresponding
energy has zero Hessian.

---

## 7. Compositional Dirichlet fix — and the dual flat-field that replaces it

`PROPOSAL_COMPOSITIONAL_EQM.md §3` / `RESULTS_DIRICHLET.md` replaces
deterministic CLR with **per-batch Dirichlet sampling**: instead of
$x_1 = \mathrm{CLR}(\text{onehot}(y))$, draw
$p \sim \mathrm{Dir}(\alpha_{\text{base}}\mathbf{1} + \alpha_{\text{peak}}e_y)$
and set $x_1 = \mathrm{CLR}(p)$ (knobs:
`transformation.dirichlet_sampling`, `dirichlet_alpha_peak≈10`,
`dirichlet_alpha_base≈0.1`).

This **moves the variance floor**. The new floor is

$$\mathrm{floor}(\gamma) \;=\; c(\gamma)^2\cdot \mathrm{Tr}\,\mathrm{Cov}_{\mathrm{Dir}}(x_1 \mid y),$$

times the residual cross-mode mixture term — but crucially it
**vanishes at $\gamma=1$ because $c(1)=0$**, regardless of how thick
$x_1$ is. So at the data manifold the optimal $g^\star$ has zero
residual and the energy can carve a real basin per token (rather than
fitting irreducible noise with a flat constant).

### But the symmetric failure reappears at $\gamma=1$

The fix is one-sided: it removes the at-data flatness but the
$c(\gamma)=0$ factor now **trivialises the task at large $\gamma$**.
For $\gamma\to 1$:

$$\bigl\|c(\gamma)(x_0-x_1)\bigr\| \;=\; \tfrac{1-\gamma}{\lambda}\,\|x_0-x_1\| \;\to\; 0,$$

so every $g$ — including $g\equiv 0$ — achieves the same loss as the
true field. The gradient with respect to $\theta$ has magnitude
$\propto c(\gamma)$ and vanishes. The model can sit at *any* function
at $\gamma\approx 1$ and pay zero penalty. Empirically this is
exactly the dual of the §6 problem: **flatness migrates from
"everywhere" to "the data manifold itself"**, and field probes confirm
$f_\theta(\cdot,\gamma{=}1)\approx 0$ (the runtime check at
`eqm.py:_compute_grad` warns about $\gamma=1$ being degenerate for the
same reason).

This is why all sampling code samples at `sample_gamma` strictly
**inside** $(0,1)$ — at the edges the conservative gradient is
informationless.

### The two flat-field regimes

| representation | flat at | mechanism |
|---|---|---|
| One-hot CLR | **all $\gamma$** (lazy init) | irreducible Bayes risk $\Rightarrow$ best affine fit is constant |
| Dirichlet CLR | **$\gamma\to 1$** | $c(\gamma)\to 0$ kills the loss gradient |

The Dirichlet path wins net because the unconditional bad slab is
narrow (a thin shell at $\gamma\approx 1$) and importance sampling can
avoid it, while the one-hot bad slab is the whole path.

---

## 8. Hilbert metric — marginal improvement

`losses.py:SoftHilbertLoss` replaces MSE with an LSE-smoothed variation
seminorm:

$$\ell_{\mathrm{Hilbert}}(p,t) \;=\; \tfrac{1}{\alpha}\bigl[\mathrm{LSE}(\alpha\,(p-t)) + \mathrm{LSE}(-\alpha\,(p-t))\bigr]
\;\xrightarrow{\alpha\to\infty}\; \max_k(p-t)_k - \min_k(p-t)_k.$$

**Why this should help conceptually.** The Hilbert seminorm is the
intrinsic metric of the Aitchison simplex: it is invariant under
translations $u \mapsto u + c\mathbf{1}$ (which are gauge moves in CLR
space) and depends only on *contrasts* between components. MSE
penalises the gauge direction equally with the informative directions,
diluting the gradient signal.

**Why it barely helps empirically.** From `RESULTS_LATENT.md`
sanity-check (3-Dirichlet mixture in $S_5$):

| loss | forward KL |
|---|---|
| MSE | $4.989 \pm 0.175$ |
| Hilbert (soft) | $4.843 \pm 0.120$ |

~3 % gain, well within one $\sigma$. The reason it's so small is that
the dominant pathology (§§3,6) is not metric choice — it's the
single-basin attractor that *any* convex pointwise loss on the FM
target shares. Hilbert improves the geometry of the regression in
the right direction but doesn't change what the regression is *for*.
You need a non-quadratic, multi-modal term (the aux CE) to actually
carve basins; once that's in place, Hilbert vs MSE is second-order.

---

## 9. Why NAG underperforms SDE — a direct consequence of §3

`SAMPLER_FINDINGS.md §1–4`.

**NAG-GD** (`eqm.py:sample`) is deterministic Nesterov on
$\nabla E_\theta$:

$$x_{k+1} = x_k - \eta\,\nabla E_\theta\bigl(x_k + \mu(x_k - x_{k-1})\bigr).$$

**SDE** (`sampling/sde.py`) is Langevin / annealed Euler:

$$x_{k+1} = x_k - h\,\nabla E_\theta(x_k) + \sqrt{2\alpha h}\,\xi_k,
\qquad \xi_k \sim \mathcal{N}(0,P).$$

Three observations explain the gap:

**9.1 The trained energy is still mostly the §3 quadratic plus thin
carved basins.** From the analysis above, even after the aux CE has
done its work, $E_\theta$ has *one wide bowl centred near $\mu_1$* with
small per-token pits superimposed. The basin volumes obey
$|\Omega_{\mu_1}| \gg \sum_v |\Omega_v|$.

**9.2 Deterministic descent has zero escape rate.** A NAG iterate
converges to whichever critical point it falls into; the catch-radius
of $\mu_1$ dominates by volume, so most chains drain there. Even when
a chain starts inside a token pit, NAG's momentum makes it overshoot
on the first encounter with a basin wall, and once it leaves the pit
it sees the wide unigram bowl and converges to it. This is documented
empirically in `SAMPLER_FINDINGS.md §2` ("overshoot 2.95 → 0.31 with
grad clip" — overshoot is *intrinsic to the field*, not just to NAG).

**9.3 Langevin noise has a Boltzmann stationary distribution.**
The SDE has stationary $\pi(x) \propto e^{-E_\theta(x)/\alpha}$ (modulo
the $V_d$ projection). The relative mass of a token basin to the
unigram bowl is

$$\frac{\pi(\Omega_v)}{\pi(\Omega_{\mu_1})} \;\approx\; \frac{|\Omega_v|}{|\Omega_{\mu_1}|}\,\exp\!\bigl(-(E_v - E_{\mu_1})/\alpha\bigr),$$

and with $E_v < E_{\mu_1}$ (token pits are deeper) the exponential
can dominate the volume ratio at the right temperature. NAG only sees
volume; SDE sees volume **and** depth.

**9.4 Annealing matches the $\gamma$ schedule.** SDE sweeps
$\gamma: 0 \to 1$, querying $f_\theta(\cdot,\gamma_k)$ at each step.
At small $\gamma$ the field points coarsely toward the data manifold
(the unigram bowl); at large $\gamma$ it resolves token basins. The
chain follows curriculum-style sharpening, which is exactly how
diffusion samplers escape spurious minima. NAG at fixed
$\texttt{sample\_gamma}$ has access to one slice only.

**Net.** NAG underperforms not because its update rule is wrong but
because the underlying energy is still §3-shaped: one wide attractor,
many small pits. Deterministic descent on such a field is an
exponentially biased estimator of $p_{\text{data}}$. Langevin
annealing is the textbook fix and matches what worked for image
EBMs for the same reason.

---

## 10. Latent embeddings smooth the field — preliminary evidence

`RESULTS_LATENT.md` / `RESULTS_LATENT_FINAL.md`. We replace
one-hot CLR with a learned (or fixed) embedding
$\phi:\{1,\ldots,K\}\to\mathbb{R}^d$ and run EqM in the embedding
space. Variants tested:

- trainable joint-with-CE embeddings ($d\in\{32,128,256\}$),
- fixed PPMI-SVD ($d\in\{16,27\}$),
- fixed skip-gram on text8 char windows ($d\in\{27,32\}$, pairwise min
  cosine $0.92$ — most discriminative).

**Why this should help (mechanistic).** Two of the §§3,6 failure modes
weaken:

1. The data mean in embedding space is no longer maximally far from
   the data points. Skip-gram and PPMI embed semantically related
   tokens near each other, so $\mu_1$ sits inside a denser region of
   the support. The §4 catastrophe ($\mu_1$ at Hilbert distance
   $\Theta(\sqrt{L})$ from every vertex) softens to a constant.
2. The Bayes-risk floor (§6) drops because $\mathrm{Cov}(\phi(y))$ is
   spectrally concentrated rather than spanning all $K-1$ contrasts
   uniformly. The optimal affine field is less flat and more
   directional.

**What we see.** Field-orientation probes at $\gamma\in[0.25,1.0]$
report `cos(g_\theta, g^\star) \geq 0.9`. The pathological
$\gamma\to 0$ slab still has `cos \approx 0.11` (centroid trap
intact at the source endpoint), but the useful interior is broader.

**Conditional recovery vs unconditional generation.** Recovery is
solid (100 % at perturbation $\alpha \leq 0.2$, 60 % at $\alpha = 0.5$);
**unconditional generation is still broken** (KL$_\mathrm{bi}=1.19$ at
$L=40$, $d=128$). This is the §3 + sibling-note story: the field is
trained on the noise→data interpolant, not on points far from it;
unconditional sampling starts in the untrained region. Embeddings
make the trained region geometrically nicer but do not extend it.

**Sequence length helps — surprisingly.** Going $L=40 \to L=128$ at
matched step count drops KL$_\mathrm{bi}$ from $1.48$ to $0.92$
($-38\%$). Mechanism: each window contributes $L-1$ bigram contexts;
the per-batch coverage of the $K^2$ bigram support grows linearly,
which dampens the high-frequency variance in $g^\star$ across nearby
$x_\gamma$'s. In the §3 picture, larger $L$ shrinks the Bayes-risk
floor at fixed $\gamma$ by averaging over more contexts before the
affine fit kicks in. We also see field probes get smoother at long
$L$ — consistent with the floor argument rather than just "more data."

**Caveat — these results are partial.** The fixed-embedding sweeps
finished; the trainable-embedding cluster runs at $L=256$ are in
flight as of this writing. The numbers above will be updated in
`RESULTS_LATENT_FINAL.md` when those land.

---

## 11. Summary

The unigram-collapse failure mode is a single mathematical fact
expressed five different ways:

| where | how it shows up |
|---|---|
| §2 Bayes-optimum | $g^\star \to -c(\gamma)\mu_1$ when $I(x_1;x_\gamma)$ is small |
| §3 lazy init | realisable energies are quadratic, single-basin |
| §5 EqM-specific | bilinear $\langle x,f\rangle$ constraint locks the function class |
| §6 one-hot CLR | irreducible Bayes-risk floor $\Rightarrow$ best affine fit is flat |
| §7 Dirichlet CLR | floor moves to $\gamma=1$, where $c(\gamma)=0$ trivialises the loss |

The fixes that actually work share one feature: they introduce a
non-MSE, non-quadratic term whose minimisers form a set of
cardinality $\Omega(K^L)$. The aux CE on implied-$x_1$ is the
canonical example. Everything else (Hilbert metric, latent
embeddings, SDE sampling, longer sequences) shapes how cleanly that
multi-modal carving propagates, but cannot substitute for it.
