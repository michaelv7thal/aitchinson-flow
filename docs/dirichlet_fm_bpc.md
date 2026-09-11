# A peer-comparable BPC for Dirichlet FM (variational dequantization ELBO)


> **Status (2026-08-20): the spec of `scripts/eval_dirichletfm_elbo_bpc.py`, for a number the paper does not report.** This note is about **Dirichlet Flow Matching**, not the Discrete Flow Matching arm that shares the "DFM" abbreviation. The derivation and especially the measure conventions below are correct and are the only record of them, and the script still implements them.
> What changed: the paper reports **no bits-per-character figure for any of our models**. `tab:config` marks `bpd()` "diagnostic only" for both Dirichlet FM and Fisher FM, and the only BPC in the paper is Statistical Flow Matching's *published* 1.39. An earlier draft carried a BPC column and a `tab:bpc-frontier` table; both were cut. No output artifact from this estimator exists in the repository, so treat it as an unrun design and not as a source of a number.

DirichletFM has no closed-form discrete likelihood (unlike the `DFM` D3PM-uniform
arm). Its forward process is a **continuous Dirichlet path** on the simplex, so
the only honest bits-per-character is a **continuous-flow (CNF) likelihood**
wrapped in a **variational dequantization ELBO** that turns the simplex density
into an upper bound on the discrete-token NLL. This note pins the math and — most
importantly — the **measure conventions**, which are where this kind of estimator
usually goes wrong.

## Setup

- Vocab `K=27`, sequence length `L`. Tokens `c = (c_1..c_L) ∈ {0..K-1}^L`.
- Simplex `Δ = {x ∈ R^K : x_k ≥ 0, Σ_k x_k = 1}`, dimension `K-1`.
- Model: a denoiser `p_θ(x₁=k | x_t, t) = softmax(f(x_t, t))_k` and the induced
  **marginal velocity field** `v(x,t)` (exactly the one `DirichletFM.sample()`
  integrates: `v = w − x·Σw`, `w = p₁·ċ_t/(1−x)`), `t ∈ [1, t_max]`.
- Forward (conditional) path: `x_t ~ Dir(β(t,c))`, `β` = ones with `t` on the
  token coordinate. `t=1 ⇒ Dir(1..1)` (uniform prior); `t=t_max ⇒` concentrated.

## The bound

Treat the continuous simplex point `x` (per position) as a **latent**, the token
`c` as **observed**. Generative model `P(c) = ∫ p_flow(x)·p_dec(c|x) dx` with a
variational posterior `q(x|c)`. For any `q`, Jensen gives the per-position bound

```
log P(c) ≥ E_{x~q(·|c)} [ log p_flow(x) + log p_dec(c|x) − log q(x|c) ]  =: ELBO(c)
```

Choices:
- **q(x|c) = Dir(β(t_max, c))** — the model's own clean-time conditional. Tractable
  to sample and to score.
- **p_dec(c|x) = softmax(f(x, t_max))_c** — the denoiser read at clean time. A
  proper categorical ⇒ valid `log p_dec`.
- **p_flow(x)** — the flow's marginal density at `t_max`, via the CNF below.

For a **sequence**, the flow is joint over all `L` positions (the transformer
mixes them), while `q` and `p_dec` factorize per position:

```
ELBO(seq) = E_{x~q} [ log p_flow(x_{1:L}) + Σ_l log p_dec(c_l|x) − Σ_l log q(x_l|c_l) ]
BPC = − ELBO(seq) / (L · ln 2),   averaged over sequences.
```

Because `ELBO ≤ log P`, **BPC is a valid UPPER bound** on the true bits/char — the
safe direction to report, and the same kind of object as `DFM.elbo_bpc`.

## The CNF density (probability-flow ODE)

Instantaneous change of variables along `ẋ = v(x,t)`, `t: 1 → t_max`:

```
log p_flow(x_{t_max}) = log p_prior(x_1) − ∫_1^{t_max} (∇·v)(x_t, t) dt
```

`p_prior = Dir(1..1)`. To score a given clean sample `x* @ t_max`, integrate the
ODE **backward** `t_max → 1` to get `x_1`, accumulating `∫ ∇·v dt` along the way.

### Measure convention (the load-bearing part)

Everything is done in the **first `K−1` simplex coordinates** `x̃ = (x_1..x_{K-1})`,
with `x_K = 1 − Σ x̃`. This is exactly the convention `torch.distributions.
Dirichlet.log_prob` uses, so:

- `log p_prior`, `log q` are just `Dirichlet(...).log_prob(x)` — **same measure**.
- The divergence is the trace in these `K−1` free coordinates. Define the reduced
  field `g: R^{K-1} → R^{K-1}`, `g_i(x̃) = v_i( [x̃; 1−Σx̃], t )`, and

```
∇·v  :=  Σ_{i=1}^{K-1} ∂g_i/∂x̃_i  =  Tr(∂g/∂x̃)
```

  estimated by Hutchinson: `Tr(J) ≈ (1/M) Σ_m ε_mᵀ J ε_m`, `ε_m ~ N(0, I_{K-1})`.
  `ċ_t` is evaluated via scipy (regularised incomplete beta) inside `no_grad` and
  is **not autograd-differentiable**, so `J ε` is taken by **central finite
  differences** `Jε ≈ (g(x̃+δε) − g(x̃−δε)) / 2δ` (forward-evals only; perturbations
  propagated to `x_K = 1−Σx̃` so the probe stays tangent). `δ≈1e-3`, float64.
  **No ILR, no Jacobian-determinant terms** — using one
  coordinate system for prior, divergence, and `q` makes all the change-of-variable
  constants cancel in the `log p_flow − log q` ratio automatically.

  (Why this is the right trace: `v` is tangent to the simplex, `v_K = −Σ_{i<K} v_i`,
  so the dynamics on the `K−1`-dim manifold are fully described by `g`, and the
  Liouville term is its Euclidean divergence in those coordinates.)

## Validation gates (must pass before trusting any BPC)

1. **Analytic CNF on a toy field.** Replace `v` with a linear tangent field
   `v(x) = A·(x − 1/K)` (A built sum-zero so it stays tangent). The reduced field
   is linear with constant Jacobian, so `∫∇·v dt = Tr(reduced A)·(t_max−1)` is
   exact — check the Hutchinson+ODE estimate matches to MC error. Gate: relative
   error < a few %.
2. **Density normalization (soft).** For the toy, the implied `log p_flow` change
   should match the analytic Gaussian/linear pushforward.
3. **Bound sanity on the real model.** `0 < BPC_elbo < 4.75` (unigram floor
   `log2 27`); a sane model lands well below the floor. It must be **far above**
   the `~0.05` identity artifact and should be a smooth function of ODE steps
   (`nfe`) and MC count (decreasing variance, stable mean).

## Cost

Per sequence: `n_mc_q` posterior samples × `nfe` ODE steps × `M` Hutchinson probes
× (one JVP through the backbone each). Heavy — target a modest eval set
(≈64–128 seqs), `nfe≈50–100`, `M≈1–2`, antithetic `ε`. Reported BPC is the mean
over sequences with a seq-level std for the MC error bar.

## Files

- `scripts/eval_dirichletfm_elbo_bpc.py` — estimator + `--self-test` (gate 1).
- This note — derivation and conventions.
