# BayesLinHead — Bayesian linear OOD detector (math)


> **Status (2026-08-20): current for the head's algebra, stale on the read-out configuration.** The energy head, the hinge with margin 4, the closed-form variance and the exact-pooling identity below are what the paper uses and are the fuller derivation of `chapters/methods.tex` §BayesLinHead. Three things have moved since this was written (2026-06-12):
> 1. **Two passes, not one.** The energy is read at `t_eval=4.5`; the variance is read at a *second* pass at `t_var=7.5` on near-one-hot input, with its own standardization. Read at 4.5 the variance signal *inverts*. See the paper's `tab:config`, row BayesLinHead.
> 2. **The backbone is d_model=1280**, the scaled full-corpus run, not the d_model=1024 benchmark backbone assumed in the notation above.
> 3. **The corruption ladder has four schemes** (replace, shuffle, false information, plausible), not the three below; "both" is now only one of the negative sets for the BLR_all variant.
> §6's claim that the variance separates only under heavy corruption was true of the single-pass version and is not true of the two-pass one (it matches the energy at rate 0.15 sequence-level and beats it per token on false information).

Reference for `scripts/ood_bayes_linear.py`. The detector bolts a **linear energy
head** plus a **Laplace / Bayesian-linear posterior** onto the frozen `DirichletFM`
backbone. It produces two scores at two granularities from a single feature pass:

- **Energy** $E$ — *discriminative*: "is this token invalid?" (trained by a per-token hinge).
- **Variance** $\operatorname{Var}$ — *epistemic uncertainty*: "how far off the in-distribution manifold?" (closed-form, no training).

Notation: $K=27$ (text8 vocab), $d$ = backbone width (`d_model`, 1024), $m$ = final
feature dim (`feat_dim`; $m=d$, or $m=$ `pca_dim` if PCA is on), $L$ = sequence length,
$B$ = batch. Evaluation time $t \equiv$ `t_eval` $\approx 4.5$.

---

## 1. Features — deterministic Dirichlet-mean encoding

For a token $c \in \{0,\dots,K-1\}$ at one position, build a Dirichlet concentration
vector by boosting that token's coordinate to $t$:

$$
\beta \;=\; \mathbf{1}_K + (t-1)\,e_c \;\in\; \mathbb{R}^K,
\qquad
\beta_k =
\begin{cases}
t, & k = c\\
1, & k \neq c
\end{cases}
$$

Feed the **Dirichlet mean** (not a sample — this is the load-bearing choice) through
the backbone $f_\theta$:

$$
x \;=\; \frac{\beta}{\mathbf{1}_K^\top \beta} \;=\; \frac{\beta}{K-1+t},
\qquad
h \;=\; f_\theta(x,\,t) \;\in\; \mathbb{R}^{d}.
$$

So $x_c = \dfrac{t}{K-1+t}$ and $x_{k\neq c} = \dfrac{1}{K-1+t}$. A stochastic
$x\sim\mathrm{Dir}(\beta)$ would inject per-position noise that collapses clean and
corrupted features together; the mean keeps $f_\theta$ a clean text→representation
encoder (≈3× larger clean-vs-corrupt feature shift at $t\!=\!4.5$).

**Standardize, then optionally PCA.** Using per-dim mean $\mu$ and std $\sigma$ from the
*clean fit set*, and (optionally) the top-$p$ principal axes $V\in\mathbb{R}^{d\times p}$
of the standardized clean cloud:

$$
z \;=\; \big(\operatorname{diag}\sigma\big)^{-1}(h-\mu),
\qquad
z \;\leftarrow\; V^\top z \ \ (\text{if PCA on}),
\qquad
z \in \mathbb{R}^{m}.
$$

(code: `_perpos_feats`, `_proj`; $\mu,\sigma$ at L142–143, $V$ at L145–150.)

---

## 2. Energy head + per-token hinge

A scalar **linear energy** on the feature:

$$
E(z) \;=\; w^\top z + b, \qquad w\in\mathbb{R}^m,\; b\in\mathbb{R}.
$$

Build per-token training sets from the fit sequences and a 50%-replace-corrupted copy:

- **Positives** $\mathcal P$ — valid chars: all clean tokens $\cup$ the *unchanged* tokens of corrupted sequences.
- **Negatives** $\mathcal N$ — the tokens actually replaced.

Train $w,b$ with a two-sided hinge (valid $\to E\approx 0$, corrupt $\to E\ge m_0$):

$$
\mathcal{L}(w,b)
=\underbrace{\frac{1}{|\mathcal P|}\sum_{z\in\mathcal P} E(z)^2}_{\text{pull valid to }0}
\;+\;
\underbrace{\frac{1}{|\mathcal N|}\sum_{z\in\mathcal N}\big(m_0 - E(z)\big)_+}_{\text{push corrupt above margin}},
\qquad (u)_+=\max(0,u),
$$

with margin $m_0\equiv$ `margin` $=4$, optimized by Adam. (code: L165–183.)

---

## 3. Laplace posterior covariance + predictive variance

Fit a **Bayesian linear regression** view on the *clean* per-token features only.
With clean feature matrix $Z_c\in\mathbb{R}^{N_c\times m}$, form the (uncentered)
second-moment matrix and a trace-scaled ridge:

$$
\Phi \;=\; \frac{1}{N_c} Z_c^\top Z_c \;\in\; \mathbb{R}^{m\times m},
\qquad
\lambda \;=\; \texttt{ridge}\cdot \frac{\operatorname{tr}\Phi}{m}.
$$

The Gaussian (Laplace) posterior precision and covariance on the head weights are

$$
\Sigma_w^{-1} \;=\; \Phi + \lambda I,
\qquad
\Sigma_w \;=\; \big(\Phi + \lambda I\big)^{-1}.
$$

This is the standard ridge-regularized linear-Gaussian posterior covariance: prior
$w\sim\mathcal N(0,\lambda^{-1}I)$, data term $\Phi$ (noise precision folded into the
$1/N_c$ scaling). The **epistemic predictive variance** of the energy at a query $z$ is
the quadratic form

$$
\boxed{\;\operatorname{Var}\big(E(z)\big) \;=\; z^\top \Sigma_w\, z\;}
$$

Computed stably via the Cholesky factor $\Sigma_w^{-1}=LL^\top$ and one triangular solve:

$$
\operatorname{Var}(z) \;=\; z^\top (LL^\top)^{-1} z \;=\; \big\lVert L^{-1} z \big\rVert_2^2 .
$$

(code: $\Phi,\lambda$ L187–188; $L$ L190; $\lVert L^{-1}z\rVert^2$ via `solve_triangular`
L204–205.)

### 3.1 Two equivalent readings

**Mahalanobis / density.** Diagonalizing $\Phi=\sum_i \rho_i\, u_i u_i^\top$ (eigenpairs $\rho_i,u_i$),

$$
\operatorname{Var}(z) \;=\; \sum_{i=1}^{m} \frac{(u_i^\top z)^2}{\rho_i + \lambda}.
$$

Projections along **high-variance ID directions** ($\rho_i$ large) are down-weighted;
projections along **rare/off-manifold directions** ($\rho_i$ small) blow up. High
$\operatorname{Var}$ ⇔ far from the in-distribution feature cloud. It is exactly a
zero-centered Mahalanobis quadratic form in the metric $(\Phi+\lambda I)^{-1}$.

**Decoupling (important).** $\operatorname{Var}(z)$ depends **only** on $\Phi$ (the ID
feature geometry) and $z$ — **not** on the trained $w,b$. So the variance head is an
*unsupervised density* score and would be identical even if the hinge head were never
trained. Energy and variance are two independent views of the same features:

| Score | Supervised? | Depends on | Question |
|---|---|---|---|
| $E(z)=w^\top z+b$ | yes (hinge) | $w,b,z$ | "looks corrupted" |
| $\operatorname{Var}(z)=z^\top\Sigma_w z$ | no (closed form) | $\Phi,z$ | "looks unfamiliar" |

---

## 4. Granularity: per-token vs sequence

The same $w,b,\Sigma_w$ give both granularities. Per **token** $z_t$ ($t=1,\dots,L$):

$$
E_t = w^\top z_t + b, \qquad \operatorname{Var}_t = z_t^\top \Sigma_w z_t
\qquad \text{(shape } B\times L,\ \text{plotted as the heatmaps)}.
$$

The **sequence** scores used by the AUROC table are the **position-means** of these
arrays (code: `Ep.mean(1)`, `Vp.mean(1)`, L220):

$$
E_{\text{seq}} = \frac{1}{L}\sum_t E_t,
\qquad
\operatorname{Var}_{\text{seq}} = \frac{1}{L}\sum_t \operatorname{Var}_t .
$$

**Energy pooling is exact.** Since $E$ is linear, with $\bar z=\frac1L\sum_t z_t$,

$$
E_{\text{seq}} = \frac{1}{L}\sum_t (w^\top z_t + b) = w^\top \bar z + b .
$$

**Variance pooling is *not* the form-of-mean.** Beware: the code uses the *mean of
per-token variances*, which exceeds the quadratic form on the pooled feature by a
Jensen gap (a quadratic form is convex):

$$
\underbrace{\frac{1}{L}\sum_t z_t^\top \Sigma_w z_t}_{\text{code }=\ \text{mean of heatmap}}
\;=\;
\underbrace{\bar z^\top \Sigma_w\, \bar z}_{\text{"variance of pooled feature"}}
\;+\;
\operatorname{tr}\!\big(\Sigma_w\,\operatorname{Cov}_t(z_t)\big)
\;\ge\; \bar z^\top \Sigma_w\, \bar z,
$$

with $\operatorname{Cov}_t(z_t)=\frac1L\sum_t (z_t-\bar z)(z_t-\bar z)^\top$ the
within-sequence positional spread. The extra term $\ge 0$, with equality iff all
positions share the same feature. So: **the heatmap is the un-pooled view of exactly
the score the AUROC uses** (mean-of-per-token), which differs from the docstring's
$z_{\text{seq}}^\top\Sigma_w z_{\text{seq}}$ phrasing. For energy there is no such gap.

---

## 5. Scoring & evaluation (corruption ladder)

Corrupt a held-out clean set $\{c\}$ by scheme × rate $r$:

- **replace** — substitute a fraction $r$ of tokens with uniform-random vocab ids.
- **shuffle** — permute a fraction $r$ of positions per sequence (**preserves the token multiset** ⇒ the hard *order* axis).
- **both** — compose replace then shuffle.

Both scores are oriented so **larger ⇒ more OOD**. For each (scheme, rate):

**Sequence-level AUROC** — clean positives vs corrupted sequences, ranked by
$E_{\text{seq}}$ (`auroc_seq_energy`) and by $\operatorname{Var}_{\text{seq}}$
(`auroc_seq_uncertainty`). With clean scores $\{s^+\}$ and corrupt $\{s^-\}$,

$$
\mathrm{AUROC} \;=\; \Pr\big(s^- > s^+\big)
\;=\; \frac{1}{|{+}||{-}|}\sum_{i\in -}\sum_{j\in +}\mathbf 1[s_i>s_j].
$$

**Token-localization AUROC** — *within* corrupted sequences, label each position
changed (1) vs unchanged (0) and rank by per-token $E_t$ / $\operatorname{Var}_t$
(`auroc_token_*`). Undefined (NaN) when every / no position changed (e.g. replace at
$r{=}1$). Note shuffle localization is intrinsically weak: a swap moves two *valid*
chars, so both endpoints are flagged "changed" though each token is individually legal.
(code: L210–264.)

---

## 6. One-line summary

$$
\underbrace{E(z)=w^\top z+b}_{\text{hinge-trained, discriminative}}
\qquad\text{and}\qquad
\underbrace{\operatorname{Var}(z)=z^\top(\Phi+\lambda I)^{-1}z}_{\text{closed-form Mahalanobis-to-ID}}
$$

on deterministic-Dirichlet-mean backbone features, read per-token (heatmaps) or
position-averaged (sequence), then swept over a replace/shuffle/both corruption ladder
and scored by AUROC. Energy carries the signal; variance is an independent,
training-free density backstop that only separates well once a large fraction of the
sequence is corrupted (concentration of measure in $\mathbb{R}^m$).

---

### Code map

| Quantity | Symbol | Code |
|---|---|---|
| Dirichlet-mean feature | $h=f_\theta(x,t)$ | `_perpos_feats` |
| standardize (+PCA) | $z=V^\top(h-\mu)/\sigma$ | `_proj` (L153–155) |
| energy head | $E=w^\top z+b$ | `nn.Linear`, L166 |
| hinge loss | $\mathcal L$ | L176 |
| ID second moment | $\Phi$ | L187 |
| ridge jitter | $\lambda$ | L188 |
| posterior precision | $\Sigma_w^{-1}=\Phi+\lambda I$ | L189 |
| Cholesky | $\Sigma_w^{-1}=LL^\top$ | L190 |
| per-token energy/variance | $E_t,\operatorname{Var}_t$ | `score`, L203–205 |
| sequence scores | $E_{\text{seq}},\operatorname{Var}_{\text{seq}}$ | `.mean(1)`, L220 |
| AUROC sweep | — | L236–264 |
