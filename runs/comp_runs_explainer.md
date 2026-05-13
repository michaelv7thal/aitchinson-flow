# `runs/comp_*` — model mechanics

Companion to [`compositional_eqm_test_summary.md`](compositional_eqm_test_summary.md)
(results) and [`../PROPOSAL_COMPOSITIONAL_EQM.md`](../PROPOSAL_COMPOSITIONAL_EQM.md)
(motivation). This note documents **what exactly each model was** —
training recipe, what made it "compositional", what differed across
cells, and how it was evaluated. Sweep spec:
[`sweeps/compositional_eqm_test.yaml`](../sweeps/compositional_eqm_test.yaml).

All eight cells are the **same EqM model class** (`models/eqm.py`,
`build_eqm`), trained with the same step `_eqm_loss`, sampled with the
same `EquilibriumFlowMatching.sample` (NAG-GD). The only things that
change between cells are:

1. whether $x_1$ is **Dirichlet-thickened** before being fed to the
   FM regression (the "compositional" knob),
2. whether the regression loss is **MSE** or **soft-Hilbert**,
3. the training seed.

That's it. Everything else — backbone, schedule, optimiser, sampler,
aux losses, eval — is identical.

---

## 1. The 8 cells

| cell                       | dirichlet | loss          | seed | epochs | windows | backbone                          |
|----------------------------|:---------:|---------------|:----:|:------:|:-------:|-----------------------------------|
| `comp_smoke`               | ✅        | hilbert_soft  | 42*  | 1      | 1024    | d=128, 2 layers, 4 heads          |
| `comp_mse_seed42/43/44`    | ✅        | mse           | 42/43/44 | 5  | 10000   | d=1024, 8 layers, 8 heads (default) |
| `comp_hilbert_seed42/43/44`| ✅        | hilbert_soft  | 42/43/44 | 5  | 10000   | d=1024, 8 layers, 8 heads (default) |
| `comp_ref_det_mse`         | ❌        | mse           | 42   | 5      | 10000   | d=1024, 8 layers, 8 heads         |
| `comp_ref_det_hilbert`     | ❌        | hilbert_soft  | 42   | 5      | 10000   | d=1024, 8 layers, 8 heads         |

`*` smoke uses the sweep's default seed; it's a 2-minute fail-fast
pipeline check, not a result-bearing cell.

---

## 2. Shared training setup (all cells)

From `runs/comp_*/config.json`:

- **Dataset.** text8 via HuggingFace (`afmck/text8`), 27-char alphabet
  (lowercase a-z + space), windows of length L=40, batch 64. 10k
  train / 5k eval windows. Label smoothing on CLR features = 1e-4.
- **Model class.** `EquilibriumFlowMatching` — conservative gradient of
  $E_\theta(x)=\langle x, f_\theta(x)\rangle$, no time conditioning,
  no context conditioning, no bigram/trigram/hinge/auditor heads, no
  trainable embeddings.
- **Backbone.** Default `TransformerBackbone` with $d_{\text{model}}=1024$,
  8 layers, 8 heads, no dropout. (smoke: $d_{\text{model}}=128$, 2 layers.)
- **Optimiser.** AdamW with cosine schedule, $\eta=3\times 10^{-4}$,
  $\eta_{\min}=0$, warmup off, grad-clip 1.0.
- **EqM training step** (`_eqm_loss`):
  - $x_0 = \sigma\cdot\mathcal{N}(0, P)$, $\sigma = 0.1$, projected to
    the mean-zero hyperplane (`source_sigma=0.1`).
  - $\gamma\sim U(0,1)^{p}$, $p=$ `gamma_power` — 1.5 for MSE / smoke,
    1.0 for Hilbert (this matches the existing
    `dphase5_hilbert_*` cells).
  - $x_\gamma = (1-\gamma)x_0 + \gamma x_1$, target
    $u_{\text{tgt}} = c(\gamma)(x_0 - x_1)$ with $c(\gamma)=(1-\gamma)/\lambda$,
    $\lambda$=`gradient_lambda`$=1.0$, linear decay.
  - Forward $v=f_\theta(x_\gamma)$; conservative gradient
    $g = \nabla_{x_\gamma}\langle x_\gamma, v\rangle$ via autograd
    with `create_graph=True`.
  - **Flow loss**: `loss_fn(g, u_tgt)` (MSE or soft-Hilbert — see §4).
  - **Aux CE on implied $x_1$**: $\text{pred}_{x_1} = x_\gamma - \lambda g$,
    log-softmaxed, NLL against `token_ids`. Weight `lambda_ce=0.5`,
    masked to $\gamma \ge$ `ce_min_gamma`$=0.5$.
  - No bigram/trigram/hinge auxiliaries (all $\lambda=0$).
- **Probe every epoch.** `sample_eval_every=1`, n=64, 100 NAG steps —
  unigram-KL probe runs at the end of each epoch.

---

## 3. The "compositional" knob — Dirichlet thickening of $x_1$

`transformation.dirichlet_sampling`. Source: `PROPOSAL_COMPOSITIONAL_EQM.md §2-3`.

**Off (deterministic-CLR, the standard recipe):**
$x_1 = \mathrm{CLR}(\text{one-hot}(y) + \text{label\_smoothing})$ — a
*fixed* CLR vector per token id. Every time the token "e" appears, the
same $x_1$ row goes into the FM regression.

**On (compositional):** at each batch, sample
$$p \sim \mathrm{Dir}\bigl(\alpha_{\text{base}}\mathbf{1} + \alpha_{\text{peak}}\,e_y\bigr),
\qquad x_1 = \mathrm{CLR}(p),$$
with `dirichlet_alpha_peak=10.0`, `dirichlet_alpha_base=0.1` (K=27, so
$\alpha_{\text{base}}\mathbf{1}$ is a 27-vector summing to $\approx 2.7$;
the peak component dominates). Effect:

- $x_1$ is no longer a delta at the CLR vertex — it's a *cloud* around
  the vertex with finite variance.
- Token argmax is preserved with ≥ 99.5 % probability at these
  $\alpha$'s (the calibration check `scripts/check_dirichlet_data.py`
  verified this before the sweep).
- The FM regression sees a *distribution* of $x_1$'s per token, which
  means the Bayes-risk floor
  $c(\gamma)^2\,\mathrm{Tr}\,\mathrm{Cov}(x_1\mid x_\gamma)$ shrinks
  to zero at $\gamma=1$ (since $c(1)=0$), instead of being a positive
  constant at every $\gamma$ (as in the one-hot CLR case).

The mechanistic story is in
[`../NOTE_WHY_EBM_INIT_STUCK.md`](../NOTE_WHY_EBM_INIT_STUCK.md) §§6-7:
Dirichlet thickening lets the model learn a non-flat field across
the path interior; without it the optimal regressor in the realisable
function class is a constant vector field.

`comp_ref_det_*` are exactly the same recipe with this knob flipped
off — they isolate the binary "does Dirichlet thicken matter?".

---

## 4. The MSE branch — `comp_mse_seed{42,43,44}`

`loss.mode = "mse"`. The flow loss is
$$\mathcal{L}_{\text{flow}}(\theta) = \frac{1}{BLK}\sum_{b,\ell,k} \bigl(g_{b\ell k} - u_{\text{tgt},b\ell k}\bigr)^2,$$
mean-reduced over all $(B,L,K)$ entries. Plus the aux CE term
($\lambda_{\text{ce}}=0.5$).

Cells are identical except for the seed (42/43/44). `gamma_power=1.5`
upweights $\gamma\approx 1$ aggressively.

---

## 5. The Hilbert branch — `comp_hilbert_seed{42,43,44}`

`loss.mode = "hilbert_soft"`, `loss.hilbert_alpha = 1.0`. The flow
loss is the LSE-smoothed variation seminorm on each $(b,\ell)$ slice:
$$\ell_{\text{H}}(g, u) = \frac{1}{\alpha}\Bigl[\mathrm{LSE}_k\bigl(\alpha\,(g - u)\bigr) + \mathrm{LSE}_k\bigl(-\alpha\,(g - u)\bigr)\Bigr]
\;\xrightarrow{\alpha\to\infty}\; \max_k(g-u)_k - \min_k(g-u)_k,$$
mean-averaged over $(b,\ell)$. This is the Aitchison-simplex
intrinsic metric: invariant under $g \mapsto g + c\mathbf{1}$ (the
CLR gauge), so it doesn't waste gradient on the redundant direction
that MSE penalises.

Hilbert cells use `gamma_power=1.0` (not 1.5) — this matches the
existing `dphase5_hilbert_*` cells the proposal cross-references.
Everything else is identical to the MSE branch.

Result: Hilbert and MSE are within seed noise on recovery (Δ@0.50 of
+0.054 vs +0.060) — see the summary's headline table.

---

## 6. The deterministic-CLR references — `comp_ref_det_{mse,hilbert}`

Same as the corresponding compositional cells **with `dirichlet_sampling`
flipped to `false`**. Seed 42 only (no triplicate; these are sanity
references, not result-bearing).

What they should show, per the proposal:

- Unigram-KL looks *better* than the compositional cells (KL_uni $\approx 0.04$
  vs $\approx 0.67$) — the deterministic field collapses to the unigram
  peak, which is a fine match to the corpus unigram distribution but
  carries no per-token structure.
- Recovery Δ@α = 0 across all α — the sampler is a no-op because the
  field is the §6/NOTE flat-field attractor. This is the headline
  failure mode the Dirichlet recipe is designed to fix.

This is what the headline table in the summary reports.

---

## 7. Sampling — NAG-GD on the conservative gradient

`EquilibriumFlowMatching.sample` (eqm.py:367). Same for every cell:
$$x_{k+1} = x_k - \eta\,\nabla E_\theta\!\bigl(x_k + \mu(x_k - x_{k-1})\bigr),$$
with `sample_eta=0.1`, `sample_mu=0.9`, max 200 steps, early-exit when
$\max_b\|\nabla E\|_b < g_{\min}=0.01$, per-position L2 clip
`sample_grad_clip=1.0`, **return the best-norm iterate** seen along
the trajectory (`sample_return_best=true`). $x_0$ initialised at
$\sigma=0.1$ matching training.

`sample_gamma=1.5` is in the config but irrelevant for these cells —
`time_conditioning="off"` means the backbone ignores $\gamma$, so the
field is queried at a single $\gamma$-independent point.

The eval `recovery_check` calls `sample(...)` with `x_init = x_1 + α·ξ`,
where $\xi$ is centred Gaussian and $\alpha\|\xi\| \approx \alpha\|x_1\|$
controls perturbation magnitude.

---

## 8. Evaluation protocol

Two diagnostics on each `epoch_final.pt`:

**Unconditional** (in `eval.json`, also probed every epoch during training):
- $B$ samples drawn from $\mathcal{N}(0,\sigma^2 P)$ → NAG → token argmax.
- Compute KL of the empirical token distribution to the corpus
  unigram (`KL_uni`), bigram (`KL_bi`), and trigram (`KL_tri`), plus
  entropy ratio H_gen / H_corpus.

**Recovery** (in `recovery.json`):
- For $\alpha \in \{0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 1.00\}$:
  - Take 256 held-out windows, encode $x_1$ via the same data pipeline.
  - Perturb: $x_{\text{init}} = x_1 + \alpha\|x_1\|\cdot\xi$, centred.
  - NAG sample 200 steps from $x_{\text{init}}$ → $x^\star$.
  - `token_acc` = mean fraction of positions where $\arg\max x^\star = y$.
  - `token_acc_perturbed` = same for the perturbed input *before*
    sampling (the trivial baseline).
  - $\Delta@\alpha$ = `token_acc − token_acc_perturbed` — the actual
    "work" the sampler did.

$\Delta@.50$ is the headline number because $\alpha\le 0.2$ is too
easy (argmax survives the noise) and $\alpha=1.0$ is too hard (input
is structurally noise). The compositional Δ@.50 of +0.05-0.06 vs the
deterministic 0.000 is the result the recipe was designed to produce.

---

## 9. Where the cells differ — at a glance

| knob                                  | comp_mse_*   | comp_hilbert_*  | comp_ref_det_mse | comp_ref_det_hilbert | comp_smoke            |
|---------------------------------------|--------------|-----------------|------------------|----------------------|------------------------|
| `transformation.dirichlet_sampling`   | true         | true            | **false**        | **false**            | true                   |
| `loss.mode`                           | mse          | hilbert_soft    | mse              | hilbert_soft         | hilbert_soft           |
| `loss.hilbert_alpha`                  | 5.0 (unused) | **1.0**         | 5.0 (unused)     | **1.0**              | 1.0                    |
| `eqm.gamma_power`                     | 1.5          | **1.0**         | 1.5              | **1.0**              | 1.5                    |
| `training.seed`                       | 42 / 43 / 44 | 42 / 43 / 44    | 42               | 42                   | (default)              |
| backbone size                          | d=1024 / 8L  | d=1024 / 8L     | d=1024 / 8L      | d=1024 / 8L          | **d=128 / 2L**         |
| training windows / epochs              | 10k / 5      | 10k / 5         | 10k / 5          | 10k / 5              | **1k / 1**             |

Everything else (LR, optimiser, $\sigma$, $\lambda_{\text{ce}}$,
`ce_min_gamma`, $\lambda$, sampler kwargs, eval-bpd off, no time
conditioning, no embeddings, no auxiliaries, no auditor) is identical
across all eight cells.
