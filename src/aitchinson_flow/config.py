from __future__ import annotations

import torch

from dataclasses import dataclass, field


@dataclass
class TrainingConfigs:
    lr: float = 3e-4
    device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    lr_sheduler: str | None = "cosine"
    """One of: None, ``constant``, ``cosine``."""

    epochs: int = 5

    scheduler_warmup_epochs: int = 0
    cosine_eta_min: float = 0.0
    cosine_t_max_epochs: int | None = None
    scheduler_warmup_start_factor: float = 0.01
    seed: int = 42
    model_name: str = "EqM"
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 5
    log_every: int = 5
    grad_clip_norm: float | None = 1.0
    eval_every: int = 5
    sample_eval_every: int | None = 1  # epochs between unigram-KL probes; None disables
    sample_eval_n: int = 64  # sequences per probe
    sample_eval_steps: int = 100  # NAG-GD steps per probe
    eval_bpd: bool = (
        False  # BPD requires full sampling per batch; disable for fast training evals
    )
    eval_bpd_max_steps: int = 200  # max NAG steps when eval_bpd is True
    # Early stopping on the val loss (runbook §1). None = off (fixed-epoch loop,
    # the historical default). When set, fit() tracks the best val loss across
    # val evals (gated by eval_every) and stops after `early_stop_patience`
    # consecutive evals with no improvement > early_stop_min_delta, restoring the
    # best checkpoint as epoch_final.pt. Requires eval_every to actually trigger
    # val eval (i.e. a val loader + eval_every <= epochs).
    early_stop_patience: int | None = None
    early_stop_min_delta: float = 0.0
    # Hard per-run wall-clock cap in hours (runbook §1 = 36 h). None = off.
    max_wall_clock_hours: float | None = None
    L: int = 40
    K: int = 27
    B: int = 64
    source: str = "raw_text"  # raw_text and or topk

    def __post_init__(self) -> None:
        if not isinstance(self.lr, float):
            raise ValueError(
                f"Learning rate must be a float, received: {self.lr}, type: {type(self.lr)}"
            )


@dataclass
class Text8DataConfig:
    K: int = TrainingConfigs.K  # vocabulary size
    L: int = TrainingConfigs.L  # sequence length
    batch_size: int = TrainingConfigs.B
    enabled: bool = False
    train_corrupt_rate: float = 0.15
    eval_corrupt_rate: float = 0.15
    train_order_mix_rate: float = 0.0
    eval_order_mix_rate: float = 0.0
    order_mix_prob: float = 0.0
    max_train_windows: int | None = 10_000
    max_eval_windows: int | None = 5_000
    corruption_seed: int = 1234
    # Lazy CLR-feature computation. Eager mode (default) precomputes the full
    # (N, L, K) float32 CLR tensor in CharWindowDataset.__init__ — ~9.7 GB at
    # the full ~90M-char split. With lazy_features=True the dataset stores only
    # the integer token-id windows and computes CLR features per-window in
    # __getitem__, so memory scales with batch not corpus. Outputs are identical
    # to eager mode. Auto-enabled when max_train_windows is None or very large
    # (see CharWindowDataset). Default False keeps the eager path byte-identical.
    lazy_features: bool = False
    # Phase S — alphabet collapse to K=2 (vowel/consonant). When set to
    # "binary" the windows are post-processed by
    # data/text8_binary.py::char_id_to_binary_class. The user is responsible
    # for also setting K=2 (training.K and text8_dataset.K).
    alphabet: str = "full"  # "full" | "binary"

    # Variable-L training for AE/EqMAE. When True, the training set yields
    # samples with L ~ U(L_min, L_max) per sample (collate pads + emits
    # pad_mask). Validation stays at fixed L (= cfg.text8_dataset.L) so
    # eval metrics are comparable across cells. The model's positional
    # embedding tables must be sized to L_max.
    variable_length: bool = False
    L_min: int = 40
    L_max: int = 128

    provider: str = "huggingface"
    source_ref: str = "afmck/text8"
    dataset_name: str | None = None
    split_train: str = "train"
    split_val: str | None = "validation"
    split_test: str | None = "test"
    text_column: str = "text"
    cache_dir: str | None = None
    trust_remote_code: bool = False
    streaming: bool = False
    revision: str | None = None


@dataclass
class TransformationConfig:
    label_smoothing: float = 1e-4
    # Dirichlet-sampled data encoding (Stark-FM-style smoothing of the per-token
    # one-hot). When ``True``, batch["x"] is replaced in CorruptingCollate with
    # a fresh draw from Dir(α_base + α_peak·e_token) → CLR each step. The
    # deterministic ``token_ids_to_features`` is still used for label_smoothing-
    # only paths (e.g. EqM.bpd reconstruction). Default False reproduces the
    # original deterministic-CLR pipeline.
    dirichlet_sampling: bool = False
    # Concentration on the target class. Larger ⇒ sharper peak at e_token,
    # closer to the deterministic limit. Calibrated via Phase 2 of
    # scripts/check_dirichlet_data.py: at α_base=0.1 the smallest α_peak
    # giving stable ≥99.5% argmax recovery across seeds is 10.
    dirichlet_alpha_peak: float = 10.0
    # Concentration on each off-target class. Smaller ⇒ heavier-tailed off-
    # target spread (more aggressive smoothing); too small and the simplex
    # samples become bimodal/unstable.
    dirichlet_alpha_base: float = 0.1
    # Stochastic-interpolant noise on the FM path (Albergo, Boffi,
    # Vanden-Eijnden 2023, arXiv:2303.08797). When >0, replaces the
    # deterministic linear interpolant with
    #     x_γ = (1-γ)·x_0 + γ·x_1 + σ(γ)·z,   z ~ N(0, I)|_{V_d},
    # and adds the σ'(γ)·z term to the FM target. σ(γ) = σ_max·sin(πγ)
    # vanishes at γ=0 and γ=1 (so endpoint marginals are preserved) and
    # peaks at γ=0.5 — creating a per-(x_γ,γ) variance floor in the path
    # interior without requiring Dirichlet thickening of x_1.
    sigma_interpolant_max: float = 0.0
    sigma_interpolant_schedule: str = "sin"


@dataclass
class TransformerConfig:
    d_model: int = 1024  # Dimension of the model
    nhead: int = 8  # Number of attention heads
    num_layers: int = 8  # Number of layers
    # Activation (gradient) checkpointing on the encoder layers. Trades ~30%
    # compute for a large drop in activation memory — the memory-fallback
    # lever for the second-order / multi-forward arms (EqM, SFLMEBM_FM,
    # SFLMEBM's hinge) at L=256 on the 20 GB MIG. use_reentrant=False so it
    # composes with create_graph=True. Numerically identical to off.
    grad_checkpointing: bool = False
    d_latent: int = 1024  # Dimension of the latent space
    dropout: float = 0.0  # Dropout rate


@dataclass
class EqM:
    decay_strategy: str = "linear"
    decay_a: float = 0.2
    decay_b: float = 1.0
    gradient_lambda: float = 1.0
    lambda_vol: float = 1e-3  # unused (volume penalty not wired into _eqm_loss)
    lambda_mse: float = 5e-3  # unused
    alpha: float = 1.0  # unused
    # Aux CE on the implied x1 reconstructed from grad_g (anchors per-token attractors).
    # x1 ≈ x_γ − λ·grad_g for linear decay; CE(decode(x1_pred), token_ids).
    lambda_ce: float = 0.5
    ce_min_gamma: float = 0.5  # only apply CE where gamma >= this (signal regime)
    # Optional n-gram likelihood terms on the same implied-x1 reconstruction.
    # Active only at gamma >= ce_min_gamma. Set to 0 to disable. See Phase 5
    # in TRAINING_PLAN.md.
    lambda_bigram: float = 0.0
    lambda_trigram: float = 0.0
    # Optional γ time-conditioning passed to the transformer backbone.
    # "off"     — backbone takes only x_γ (default, original behaviour).
    # "add"     — sinusoidal γ embedding added to per-token hidden state.
    # "concat"  — γ embedding concatenated to hidden, projected back to d_model.
    # When time-conditioned, sample_gamma sets the fixed γ used at sample time.
    # γ=1 is degenerate because c(γ=1)·(x0-x1)=0 ⇒ velocity ≈ 0 ⇒ flat
    # energy field at sample time. γ=0.5 sits in the signal regime (matches
    # ce_min_gamma).
    time_conditioning: str = "off"
    sample_gamma: float = 0.5
    # γ importance sampling: gamma = U(0,1)**gamma_power.
    # gamma_power=1.0 → uniform; <1 pushes mass toward γ≈1 (more signal),
    # >1 toward γ≈0 (noise). 0.5 (mean γ≈0.67, mass in the signal regime) is
    # the load-bearing anti-mode-collapse value documented in SESSION_SUMMARY.md
    # fix #4 / CLAUDE.md. NB: commit 2e15b3f silently flipped this (and
    # sample_gamma) to 1.5 — the *noise* regime, the inverse of the fix —
    # which was restored here. Don't re-flip without re-reading SESSION_SUMMARY.
    gamma_power: float = 0.5
    # x0 source noise scale (used at both train and inference for distribution match).
    source_sigma: float = 0.1
    # NAG-GD sampling hyperparameters (Algorithm 2)
    sample_eta: float = 0.1  # step size η
    sample_mu: float = 0.9  # NAG look-ahead factor μ
    sample_g_min: float = 0.01  # gradient-norm stopping threshold
    sample_max_steps: int = (
        200  # hard cap on iterations (no benefit beyond ~150 once clipped)
    )
    # Per-position L2 clip on ∇E during sampling. Kills the cold-start gradient
    # spike (||∇E||_pos ≈ 4.6 → 9.7 at step 1) that NAG amplifies into a basin
    # overshoot. None disables clipping. See SAMPLER_FINDINGS.md.
    sample_grad_clip: float | None = 1.0
    # Return the lowest-‖∇E‖ iterate seen, not the trajectory's endpoint. NAG
    # overshoots the basin around step 60 and drifts away; the trajectory
    # minimum is the right thing to return.
    sample_return_best: bool = True
    # Sampler selection: "nag" (NAG-GD on the conservative gradient — the
    # original behaviour) or "euler" (FM-style Euler integrator over γ on the
    # raw velocity f, see RESULTS.md §1). Training is unaffected; only
    # inference changes.
    sampler: str = "nag"
    # Number of Euler steps when sampler="euler". Maps from runner.py's
    # max_steps so the existing eval probe wiring works untouched.
    euler_nfe: int = 64
    # Ablation: if True, the Euler sampler uses ∇⟨x,f⟩ (the conservative
    # gradient) at each step instead of raw f. Lets the writeup separate
    # "FM-style sampler" from "raw-f vs grad-of-energy".
    euler_use_grad: bool = False
    # Optional override of source_sigma at sample time. None ⇒ use source_sigma.
    # The OOD-at-γ≈0 mismatch hypothesis says reducing this may help the Euler
    # sampler that starts at γ=0 (where training sees x_γ ≈ x0 = σ-noise).
    sample_sigma_init: float | None = None
    # Langevin diffusion coefficient α for the SDE sampler (sampler="sde").
    # Per-step noise scale is sqrt(2·α·h) where h = 1/nfe. α = 0 reduces to
    # deterministic Euler; small α (~0.05–0.2) adds mixing without
    # destroying the trajectory; α > 0.5 is essentially Brownian motion.
    sde_alpha: float = 0.0
    # Joint-head bigram NLL weight (W2). 0 disables. Distinct from the
    # factorised lambda_bigram above (which Phase 5 showed is a re-weighted
    # unigram CE — kept here only for reproducibility of that negative).
    lambda_bigram_joint: float = 0.0
    # Auditor hinge loss (Phase F — TRAINING_PROTOCOL.md §6).
    # Active when ``lambda_E_hinge > 0`` AND the batch carries an
    # ``x_invalid`` paired-input tensor. Trains the field so that
    # per-sequence grad-norm² ``Σ‖∇⟨x,f(x;γ_aud)⟩‖²`` is *small* on clean
    # inputs and *at least margin_energy²* on invalid inputs. This
    # repurposes the EqM energy as a binary clean/invalid discriminator;
    # FM regression on the clean samples is still computed, so the field
    # is anchored on the data manifold.
    lambda_E_hinge: float = 0.0
    margin_energy: float = 2.0
    # γ at which the auditor's energy proxy is evaluated. γ=1 is the
    # data-manifold endpoint where FM training drives grad_g → 0 on
    # clean inputs. Use γ=1.0 for time-conditioned auditors; for
    # untimed it's ignored (the backbone has no γ input).
    auditor_gamma: float = 1.0
    # Context conditioning mode for the auditor backbone.
    # "off"             — backbone sees only the K-dim simplex (default).
    # "hidden_only"     — backbone sees only the LM's last-hidden state
    #                     (the simplex is dropped). Auditor decides
    #                     purely from the LM's representation.
    # "product_concat"  — concat the K-dim simplex with a learned
    #                     projection of the LM hidden state, then
    #                     project to d_model. Both signals are visible.
    # Activates when the batch carries ``h_clean`` / ``h_invalid``.
    context_features: str = "off"
    # LM hidden-state dim. GPT-2 small=768, GPT-2 medium=1024,
    # Qwen2.5-1.5B=1536. Set to match the cache producer.
    ctx_hidden: int = 768
    # Width of the learned projection applied to ``h_LLM`` before it is
    # concatenated to the simplex input in "product_concat" mode. None
    # defaults to d_model // 2.
    ctx_proj_dim: int | None = None
    # Aux loss family on the implied-x1 reconstruction.
    # "softmax"   — log_softmax + NLL (the default, used everywhere prior).
    # "sparsemax" — Martins & Astudillo (2016) sparsemax loss; the implied-x1
    #               distribution is computed via sparsemax instead of softmax,
    #               producing sparse posteriors over tokens. Combined with the
    #               existing bigram_joint head this is a discrete-side
    #               symmetry-break alternative to Dirichlet x_1 thickening.
    aux_kind: str = "softmax"


@dataclass(frozen=True)
class LoaderSettings:
    batch_size: int = TrainingConfigs.B
    num_workers: int = 8
    device_type: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )


@dataclass
class LossConfig:
    mode: str = "mse"  # "hilbert" | "hilbert_soft" | "mse"
    hilbert_alpha: float = 5.0  # target alpha for "hilbert_soft"
    hilbert_alpha_start: float = 5.0  # initial alpha; anneals up to hilbert_alpha
    hilbert_alpha_anneal_epochs: int | None = (
        None  # epochs to reach hilbert_alpha; None = all training epochs
    )


@dataclass
class DFMConfig:
    kappa_schedule: str = "quadratic"  # "linear" | "quadratic"
    dropout: float = 0.1
    sample_nfe: int = 128  # default Euler steps at eval time


@dataclass
class DirichletFMConfig:
    """Dirichlet Flow Matching (Stark et al. 2024, arXiv:2402.05841).

    Conditional probability path on the K-1 simplex:
        p_{t|1}(x | x_1) = Dir(x; β(t, x_1))   with β_i = α(t) if i=x_1 else 1
    α is parameterized directly as t ∈ [1, t_max]. The path goes from
    Dir(1,...,1) (uniform on the simplex, t=1) to a Dirichlet concentrated
    at the vertex e_{x_1} as t grows. The denoiser predicts p(x_1 | x_t, t)
    under cross-entropy. Sampling integrates the marginal vector field
    derived in Theorem 3.1 of the paper.
    """

    t_max: float = 8.0  # paper default for moderate K; α_max in the paper
    dropout: float = 0.1
    sample_nfe: int = 100  # Euler steps for sampling (Stark uses 100)
    # If True, add small Gaussian noise to the input each step for stability
    # (the paper's stochastic variant). Off by default — pure ODE.
    sample_stochastic: bool = False
    # Auditor extension: condition the denoiser on a per-position context vector
    # (the LM's last hidden state h_LLM). "off" disables; "product_concat" mirrors
    # EqM's mode and concats h_proj(h_LLM) onto x before the input projection.
    context_features: str = "off"  # "off" | "hidden_only" | "product_concat"
    ctx_hidden: int = 768  # raw context dim (GPT-2 small = 768)
    ctx_proj_dim: int = 64  # projection size before concat
    # EBM-eval-time t (where the closed-form mixture-of-Dirichlets log p_t is
    # evaluated for UQ). Closer to t_max ⇒ peakier prior; closer to 1 ⇒ uniform.
    energy_t: float = 4.0
    # Architecture B (dual-head joint training): when ``joint_halluc=True`` the
    # encoder grows a second head that emits a scalar hallucination logit per
    # position; the training step combines slot-CE on clean rows with BCE on
    # all rows. Encoder learns features useful for both objectives.
    joint_halluc: bool = False
    lambda_slot: float = 1.0
    lambda_halluc: float = 1.0
    # Per-row pos:neg imbalance in HaluEval-QA answer-mask positions is ~6:1
    # (hallucinated answers are systematically longer than clean ones). Use a
    # pos_weight in BCE to compensate; auto-computed in the runner if <= 0.
    halluc_pos_weight: float = 0.0
    # Encoder architecture. ``"transformer"`` is the default cross-positional
    # attention model. ``"mlp"`` is a weight-shared per-position MLP — same
    # input/output API but with NO cross-positional information flow. The MLP
    # backbone is cascade-clean by construction (tok(non-ans) AUROC ≈ 0.50)
    # and is the principled choice when locality matters more than
    # row-level discrimination via cross-position cues.
    backbone_kind: str = "transformer"  # "transformer" | "mlp"
    # Slot-CE training data filter. When True (default) the slot head sees
    # only clean rows — the original "one-class density on clean" recipe.
    # When False, slot CE is trained on **all** rows without inspecting the
    # halluc label, accepting label-noise contamination from halluc rows.
    # This is the strictly self-supervised / unsupervised variant.
    train_clean_only: bool = True
    # Architecture A (post-hoc SVGP head): when ``svgp_head=True`` the
    # auditor exposes a sparse variational GP over encoder features
    # (gpytorch ApproximateGP + RBF + Bernoulli likelihood). It is **not**
    # jointly trained with the encoder — main optimizer ignores its
    # parameters; it is fitted post-hoc via ``model.fit_svgp(loader)`` and
    # queried via ``model.svgp_score_at_lm(batch)`` for per-position mean +
    # variance. Same gpytorch recipe used by ``scripts/phaseF_uq.train_svgp``.
    svgp_head: bool = False
    svgp_n_inducing: int = 128
    svgp_n_iters: int = 200
    svgp_lr: float = 0.01


@dataclass
class LogitKLFlowConfig:
    """Logit-KL Flow Matching (arXiv:2411.16821).

    Linear interpolation in logit space ``l_t = (1−t)·l_0 + t·l_1`` with
    ``l_0 ~ source_sigma·N(0, I)`` and ``l_1 = gamma_l · one_hot(token_id)``.
    The denoiser regresses *clean logits* ``v̂(l_t, t) ≈ E[l_1 | l_t]`` under
    MSE. Sampling is hybrid: deterministic-ODE for ``t < sampler_split_t``,
    stochastic re-noising for ``t ≥ sampler_split_t``.
    """

    gamma_l: float = 8.0  # clean-logit magnitude (one-hot · γ_l)
    source_sigma: float = 0.1  # σ for the Gaussian noise source l_0
    sampler_split_t: float = 0.28  # det → stochastic switch threshold
    sampler_nfe: int = 64  # default Euler steps at eval time
    sampler_noise_scale: float = 0.5  # multiplier on σ_t = sqrt(1−t²)


@dataclass
class AuditorConfig:
    """Phase F (TRAINING_PROTOCOL.md §6) — EqM auditor on cached LM features.

    When ``enabled = True`` the runner builds a :class:`WikiAuditorDataModule`
    instead of the text8 module; the EqM model's ``lambda_E_hinge`` should
    also be > 0 so the paired-input branch in ``training_step`` fires. The
    cache itself is produced one-time by ``scripts/cache_wiki.py``.
    """

    enabled: bool = False
    cache_path: str = "data/wiki_cache_gpt2.pt"
    batch_size: int = 8
    train_frac: float = 0.8


@dataclass
class HalluevalDFMAuditorConfig:
    """DirichletFM auditor on cached HaluEval-QA top-K + h_LLM features.

    Pairs two caches produced for Phase K / Phase Q:
      * `topk_cache_path`   — `(topk_logp, topk_idx, E_logit, E_marg, ΔE)` per
        position; K=32, L=160 (from `scripts/cache_hallueval_topk.py`).
      * `hidden_cache_path` — `(full_ids, hidden_states, answer_mask, label,
        pair_id)` (from `scripts/cache_hallueval.py`); 20000 rows of which
        the first 4000 align with the topk cache.

    The auditor trains its denoiser only on **clean** rows (label=False). At
    eval time the closed-form mixture-of-Dirichlets EBM `−log p_t(x_LM | h)`
    is evaluated at the LM's actual top-K simplex distribution; AUROC vs the
    per-pair label is the headline number.
    """

    enabled: bool = False
    topk_cache_path: str = "data/hallueval_topk_gpt2.pt"
    hidden_cache_path: str = "data/hallueval_cache_gpt2.pt"
    batch_size: int = 16
    # Pair-level train/val split (prevents leakage across the clean/halluc
    # halves of the same prompt).
    train_frac: float = 0.8
    # Subset cap on rows used (a) for fast smoke runs and (b) when the topk
    # cache is smaller than the hidden cache (default 4000-row topk subset).
    max_rows: int = 4000


@dataclass
class EmbeddingConfig:
    """Latent-EqM (`EqMLatent` model). Replaces the deterministic CLR-on-simplex
    representation with a learnable ``nn.Embedding(K, d_embed)`` + tied output
    decoder. The flow runs in plain ``R^{d_embed}`` (no V_d projection); the
    model jointly trains embeddings to be flow-friendly *and* linearly
    separable for the CE auxiliary.

    Phase 0-3 of the simplex pipeline (Dirichlet-sampled CLR data + Hilbert vs
    Aitchison) demonstrated that the simplex *representation* — not the metric
    on it — is the bottleneck (RESULTS_DIRICHLET.md). This config feeds the
    pivot to learned embeddings.
    """

    enabled: bool = False
    d_embed: int = 32
    init_std_factor: float = 1.0  # final init std = init_std_factor / sqrt(d)
    # If > 0, freeze the embedding rows for the first N optimizer steps so the
    # flow field finds a sensible starting energy landscape before the
    # embedding rows start moving. Phase 2-3 calibration knob.
    freeze_steps: int = 0
    # Optional separate learning rate multiplier for the embedding rows.
    # ``None`` ⇒ same lr as the rest of the model (no param-group split).
    embed_lr_mult: float | None = None
    # Untied vs tied output decoder. Tied (default) shares ``embed.weight`` with
    # the readout linear layer; untied trains a separate ``nn.Linear(d, K)``.
    tie_decoder: bool = True
    # Pre-computed fixed embedding source. When set, EqMLatent loads the
    # ``embeddings`` tensor from this ``torch.save``-d file (produced by
    # ``scripts/learn_ppmi_svd_embeddings.py``) and **freezes** the embedding
    # layer (``requires_grad_(False)``). This eliminates the moving-target
    # failure mode of jointly training embeddings — x_1 = embed(token) is
    # constant throughout training, so the flow regression has a fixed
    # target field.
    fixed_path: str | None = None


@dataclass
class AutoencoderConfig:
    """TextAutoencoder — contextual denoising AE used as the latent space for
    ``EqMAE``. Architecture: token+pos embedding → N-layer Transformer encoder
    → Linear(d_latent), then Linear(d_latent) → token+pos embedding → N-layer
    Transformer encoder → Linear(K) head.

    The denoising_sigma noise on z during training is what makes the latent
    space FM-friendly: the decoder learns to be robust to small perturbations
    of z, so the EqM sampler doesn't need to land exactly on the data
    manifold for the decode to be valid text. ``latent_l2`` keeps ‖z‖ bounded
    so the EqM source noise can match the data scale without re-tuning.
    """

    d_model: int = 256
    d_latent: int = 64
    nhead: int = 4
    num_layers: int = 2
    dropout: float = 0.0
    # Denoising noise on the encoded z before decoding. Two regimes:
    #   "fixed"            — σ = denoising_sigma (absolute, per-dim)
    #   "relative_uniform" — σ = U(0, denoising_sigma) * std(z),  per-batch
    # The relative-uniform schedule is the default because (a) it scales
    # with whatever z scale the AE settles into (so the noise/signal ratio
    # is comparable across AE sizes) and (b) sweeping σ from 0 to max in
    # one training trains the decoder to be robust across the *full
    # spectrum* of residuals the downstream EqM sampler might leave behind
    # — not just one specific noise magnitude.
    denoising_schedule: str = "relative_uniform"
    denoising_sigma: float = 0.5
    latent_l2: float = 1e-3
    # Best-practice defaults: GELU FFN activation (modern transformer norm)
    # and tied I/O embeddings (decoder head shares weight with encoder
    # token_embed). Tying forces the encoder representation to be linearly
    # separable for the decoder and saves K*d_model params on the head.
    activation: str = "gelu"
    tie_embeddings: bool = True
    # AE | VAE flavour. "ae" is the deterministic encoder used so far;
    # "vae" replaces to_latent with parallel μ and logσ heads, samples
    # z = μ + σ·ε via reparameterisation, and adds a KL(q(z|x) || N(0,I))
    # term to the loss with weight ``vae_beta``. Sampling z each step
    # also gives EqMVAE training the "thickened-x1" property the
    # compositional simplex EqM gets from Dirichlet thickening, but in
    # latent space.
    mode: str = "ae"
    vae_beta: float = 0.1
    vae_beta_warmup_epochs: int = 1


@dataclass
class EqMAEConfig:
    """EqMAE — EqM in a frozen pretrained-AE latent space. Pairs with
    ``AutoencoderConfig`` for the AE architecture (must match the pretrained
    checkpoint's arch) and ``EqM`` for the flow hyperparameters."""

    ae_ckpt_path: str = ""


@dataclass
class BayesianAuditorConfig:
    """BayesianAuditorAE — GP-energy EBM with contrastive hinge.

    Implements the supervisor-suggested design: GP posterior mean as a scalar
    energy E(x), with conservative-gradient velocity v = -∇E, FM regression
    loss plus a contrastive hinge against perturbation-based negatives
    produced by ``CorruptingCollate`` (set ``text8_dataset.train_corrupt_rate``
    > 0 to enable; default 0.15).
    """

    ae_ckpt_path: str = ""
    num_inducing: int = 64
    lambda_hinge: float = 1.0      # weight on E(clean)² + relu(margin - E(invalid))
    margin_energy: float = 2.0
    lambda_kl: float = 0.01        # weight on the variational KL / B
    # (variance hinge omitted in the AE-latent variant — the GP posterior over a
    # pooled deep feature is not well-calibrated as an OOD signal without further
    # work; left for a follow-up.)
    # ── product-kernel (wiki/GPT-2) extension ─────────────────────────────
    # Used by ``PerTokenBayesianAuditorWiki`` (cfg.training.model_name).
    # The context dim ``ctx_hidden`` matches the LM's last-hidden-state dim
    # (768 for GPT-2 small); ``ctx_proj_dim`` is the projected per-position
    # context dim consumed by the product kernel's second factor.
    ctx_hidden: int = 768
    ctx_proj_dim: int = 64
    # Cache path used by the wiki/GPT-2 datamodule. Produced by
    # ``scripts/cache_wiki.py`` with --lm gpt2 --K 64.
    wiki_cache_path: str = "data/wiki_cache_gpt2.pt"


@dataclass
class DSMConfig:
    """Denoising-score-matching family (ScoreDSM, EqMDSM).

    Both models train a noise-conditional score on a frozen AE latent. They
    share this config block. The model_name selects which parameterisation
    of the score is used (direct vector field vs. ∇⟨x, f⟩ — i.e. the
    energy-gradient form that recovers EqM's conservative-field structure).

    Conditioning. Both models pass ``log(σ)`` through the backbone's
    existing γ-conditioning path (`eqm.time_conditioning` must be "add" or
    "concat"). This means the backbone is unchanged; the σ-embedding is
    just the sinusoidal embedding of the log-noise-level.
    """

    # σ schedule for training-time noise sampling.
    # Default range covers ~99% of the embedding-norm dynamic range you
    # observe on text8 AE latents (z_norm ~6–10).
    sigma_min: float = 0.05
    sigma_max: float = 5.0
    sigma_distribution: str = "log_uniform"  # "log_uniform" | "uniform"

    # ε-prediction loss weighting λ(σ). "constant" = 1 (matches
    # variance-preserving DDPM); "snr+1" = σ²+1 (Karras 2022 weighting).
    loss_weighting: str = "constant"

    # Aux CE on the implied-x1 reconstruction, same role as in EqM.
    # ``ce_min_logsig`` masks the CE to small-σ samples (where the
    # implied-x1 is reliable). Translation of EqM's ce_min_gamma but
    # measured in log-σ rather than γ.
    lambda_ce: float = 0.5
    ce_min_logsig: float = 0.0  # only CE on σ <= exp(0) = 1.0 by default

    # Annealed Langevin sampler.
    n_sigma: int = 16  # σ ladder size (geometric between σ_max and σ_min)
    steps_per_sigma: int = 8  # K Langevin steps at each σ
    sampler_eps: float = 1e-5  # base step size; α_i = eps · (σ_i/σ_min)²
    sample_use_predictor_corrector: bool = False

    # AE pretrained checkpoint path (mirrors EqMAEConfig — same AE in
    # all three cells of the sweep).
    ae_ckpt_path: str = ""


@dataclass
class SFLMEbmConfig:
    """SFLMEBM — time-free hyperspherical flow model read as an EBM.

    Tokens get a learned codebook embedding projected onto S^{d-1}. A
    time-free denoiser is trained with CE on SLERP-noised latents
    (S-FLM, arXiv:2605.11125); the implicit energy is the EqM-style
    log-sum-exp readout ``E(z) = -tau * logsumexp_v <h(z), e_v>/tau``
    (arXiv:2510.02300). The SFLM_EBM_FINDINGS.md probe showed the
    energy basin is correct (recover≈1.0) under pure CE; the
    importance schedule + contrastive hinge below are the EqM
    anti-collapse knobs, kept as one-flag ablations.

    The model ignores ``batch["x"]`` (CLR features) and does the
    embedding lookup from ``batch["token_ids"]`` internally, mirroring
    ``EqMLatent``. Uses ``cfg.transformer`` for backbone width/depth.
    """

    d_embed: int = 64
    tau: float = 0.1  # codebook-softmax / energy temperature
    # SLERP noise schedule. "uniform": alpha~U(lo,hi); "trunc": U(0.5,hi);
    # "import": hi*U(0,1)**0.5 (EqM gamma-power, mass toward the signal end).
    alpha_sched: str = "import"
    alpha_lo: float = 0.0
    alpha_hi: float = 0.95
    # CE only applied where alpha >= this (signal regime; EqM ce_min_gamma).
    ce_min_alpha: float = 0.3
    # Contrastive energy hinge: relu(margin + E_clean - E_neg) with
    # negatives = {token_ids_invalid embeddings, centroid seq, uniform
    # sphere}. 0 disables (pure-CE ablation). Needs corruption enabled
    # (text8_dataset.train_corrupt_rate > 0) for the invalid negatives.
    lambda_hinge: float = 1.0
    hinge_margin: float = 0.5
    # Riemannian-GD sampler (adaptive step in geodesic arc length).
    sample_steps: int = 200
    sample_target_step: float = 0.1
    # ----- Option 2 (conservative-gradient FM target on the sphere) -----
    # When > 0, the model additionally regresses the Riemannian gradient of
    # its own energy onto the geodesic FM target ``c(α)·log_map(z_α, z1)``
    # (Chen & Lipman 2023). This makes the same energy density-faithful
    # *and* generatively traversable — the existing energy-GD ``sample()``
    # then transports noise→data instead of failing to (see
    # SFLM_EBM_FINDINGS.md option 2). Requires second-order autograd
    # (create_graph=True) — heavier than CE-only. Recommend running with
    # lambda_hinge=0 in this regime to test whether FM supervision alone
    # gives both generation and OOD; the hinge can be re-enabled later
    # as an ablation. 0 disables → exactly current SFLMEBM behaviour.
    lambda_fm: float = 0.0
    # Path-decay c(α). "linear": c(α) = 1−α (matches EqM default).
    fm_c_decay: str = "linear"


@dataclass
class SFLMConfig:
    """Proper S-FLM Stage-1 generator (arXiv:2605.11125 in spirit, on
    text8). Time-conditioned hyperspherical denoiser trained with plain CE
    on SLERP-noised latents; sampled by Euler-over-γ on the sphere via
    x1-prediction + geodesic step. Pairs with a Stage-2 SVGP head
    (post-hoc, mirrors DirichletFMSvgp) for OOD — see
    ``SFLM_EBM_FINDINGS.md``.
    """

    d_embed: int = 128
    tau: float = 0.1
    alpha_sched: str = "import"   # γ ~ alpha_hi · U(0,1)**0.5
    alpha_lo: float = 0.0
    alpha_hi: float = 0.95
    # γ conditioning: "add" injects sinusoidal-γ via an extra projection;
    # "concat" concatenates it onto the per-token hidden state and projects
    # back. Both are standard; "add" is cheaper.
    time_conditioning: str = "add"
    # Eval-time γ used when the bench / recovery_check calls
    # ``decode_to_logprobs(z)`` without a γ argument (signal-regime
    # equivalent of DFM's t≈1 evaluation).
    eval_gamma: float = 0.95
    # Euler-over-γ sampler: nfe steps, x1-prediction + geodesic SLERP step.
    sample_nfe: int = 64


@dataclass
class SFMConfig:
    """Statistical Flow Matching (Cheng et al. 2024, arXiv:2405.16441).

    Categorical flow matching on the statistical (Fisher–Rao) manifold. The
    simplex is mapped to the positive orthant of S^{K-1} by π: μ ↦ √μ (inverse
    μ = x²), under which the Fisher metric becomes the round sphere metric. The
    conditional path is the constant-speed great-circle geodesic from a
    uniform-simplex source (t=0, same prior as DirichletFM, mapped through π) to
    the data vertex e_c (t=1); a transformer velocity field v(x_t, t) in the
    tangent space is regressed against the geodesic velocity with an MSE
    flow-matching loss. Sampling integrates ẋ = v on the sphere and reads tokens
    from μ = x². The exact CNF likelihood / peer-comparable BPC (paper Eqs.
    12–14) is a separate post-hoc readout, not part of training.
    """

    sample_nfe: int = 100   # exp-map Euler steps for the sampling ODE
    t_eps: float = 1e-3     # keep geodesic time in [0, 1-t_eps] (stable target)


@dataclass
class WandbConfig:
    """Optional Weights & Biases logging. Off by default; enable with --wandb."""

    enabled: bool = True
    project: str = "eqm-text8"
    entity: str | None = None
    run_name: str | None = None  # auto-derived in main.py if None
    group: str | None = None  # set by run_sweep.py to "sweep:<spec_stem>"
    tags: tuple[str, ...] = ()
    mode: str = "online"  # online | offline | disabled
    log_artifacts: bool = True  # upload epoch_final.pt
    log_samples: bool = True  # decoded text Table at each probe
    sample_count: int = 8
    step_log_every: int = 1  # throttle per-step wandb.log() calls


@dataclass
class DFMSvgpConfig:
    """DirichletFMSvgp: Dirichlet Flow Matching + post-hoc SVGP head for OOD.

    Stage 1 is identical to DirichletFlowMatching; this config controls
    only the SVGP head and its training set construction.
    """

    pooling: str = "mean"               # "mean" | "max" | "attention"
    d_embed: int = 256                  # SVGP input dim (pooled projection)
    n_inducing: int = 128               # SVGP inducing-point count
    kernel: str = "matern52"            # "matern52" | "rbf" | "matern32"
                                         # Matern-5/2 is default: C² samples,
                                         # heavier tails than RBF → better-graded
                                         # OOD variance, and consistent with
                                         # this repo's _SparseGP (which also
                                         # uses Matern-5/2). RBF (C∞ samples,
                                         # Gaussian tails) saturates variance
                                         # too quickly far from inducing pts
                                         # for OOD discrimination.
    t_eval: float = 4.5                 # Dirichlet path time at which to extract
                                         # features (mid-path; signal+noise balanced)
    n_pos: int = 4000                   # number of positive features for SVGP fit
    neg_strategy: str = "scrambled"     # "scrambled" | "random_simplex" | "mixed"
    neg_per_pos_ratio: float = 1.0      # how many negatives per positive
    n_iters: int = 200                  # SVGP optimisation iters
    lr: float = 0.01                    # SVGP optimiser LR
    train_pooler_with_dfm: bool = False # if True, pooler gradients flow through
                                         # Stage 1 (cheap regularisation); False
                                         # keeps Stage 1 identical to vanilla DFM.
    # ----- joint contrastive-hinge training (alternative to post-hoc fit) ---
    lambda_hinge: float = 0.0           # contrastive energy hinge weight;
                                         # 0 = original two-stage (post-hoc SVGP fit only)
                                         # >0 = trains SVGP jointly during Stage 1
                                         # using batch["token_ids_invalid"] from
                                         # CorruptingCollate.
    margin_energy: float = 2.0          # hinge margin: E_invalid should exceed this
    train_svgp_jointly: bool = False    # set True alongside lambda_hinge>0 to
                                         # unfreeze the SVGP params during Stage 1
                                         # (pooler also unfrozen — hinge gradient
                                         # has to reach the encoder)
    hinge_t_eval: float | None = None   # path-time t at which to compute the
                                         # hinge during training (None → use t_eval)


@dataclass
class Config:
    training: TrainingConfigs = field(default_factory=TrainingConfigs)
    text8_dataset: Text8DataConfig = field(default_factory=Text8DataConfig)
    transformation: TransformationConfig = field(default_factory=TransformationConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    eqm: EqM = field(default_factory=EqM)
    dfm: DFMConfig = field(default_factory=DFMConfig)
    dirichlet_fm: DirichletFMConfig = field(default_factory=DirichletFMConfig)
    dfm_svgp: DFMSvgpConfig = field(default_factory=DFMSvgpConfig)
    logitkl: LogitKLFlowConfig = field(default_factory=LogitKLFlowConfig)
    loader_settings: LoaderSettings = field(default_factory=LoaderSettings)
    loss: LossConfig = field(default_factory=LossConfig)
    auditor: AuditorConfig = field(default_factory=AuditorConfig)
    hallueval_dfm_auditor: HalluevalDFMAuditorConfig = field(
        default_factory=HalluevalDFMAuditorConfig
    )
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    eqm_ae: EqMAEConfig = field(default_factory=EqMAEConfig)
    dsm: DSMConfig = field(default_factory=DSMConfig)
    bayes_auditor: BayesianAuditorConfig = field(default_factory=BayesianAuditorConfig)
    sflm_ebm: SFLMEbmConfig = field(default_factory=SFLMEbmConfig)
    sflm: SFLMConfig = field(default_factory=SFLMConfig)
    sfm: SFMConfig = field(default_factory=SFMConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
