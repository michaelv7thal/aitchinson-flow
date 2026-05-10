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
    # Phase S — alphabet collapse to K=2 (vowel/consonant). When set to
    # "binary" the windows are post-processed by
    # data/text8_binary.py::char_id_to_binary_class. The user is responsible
    # for also setting K=2 (training.K and text8_dataset.K).
    alphabet: str = "full"  # "full" | "binary"

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
    sample_gamma: float = 1.5
    # γ importance sampling: gamma = U(0,1)**gamma_power.
    # gamma_power=1.0 → uniform; <1 pushes mass toward γ=1 (more signal).
    gamma_power: float = 1.5
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
class Config:
    training: TrainingConfigs = field(default_factory=TrainingConfigs)
    text8_dataset: Text8DataConfig = field(default_factory=Text8DataConfig)
    transformation: TransformationConfig = field(default_factory=TransformationConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    eqm: EqM = field(default_factory=EqM)
    dfm: DFMConfig = field(default_factory=DFMConfig)
    dirichlet_fm: DirichletFMConfig = field(default_factory=DirichletFMConfig)
    logitkl: LogitKLFlowConfig = field(default_factory=LogitKLFlowConfig)
    loader_settings: LoaderSettings = field(default_factory=LoaderSettings)
    loss: LossConfig = field(default_factory=LossConfig)
    auditor: AuditorConfig = field(default_factory=AuditorConfig)
    hallueval_dfm_auditor: HalluevalDFMAuditorConfig = field(
        default_factory=HalluevalDFMAuditorConfig
    )
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
