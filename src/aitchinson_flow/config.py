from __future__ import annotations

import math
import torch

from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class DatasetConfig:
    dataset: str = "text8"  # "dna", "text8", "wiki"
    K: int = 27  # Vocabulary size (4=DNA, 27=text8, 64=wiki top-k)
    L: int = 30  # Sequence length


@dataclass
class TransformerConfig:
    d_model: int = 1280  # Dimension of the model
    nhead: int = 8  # Number of attention heads
    num_layers: int = 8  # Number of layers
    d_latent: int = 1280  # Dimension of the latent space
    dropout: float = 0.0  # Dropout rate
    time_conditioned: bool = False  # Whether to condition on time
    pretrained_backbone: str | None = None  # e.g. "gpt2", "gpt2-medium", "gpt2-large"

    def __post_init__(self) -> None:
        if self.d_model % self.nhead != 0:
            raise ValueError(
                f"TransformerConfig.d_model ({self.d_model}) must be divisible by "
                f"nhead ({self.nhead})"
            )


@dataclass
class GPConfig:
    num_inducing: int = 500  # Number of inducing points
    lambda_kl: float = 1.0  # KL regularization parameter
    lambda_contrastive: float = 0.1  # Stage 2 random-negative energy hinge weight
    lambda_var: float = 2.0
    """Anchor weight on valid GP mean in the composed ``BayesianAuditor`` loss.

    Keeps ``E[valid]`` near zero so the contrastive hinge does not pull the
    whole energy surface upward. The historical name is retained for
    checkpoint compatibility; see also ``lambda_anchor`` for the Stage 2
    analogue.
    """
    lambda_anchor: float = 0.2
    """Stage 2 anchor weight on ``E[valid]^2`` (defaults to off).

    Mirrors ``lambda_var`` in the composed auditor. Enable when relaxing the
    contrastive hinge (C2 fix removes the ``.detach()``) causes valid energy
    to drift.
    """
    margin_E: float = 1.0  # Hinge margin for L_energy score
    margin_V: float = 1.0
    """Hinge margin for variance separation.

    Used only by ``per_token_bayesian_auditor`` in the single-stage codepath;
    has no effect in the two-stage pipeline (Stage 2 has no variance hinge).
    """
    init_log_noise_var: float = math.log(0.1)
    """Initial value for the GP's ``log_noise_var`` parameter (log-space).

    Defaults to ``log(0.1)`` → noise variance of ``softplus(log(0.1)) ≈ 0.1``.
    Increase (e.g. ``log(1.0)``) for noisier data or when the NLL saturates
    early in training.
    """
    min_log_noise_var: float = math.log(0.01)
    """Lower clamp applied to ``log_noise_var`` before ``softplus``.

    Prevents the learned aleatoric noise from underflowing to zero, which
    would make the NLL numerically unstable.  Defaults to ``log(1e-6)``
    so the minimum representable noise variance is ≈ 1e-6.
    """

    score_answer_tokens_only: bool = False
    """If True, restrict Stage 2 token-level NLL and contrastive loss to
    answer-span positions (requires ``batch["answer_mask"]``).

    Used by the Q+A hallucination auditor (Path B): question tokens are
    identical across correct/incorrect pairs and would dilute the
    contrastive signal. Defaults to ``False`` so existing text8/Path A
    batches (which lack ``answer_mask``) behave unchanged.
    """

    def __post_init__(self) -> None:
        if self.num_inducing < 1:
            raise ValueError(f"GPConfig.num_inducing must be >= 1, got {self.num_inducing}")
        if self.margin_E <= 0.0:
            raise ValueError(
                f"GPConfig.margin_E must be > 0 (a non-positive margin makes the "
                f"contrastive hinge vacuous), got {self.margin_E}"
            )
        if self.margin_V <= 0.0:
            raise ValueError(
                f"GPConfig.margin_V must be > 0 (a non-positive margin makes the "
                f"variance hinge vacuous), got {self.margin_V}"
            )
        for name in ("lambda_kl", "lambda_contrastive", "lambda_var", "lambda_anchor"):
            v = getattr(self, name)
            if v < 0.0:
                raise ValueError(
                    f"GPConfig.{name} must be >= 0 (negative values invert the "
                    f"loss direction), got {v}"
                )


@dataclass
class TrainingConfig:
    model_name: str = "flow_matching"
    """Registered model key (see ``aitchinson_flow.models.factory``). Known values:

    - ``"flow_matching"``: time-conditioned CFM-style auditor (velocity head).
    - ``"equilibrium"``: standalone Equilibrium Matching auditor (velocity head).
    - ``"bayesian_generator"``: joint flow + GP generator (no contrastive terms).
    - ``"bayesian_auditor"``: joint flow + GP + contrastive (single-stage).
    - ``"bayesian_auditor_stage1"``: **Stage 1** of the two-stage auditor —
      EqM + Hilbert backbone training. Requires ``velocity_loss`` set to a
      Hilbert-family loss (``soft_hilbert``/``hard_hilbert``/``clr_mse``/
      ``ilr_mse``). Batches only need ``log_x``.
    - ``"bayesian_auditor_stage2"``: **Stage 2** of the two-stage auditor —
      mandatory-frozen backbone + contrastive GP head. Batches need
      ``log_x`` and ``log_x_invalid``.
    - ``"per_token_bayesian_auditor"``, ``"frozen_backbone_auditor"``: see
      their respective modules.

    Two-stage workflow:
        1. Train ``bayesian_auditor_stage1`` on valid sequences (EqM + Hilbert).
        2. Independently train ``bayesian_auditor_stage2`` with valid/invalid
           contrastive pairs (optionally starting from a random backbone if
           no Stage 1 checkpoint is used — representation freezing is still
           enforced, matching the ablation setting documented in the plan).
        3. Fuse checkpoints for inference via
           ``aitchinson_flow.models.load_auditor_from_stage_checkpoints`` or
           ``compose_auditor_from_stages``; the resulting ``BayesianAuditor``
           supports the full inference API (``forward``/``_velocity``,
           ``ood_score``, ``per_token_uq``, ``audit``).
    """
    B: int = 128  # Batch size
    epochs: int = 15  # Number of epochs
    lr: float = 5e-4  # Learning rate
    loss: str = "hilbert"  # "hilbert" or "mse"
    device: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    seed: int = 0
    grad_clip_norm: float | None = None  # if set, clip after backward
    log_every: int = 50
    eval_every: int = 5  # epochs between validation
    checkpoint_every: int = 5
    checkpoint_dir: str = "checkpoints"
    num_workers: int = 8
    weight_decay: float = 0.0
    # Step once per epoch in ``runner.fit`` (see ``build_scheduler``).
    lr_scheduler: str | None = "cosine"
    """One of: None, ``constant``, ``cosine``, ``cosine_restarts``, ``onecycle``, ``linear``,
    ``polynomial``, ``exponential``, ``multistep``."""

    scheduler_warmup_epochs: int = 0
    """Linear warmup before the main schedule (paired with ``cosine``)."""

    scheduler_warmup_start_factor: float = 0.01
    """Initial LR multiplier during warmup (``LinearLR`` ``start_factor``; must be in (0, 1])."""

    cosine_eta_min: float = 0.0
    cosine_t_max_epochs: int | None = None
    """Cosine half-period in epoch steps; default: ``epochs - warmup`` or ``epochs``."""

    cosine_restart_t0_epochs: int = 10
    cosine_restart_t_mult: int = 1

    onecycle_pct_start: float = 0.3
    onecycle_div_factor: float = 25.0
    onecycle_final_div_factor: float = 1e4
    onecycle_three_phase: bool = False
    onecycle_cycle_momentum: bool = False
    """Set False for AdamW (no SGD-style momentum cycling)."""

    linear_end_factor: float = 0.0
    """``LinearLR`` ending multiplier (in [0, 1])."""

    polynomial_power: float = 1.0

    exponential_gamma: float = 0.95

    multistep_milestones: tuple[int, ...] = ()
    multistep_gamma: float = 0.1

    velocity_loss: str = "mse"
    soft_hilbert_alpha: float = 1.0

    lambda_mask: float = 0.0
    """Weight for the Stage 1 masked-reconstruction auxiliary loss.

    When ``> 0`` ``BayesianAuditorStage1`` adds an MLM-style masking
    objective on clean ``log_x`` via a separate ``MaskReconHead`` that
    shares the backbone with the EqM velocity head. The default ``0.0``
    keeps training behavior identical to the pre-masking Stage 1.
    """

    mask_rate: float = 0.15
    """Fraction of sequence positions zeroed out when computing the Stage 1
    masked-reconstruction loss. Only read when ``lambda_mask > 0``.
    """

    stage1_answer_tokens_only: bool = False
    """If True, Stage 1 EqM/Hilbert loss is restricted to ``answer_mask`` tokens.

    Intended for QA training on concatenated ``[Q][A]`` where question tokens
    are context and gradients should focus on answer spans.
    """

    use_tqdm: bool = True
    """Show tqdm progress bars for training/validation loops."""

    # --- Weights & Biases ---
    wandb_enabled: bool = False
    """Enable Weights & Biases experiment tracking."""
    wandb_mode: str = "online"
    """W&B mode: ``online`` / ``offline`` / ``disabled``."""
    wandb_project: str = "aitchinson-flow"
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_run_name: str | None = "new-plots"
    wandb_job_type: str | None = None
    wandb_notes: str | None = None
    wandb_tags: list[str] = field(default_factory=list)
    wandb_log_model: bool = False
    """If True, upload model artifacts/checkpoints."""
    wandb_watch_model: bool = False
    """If True, call ``wandb.watch`` for gradient/parameter stats."""
    wandb_watch_log: str = "gradients"
    """Watch mode passed to ``wandb.watch`` (``gradients``/``parameters``/``all``)."""
    wandb_watch_log_freq: int = 100
    wandb_log_steps: bool = False
    """If True, emit ``step/train/*`` W&B metrics every ``log_every`` updates."""


@dataclass
class EquilibriumFlowConfig:
    """Equilibrium Matching (EqM): time-independent velocity field on the simplex path."""

    # Interpolation endpoint: scale uses 1 / (1 - eqm_interp); keep < 1.0.
    eqm_interp: float = 0.99
    # Cap on the c_t schedule (legacy EqM).
    eqm_start: float = 0.2
    # Decay strategy for c(gamma) in EqM target u_tgt = c(gamma) * (log_x0 - log_x1).
    # The target velocity points from data toward noise; inference integrators
    # subtract v (x ← x - v(x) * dt) to flow noise → data.
    # - "legacy": min(eqm_start, scale * (1-gamma)) * 4 (original behavior)
    # - "linear": 1 - gamma
    # - "truncated": 1 when gamma <= a, else (1-gamma)/(1-a)
    # - "piecewise": b - ((b-1)/a)*gamma when gamma <= a, else (1-gamma)/(1-a)
    eqm_decay_strategy: str = "linear"
    # 'a' breakpoint for truncated/piecewise schedules; must satisfy 0 < a < 1 for piecewise,
    # and 0 <= a < 1 for truncated.
    eqm_decay_a: float = 0.2
    # 'b' intercept for piecewise schedule (typically >= 1).
    eqm_decay_b: float = 2.0
    # Optional gradient multiplier (any schedule): c(gamma) <- c(gamma) / eqm_gradient_lambda
    # (set to 1.0 for no scaling).
    eqm_gradient_lambda: float = 1.0
    # Sampling / fixed-point generation from uniform noise in log-space.
    generate_steps: int = 250
    generate_stepsize: float = 0.004
    generate_init_noise: float = 0.01


@dataclass
class BayesianGeneratorConfig:
    """Hyperparameters specific to GP + flow-matching energy model."""

    corrupt_source_prob: float = 0.0
    corrupt_mode: str = "faulty"  # "missing" | "faulty"
    use_gp_variance_weighting: bool = False
    ode_steps: int = 50
    ode_init_noise: float = 0.01
    volume_penalty_alpha: float = 10.0  # passed to geometry.volume_penalty if you thread it


@dataclass
class HFDatasetConfig:
    """Hugging Face Hub / local script loading (raw splits only — no transforms)."""

    enabled: bool = False
    path: str = "afmck/text8"  # Hub repo id or local path
    name: str | None = None  # subset / config name for load_dataset
    revision: str | None = None  # reproducible Hub snapshot
    split_train: str = "train"
    split_val: str | None = "validation"  # None → no val split loaded
    split_test: str | None = "test"  # None → no test split loaded

    streaming: bool = False
    trust_remote_code: bool = False
    max_samples_train: int | None = None
    max_samples_val: int | None = None
    max_samples_test: int | None = None
    shuffle_seed: int = 0
    shuffle_buffer_size: int = 10_000  # streaming shuffle
    num_proc: int | None = None  # for later dataset.map

    row_input_key: str = "input_ids"
    pad_token_id: int = 0
    log_simplex_eps: float = 1e-8
    label_smoothing: float = 0.01
    """Explicit label smoothing strength α used by the discrete→simplex transform.

    When ``α > 0`` the one-hot row is replaced by ``(1 - α) · one_hot + α / K``
    (a true label smoothing mix with the uniform distribution on the K-vocab).
    The smoothed row sums to 1 exactly, lives in the open simplex interior, and
    the additive ``log_simplex_eps`` is **not** applied on top — label
    smoothing alone moves probabilities off the simplex boundary in a
    geometrically meaningful way (matching the design described in the
    two-stage auditor plan).

    Set to ``0.0`` (default) to keep the legacy ``one_hot + log_simplex_eps``
    path. Both paths produce ILR or CLR coordinates depending on
    ``transform_mode``.
    """

    transform_mode: str = "ilr"
    """Discrete→continuous transform applied per token.

    * ``"ilr"`` (default): isometric log-ratio, output shape ``(L, K-1)``.
      Removes the sum-to-zero constraint so distances and kernels are
      Euclidean. This is the geometrically correct chart for the open
      simplex and is what the two-stage auditor was designed for.
    * ``"clr"``: centered log-ratio, output shape ``(L, K)``. Constrained
      (rows sum to zero) — used for the **without-ILR** ablation in the
      benchmarks (Plan point 5: effect of proper simplex geometry on the
      GP kernel distances).
    """


@dataclass
class Text8DatasetConfig:
    """Char-level text8 corpus: download, chunk, corrupt.

    Uses the dataset's native ``train``, ``validation``, and ``test`` splits from
    ``afmck/text8`` (or the fallback ``afm-intelligence/text8``) and applies
    corruption independently per split.
    """

    enabled: bool = False
    cache_dir: str | None = None
    train_corrupt_rate: float = 0.15
    eval_corrupt_rate: float = 0.15
    train_order_mix_rate: float = 0.0
    eval_order_mix_rate: float = 0.0
    order_mix_prob: float = 0.0
    max_train_windows: int | None = 10_000
    max_eval_windows: int | None = 5_000
    corruption_seed: int = 1234


@dataclass
class RawTextDatasetConfig:
    """Unified source selector for raw text datasets.

    ``provider`` controls how ``source_ref`` is interpreted:
    - ``"huggingface"``: ``source_ref`` is a HF dataset id (or local load script path).
    - ``"manual"``: ``source_ref`` is a relative/absolute local dataset path.
    """

    provider: str = "huggingface"
    source_ref: str = "afmck/text8"
    dataset_name: str | None = None
    revision: str | None = None
    split_train: str = "train"
    split_val: str | None = "validation"
    split_test: str | None = "test"
    text_column: str = "text"
    cache_dir: str | None = None
    trust_remote_code: bool = False
    streaming: bool = False

    _VALID_PROVIDERS: ClassVar[tuple[str, ...]] = ("huggingface", "manual")

    def __post_init__(self) -> None:
        if self.provider not in self._VALID_PROVIDERS:
            raise ValueError(
                f"RawTextDatasetConfig.provider={self.provider!r} must be one of "
                f"{self._VALID_PROVIDERS}"
            )
        if not self.source_ref:
            raise ValueError("RawTextDatasetConfig.source_ref must be a non-empty string")
        if not self.text_column:
            raise ValueError("RawTextDatasetConfig.text_column must be non-empty")


@dataclass
class TeacherConfig:
    enabled: bool = False
    model_id: str = "gpt2"
    revision: str | None = None
    trust_remote_code: bool = False
    dtype: str = "float32"  # or "bfloat16", "float16"
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    max_length: int | None = None  # default: cfg.dataset.L
    # Output: store teacher distribution in same K as student (top-k slice or full vocab project)
    output_key: str = "log_x_teacher"  # column name after map
    top_k: int | None = None  # if set, take top-k logits per position → student K
    cache_on_disk: bool = True  # dataset.map cache_file_name


@dataclass
class QADatasetConfig:
    """Generic HuggingFace Q+A datamodule config (Path B).

    Answers and questions are byte-encoded (UTF-8, K=256) and concatenated
    as ``[STX] question [ETX] answer [EOT]`` padded to ``cfg.dataset.L``.
    """

    hf_path: str = "mandarjoshi/trivia_qa"
    name: str | None = "rc.nocontext"
    revision: str | None = None
    split_train: str = "train"
    split_val: str | None = "validation"
    split_test: str | None = None

    question_col: str = "question"
    answer_col: str = "answer.value"
    """Dotted path to the ground-truth answer field. For TriviaQA use
    ``"answer.value"`` or ``"answer.aliases"`` (first alias)."""

    aliases_col: str | None = "answer.aliases"
    """Optional dotted path to a list of acceptable alias answers, used
    for label matching when scoring LLM-generated outputs. ``None``
    disables alias matching (fall back to exact value match)."""

    max_train_samples: int | None = 10_000
    max_val_samples: int | None = 2_000
    max_test_samples: int | None = 2_000

    max_question_bytes: int = 96
    max_answer_bytes: int = 28
    shuffle_seed: int = 0
    log_simplex_eps: float = 1e-8

    trust_remote_code: bool = True


@dataclass
class AnswerGeneratorConfig:
    """LLM answer-generator settings for Path B eval.

    Instantiates a HuggingFace causal LM at eval time, asks it to answer
    val/test questions, and labels the decoded output against the
    ground-truth aliases. The auditor then scores ``[Q][A_llm]`` pairs.
    """

    lm_key: str = "hf_causal"
    """Registry key (see ``aitchinson_flow.llms.registry``)."""

    model_id: str = "gpt2"
    revision: str | None = None
    trust_remote_code: bool = False
    dtype: str = "float32"
    device: str = "auto"

    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 24
    generation_seed: int = 0

    prompt_template: str = "Q: {q}\nA:"
    """Format string for the prompt fed to the LLM. ``{q}`` is substituted."""

    cache_dir: str | None = None
    """If set, cache pre-generated val/test answers under this directory."""


@dataclass
class LLMEmbeddingDatasetConfig:
    """Path B datasource: raw text -> LLM token embeddings -> learned projection -> ILR.

    Each raw text window is tokenized by the pretrained LLM's tokenizer; the
    LLM's **frozen input embedding matrix** is then indexed to produce per-token
    embeddings of shape ``(L, d_embed)``. The embeddings are handed to a small
    learned ``Linear(d_embed, K) + log_softmax`` head on the auditor model
    (``TokenEmbeddingToSimplex``), whose output is projected through ILR/CLR to
    yield ``(L, K-1)`` simplex coordinates. The embedding lookup lives on the
    datamodule (LLM stays off the model checkpoint); the projection weights
    live on the auditor and are trained in Stage 1, then frozen for Stage 2.

    ``K = cfg.dataset.K`` is the **simplex output dimension of the learned
    projection** — it has no relationship to LLM vocabulary size. Same
    ``token_id`` always maps to the same simplex point, so identical text
    produces identical features (deterministic, unlike top-K sampling).

    Invalid (OOD) sequences are produced by applying Path A's char-level
    corruption (via :func:`aitchinson_flow.data.corruption.corrupt_token_ids`)
    to the same raw text window *before* tokenization, so the projection sees
    the LLM's reaction to corrupted text rather than a feature-space
    perturbation.
    """

    lm_key: str = "hf_causal"
    """LLM registry key used to build the pretrained LM."""

    raw_text_backend: str = "text8"
    """Raw-text source: ``"text8"`` or ``"hf"`` (same semantics as
    ``TrainingDataConfig.raw_dataset``)."""

    char_window_length: int = 256
    """Characters per raw-text window before tokenization. Picked large enough
    that the LLM tokenizer produces at least ``cfg.dataset.L`` tokens after
    truncation; excess tokens are truncated."""

    n_batches_per_epoch: int = 200
    """Number of minibatches per training epoch (iterable datamodule)."""

    corrupt_rate: float | None = None
    """Char-level corruption rate for invalid batches. ``None`` falls back to
    ``cfg.text8_dataset.train_corrupt_rate``."""

    generation_seed: int = 0
    """Seed for the per-batch corruption RNG."""

    llm_embed_dim: int | None = None
    """LLM input-embedding dimension (d_embed). Populated by the datamodule at
    build time from the selected LLM (e.g. 768 for GPT-2, 896 for Qwen2.5-0.5B).
    The auditor reads this at ``__init__`` to size the learned
    ``TokenEmbeddingToSimplex`` projection — leave ``None`` and the datamodule
    will fill it in via :func:`~aitchinson_flow.training.data_sources.build_training_datamodule`."""

    _VALID_BACKENDS: ClassVar[tuple[str, ...]] = ("text8", "hf")

    def __post_init__(self) -> None:
        if self.raw_text_backend not in self._VALID_BACKENDS:
            raise ValueError(
                f"LLMEmbeddingDatasetConfig.raw_text_backend={self.raw_text_backend!r} must "
                f"be one of {self._VALID_BACKENDS}"
            )
        if self.char_window_length < 1:
            raise ValueError(
                f"LLMEmbeddingDatasetConfig.char_window_length must be >= 1, got "
                f"{self.char_window_length}"
            )
        if self.n_batches_per_epoch < 1:
            raise ValueError(
                f"LLMEmbeddingDatasetConfig.n_batches_per_epoch must be >= 1, got "
                f"{self.n_batches_per_epoch}"
            )
        if self.corrupt_rate is not None and not 0.0 <= self.corrupt_rate <= 1.0:
            raise ValueError(
                f"LLMEmbeddingDatasetConfig.corrupt_rate must be in [0, 1], got {self.corrupt_rate}"
            )
        if self.llm_embed_dim is not None and self.llm_embed_dim < 1:
            raise ValueError(
                f"LLMEmbeddingDatasetConfig.llm_embed_dim must be >= 1 when set, got "
                f"{self.llm_embed_dim}"
            )


@dataclass
class LLMTopKProbsConfig:
    """Component 2 datasource: raw text → LLM top-K softmax probabilities → ILR.

    Each raw text window is tokenized by the LLM's tokenizer; the frozen LLM's
    forward pass extracts per-token softmax probabilities over the full
    vocabulary, then the top-K entries are selected and re-normalized to form a
    K-dimensional probability simplex. ILR projection yields (L, K-1) Aitchison
    coordinates for the flow-matching backbone.

    ``K = cfg.dataset.K`` controls how many top-K vocabulary slots are retained.
    Valid text produces exponential-decay-like top-K distributions (one token
    dominates); corrupted text produces flatter distributions — this geometric
    difference is what the Stage 1 + Stage 2 pipeline learns to detect.
    """

    lm_key: str = "hf_causal"
    """LLM registry key used to build the pretrained LM."""

    renormalize: bool = True
    """Re-normalize the top-K probabilities to sum to 1 after truncation."""

    raw_text_backend: str = "text8"
    """Raw-text source for valid sequences; currently only ``"text8"`` is supported."""

    char_window_length: int = 128
    """Characters per raw-text window before LLM tokenization. Should be large
    enough that the tokenizer produces at least ``cfg.dataset.L`` LLM tokens
    after truncation (roughly ``L * avg_chars_per_token``; for GPT-2 with L=30
    the default of 128 gives ~32 tokens with a small safety margin)."""

    corrupt_rate: float | None = None
    """Char-level corruption rate for the invalid (OOD) batches. ``None`` falls
    back to ``cfg.text8_dataset.train_corrupt_rate``."""

    generation_seed: int = 0
    """Base seed for the per-batch corruption RNG."""

    _VALID_BACKENDS: ClassVar[tuple[str, ...]] = ("text8",)

    def __post_init__(self) -> None:
        if self.raw_text_backend not in self._VALID_BACKENDS:
            raise ValueError(
                f"LLMTopKProbsConfig.raw_text_backend={self.raw_text_backend!r} must "
                f"be one of {self._VALID_BACKENDS}"
            )
        if self.char_window_length < 1:
            raise ValueError(
                f"LLMTopKProbsConfig.char_window_length must be >= 1, got {self.char_window_length}"
            )
        if self.corrupt_rate is not None and not 0.0 <= self.corrupt_rate <= 1.0:
            raise ValueError(
                f"LLMTopKProbsConfig.corrupt_rate must be in [0, 1], got {self.corrupt_rate}"
            )


@dataclass
class DNADatasetConfig:
    """Nucleotide sequence dataset for Component 1 structural UQ benchmark.

    Sequences are single-letter ACGT (K=4) or ACGTN (K=5) encoded, chunked into
    windows of ``cfg.dataset.L`` bases. Invalid sequences are generated by point
    mutation, frameshift insertion, or random shuffle.
    """

    enabled: bool = False
    hf_path: str | None = None
    """HuggingFace dataset path (e.g. 'InstaDeepAI/nucleotide_transformer_downstream_tasks').
    If ``None`` (default), synthetic sequences are generated from the ``gc_content`` parameter."""
    hf_name: str | None = None
    split_train: str = "train"
    split_val: str | None = "test"
    text_column: str = "sequence"
    trust_remote_code: bool = False
    use_n_base: bool = False
    """Include the ambiguous N base (K=5). When False (default) vocab is ACGT (K=4)."""
    max_train_windows: int | None = 50_000
    max_eval_windows: int | None = 10_000
    train_corrupt_rate: float = 0.5
    eval_corrupt_rate: float = 0.5
    gc_content: float = 0.5
    """GC content probability for synthetic sequence generation (AT content = 1 - gc_content)."""
    corruption_seed: int = 1234
    synthetic_seed: int = 42
    """RNG seed for synthetic sequence generation."""


@dataclass
class MedicalDatasetConfig:
    """Clinical text dataset for Component 1+2 structural + contextual UQ benchmark.

    Clinical notes are lowercased and restricted to a 47-char clinical ASCII set (K=47),
    chunked into windows of ``cfg.dataset.L`` characters. Invalid sequences are generated
    by drug-name misspelling, unit corruption, and random character substitution.
    """

    enabled: bool = False
    hf_path: str | None = None
    """HuggingFace dataset path (e.g. 'BI-Misc/med_notes_test').
    If ``None`` (default), synthetic clinical notes are generated from templates."""
    hf_name: str | None = None
    split_train: str = "train"
    split_val: str | None = "test"
    text_column: str = "text"
    trust_remote_code: bool = False
    max_train_windows: int | None = 10_000
    max_eval_windows: int | None = 2_000
    train_corrupt_rate: float = 0.5
    eval_corrupt_rate: float = 0.5
    corruption_seed: int = 2345
    synthetic_seed: int = 99
    """RNG seed for synthetic clinical note generation."""


@dataclass
class HealingConfig:
    """Phase 4 OOD healing strategy configuration.

    Three strategies (see :mod:`aitchinson_flow.healing`):

    * ``"targeted_resample"`` — mask per-token positions with energy > threshold,
      project via EqM or resample via LLM; iterate up to ``max_iter`` times.
    * ``"simplex_project"`` — run EqM on flagged sequences, project back to the
      nearest vocabulary token via ILR inverse + argmax.
    * ``"beam_rerank"`` — score beam candidates by
      ``log_p − lambda_energy · mean_energy``; return the highest-scoring one.
    * ``"eqm"`` — legacy whole-sequence EqM integration (equivalent to the
      ``HealingPipeline`` in :mod:`benchmarks.healing`).
    """

    strategy: str = "targeted_resample"
    """Healing strategy: one of ``targeted_resample``, ``simplex_project``,
    ``beam_rerank``, ``eqm``."""

    threshold: float = 1.0
    """Per-token (``targeted_resample``) or sequence-level (others) energy
    threshold for flagging sequences / positions as OOD."""

    max_iter: int = 5
    """Maximum healing iterations (``targeted_resample`` only)."""

    lambda_energy: float = 1.0
    """Energy penalty weight λ for beam reranking:
    ``score = log_p − λ · mean_energy``."""

    n_beams: int = 5
    """Number of beam candidates to generate before reranking."""

    heal_steps: int | None = None
    """EqM integration steps per healing pass (``None`` → healer model default)."""

    heal_dt: float | None = None
    """EqM step size (``None`` → healer model default)."""

    re_encode: bool = True
    """If ``True`` (default), re-encode snapped token IDs back to ILR after
    simplex projection so the output geometry matches training."""

    eps: float = 1e-8
    """Additive smoothing epsilon used when re-encoding token IDs to ILR."""

    label_smoothing: float = 0.0
    """Label-smoothing coefficient for re-encoding (mirrors
    ``cfg.hf_dataset.label_smoothing`` — set consistently)."""

    transform_mode: str = "ilr"
    """``"ilr"`` or ``"clr"`` for re-encoding token IDs (must match training)."""

    _VALID_STRATEGIES: ClassVar[tuple[str, ...]] = (
        "targeted_resample",
        "simplex_project",
        "beam_rerank",
        "eqm",
    )

    def __post_init__(self) -> None:
        if self.strategy not in self._VALID_STRATEGIES:
            raise ValueError(
                f"HealingConfig.strategy={self.strategy!r} must be one of {self._VALID_STRATEGIES}"
            )
        if self.threshold <= 0.0:
            raise ValueError(f"HealingConfig.threshold must be > 0, got {self.threshold}")
        if self.max_iter < 1:
            raise ValueError(f"HealingConfig.max_iter must be >= 1, got {self.max_iter}")
        if self.lambda_energy < 0.0:
            raise ValueError(
                f"HealingConfig.lambda_energy must be >= 0 (negative values reward "
                f"anomalous sequences), got {self.lambda_energy}"
            )
        if self.n_beams < 1:
            raise ValueError(f"HealingConfig.n_beams must be >= 1, got {self.n_beams}")
        if self.heal_steps is not None and self.heal_steps < 1:
            raise ValueError(
                f"HealingConfig.heal_steps must be >= 1 when set, got {self.heal_steps}"
            )
        if self.transform_mode not in ("ilr", "clr"):
            raise ValueError(
                f"HealingConfig.transform_mode must be 'ilr' or 'clr', got {self.transform_mode!r}"
            )


@dataclass
class TrainingDataConfig:
    """Training-time data source selection and generation controls.

    This is intentionally separate from ``benchmark.data_source`` because
    training objectives may want different input pipelines than benchmark
    sweeps.

    Supported sources (dispatched by
    ``aitchinson_flow.training.data_sources.build_training_datamodule``):

    * ``source="raw_text"`` — Path A: load a raw corpus (``text8`` or the
      HF-hub datamodule) and feed it through the standard discrete→ILR
      transform.
    * ``source="llm_topk"`` — Path B: tokenize raw text with a pretrained LLM
      tokenizer, look up the frozen input embeddings, and project them into
      the simplex with a learned ``Linear + log_softmax + ILR`` head
      (``TokenEmbeddingToSimplex``) trained jointly with the auditor. See
      :class:`LLMEmbeddingDatasetConfig`.
    * ``source="llm_generated"`` — ask a registered causal LM to *generate*
      batches on the fly. Generation is seeded per-batch via
      ``generation_seed``; ``use_text8_prompts`` optionally seeds every
      generation from a text8 prompt window.
    * ``source="qa_pairs"`` — Path C: byte-level Q+A (trivia) auditor using
      cross-question-swap negatives and an LLM answer generator.
    * ``source="llm_topk_probs"`` — Component 2: tokenize raw text, run frozen
      LLM forward pass, extract top-K softmax probabilities per position,
      re-normalize, then apply ILR. Captures contextual semantic uncertainty.
      See :class:`LLMTopKProbsConfig`.
    """

    source: str = "raw_text"
    """One of: ``"raw_text"``, ``"llm_topk"``, ``"llm_generated"``, ``"qa_pairs"``,
    ``"llm_topk_probs"``."""

    raw_dataset: str = "text8"
    """Raw-data backend when ``source='raw_text'``.

    Supported: ``"text8"``, ``"hf"``.
    """

    lm_key: str = "hf_causal"
    """LLM registry key used when ``source='llm_generated'``."""

    n_batches: int = 200
    """Number of generated batches per epoch-like pass for iterable teacher data."""

    use_text8_prompts: bool = False
    text8_prompt_length: int = 32
    generation_seed: int = 0
    generation_temperature: float = 1.0
    generation_top_p: float = 1.0

    llm_feature_mode: str = "token_ids"
    """Feature path for ``source='llm_generated'``.

    - ``"token_ids"``: generated ids -> discrete smoothing -> ILR/CLR.
    - ``"token_probs"``: LM token distributions -> simplex projection -> ILR/CLR.
    """

    qa_skip_llm_eval: bool = False
    """If True and ``source='qa_pairs'``, skip instantiating the answer-generator
    LLM and fall back to cross-question-swap placeholders for eval negatives.
    Useful for smoke tests that exercise the plumbing without a real LLM."""

    _VALID_SOURCES: ClassVar[tuple[str, ...]] = (
        "raw_text",
        "llm_topk",
        "llm_generated",
        "qa_pairs",
        "llm_topk_probs",
        "dna",
        "medical",
    )
    _VALID_RAW_DATASETS: ClassVar[tuple[str, ...]] = ("text8", "hf")
    _VALID_LLM_FEATURE_MODES: ClassVar[tuple[str, ...]] = ("token_ids", "token_probs")

    def __post_init__(self) -> None:
        if self.source not in self._VALID_SOURCES:
            raise ValueError(
                f"TrainingDataConfig.source={self.source!r} must be one of {self._VALID_SOURCES}; "
                f"use 'llm_topk_probs' for Component 2, 'dna' for DNA sequences, "
                f"'medical' for clinical text"
            )
        if self.raw_dataset not in self._VALID_RAW_DATASETS:
            raise ValueError(
                f"TrainingDataConfig.raw_dataset={self.raw_dataset!r} must be one of "
                f"{self._VALID_RAW_DATASETS}"
            )
        if self.llm_feature_mode not in self._VALID_LLM_FEATURE_MODES:
            raise ValueError(
                f"TrainingDataConfig.llm_feature_mode={self.llm_feature_mode!r} must be one of "
                f"{self._VALID_LLM_FEATURE_MODES}"
            )
        if self.n_batches < 1:
            raise ValueError(f"TrainingDataConfig.n_batches must be >= 1, got {self.n_batches}")
        if self.text8_prompt_length < 1:
            raise ValueError(
                f"TrainingDataConfig.text8_prompt_length must be >= 1, got "
                f"{self.text8_prompt_length}"
            )
        if self.generation_temperature <= 0.0:
            raise ValueError(
                f"TrainingDataConfig.generation_temperature must be > 0 (temperature=0 "
                f"is not sampling — use a greedy mode instead), got "
                f"{self.generation_temperature}"
            )
        if not 0.0 < self.generation_top_p <= 1.0:
            raise ValueError(
                f"TrainingDataConfig.generation_top_p must be in (0, 1], got "
                f"{self.generation_top_p}"
            )


@dataclass
class BenchmarkConfig:
    """Benchmark runner settings for model/task/scale sweeps."""

    task_name: str = "text_audit"
    lm_key: str = "hf_causal"

    # Which datamodule to use: "lm_teacher" (GPT-2 synthetic) or "text8" (char-level).
    data_source: str = "text8"

    # How much synthetic data to generate per benchmark run.
    n_batches: int = 50

    # Scale sweep over transformer backbone.
    # Each entry should include: d_model, num_layers, nhead.
    scale_grid: list[dict[str, int]] = field(
        default_factory=lambda: [
            {"d_model": 128, "num_layers": 4, "nhead": 8},
            {"d_model": 256, "num_layers": 6, "nhead": 8},
            {"d_model": 512, "num_layers": 8, "nhead": 8},
        ]
    )

    results_dir: str = "results/benchmark"

    # --- text8-conditioned generation ---
    use_text8_prompts: bool = True
    """If True, seed GPT-2 generation from text8 chunks instead of BOS-only."""
    text8_prompt_length: int = 32
    """Number of GPT-2 tokens per text8 prompt prefix."""

    # --- invalid sample corruption ---
    corrupt_rate: float = 0.15
    """Fraction of positions replaced with random vocab ids for invalid samples."""
    order_mix_rate: float = 0.15
    """Fraction of positions partially shuffled for invalid samples."""
    order_mix_prob: float = 0.5
    """Probability of applying partial order mixing for a batch."""
    corruption_seed: int = 42

    # --- spilled energy baseline ---
    compute_spilled_energy: bool = True
    """Compute training-free spilled-energy anomaly scores alongside auditor metrics."""
    spilled_hard_neg_top_k: int = 5
    """Top-k for optional hard-negative ablation (not used in default random corruption)."""
    use_tqdm: bool = True
    """Show tqdm progress bars for benchmark scale sweeps."""

    # --- training inside the benchmark sweep ---
    train_before_eval: bool = True
    """If True, train each scale's model via ``fit`` before running the audit task."""
    train_epochs: int | None = 20
    """Override ``training.epochs`` during the benchmark sweep; None keeps the global value."""

    # --- plotting ---
    save_plots: bool = True
    """If True, save per-scale ROC, score-histogram, loss-curve, and per-sequence PNGs."""

    # --- corrupt_rate sweep (Experiment 3) ---
    corrupt_rate_sweep: list[float] | None = None
    """If set, TextAuditTask loops over these rates and emits per-rate AUROC metrics."""

    # --- self-healing (Experiment 4) ---
    healer_ckpt: str | None = None
    """Path to a trained EquilibriumAuditor checkpoint used by HealingAuditTask."""
    healing_var_threshold: float = 0.5
    """Variance threshold above which a sequence is flagged for healing."""
    healing_steps: int = 20
    """Number of Euler integration steps used during healing."""


@dataclass
class PerTokenAuditorConfig:
    """Per-token Bayesian auditor (GP at each position)."""

    use_context: bool = False
    """If True, use ProductSparseGP with teacher hidden states; batch must include ctx_1 / ctx_1_invalid."""

    ctx_hidden: int = 768
    """Last-hidden size from the teacher (e.g. GPT-2 = 768). Projected to d_latent via ctx_proj."""


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    hf_dataset: HFDatasetConfig = field(default_factory=HFDatasetConfig)
    text8_dataset: Text8DatasetConfig = field(default_factory=Text8DatasetConfig)
    raw_text_dataset: RawTextDatasetConfig = field(default_factory=RawTextDatasetConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    gp: GPConfig = field(default_factory=GPConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    bayesian_generator: BayesianGeneratorConfig = field(default_factory=BayesianGeneratorConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    training_data: TrainingDataConfig = field(default_factory=TrainingDataConfig)
    llm_embedding_dataset: LLMEmbeddingDatasetConfig = field(
        default_factory=LLMEmbeddingDatasetConfig
    )
    llm_topk_probs: LLMTopKProbsConfig = field(default_factory=LLMTopKProbsConfig)
    dna_dataset: DNADatasetConfig = field(default_factory=DNADatasetConfig)
    medical_dataset: MedicalDatasetConfig = field(default_factory=MedicalDatasetConfig)
    qa_dataset: QADatasetConfig = field(default_factory=QADatasetConfig)
    answer_generator: AnswerGeneratorConfig = field(default_factory=AnswerGeneratorConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    per_token_auditor: PerTokenAuditorConfig = field(default_factory=PerTokenAuditorConfig)
    equilibrium: EquilibriumFlowConfig = field(default_factory=EquilibriumFlowConfig)
    healing: HealingConfig = field(default_factory=HealingConfig)

    def __post_init__(self) -> None:
        # Backward-compatible aliases: if the new unified selector stays at defaults,
        # inherit legacy HF knobs for raw_dataset="hf".
        if (
            self.raw_text_dataset.source_ref == "afmck/text8"
            and self.raw_text_dataset.provider == "huggingface"
            and self.training_data.raw_dataset == "hf"
            and self.hf_dataset.path
        ):
            self.raw_text_dataset.source_ref = self.hf_dataset.path
            self.raw_text_dataset.dataset_name = self.hf_dataset.name
            self.raw_text_dataset.revision = self.hf_dataset.revision
            self.raw_text_dataset.split_train = self.hf_dataset.split_train
            self.raw_text_dataset.split_val = self.hf_dataset.split_val
            self.raw_text_dataset.split_test = self.hf_dataset.split_test
            self.raw_text_dataset.streaming = self.hf_dataset.streaming
            self.raw_text_dataset.trust_remote_code = self.hf_dataset.trust_remote_code
            if self.hf_dataset.path.startswith(("/", "./", "../")):
                self.raw_text_dataset.provider = "manual"
