from __future__ import annotations

import math
import torch

from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    dataset: str = "text8"  # "dna", "text8", "wiki"
    K: int = 27  # Vocabulary size (4=DNA, 27=text8, 64=wiki top-k)
    L: int = 30  # Sequence length


@dataclass
class TransformerConfig:
    d_model: int = 1440  # Dimension of the model
    nhead: int = 8  # Number of attention heads
    num_layers: int = 8  # Number of layers
    d_latent: int = 1440  # Dimension of the latent space
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
    lambda_contrastive: float = 1.0  # Stage 2 random-negative energy hinge weight
    lambda_var: float = 2.0
    """Anchor weight on valid GP mean in the composed ``BayesianAuditor`` loss.

    Keeps ``E[valid]`` near zero so the contrastive hinge does not pull the
    whole energy surface upward. The historical name is retained for
    checkpoint compatibility; see also ``lambda_anchor`` for the Stage 2
    analogue.
    """
    lambda_anchor: float = 0.01
    """Stage 2 anchor weight on ``E[valid]^2`` (defaults to off).

    Mirrors ``lambda_var`` in the composed auditor. Enable when relaxing the
    contrastive hinge (C2 fix removes the ``.detach()``) causes valid energy
    to drift.
    """
    margin_E: float = 2.0  # Hinge margin for L_energy score
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
    epochs: int = 10_000  # Number of epochs
    lr: float = 1e-3  # Learning rate
    loss: str = "hilbert"  # "hilbert" or "mse"
    device: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    seed: int = 0
    grad_clip_norm: float | None = None  # if set, clip after backward
    log_every: int = 50
    eval_every: int = 1  # epochs between validation
    checkpoint_every: int = 1
    checkpoint_dir: str = "checkpoints"
    num_workers: int = 8
    weight_decay: float = 0.0
    # Step once per epoch in ``runner.fit`` (see ``build_scheduler``).
    lr_scheduler: str | None = "cosine"
    """One of: None, ``constant``, ``cosine``, ``cosine_restarts``, ``onecycle``, ``linear``,
    ``polynomial``, ``exponential``, ``multistep``."""

    scheduler_warmup_epochs: int = 2
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

    velocity_loss: str = "soft_hilbert"
    soft_hilbert_alpha: float = 1.0

    lambda_mask: float = 0.2
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

    use_tqdm: bool = True
    """Show tqdm progress bars for training/validation loops."""

    # --- Weights & Biases ---
    wandb_enabled: bool = True
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
    path: str = ""  # Hub repo id or local path
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
    eval_corrupt_rate: float = 0.30
    train_order_mix_rate: float = 0.15
    eval_order_mix_rate: float = 0.0
    order_mix_prob: float = 0.0
    max_train_windows: int | None = 10_000
    max_eval_windows: int | None = 5_000
    corruption_seed: int = 1234


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
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    gp: GPConfig = field(default_factory=GPConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    bayesian_generator: BayesianGeneratorConfig = field(default_factory=BayesianGeneratorConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    per_token_auditor: PerTokenAuditorConfig = field(default_factory=PerTokenAuditorConfig)
    equilibrium: EquilibriumFlowConfig = field(default_factory=EquilibriumFlowConfig)
