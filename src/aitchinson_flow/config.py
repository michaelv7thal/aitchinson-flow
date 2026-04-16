from __future__ import annotations

import torch

from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    dataset: str = "text8"  # "dna", "text8", "wiki"
    K: int = 27  # Vocabulary size (4=DNA, 27=text8, 64=wiki top-k)
    L: int = 20  # Sequence length


@dataclass
class TransformerConfig:
    d_model: int = 128  # Dimension of the model
    nhead: int = 8  # Number of attention heads
    num_layers: int = 6  # Number of layers
    d_latent: int = 128  # Dimension of the latent space
    dropout: float = 0.0  # Dropout rate
    time_conditioned: bool = False  # Whether to condition on time


@dataclass
class GPConfig:
    num_inducing: int = 500  # Number of inducing points
    lambda_kl: float = 1e-4  # KL regularization parameter
    lambda_var: float = 2.0  # Variance regularization parameter
    margin_E: float = 2.0  # Hinge margin for L_energy score
    margin_V: float = 1.0  # Hinge margin for L_variance score


@dataclass
class TrainingConfig:
    model_name: str = "flow_matching"
    B: int = 128  # Batch size
    epochs: int = 10_000  # Number of epochs
    lr: float = 5e-4  # Learning rate
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
    num_workers: int = 0
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

    velocity_loss: str = "soft_hilbert"
    soft_hilbert_alpha: float = 1.0
    use_tqdm: bool = True
    """Show tqdm progress bars for training/validation loops."""


@dataclass
class EquilibriumFlowConfig:
    """Equilibrium Matching (EqM): time-independent velocity field on the simplex path."""

    # Interpolation endpoint: scale uses 1 / (1 - eqm_interp); keep < 1.0.
    eqm_interp: float = 0.99
    # Cap on the c_t schedule (legacy EqM).
    eqm_start: float = 0.2
    # Sampling / fixed-point generation from uniform noise in log-space.
    generate_steps: int = 250
    generate_stepsize: float = 0.004
    generate_init_noise: float = 0.01


@dataclass
class BayesianGeneratorConfig:
    """Hyperparameters specific to GP + flow-matching energy model."""

    corrupt_source_prob: float = 0.0
    corrupt_mode: str = "missing"  # "missing" | "faulty"
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
    eval_order_mix_rate: float = 0.30
    order_mix_prob: float = 0.5
    max_train_windows: int | None = None
    max_eval_windows: int | None = 2000
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
