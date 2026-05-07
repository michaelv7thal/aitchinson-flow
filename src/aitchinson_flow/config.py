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
    sample_gamma: float = 0.5
    # γ importance sampling: gamma = U(0,1)**gamma_power.
    # gamma_power=1.0 → uniform; <1 pushes mass toward γ=1 (more signal).
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
class WandbConfig:
    """Optional Weights & Biases logging. Off by default; enable with --wandb."""

    enabled: bool = False
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
    loader_settings: LoaderSettings = field(default_factory=LoaderSettings)
    loss: LossConfig = field(default_factory=LossConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
