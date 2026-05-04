from __future__ import annotations

import torch

from dataclasses import dataclass, field


@dataclass
class TrainingConfigs:
    lr: float = 1e-5
    device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    lr_sheduler: str | None = "cosine"
    """One of: None, ``constant``, ``cosine``, ``cosine_restarts``, ``onecycle``, ``linear``, ``polynomial``, ``exponential``, ``multistep``."""

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
    grad_clip_norm: float | None = None
    eval_every: int = 5
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
    lambda_vol: float = 1e-3
    lambda_mse: float = 5e-3
    alpha: float = 1.0
    # NAG-GD sampling hyperparameters (Algorithm 2)
    sample_eta: float = 0.1  # step size η
    sample_mu: float = 0.9  # NAG look-ahead factor μ
    sample_g_min: float = 0.01  # gradient-norm stopping threshold
    sample_max_steps: int = 500  # hard cap on iterations


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
    mode: str = "hilbert_soft"  # "hilbert" | "hilbert_soft" | "mse"
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
class Config:
    training: TrainingConfigs = field(default_factory=TrainingConfigs)
    text8_dataset: Text8DataConfig = field(default_factory=Text8DataConfig)
    transformation: TransformationConfig = field(default_factory=TransformationConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    eqm: EqM = field(default_factory=EqM)
    dfm: DFMConfig = field(default_factory=DFMConfig)
    loader_settings: LoaderSettings = field(default_factory=LoaderSettings)
    loss: LossConfig = field(default_factory=LossConfig)
