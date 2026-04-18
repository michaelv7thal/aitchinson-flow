"""End-to-end smoke test for ``scripts/two_stage_train.run_two_stage``.

Verifies that the orchestration script runs Stage 1 → Stage 2 → compose on a
tiny synthetic corpus and produces ``stage1.pt``, ``stage2.pt``, ``fused.pt``,
and ``orchestration.json`` that loads back into a usable
``BayesianAuditor``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Allow `pytest tests/` without an editable install (script lives outside the package).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from two_stage_train import _smoke_config, run_two_stage  # noqa: E402

from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.data.feature_dim import feature_dim  # noqa: E402
from aitchinson_flow.models.bayesian_auditor import BayesianAuditor  # noqa: E402


class _SyntheticTwoStageDataset(Dataset):
    """Yields ``{"log_x", "log_x_invalid"}`` batches in ILR coords."""

    def __init__(self, cfg: Config, n: int = 32) -> None:
        rng = torch.Generator().manual_seed(0)
        D = feature_dim(cfg)
        self._valid = 0.05 * torch.randn(n, cfg.dataset.L, D, generator=rng)
        # Invalid: shifted distribution so contrastive loss has a real signal.
        self._invalid = 0.5 + 0.05 * torch.randn(n, cfg.dataset.L, D, generator=rng)

    def __len__(self) -> int:
        return self._valid.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "log_x": self._valid[idx],
            "log_x_invalid": self._invalid[idx],
        }


class _SyntheticDataModule:
    def __init__(self, cfg: Config) -> None:
        ds = _SyntheticTwoStageDataset(cfg, n=32)
        self._loader = DataLoader(ds, batch_size=cfg.training.B, shuffle=False)

    def train_dataloader(self) -> DataLoader:
        return self._loader

    def val_dataloader(self) -> DataLoader | None:
        return None

    def test_dataloader(self) -> DataLoader | None:
        return self._loader


class _FakeWandbLogger:
    instances: list["_FakeWandbLogger"] = []

    def __init__(
        self,
        cfg: Config,
        *,
        run_name: str | None = None,
        group: str | None = None,
        job_type: str | None = None,
        tags: list[str] | None = None,
        extra_config: dict[str, Any] | None = None,
    ) -> None:
        self.cfg = cfg
        self.run_name = run_name
        self.group = group
        self.job_type = job_type
        self.tags = tags or []
        self.extra_config = extra_config or {}
        self.logged_metrics: list[tuple[dict[str, float], int | None]] = []
        self.logged_artifacts: list[dict[str, Any]] = []
        self.watch_calls = 0
        self.finished = False
        self.active = True
        self.__class__.instances.append(self)

    def watch_model(self, model: Any) -> None:
        self.watch_calls += 1

    def log_metrics(self, metrics: dict[str, float], *, step: int | None = None) -> None:
        self.logged_metrics.append((metrics, step))

    def log_artifact(
        self,
        path: str | Path,
        *,
        name: str,
        artifact_type: str,
        aliases: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.logged_artifacts.append(
            {
                "path": str(path),
                "name": name,
                "artifact_type": artifact_type,
                "aliases": aliases or [],
                "metadata": metadata or {},
            }
        )

    def finish(self) -> None:
        self.finished = True


class _FakeAuditTask:
    def run(self, model: Any, datamodule: Any, cfg: Config) -> dict[str, Any]:
        del model, datamodule, cfg
        return {
            "auroc_auditor": 0.82,
            "auroc_spilled": 0.76,
            "pearson_r_valid": 0.11,
            "_scores": {
                "auditor_energy_seq_valid": np.array([0.2, 0.1]),
                "auditor_energy_seq_invalid": np.array([0.9, 1.2]),
                "auditor_var_seq_valid": np.array([0.05, 0.06]),
                "auditor_var_seq_invalid": np.array([0.3, 0.4]),
                "latent_tokens_valid": np.array([0.2, 0.3]),
                "latent_tokens_invalid": np.array([0.9, 1.1]),
                "inducing_points": np.array([0.1, 0.2]),
            },
        }


def test_two_stage_orchestration_smoke(tmp_path: Path) -> None:
    cfg = _smoke_config()
    dm = _SyntheticDataModule(cfg)

    manifest = run_two_stage(
        cfg,
        out_dir=tmp_path,
        stage1_epochs=1,
        stage2_epochs=1,
        datamodule=dm,
        save_plots=False,
    )

    for key in ("stage1_ckpt", "stage2_ckpt", "fused_ckpt"):
        p = Path(manifest[key])
        assert p.exists(), f"missing artifact: {key} -> {p}"

    orch = json.loads((tmp_path / "orchestration.json").read_text())
    assert orch["stage1_epochs"] == 1
    assert orch["stage2_epochs"] == 1
    assert orch["random_stage2_backbone"] is False

    # Fused model must load with the inference auditor architecture.
    fused = BayesianAuditor(cfg)
    state = torch.load(manifest["fused_ckpt"], map_location="cpu", weights_only=False)
    state_dict = state["model_state_dict"] if "model_state_dict" in state else state
    missing, unexpected = fused.load_state_dict(state_dict, strict=False)
    # Allow extra/missing optional buffers but require backbone + GP keys to match.
    must_have = {"backbone.", "latent_head.", "gp."}
    fused_keys = set(state_dict.keys())
    for prefix in must_have:
        assert any(k.startswith(prefix) for k in fused_keys), (
            f"fused checkpoint missing any '{prefix}*' keys"
        )
    # Sanity: missing list shouldn't contain backbone weights.
    assert not any(k.startswith("backbone.") for k in missing), (
        f"missing backbone keys after fused load: {missing}"
    )


def test_two_stage_orchestration_random_backbone_ablation(tmp_path: Path) -> None:
    cfg = _smoke_config()
    dm = _SyntheticDataModule(cfg)
    manifest = run_two_stage(
        cfg,
        out_dir=tmp_path,
        stage1_epochs=1,
        stage2_epochs=1,
        datamodule=dm,
        random_stage2_backbone=True,
        save_plots=False,
    )
    assert manifest["random_stage2_backbone"] is True
    assert Path(manifest["fused_ckpt"]).exists()


def test_two_stage_orchestration_with_wandb_hooks(tmp_path: Path, monkeypatch: Any) -> None:
    _FakeWandbLogger.instances = []
    monkeypatch.setattr("aitchinson_flow.training.runner.WandbLogger", _FakeWandbLogger)
    monkeypatch.setattr("two_stage_train.WandbLogger", _FakeWandbLogger)

    cfg = _smoke_config()
    cfg.training.wandb_enabled = True
    cfg.training.wandb_log_model = True
    cfg.training.log_every = 1
    dm = _SyntheticDataModule(cfg)

    manifest = run_two_stage(
        cfg,
        out_dir=tmp_path,
        stage1_epochs=1,
        stage2_epochs=1,
        datamodule=dm,
        save_plots=False,
    )
    assert Path(manifest["stage1_ckpt"]).exists()

    by_job_type = {inst.job_type: inst for inst in _FakeWandbLogger.instances if inst.job_type}
    assert "stage1_train" in by_job_type
    assert "stage2_train" in by_job_type
    assert "two_stage_orchestration" in by_job_type

    stage1_logger = by_job_type["stage1_train"]
    stage2_logger = by_job_type["stage2_train"]
    orchestration_logger = by_job_type["two_stage_orchestration"]

    assert stage1_logger.watch_calls == 1
    assert stage2_logger.watch_calls == 1
    assert stage1_logger.logged_metrics
    assert stage2_logger.logged_metrics
    assert not any(
        key.startswith("step/train/")
        for payload, _ in stage1_logger.logged_metrics
        for key in payload
    )
    assert not any(
        key.startswith("step/train/")
        for payload, _ in stage2_logger.logged_metrics
        for key in payload
    )
    assert any(a["artifact_type"] == "model" for a in stage1_logger.logged_artifacts)
    assert any(a["artifact_type"] == "model" for a in stage2_logger.logged_artifacts)
    assert any(
        a["artifact_type"] == "manifest" for a in orchestration_logger.logged_artifacts
    )
    assert all(inst.finished for inst in _FakeWandbLogger.instances)


def test_two_stage_orchestration_logs_stage_performance_and_step_metrics(
    tmp_path: Path, monkeypatch: Any
) -> None:
    _FakeWandbLogger.instances = []
    monkeypatch.setattr("aitchinson_flow.training.runner.WandbLogger", _FakeWandbLogger)
    monkeypatch.setattr("two_stage_train.WandbLogger", _FakeWandbLogger)
    monkeypatch.setattr("benchmarks.tasks.registry.build_task", lambda _name: _FakeAuditTask())
    monkeypatch.setattr(
        "aitchinson_flow.plots.save_stage_plots",
        lambda out_dir, stage_data: {"out_dir": str(out_dir), "stage": stage_data.stage},
    )

    cfg = _smoke_config()
    cfg.training.wandb_enabled = True
    cfg.training.wandb_log_model = False
    cfg.training.wandb_log_steps = True
    cfg.training.log_every = 1
    dm = _SyntheticDataModule(cfg)

    run_two_stage(
        cfg,
        out_dir=tmp_path,
        stage1_epochs=1,
        stage2_epochs=1,
        datamodule=dm,
        save_plots=True,
    )

    by_job_type = {inst.job_type: inst for inst in _FakeWandbLogger.instances if inst.job_type}
    stage1_logger = by_job_type["stage1_train"]
    stage2_logger = by_job_type["stage2_train"]
    orchestration_logger = by_job_type["two_stage_orchestration"]

    assert any(
        key.startswith("step/train/")
        for payload, _ in stage1_logger.logged_metrics
        for key in payload
    )
    assert any(
        key.startswith("step/train/")
        for payload, _ in stage2_logger.logged_metrics
        for key in payload
    )
    orchestration_metric_payloads = [payload for payload, _ in orchestration_logger.logged_metrics]
    merged = {k: v for payload in orchestration_metric_payloads for k, v in payload.items()}
    assert merged["stage1/auroc_auditor"] == 0.82
    assert merged["stage2/auroc_spilled"] == 0.76
    assert merged["stage1/pearson_r_valid"] == 0.11
