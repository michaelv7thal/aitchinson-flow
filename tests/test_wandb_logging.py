from __future__ import annotations

from pathlib import Path
from typing import Any
import sys

# Allow `pytest tests/` without an editable install.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from aitchinson_flow.config import Config
from aitchinson_flow.training.wandb_logger import WandbLogger


def test_wandb_step_logging_is_disabled_by_default() -> None:
    cfg = Config()
    assert cfg.training.wandb_log_steps is False


class _FakeArtifact:
    def __init__(self, *, name: str, type: str, metadata: dict[str, Any]) -> None:
        self.name = name
        self.type = type
        self.metadata = metadata
        self.files: list[str] = []
        self.dirs: list[str] = []

    def add_file(self, path: str) -> None:
        self.files.append(path)

    def add_dir(self, path: str) -> None:
        self.dirs.append(path)


class _FakeRun:
    def __init__(self) -> None:
        self.artifacts: list[tuple[_FakeArtifact, list[str]]] = []
        self.finished = False

    def log_artifact(self, artifact: _FakeArtifact, aliases: list[str]) -> None:
        self.artifacts.append((artifact, aliases))

    def finish(self) -> None:
        self.finished = True


class _FakeWandb:
    def __init__(self) -> None:
        self.init_calls: list[dict[str, Any]] = []
        self.log_calls: list[tuple[dict[str, float], int | None]] = []
        self.watch_calls: list[dict[str, Any]] = []
        self.latest_run: _FakeRun | None = None

    def init(self, **kwargs: Any) -> _FakeRun:
        self.init_calls.append(kwargs)
        self.latest_run = _FakeRun()
        return self.latest_run

    def log(self, payload: dict[str, float], step: int | None = None) -> None:
        self.log_calls.append((payload, step))

    def watch(self, model: Any, *, log: str, log_freq: int) -> None:
        self.watch_calls.append({"model": model, "log": log, "log_freq": log_freq})

    def Artifact(self, *, name: str, type: str, metadata: dict[str, Any]) -> _FakeArtifact:
        return _FakeArtifact(name=name, type=type, metadata=metadata)


def test_wandb_logger_is_noop_when_disabled(monkeypatch: Any) -> None:
    called = {"imported": False}

    def _fake_import() -> Any:
        called["imported"] = True
        return _FakeWandb()

    monkeypatch.setattr("aitchinson_flow.training.wandb_logger._import_wandb", _fake_import)

    cfg = Config()
    cfg.training.wandb_enabled = False
    logger = WandbLogger(cfg)

    assert logger.active is False
    assert called["imported"] is False
    logger.log_metrics({"train/loss": 1.0}, step=1)
    logger.finish()


def test_wandb_logger_logs_metrics_watch_and_artifacts(
    tmp_path: Path, monkeypatch: Any
) -> None:
    fake = _FakeWandb()
    monkeypatch.setattr("aitchinson_flow.training.wandb_logger._import_wandb", lambda: fake)

    cfg = Config()
    cfg.training.wandb_enabled = True
    cfg.training.wandb_project = "proj"
    cfg.training.wandb_mode = "offline"
    cfg.training.wandb_watch_model = True
    cfg.training.wandb_watch_log = "all"
    cfg.training.wandb_watch_log_freq = 10

    logger = WandbLogger(
        cfg,
        run_name="run-1",
        group="group-1",
        job_type="train",
        tags=["custom-tag"],
        extra_config={"seed": 123},
    )
    assert logger.active
    assert len(fake.init_calls) == 1
    assert fake.init_calls[0]["name"] == "run-1"
    assert fake.init_calls[0]["group"] == "group-1"
    assert "custom-tag" in fake.init_calls[0]["tags"]

    model = object()
    logger.watch_model(model)
    assert fake.watch_calls and fake.watch_calls[0]["model"] is model

    logger.log_metrics({"train/loss": 0.5}, step=12)
    assert fake.log_calls == [({"train/loss": 0.5}, 12)]

    model_path = tmp_path / "epoch_1.pt"
    model_path.write_text("x", encoding="utf-8")
    plot_dir = tmp_path / "plots"
    plot_dir.mkdir()
    (plot_dir / "plot.png").write_text("png", encoding="utf-8")

    logger.log_artifact(
        model_path,
        name="model-ckpt",
        artifact_type="model",
        aliases=["latest"],
        metadata={"epoch": 1},
    )
    logger.log_artifact(
        plot_dir,
        name="plots",
        artifact_type="plots",
        aliases=["latest"],
    )
    logger.finish()

    run = fake.latest_run
    assert run is not None
    assert run.finished is True
    assert len(run.artifacts) == 2
    first_artifact = run.artifacts[0][0]
    second_artifact = run.artifacts[1][0]
    assert str(model_path) in first_artifact.files
    assert str(plot_dir) in second_artifact.dirs
