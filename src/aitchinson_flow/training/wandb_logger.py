from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from aitchinson_flow.config import Config
from aitchinson_flow.training.checkpoint import config_checkpoint_dict


def _import_wandb() -> Any | None:
    try:
        import wandb
    except Exception as exc:  # pragma: no cover - import environment dependent
        warnings.warn(
            f"W&B tracking requested but wandb could not be imported: {exc}",
            stacklevel=2,
        )
        return None
    return wandb


def _clean_wandb_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _clean_wandb_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_wandb_value(v) for v in value]
    if isinstance(value, tuple):
        return [_clean_wandb_value(v) for v in value]
    if isinstance(value, set):
        return sorted(_clean_wandb_value(v) for v in value)
    return value


def _build_run_config(cfg: Config, extra_config: Mapping[str, Any] | None) -> dict[str, Any]:
    payload: dict[str, Any] = {"config": config_checkpoint_dict(cfg)}
    if extra_config:
        payload["context"] = _clean_wandb_value(dict(extra_config))
    return payload


class WandbLogger:
    """Best-effort W&B logger with safe no-op behavior."""

    def __init__(
        self,
        cfg: Config,
        *,
        run_name: str | None = None,
        group: str | None = None,
        job_type: str | None = None,
        tags: Sequence[str] | None = None,
        extra_config: Mapping[str, Any] | None = None,
    ) -> None:
        self._cfg = cfg
        self._wandb: Any | None = None
        self._run: Any | None = None
        self._active = False

        if not cfg.training.wandb_enabled or cfg.training.wandb_mode == "disabled":
            return

        wandb = _import_wandb()
        if wandb is None:
            return

        merged_tags = list(cfg.training.wandb_tags)
        if tags:
            merged_tags.extend(tags)
        # Keep ordering stable while removing duplicates.
        merged_tags = list(dict.fromkeys(merged_tags))

        init_kwargs: dict[str, Any] = {
            "project": cfg.training.wandb_project,
            "entity": cfg.training.wandb_entity,
            "group": group if group is not None else cfg.training.wandb_group,
            "name": run_name if run_name is not None else cfg.training.wandb_run_name,
            "job_type": job_type if job_type is not None else cfg.training.wandb_job_type,
            "notes": cfg.training.wandb_notes,
            "mode": cfg.training.wandb_mode,
            "tags": merged_tags if merged_tags else None,
            "config": _build_run_config(cfg, extra_config),
            "reinit": True,
        }
        init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

        try:
            self._run = wandb.init(**init_kwargs)
        except Exception as exc:
            warnings.warn(
                f"W&B init failed; continuing without tracking: {exc}",
                stacklevel=2,
            )
            return

        self._wandb = wandb
        self._active = self._run is not None

    @property
    def active(self) -> bool:
        return self._active and self._run is not None and self._wandb is not None

    def watch_model(self, model: Any) -> None:
        if not self.active or not self._cfg.training.wandb_watch_model:
            return
        try:
            self._wandb.watch(
                model,
                log=self._cfg.training.wandb_watch_log,
                log_freq=self._cfg.training.wandb_watch_log_freq,
            )
        except Exception as exc:
            warnings.warn(f"W&B watch failed: {exc}", stacklevel=2)

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        if not self.active or not metrics:
            return
        try:
            payload = {k: float(v) for k, v in metrics.items()}
            self._wandb.log(payload, step=step)
        except Exception as exc:
            warnings.warn(f"W&B metric logging failed: {exc}", stacklevel=2)

    def log_artifact(
        self,
        path: str | Path,
        *,
        name: str,
        artifact_type: str,
        aliases: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not self.active:
            return
        p = Path(path)
        if not p.exists():
            return
        try:
            artifact = self._wandb.Artifact(
                name=name,
                type=artifact_type,
                metadata=_clean_wandb_value(dict(metadata or {})),
            )
            if p.is_dir():
                artifact.add_dir(str(p))
            else:
                artifact.add_file(str(p))
            self._run.log_artifact(artifact, aliases=list(aliases or []))
        except Exception as exc:
            warnings.warn(f"W&B artifact logging failed: {exc}", stacklevel=2)

    def finish(self) -> None:
        if not self.active:
            return
        try:
            self._run.finish()
        except Exception as exc:
            warnings.warn(f"W&B finish failed: {exc}", stacklevel=2)
        finally:
            self._active = False
