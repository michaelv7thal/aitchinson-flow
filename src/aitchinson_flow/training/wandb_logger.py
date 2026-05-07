"""Optional Weights & Biases logger.

Constructed once per training run. When ``enabled`` is False the logger is a
silent no-op; when True the methods forward to ``wandb``. If ``wandb`` is not
installed, ``wandb.init`` raises, or no API key is available, the logger
auto-disables and prints a single warning — training itself never crashes
because of wandb.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from aitchinson_flow.config import Config


_log = logging.getLogger(__name__)

# Key prefix routing for log_epoch. Anything starting with one of these
# substrings goes under that namespace; everything else falls into "epoch/".
_VAL_PREFIX = "val_"
_PROBE_KEYS = ("unigram_kl", "bigram_kl", "trigram_kl", "H_gen", "H_gt")


def _config_to_plain_dict(cfg: Config) -> dict[str, Any]:
    """Stringify non-JSON-able fields (torch.device) so wandb.config accepts it."""
    d = asdict(cfg)
    if "training" in d and "device" in d["training"]:
        d["training"]["device"] = str(cfg.training.device)
    if "loader_settings" in d and "device_type" in d["loader_settings"]:
        d["loader_settings"]["device_type"] = str(cfg.loader_settings.device_type)
    return d


class WandbLogger:
    """Thin wrapper around ``wandb`` that no-ops when disabled."""

    def __init__(
        self,
        *,
        enabled: bool,
        project: str,
        name: str | None,
        entity: str | None,
        group: str | None,
        tags: Iterable[str],
        mode: str,
        config_dict: dict[str, Any],
        run_dir: Path | str | None,
    ) -> None:
        self._enabled = bool(enabled)
        self._wandb = None
        self._run = None

        if not self._enabled:
            return

        try:
            import wandb  # noqa: PLC0415 — lazy import keeps wandb optional
        except ImportError as e:
            _log.warning("wandb requested but not installed (%s); disabling.", e)
            self._enabled = False
            return

        try:
            self._run = wandb.init(
                project=project,
                name=name,
                entity=entity,
                group=group,
                tags=list(tags),
                mode=mode,
                config=config_dict,
                dir=str(run_dir) if run_dir is not None else None,
                reinit=True,
            )
            self._wandb = wandb
        except Exception as e:  # noqa: BLE001 — never let wandb kill training
            _log.warning("wandb.init failed (%s); disabling logger.", e)
            self._enabled = False
            self._wandb = None
            self._run = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def run_name(self) -> str | None:
        if self._run is None:
            return None
        return getattr(self._run, "name", None)

    def log_step(self, step: int, metrics: dict[str, float]) -> None:
        if not self._enabled or self._wandb is None:
            return
        prefixed = {
            k if "/" in k else f"train/{k}": float(v) for k, v in metrics.items()
        }
        self._wandb.log(prefixed, step=step)

    def log_epoch(self, epoch: int, metrics: dict[str, float]) -> None:
        if not self._enabled or self._wandb is None:
            return
        out: dict[str, float] = {"epoch": int(epoch)}
        for k, v in metrics.items():
            if k.startswith(_VAL_PREFIX):
                out[f"val/{k[len(_VAL_PREFIX) :]}"] = float(v)
            elif k in _PROBE_KEYS:
                out[f"probe/{k}"] = float(v)
            else:
                out[f"epoch/{k}"] = float(v)
        self._wandb.log(out)

    def log_samples(self, epoch: int, decodes: list[str]) -> None:
        if not self._enabled or self._wandb is None or not decodes:
            return
        try:
            table = self._wandb.Table(columns=["epoch", "idx", "decode"])
            for i, s in enumerate(decodes):
                table.add_data(int(epoch), i, s)
            self._wandb.log({"samples/decodes": table})
        except Exception as e:  # noqa: BLE001
            _log.warning("wandb log_samples failed (%s); continuing.", e)

    def log_artifact(self, path: Path | str, *, name: str, type_: str) -> None:
        if not self._enabled or self._wandb is None:
            return
        p = Path(path)
        if not p.exists():
            _log.warning("wandb log_artifact: %s does not exist; skipping.", p)
            return
        try:
            artifact = self._wandb.Artifact(name=name, type=type_)
            artifact.add_file(str(p))
            self._wandb.log_artifact(artifact)
        except Exception as e:  # noqa: BLE001
            _log.warning("wandb log_artifact failed (%s); continuing.", e)

    def finish(self, exit_code: int = 0) -> None:
        if not self._enabled or self._wandb is None:
            return
        try:
            self._wandb.finish(exit_code=exit_code)
        except Exception as e:  # noqa: BLE001
            _log.warning("wandb.finish failed (%s).", e)
        finally:
            self._enabled = False
            self._wandb = None
            self._run = None


def build_wandb_logger(
    cfg: Config,
    *,
    run_dir: Path | str | None = None,
    extra_tags: Iterable[str] = (),
) -> WandbLogger:
    """Construct a WandbLogger from cfg.wandb. Honors WANDB_* env vars as fallbacks."""
    wcfg = cfg.wandb
    enabled = wcfg.enabled or bool(os.environ.get("WANDB_PROJECT"))
    project = os.environ.get("WANDB_PROJECT") or wcfg.project
    entity = os.environ.get("WANDB_ENTITY") or wcfg.entity
    name = os.environ.get("WANDB_NAME") or wcfg.run_name

    tags = tuple(wcfg.tags) + tuple(extra_tags)

    return WandbLogger(
        enabled=enabled,
        project=project,
        name=name,
        entity=entity,
        group=wcfg.group,
        tags=tags,
        mode=wcfg.mode,
        config_dict=_config_to_plain_dict(cfg),
        run_dir=run_dir,
    )
