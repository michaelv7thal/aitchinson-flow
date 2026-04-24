"""Typed schema for unified benchmark outputs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkResultRow:
    """One benchmark observation for a (component, task, scale) triple."""

    component: str
    task: str
    scale: str
    auroc_energy: float | None = None
    auroc_variance: float | None = None
    auroc_combined: float | None = None
    auroc_spilled: float | None = None
    energy_gap: float | None = None
    variance_ratio: float | None = None
    per_token_auroc_at_corruption: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BenchmarkResultRow":
        """Build a row from JSON payload."""
        return cls(
            component=str(payload["component"]),
            task=str(payload["task"]),
            scale=str(payload["scale"]),
            auroc_energy=_as_optional_float(payload.get("auroc_energy")),
            auroc_variance=_as_optional_float(payload.get("auroc_variance")),
            auroc_combined=_as_optional_float(payload.get("auroc_combined")),
            auroc_spilled=_as_optional_float(payload.get("auroc_spilled")),
            energy_gap=_as_optional_float(payload.get("energy_gap")),
            variance_ratio=_as_optional_float(payload.get("variance_ratio")),
            per_token_auroc_at_corruption=_as_optional_float(
                payload.get("per_token_auroc_at_corruption")
            ),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass
class FullBenchmarkResults:
    """Top-level full benchmark artifact."""

    rows: list[BenchmarkResultRow]
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the whole artifact."""
        return {
            "rows": [row.to_dict() for row in self.rows],
            "created_at": self.created_at,
            "config_snapshot": self.config_snapshot,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FullBenchmarkResults":
        """Build an artifact from JSON payload."""
        rows_raw = payload.get("rows") or []
        rows = [BenchmarkResultRow.from_dict(item) for item in rows_raw]
        return cls(
            rows=rows,
            created_at=str(payload.get("created_at") or ""),
            config_snapshot=dict(payload.get("config_snapshot") or {}),
        )

    def to_json_file(self, path: str | Path) -> Path:
        """Write this artifact to a JSON file."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return out

    @classmethod
    def from_json_file(cls, path: str | Path) -> "FullBenchmarkResults":
        """Read a full benchmark artifact from disk."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)

