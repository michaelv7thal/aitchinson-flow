"""Common script bootstrap helpers."""

from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_repo_paths(anchor: Path) -> Path:
    """Add ``src`` and repo root to ``sys.path`` for direct script execution."""
    repo_root = anchor.resolve().parents[1]
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))
    return repo_root


__all__ = ["bootstrap_repo_paths"]
