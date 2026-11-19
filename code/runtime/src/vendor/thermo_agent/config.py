from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_project_config(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate the project JSON configuration."""
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    required = {"schema_version", "workspace_root", "energyplus", "optics", "cities"}
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Project configuration is missing keys: {missing}")
    return config


def workspace_path(config: dict[str, Any], relative_path: str | Path) -> Path:
    """Resolve a configured workspace-relative path without changing the CWD."""
    root = Path(config["workspace_root"]).resolve()
    candidate = (root / Path(relative_path)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path escapes workspace root: {relative_path}") from exc
    return candidate
