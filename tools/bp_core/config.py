from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load and validate a pipeline configuration.

    Relative paths are resolved against the current working directory so the
    documented commands behave consistently from the project root.
    """
    config_path = Path(path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("BP pipeline config must have schema_version=1")
    for section in ("datasets", "signal", "quality", "models"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"BP pipeline config is missing object: {section}")
    minimum_windows = config["quality"].get("minimum_accepted_windows_per_occasion")
    minimum_coverage = config["quality"].get("minimum_unique_clean_coverage_seconds")
    if not isinstance(minimum_windows, int) or isinstance(minimum_windows, bool) or minimum_windows < 1:
        raise ValueError("quality.minimum_accepted_windows_per_occasion must be a positive integer")
    if not isinstance(minimum_coverage, (int, float)) or isinstance(minimum_coverage, bool) or minimum_coverage <= 0:
        raise ValueError("quality.minimum_unique_clean_coverage_seconds must be positive")
    upper_arm_gate = config["quality"].get("require_upper_arm_analyzer_acceptance")
    if not isinstance(upper_arm_gate, bool):
        raise ValueError("quality.require_upper_arm_analyzer_acceptance must be true or false")
    return config, config_path


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (Path.cwd() / path).resolve()
