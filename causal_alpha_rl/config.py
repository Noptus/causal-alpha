"""Experiment configuration helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ExperimentConfig:
    name: str
    payload: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)


def load_config(config_name: str, config_dir: Path | None = None) -> ExperimentConfig:
    base_dir = config_dir or Path("experiments") / "configs"
    config_path = base_dir / f"{config_name}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Unknown config '{config_name}' at {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return ExperimentConfig(name=config_name, payload=payload)

