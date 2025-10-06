from __future__ import annotations

import dataclasses
import datetime as dt
from pathlib import Path
from typing import Any, Dict

import yaml

@dataclasses.dataclass(slots=True)
class Config:
    """Container for structured experiment configuration."""

    data: Dict[str, Any]
    splits: Dict[str, Any]
    window: Dict[str, Any]
    loader: Dict[str, Any]
    environment: Dict[str, Any] = dataclasses.field(default_factory=dict)
    agents: Dict[str, Any] = dataclasses.field(default_factory=dict)
    logging: Dict[str, Any] = dataclasses.field(default_factory=dict)
    preprocessing: Dict[str, Any] = dataclasses.field(default_factory=dict)
    training: Dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def data_path(self) -> Path:
        path = Path(self.data["path"])
        if not path.exists():
            raise FileNotFoundError(f"Data file missing: {path}")
        return path

    def split_range(self, split: str) -> tuple[dt.date, dt.date]:
        if split not in self.splits:
            raise KeyError(f"Unknown split '{split}' in configuration")
        raw_start = self.splits[split]["start"]
        raw_end = self.splits[split]["end"]
        start = self._coerce_date(raw_start)
        end = self._coerce_date(raw_end)
        if start > end:
            raise ValueError(f"Split '{split}' start {start} is after end {end}")
        return start, end

    def environment_settings(self) -> Dict[str, Any]:
        return dict(self.environment)

    def logging_settings(self) -> Dict[str, Any]:
        return dict(self.logging)

    def agent_settings(self, name: str | None = None) -> Dict[str, Any]:
        agents = self.agents.get("policies", {})
        if name is None:
            name = self.agents.get("default")
        if not name:
            raise KeyError("Agent configuration requires a policy name")
        if name not in agents:
            raise KeyError(f"Agent '{name}' not defined in configuration")
        return dict(agents[name])

    def training_settings(self) -> Dict[str, Any]:
        return dict(self.training)

    @staticmethod
    def _coerce_date(value: Any) -> dt.date:
        if isinstance(value, dt.date):
            return value
        if isinstance(value, str):
            return dt.date.fromisoformat(value)
        raise TypeError(f"Unsupported date type: {type(value).__name__}")


def load_config(path: str | Path) -> Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    required_keys = {"data", "splits", "window", "loader"}
    missing = required_keys - raw.keys()
    if missing:
        keys = ", ".join(sorted(missing))
        raise KeyError(f"Config missing sections: {keys}")
    optional_keys = {"environment", "agents", "logging", "preprocessing", "training"}
    payload: Dict[str, Any] = {key: raw.get(key, {}) for key in optional_keys}
    init_kwargs = {key: raw[key] for key in required_keys}
    init_kwargs.update(payload)
    return Config(**init_kwargs)
