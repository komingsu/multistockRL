"""Simple CSV + YAML experiment logging."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

from src.utils.config import Config


@dataclass(slots=True)
class ExperimentLogger:
    """Append-only logger for metrics, events, and configuration snapshots."""

    root_dir: Path
    run_name: str
    metrics: Iterable[str] = field(default_factory=list)
    flush_interval: int = 100
    destination: str = "file"
    timestamp_format: str = "%Y-%m-%d %H:%M:%S"
    run_dir: Path = field(init=False)
    metrics_path: Path = field(init=False)
    config_path: Path = field(init=False)
    log_path: Path = field(init=False)
    _buffer: List[Dict[str, Any]] = field(init=False, default_factory=list)
    _header_written: bool = field(init=False, default=False)
    _fieldnames: List[str] = field(init=False, default_factory=list)
    _written_fieldnames: List[str] = field(init=False, default_factory=list)
    _emit_file: bool = field(init=False, default=True)
    _emit_console: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        dest = (self.destination or "file").strip().lower()
        if dest in {"stdout", "console", "print"}:
            self._emit_file = False
            self._emit_console = True
            self.destination = "stdout"
        elif dest in {"both", "all"}:
            self._emit_file = True
            self._emit_console = True
            self.destination = "both"
        else:
            self._emit_file = True
            self._emit_console = False
            self.destination = "file"

        now = datetime.utcnow()
        timestamp = now.strftime("%Y%m%d-%H%M%S")
        date_stamp = now.strftime("%Y%m%d")
        safe_name = self.run_name.replace(" ", "_")
        self.run_dir = self.root_dir / f"{timestamp}_{safe_name}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.csv"
        self.config_path = self.run_dir / "config.yaml"
        self.log_path = self.run_dir / f"{date_stamp}_{safe_name}.log"
        self.metrics = list(self.metrics)

        header_line = f"[{now.strftime(self.timestamp_format)}] alias={self.run_name}"
        if self._emit_file:
            self.log_path.write_text(header_line + "\n", encoding="utf-8")
        if self._emit_console:
            print(header_line, flush=True)

        self._fieldnames = ["step", "timestamp"]
        for metric in self.metrics:
            if metric not in self._fieldnames:
                self._fieldnames.append(metric)

    def log_config(self, config: Config) -> None:
        payload = {
            "data": config.data,
            "splits": config.splits,
            "window": config.window,
            "loader": config.loader,
            "environment": config.environment,
            "agents": config.agents,
            "logging": config.logging,
            "preprocessing": config.preprocessing,
            "training": config.training,
        }
        with self.config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
        self.log_event("Config snapshot written to config.yaml")

    def log_event(self, message: str) -> None:
        timestamp = datetime.utcnow().strftime(self.timestamp_format)
        self._emit_line(f"[{timestamp}] {message}")

    def log_metrics(self, step: int, values: Dict[str, Any]) -> None:
        row = {
            "step": step,
            "timestamp": datetime.utcnow().strftime(self.timestamp_format),
        }
        for metric in self.metrics:
            row[metric] = values.get(metric)
        for key, value in values.items():
            if key not in row:
                row[key] = value
        self._register_fieldnames(row)
        self._buffer.append(row)
        if len(self._buffer) >= self.flush_interval:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        fieldnames = list(self._fieldnames)
        if self._header_written and fieldnames != self._written_fieldnames:
            self._rewrite_metrics(fieldnames)
        mode = "a" if self._header_written else "w"
        with self.metrics_path.open(mode, encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            for row in self._buffer:
                sanitized_row = {name: row.get(name) for name in fieldnames}
                writer.writerow(sanitized_row)
        if self._emit_console:
            latest = self._buffer[-1]
            metrics_summary = ", ".join(
                f"{metric}={latest.get(metric)}"
                for metric in self.metrics
                if latest.get(metric) is not None
            )
            summary = f"[{latest.get('timestamp')}] step={latest.get('step')}"
            if metrics_summary:
                summary += f" {metrics_summary}"
            print(summary, flush=True)
        self._written_fieldnames = list(fieldnames)
        self._buffer.clear()

    def _register_fieldnames(self, row: Dict[str, Any]) -> None:
        for key in row.keys():
            if key not in self._fieldnames:
                self._fieldnames.append(key)

    def _rewrite_metrics(self, fieldnames: List[str]) -> None:
        existing_rows: List[Dict[str, Any]] = []
        if self.metrics_path.exists():
            with self.metrics_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames:
                    existing_rows = list(reader)
        with self.metrics_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in existing_rows:
                writer.writerow({name: row.get(name) for name in fieldnames})
        self._written_fieldnames = list(fieldnames)
        self._header_written = True

    def finalize_summary(self, summary: Dict[str, Any]) -> None:
        self.log_event("Performance summary:")
        for key, value in summary.items():
            self.log_event(f"{key}={value}")

    def close(self) -> None:
        self.flush()

    def as_run_directory(self) -> Path:
        return self.run_dir

    def as_log_path(self) -> Path:
        return self.log_path

    def _emit_line(self, line: str) -> None:
        if self._emit_file:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        if self._emit_console:
            print(line, flush=True)
