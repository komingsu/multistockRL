"""Preprocessing transforms for sliding-window dataset generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd


def _ensure_sequence(values: Iterable[int]) -> List[int]:
    seq = list(values)
    if not seq:
        raise ValueError("Lag specification requires at least one lag value")
    if any(lag <= 0 for lag in seq):
        raise ValueError("Lag values must be positive integers")
    return seq

@dataclass(slots=True)
class LagSpec:
    """Specification for generating lag-based derived features."""

    column: str
    lags: tuple[int, ...]
    transform: str = "raw"
    feature_group: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        self.lags = tuple(_ensure_sequence(self.lags))
        supported = {"raw", "delta", "pct_change", "log_return"}
        if self.transform not in supported:
            raise ValueError(
                f"Unsupported transform '{self.transform}'. Expected one of {sorted(supported)}"
            )

    def column_name(self, lag: int) -> str:
        suffix_map = {
            "raw": f"lag_{lag}",
            "delta": f"delta_lag_{lag}",
            "pct_change": f"pct_change_lag_{lag}",
            "log_return": f"log_return_lag_{lag}",
        }
        suffix = suffix_map[self.transform]
        return f"{self.column}_{suffix}"


def apply_lag_features(
    frame: pd.DataFrame,
    specs: Iterable[LagSpec],
    *,
    symbol_column: str,
) -> List[str]:
    """Apply lag specifications to the frame and return generated column names."""

    generated: List[str] = []
    specs = list(specs)
    if not specs:
        return generated

    grouped = frame.groupby(symbol_column, sort=False)

    for spec in specs:
        if spec.column not in frame.columns:
            raise KeyError(f"Lag spec references missing column '{spec.column}'")

        current = frame[spec.column].astype(float)
        for lag in spec.lags:
            shifted = grouped[spec.column].shift(lag).astype(float)
            if spec.transform == "raw":
                values = shifted
            elif spec.transform == "delta":
                values = current - shifted
            elif spec.transform == "pct_change":
                with np.errstate(divide="ignore", invalid="ignore"):
                    values = (current - shifted) / shifted
                values = values.where(~np.isclose(shifted, 0.0), np.nan)
            elif spec.transform == "log_return":
                ratio = np.where((current > 0.0) & (shifted > 0.0), current / shifted, np.nan)
                values = pd.Series(np.log(ratio), index=current.index)
            else:  # defensive fallback
                raise RuntimeError(f"Unexpected transform '{spec.transform}'")

            new_col = spec.column_name(lag)
            frame[new_col] = values
            generated.append(new_col)
    return generated


def compute_trading_gap_indicator(
    frame: pd.DataFrame,
    *,
    symbol_column: str,
    date_column: str,
) -> pd.Series:
    """Return a Series indicating gap length (in days) between consecutive rows."""

    date_series = pd.to_datetime(frame[date_column])
    gap = (
        date_series.groupby(frame[symbol_column]).diff().dt.days.fillna(1).astype(int)
    )
    gap[gap <= 0] = 1
    return gap


__all__ = [
    "LagSpec",
    "apply_lag_features",
    "compute_trading_gap_indicator",
]
