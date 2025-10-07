from __future__ import annotations

import dataclasses
import datetime as dt
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from .schema import validate_dataframe
from .transforms import LagSpec, apply_lag_features, compute_trading_gap_indicator
from ..utils.config import Config


@dataclasses.dataclass(slots=True)
class WindowSample:
    """Single sliding-window sample with metadata."""

    features: np.ndarray  # shape (window, feature_dim)
    target: np.ndarray  # shape (target_dim,)
    symbol: str
    start_date: dt.date
    end_date: dt.date
    mask: np.ndarray  # shape (window,)


class SlidingWindowDataset:
    """Materializes sliding-window samples per split without leaking future data."""

    GAP_COLUMN = "__gap_days"

    def __init__(self, config: Config) -> None:
        self.config = config
        self.trading_gap_mask = bool(
            self.config.preprocessing.get("masks", {}).get("trading_gap", False)
        )
        self.df = self._load_frame()
        self.lag_specs = self._parse_lag_specs()
        self.derived_feature_columns = self._apply_lag_transforms()
        self.feature_columns = self._collect_feature_columns()
        if self.derived_feature_columns:
            self.feature_columns.extend(self.derived_feature_columns)
        self.feature_metadata = self._build_feature_metadata()
        self.target_column = self.config.data["target_column"]
        self.horizon = int(self.config.window.get("horizon", 1))
        self.window_size = int(self.config.window.get("size", 1))
        if self.window_size <= 0:
            raise ValueError("window.size must be positive")
        if self.horizon <= 0:
            raise ValueError("window.horizon must be positive")
        self.step = int(self.config.window.get("step", 1))
        if self.step <= 0:
            raise ValueError("window.step must be positive")

        self._split_cache: Dict[str, List[WindowSample]] = {}
        self._frame_cache: Dict[str, pd.DataFrame] = {}

    def _collect_feature_columns(self) -> List[str]:
        columns = self.config.data.get("feature_columns", {})
        if isinstance(columns, list):
            return list(columns)
        ordered: List[str] = []
        for group in ("price", "flows", "technical"):
            ordered.extend(columns.get(group, []))
        return ordered

    def _parse_lag_specs(self) -> List[LagSpec]:
        raw_specs = self.config.preprocessing.get("lag_features", [])
        specs: List[LagSpec] = []
        for entry in raw_specs:
            if not isinstance(entry, dict):
                raise TypeError("Lag feature specification must be a mapping")
            column = entry.get("column")
            if column is None:
                raise KeyError("Lag feature specification missing 'column'")
            lags = tuple(entry.get("lags", ()))
            spec = LagSpec(
                column=column,
                lags=lags,
                transform=entry.get("transform", "raw"),
                feature_group=entry.get("feature_group"),
                description=entry.get("description"),
            )
            specs.append(spec)
        return specs

    def _apply_lag_transforms(self) -> List[str]:
        if not self.lag_specs:
            return []
        symbol_col = self.config.data["symbol_column"]
        generated = apply_lag_features(self.df, self.lag_specs, symbol_column=symbol_col)
        return generated

    def _build_feature_metadata(self) -> Dict[str, str]:
        metadata: Dict[str, str] = {}
        for spec in self.lag_specs:
            for lag in spec.lags:
                key = spec.column_name(lag)
                if spec.description:
                    metadata[key] = spec.description
                    continue
                transform_label = {
                    "raw": "Level lag",
                    "delta": "Absolute change",
                    "pct_change": "Percent change",
                    "log_return": "Log return",
                }[spec.transform]
                group_hint = spec.feature_group or "feature"
                metadata[key] = (
                    f"{transform_label} of {spec.column} over {lag} day lag within the same symbol window"
                    f" (group: {group_hint})."
                )
        return metadata

    def _load_frame(self) -> pd.DataFrame:
        cfg = self.config
        df = pd.read_csv(cfg.data_path)
        violations = validate_dataframe(df)
        if violations:
            joined = "; ".join(violations)
            raise ValueError(f"Schema validation failed: {joined}")

        date_col = cfg.data["date_column"]
        symbol_col = cfg.data["symbol_column"]
        df[date_col] = pd.to_datetime(df[date_col])
        df[symbol_col] = df[symbol_col].astype(str)

        categorical = cfg.data.get("categorical_columns", {})
        for col_name, options in categorical.items():
            token = options.get("missing_token", f"UNKNOWN_{col_name.upper()}")
            if col_name in df.columns:
                df[col_name] = df[col_name].fillna(token)

        if cfg.loader.get("enforce_sort", True):
            df = df.sort_values([symbol_col, date_col]).reset_index(drop=True)

        if self.trading_gap_mask:
            df[self.GAP_COLUMN] = compute_trading_gap_indicator(
                df,
                symbol_column=symbol_col,
                date_column=date_col,
            )
        return df

    def frame_for_split(self, split: str) -> pd.DataFrame:
        """Return a cached dataframe for the requested split."""

        if split in self._frame_cache:
            return self._frame_cache[split].copy()
        frame = self._slice_split(split)
        self._frame_cache[split] = frame
        return frame.copy()

    def _slice_split(self, name: str) -> pd.DataFrame:
        start, end = self.config.split_range(name)
        date_col = self.config.data["date_column"]
        mask = (self.df[date_col].dt.date >= start) & (self.df[date_col].dt.date <= end)
        return self.df.loc[mask].copy()

    def iter_split(self, split: str) -> Iterable[WindowSample]:
        if split in self._split_cache:
            return self._split_cache[split]

        panel = self._slice_split(split)
        samples: List[WindowSample] = []
        if panel.empty:
            self._split_cache[split] = samples
            return samples

        symbol_col = self.config.data["symbol_column"]
        date_col = self.config.data["date_column"]
        target_transform = self.config.data.get("target_transform", "log_return")
        float_dtype = self.config.loader.get("float_dtype", "float32")

        feature_cols = self.feature_columns
        if self.target_column not in panel.columns:
            raise KeyError(f"Target column '{self.target_column}' not present in data")

        for symbol, symbol_df in panel.groupby(symbol_col, sort=False):
            symbol_df = symbol_df.reset_index(drop=True)
            features_arr = symbol_df[feature_cols]
            close_series = symbol_df[self.target_column]
            if self.trading_gap_mask and self.GAP_COLUMN in symbol_df.columns:
                gap_array = symbol_df[self.GAP_COLUMN].to_numpy(dtype=int)
            else:
                gap_array = None

            for start_idx in range(0, len(symbol_df) - self.window_size - self.horizon + 1, self.step):
                end_idx = start_idx + self.window_size
                target_idx = end_idx - 1
                future_idx = target_idx + self.horizon
                window_frame = features_arr.iloc[start_idx:end_idx]

                if window_frame.isna().any().any():
                    continue
                if future_idx >= len(symbol_df):
                    break

                target_value = self._compute_target(
                    current=float(close_series.iloc[target_idx]),
                    future=float(close_series.iloc[future_idx]),
                    transform=target_transform,
                )
                if target_value is None or np.isnan(target_value):
                    continue

                window_array = window_frame.to_numpy(dtype=float_dtype)
                window_array = self._normalize(window_array).astype(float_dtype, copy=False)

                if gap_array is not None:
                    gap_window = gap_array[start_idx:end_idx]
                    mask_window = np.where(gap_window == 1, 1.0, 0.0).astype(float_dtype, copy=False)
                else:
                    mask_window = np.ones(self.window_size, dtype=float_dtype)

                sample = WindowSample(
                    features=window_array,
                    target=np.array([target_value], dtype=float_dtype),
                    symbol=str(symbol),
                    start_date=symbol_df.loc[start_idx, date_col].date(),
                    end_date=symbol_df.loc[target_idx, date_col].date(),
                    mask=mask_window,
                )
                samples.append(sample)

        self._split_cache[split] = samples
        return samples

    def split_size(self, split: str) -> int:
        """Number of sliding-window samples available for a split."""

        return len(self.iter_split(split))

    def _normalize(self, window_array: np.ndarray) -> np.ndarray:
        normalization = self.config.window.get("normalization", {})
        method = normalization.get("method", "none")
        scope = normalization.get("scope", "window")
        if method == "none":
            return window_array
        if method != "zscore":
            raise NotImplementedError(f"Unsupported normalization method '{method}'")
        if scope != "window":
            raise NotImplementedError("Only per-window normalization is implemented")
        mean = window_array.mean(axis=0, keepdims=True)
        std = window_array.std(axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        normalized = (window_array - mean) / std
        return normalized

    @staticmethod
    def _compute_target(current: float, future: float, transform: str) -> float | None:
        if transform == "log_return":
            if current <= 0 or future <= 0:
                return None
            return float(np.log(future / current))
        if transform == "pct_change":
            if current == 0:
                return None
            return float((future - current) / current)
        if transform == "close":
            return future
        raise NotImplementedError(f"Unsupported target transform '{transform}'")

    def build(self) -> Dict[str, List[WindowSample]]:
        datasets: Dict[str, List[WindowSample]] = {}
        for split in self.config.splits.keys():
            samples = list(self.iter_split(split))
            if self.config.loader.get("drop_na_targets", True):
                samples = [sample for sample in samples if not np.isnan(sample.target).any()]
            datasets[split] = samples
        return datasets
