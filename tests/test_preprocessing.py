from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.loader import SlidingWindowDataset
from src.data.transforms import LagSpec, apply_lag_features, compute_trading_gap_indicator
from src.utils.config import Config


def test_apply_lag_features_creates_expected_columns():
    frame = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "AAA", "AAA"],
            "date": pd.date_range("2025-01-01", periods=4, freq="B"),
            "close": [100.0, 102.0, 104.0, 103.0],
            "volume": [1000.0, 1100.0, 900.0, 950.0],
        }
    )
    spec_close = LagSpec(column="close", lags=(1,), transform="log_return", feature_group="price")
    spec_volume = LagSpec(column="volume", lags=(1,), transform="pct_change", feature_group="liquidity")

    generated = apply_lag_features(frame, [spec_close, spec_volume], symbol_column="symbol")

    assert generated == ["close_log_return_lag_1", "volume_pct_change_lag_1"]
    expected_close_log = np.log(102.0 / 100.0)
    assert frame.loc[1, "close_log_return_lag_1"] == pytest.approx(expected_close_log)
    expected_volume_change = (1100.0 - 1000.0) / 1000.0
    assert frame.loc[1, "volume_pct_change_lag_1"] == pytest.approx(expected_volume_change)


def test_sliding_window_dataset_emits_masks_and_lagged_features(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "AAA", "AAA"],
            "date": [
                "2025-01-01",
                "2025-01-02",
                "2025-01-06",
                "2025-01-07",
            ],
            "close": [100.0, 102.0, 104.0, 103.0],
            "volume": [1000.0, 1100.0, 800.0, 900.0],
        }
    )
    csv_path = tmp_path / "toy.csv"
    data.to_csv(csv_path, index=False)

    config = Config(
        data={
            "path": str(csv_path),
            "date_column": "date",
            "symbol_column": "symbol",
            "feature_columns": {"price": ["close", "volume"], "flows": [], "technical": []},
            "target_column": "close",
            "target_transform": "log_return",
        },
        splits={
            "train": {"start": "2025-01-01", "end": "2025-01-31"},
        },
        window={
            "size": 2,
            "horizon": 1,
            "step": 1,
            "normalization": {"method": "zscore", "scope": "window"},
            "include_future_targets": False,
        },
        loader={
            "drop_na_targets": True,
            "enforce_sort": True,
            "float_dtype": "float32",
            "cache": {"enabled": False},
        },
        environment={},
        agents={},
        logging={},
        preprocessing={
            "lag_features": [
                {"column": "close", "lags": [1], "transform": "log_return", "feature_group": "price"},
                {"column": "volume", "lags": [1], "transform": "pct_change", "feature_group": "liquidity"},
            ],
            "masks": {"trading_gap": True},
        },
    )

    monkeypatch.setattr("src.data.loader.validate_dataframe", lambda df: [])

    dataset = SlidingWindowDataset(config)

    lag_col = "close_log_return_lag_1"
    volume_col = "volume_pct_change_lag_1"
    assert lag_col in dataset.feature_columns
    assert volume_col in dataset.feature_columns

    expected_log_return = np.log(104.0 / 102.0)
    assert dataset.df.loc[2, lag_col] == pytest.approx(expected_log_return)
    expected_volume_change = (800.0 - 1100.0) / 1100.0
    assert dataset.df.loc[2, volume_col] == pytest.approx(expected_volume_change)

    samples = list(dataset.iter_split("train"))
    assert len(samples) == 1
    sample = samples[0]
    assert sample.features.shape == (config.window["size"], len(dataset.feature_columns))
    assert sample.mask.shape == (config.window["size"],)
    assert sample.mask.dtype == np.float32
    assert sample.mask[0] == pytest.approx(1.0)
    assert sample.mask[1] == pytest.approx(0.0)

    gap_series = compute_trading_gap_indicator(
        dataset.df,
        symbol_column="symbol",
        date_column="date",
    )
    assert int(gap_series.iloc[2]) == 4
