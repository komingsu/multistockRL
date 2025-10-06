from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipelines.training import run_training
from src.utils.config import Config

def _synthetic_prices(num_days: int) -> tuple[list[str], list[float], list[float]]:
    dates = pd.date_range("2025-01-01", periods=num_days, freq="B").strftime("%Y-%m-%d").tolist()
    close = [100.0 + i for i in range(num_days)]
    volume = [1_000.0 + 10 * i for i in range(num_days)]
    return dates, close, volume


def _base_config(csv_path: Path, dates: list[str]) -> Config:
    return Config(
        data={
            "path": str(csv_path),
            "date_column": "date",
            "symbol_column": "symbol",
            "feature_columns": {"price": ["close", "volume"], "flows": [], "technical": []},
            "target_column": "close",
            "target_transform": "log_return",
        },
        splits={
            "train": {"start": dates[0], "end": dates[8]},
            "validation": {"start": dates[8], "end": dates[-1]},
        },
        window={
            "size": 3,
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
        environment={
            "initial_cash": 1_000_000.0,
            "n": 5,
            "reward_scaling": 1.0,
        },
        agents={
            "default": "PPO",
            "policies": {
                "PPO": {
                    "policy": "MlpPolicy",
                    "learning_rate": 0.001,
                    "batch_size": 8,
                    "n_steps": 8,
                    "n_epochs": 1,
                }
            },
        },
        logging={
            "run_dir": str(csv_path.parent / "runs"),
            "metrics": ["train_reward", "valid_reward", "num_timesteps"],
            "flush_interval": 1,
        },
        preprocessing={
            "lag_features": [
                {"column": "close", "lags": [1], "transform": "log_return", "feature_group": "price"}
            ],
            "masks": {"trading_gap": True},
        },
        training={
            "total_epochs": 1,
            "eval_interval_epochs": 1,
            "checkpoint_interval_epochs": 1,
            "eval_episodes": 1,
            "log_interval_epochs": 1,
        },
    )


@pytest.mark.parametrize("eval_split", ["train"])
def test_run_training_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, eval_split: str):
    num_days = 12
    dates, close, volume = _synthetic_prices(num_days)
    data = pd.DataFrame(
        {
            "symbol": ["AAA"] * num_days,
            "date": dates,
            "open": close,
            "high": [c + 1 for c in close],
            "low": [c - 1 for c in close],
            "close": close,
            "volume": volume,
            "value": [c * v for c, v in zip(close, volume)],
            "market_cap": [1_000_000.0] * num_days,
            "change": np.diff([close[0] - 1] + close).tolist(),
            "change_rate": np.diff([close[0] - 1] + close).tolist(),
            "range_pct": [0.01] * num_days,
            "gap_pct": [0.0] * num_days,
        }
    )
    csv_path = tmp_path / "synthetic.csv"
    data.to_csv(csv_path, index=False)

    config = _base_config(csv_path, dates)

    monkeypatch.setattr("src.data.loader.validate_dataframe", lambda df: [])

    result = run_training(config, run_name="unit", eval_split=eval_split)

    assert result.train_epochs >= 1
    assert result.train_timesteps >= result.train_epochs
    assert result.train_reward is not None
    assert result.run_directory.exists()
    log_files = list(result.run_directory.glob("*.log"))
    assert log_files, "Expected log file"
    log_text = log_files[0].read_text(encoding="utf-8")
    assert "Performance summary" in log_text
    assert "alias" in log_text
    assert "Algorithm metadata" in log_text

    metrics_path = result.run_directory / "metrics.csv"
    metrics_df = pd.read_csv(metrics_path)
    required_columns = {"project_epoch", "total_env_steps", "episodes_completed", "updates_applied", "nav", "turnover", "drawdown"}
    assert required_columns.issubset(metrics_df.columns)
    assert metrics_df["project_epoch"].notna().any()

    trace_dir = result.run_directory / "validation_traces"
    assert trace_dir.exists()
    trace_files = sorted(trace_dir.glob("*.csv"))
    assert trace_files, "Expected validation trace CSV"
    trace_df = pd.read_csv(trace_files[0])
    assert "cash" in trace_df.columns
    assert "portfolio_value_change" in trace_df.columns
    assert "nav_return" in trace_df.columns
    assert "turnover" in trace_df.columns
    assert "drawdown" in trace_df.columns
    for symbol in sorted(data["symbol"].unique()):
        assert symbol in trace_df.columns

    final_trace_dir = result.run_directory / "final_eval_traces"
    assert final_trace_dir.exists()
    final_trace_files = sorted(final_trace_dir.glob("*.csv"))
    assert final_trace_files, "Expected final evaluation trace CSV"
    final_trace_df = pd.read_csv(final_trace_files[0])
    assert list(trace_df.columns) == list(final_trace_df.columns)

    final_checkpoint = sorted(result.checkpoint_dir.rglob("model.pt"))[-1]
    payload = torch.load(final_checkpoint, map_location="cpu", weights_only=False)
    assert "policy_state_dict" in payload
    assert payload["config"]["data"]["path"] == str(csv_path)


def test_resume_from_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    num_days = 12
    dates, close, volume = _synthetic_prices(num_days)
    data = pd.DataFrame(
        {
            "symbol": ["AAA"] * num_days,
            "date": dates,
            "open": close,
            "high": [c + 1 for c in close],
            "low": [c - 1 for c in close],
            "close": close,
            "volume": volume,
            "value": [c * v for c, v in zip(close, volume)],
            "market_cap": [1_000_000.0] * num_days,
            "change": np.diff([close[0] - 1] + close).tolist(),
            "change_rate": np.diff([close[0] - 1] + close).tolist(),
            "range_pct": [0.01] * num_days,
            "gap_pct": [0.0] * num_days,
        }
    )
    csv_path = tmp_path / "synthetic.csv"
    data.to_csv(csv_path, index=False)

    config = _base_config(csv_path, dates)

    monkeypatch.setattr("src.data.loader.validate_dataframe", lambda df: [])

    first_run = run_training(config, run_name="resume_test", eval_split="train")
    checkpoint_path = sorted(first_run.checkpoint_dir.rglob("model.pt"))[-1]

    second_run = run_training(
        config,
        run_name="resume_test",
        eval_split="train",
        resume_path=checkpoint_path,
    )

    assert second_run.train_reward is not None
    assert second_run.train_epochs >= 1
    assert second_run.train_timesteps >= second_run.train_epochs
    assert second_run.run_directory.exists()
    resume_log = list(second_run.run_directory.glob("*.log"))[0].read_text(encoding="utf-8")
    assert "Resumed from checkpoint" in resume_log
