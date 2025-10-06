"""Regression tests for evaluation trace ordering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv

from src.env.multi_stock_env import EnvironmentConfig, MultiStockTradingEnv
from src.pipelines.evaluation import evaluate_full_episodes


class _StaticPolicy:
    """Predicts hold actions for every symbol."""

    def __init__(self, action_dim: int) -> None:
        self._action_dim = action_dim

    def predict(self, observation, deterministic: bool = True):  # pragma: no cover - simple stub
        batch = observation.shape[0]
        actions = np.zeros((batch, self._action_dim), dtype=np.int64)
        return actions, None


def _make_env() -> MultiStockTradingEnv:
    dates = pd.date_range("2025-01-01", periods=5, freq="D")
    rows = []
    for symbol in ["100", "105"]:
        base = 10.0 if symbol == "100" else 20.0
        for offset, date in enumerate(dates):
            rows.append({"date": date, "symbol": symbol, "close": base + offset})
    frame = pd.DataFrame(rows)
    return MultiStockTradingEnv(
        frame,
        EnvironmentConfig(initial_cash=1_000_000.0, n=1, max_steps=len(dates) - 1),
    )


def test_evaluation_traces_are_monotonic():
    example_env = _make_env()
    action_dim = len(example_env.symbols)
    example_env.close()

    vec_env = DummyVecEnv([_make_env])
    model = _StaticPolicy(action_dim=action_dim)

    collected = []

    def writer(ep_index: int, traces: list[dict[str, float]], _env) -> None:
        collected.append(pd.DataFrame(traces))

    rewards = evaluate_full_episodes(
        model,
        vec_env,
        episodes=1,
        deterministic=True,
        trace_writer=writer,
    )

    assert rewards.shape == (1,)
    assert collected, "evaluation did not emit traces"

    frame = collected[0]
    assert not frame.empty, "trace frame should contain portfolio rows"
    assert "date" in frame, "trace frame missing date column"
    date_series = pd.to_datetime(frame["date"])
    assert date_series.is_monotonic_increasing, "trace dates must be non-decreasing"
