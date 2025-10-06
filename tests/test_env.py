
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.env.multi_stock_env import EnvironmentConfig, MultiStockTradingEnv
from src.utils.builders import build_environment
from src.utils.config import load_config


@pytest.fixture()
def configured_env():
    config = load_config("configs/base.yaml")
    config.environment.update(
        {
            "initial_cash": 1_000_000.0,
            "n": 1,
            "reward_scaling": 1.0,
            "max_steps": 2,
            "friction": {
                "commission_rate": 0.001,
                "min_commission": 0.0,
                "slippage_bps": 10.0,
            },
            "stress": {
                "enabled": False,
                "commission_multiplier": 1.0,
                "slippage_multiplier": 1.0,
            },
        }
    )

    price_frame = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "symbol": ["AAA", "AAA", "AAA"],
            "close": [100.0, 102.0, 105.0],
        }
    )

    env = build_environment(config, price_frame)
    try:
        yield env
    finally:
        env.close()


def test_transaction_costs_and_diagnostics(configured_env):
    env = configured_env
    obs, info = env.reset()
    assert obs.dtype == np.float32
    assert info["nav_return"] == pytest.approx(0.0)
    assert info["turnover"] == pytest.approx(0.0)
    assert info["drawdown"] == pytest.approx(0.0)

    action_buy = np.array([2], dtype=np.int64)
    obs, reward, terminated, truncated, info = env.step(action_buy)
    assert not terminated
    assert not truncated

    initial_cash = 1_000_000.0
    buy_price = 100.0
    next_price = 102.0
    expected_cash = initial_cash - buy_price
    expected_portfolio_value = expected_cash + next_price
    expected_nav_return = (expected_portfolio_value - initial_cash) / initial_cash
    expected_turnover = buy_price / initial_cash

    assert info["commission_cost"] == pytest.approx(0.0)
    assert info["slippage_cost"] == pytest.approx(0.0)
    assert info["portfolio_value"] == pytest.approx(expected_portfolio_value)
    assert info["nav_return"] == pytest.approx(expected_nav_return)
    assert info["turnover"] == pytest.approx(expected_turnover)
    assert info["drawdown"] == pytest.approx(0.0)
    assert env.holdings[0] == 1

    action_sell = np.array([0], dtype=np.int64)
    obs, reward, terminated, truncated, info = env.step(action_sell)
    assert terminated
    assert not truncated

    sell_price = 102.0
    next_price = 105.0
    commission_rate = 0.001
    slippage_rate = 10.0 / 10_000.0
    commission_cost = sell_price * commission_rate
    slippage_cost = sell_price * slippage_rate
    total_cost = commission_cost + slippage_cost

    prev_portfolio_value = expected_portfolio_value
    expected_cash = expected_cash + sell_price - total_cost
    expected_portfolio_value = expected_cash  # no holdings
    expected_nav_return = (expected_portfolio_value - prev_portfolio_value) / prev_portfolio_value
    expected_turnover = sell_price / prev_portfolio_value
    expected_drawdown = (prev_portfolio_value - expected_portfolio_value) / prev_portfolio_value

    assert info["commission_cost"] == pytest.approx(commission_cost, rel=1e-7)
    assert info["slippage_cost"] == pytest.approx(slippage_cost, rel=1e-7)
    assert info["cost_ratio"] == pytest.approx(total_cost / prev_portfolio_value, rel=1e-7)
    assert info["portfolio_value"] == pytest.approx(expected_portfolio_value, rel=1e-7)
    assert info["nav_return"] == pytest.approx(expected_nav_return, rel=1e-7)
    assert info["turnover"] == pytest.approx(expected_turnover, rel=1e-7)
    assert info["drawdown"] == pytest.approx(expected_drawdown, rel=1e-7)
    assert info["returns"][0] == pytest.approx((next_price - sell_price) / sell_price, rel=1e-7)
    assert info["reward_unscaled"] == pytest.approx(expected_portfolio_value - initial_cash, rel=1e-7)
    assert info["reward"] == pytest.approx(info["reward_unscaled"], rel=1e-7)


def _two_symbol_env() -> MultiStockTradingEnv:
    dates = pd.date_range("2025-01-01", periods=3, freq="D")
    rows = []
    for symbol, base in (("AAA", 10.0), ("BBB", 20.0)):
        for offset, date in enumerate(dates):
            rows.append({"date": date, "symbol": symbol, "close": base + offset})
    frame = pd.DataFrame(rows)
    return MultiStockTradingEnv(
        frame,
        EnvironmentConfig(initial_cash=1_000.0, n=5, max_steps=len(dates) - 1),
    )


def test_portfolio_override_requires_all_symbols():
    env = _two_symbol_env()
    try:
        env.set_portfolio_state({"holdings": {"AAA": 5}, "cash": 500.0})
        with pytest.raises(KeyError):
            env.reset()
    finally:
        env.close()


def test_portfolio_override_applies_explicit_zero_positions():
    env = _two_symbol_env()
    try:
        env.set_portfolio_state(
            {
                "holdings": {"AAA": 5, "BBB": 0},
                "cash": 250.0,
                "start_date": "2025-01-02",
            }
        )
        obs, info = env.reset()
        assert env.holdings.tolist() == [5, 0]
        assert info["cash"] == pytest.approx(250.0)
        symbol_index = env.symbols.index("AAA")
        price_a = env.price_matrix[env.current_idx, symbol_index]
        expected_nav = 250.0 + price_a * 5
        assert info["portfolio_value"] == pytest.approx(expected_nav)
        assert env.initial_cash == pytest.approx(expected_nav)
        # Ensure zero-holding symbol is present explicitly
        assert info["holdings"].tolist() == [5, 0]
    finally:
        env.close()
