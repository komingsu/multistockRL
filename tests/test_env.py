
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env.costs import evaluate_transaction_costs
from src.env.frictions import FrictionConfig
from src.env.multi_stock_env import EnvironmentConfig, MultiStockTradingEnv
from src.utils.builders import build_environment
from src.utils.config import load_config
from src.utils.portfolio import apply_tau_limit, integerize_allocations


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
                "commission_rate": 0.005,
                "min_commission": 0.0,
                "slippage_bps": 50.0,
            },
            "stress": {
                "enabled": False,
                "commission_multiplier": 1.0,
                "slippage_multiplier": 1.0,
            },
            "action_mode": "shares",
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

    commission_rate = 0.005
    slippage_rate = 50.0 / 10_000.0
    buy_commission = buy_price * commission_rate
    buy_slippage = buy_price * slippage_rate
    total_buy_cost = buy_commission + buy_slippage
    gross_value = (initial_cash - buy_price) + next_price
    expected_cash = initial_cash - buy_price - total_buy_cost
    expected_portfolio_value = expected_cash + next_price
    expected_nav_return = (expected_portfolio_value / initial_cash) - 1.0
    expected_turnover = buy_price / gross_value
    expected_cost_ratio = total_buy_cost / gross_value
    expected_log_return = np.log(expected_portfolio_value / initial_cash)
    lambda_turnover = env.config.lambda_turnover
    expected_reward = expected_log_return - lambda_turnover * expected_turnover

    assert info["commission_cost"] == pytest.approx(buy_commission)
    assert info["slippage_cost"] == pytest.approx(buy_slippage)
    assert info["portfolio_value"] == pytest.approx(expected_portfolio_value)
    assert info["nav_return"] == pytest.approx(expected_nav_return)
    assert info["turnover"] == pytest.approx(expected_turnover)
    assert info["cost_ratio"] == pytest.approx(expected_cost_ratio)
    assert info["drawdown"] == pytest.approx(0.0)
    assert info["log_return"] == pytest.approx(expected_log_return)
    assert info["reward_unscaled"] == pytest.approx(expected_reward)
    assert info["reward"] == pytest.approx(expected_reward)
    weights = info.get("weights")
    assert weights is not None and len(weights) == 1
    assert weights[0] == pytest.approx((expected_portfolio_value - info["cash"]) / gross_value)
    assert info["cash_fraction"] == pytest.approx(info["cash"] / gross_value)
    assert env.holdings[0] == 1

    action_sell = np.array([0], dtype=np.int64)
    obs, reward, terminated, truncated, info = env.step(action_sell)
    assert terminated
    assert not truncated

    sell_price = 102.0
    next_price = 105.0
    commission_cost = sell_price * commission_rate
    slippage_cost = sell_price * slippage_rate
    total_cost = commission_cost + slippage_cost

    prev_portfolio_value = expected_portfolio_value
    cash_pre_cost = expected_cash + sell_price
    gross_value_sell = cash_pre_cost
    expected_cash = cash_pre_cost - total_cost
    expected_portfolio_value = expected_cash  # no holdings
    expected_nav_return = (expected_portfolio_value / prev_portfolio_value) - 1.0
    expected_turnover = sell_price / gross_value_sell
    expected_drawdown = (prev_portfolio_value - expected_portfolio_value) / prev_portfolio_value
    expected_cost_ratio = total_cost / gross_value_sell
    expected_log_return = np.log(expected_portfolio_value / prev_portfolio_value)
    expected_reward = expected_log_return - lambda_turnover * expected_turnover

    assert info["commission_cost"] == pytest.approx(commission_cost, rel=1e-7)
    assert info["slippage_cost"] == pytest.approx(slippage_cost, rel=1e-7)
    assert info["cost_ratio"] == pytest.approx(expected_cost_ratio, rel=1e-7)
    assert info["portfolio_value"] == pytest.approx(expected_portfolio_value, rel=1e-7)
    assert info["nav_return"] == pytest.approx(expected_nav_return, rel=1e-7)
    assert info["turnover"] == pytest.approx(expected_turnover, rel=1e-7)
    assert info["drawdown"] == pytest.approx(expected_drawdown, rel=1e-7)
    assert info["returns"][0] == pytest.approx((next_price - sell_price) / sell_price, rel=1e-7)
    assert info["log_return"] == pytest.approx(expected_log_return, rel=1e-7)
    assert info["reward_unscaled"] == pytest.approx(expected_reward, rel=1e-7)
    assert info["reward"] == pytest.approx(expected_reward, rel=1e-7)


def _two_symbol_env() -> MultiStockTradingEnv:
    dates = pd.date_range("2025-01-01", periods=3, freq="D")
    rows = []
    for symbol, base in (("AAA", 10.0), ("BBB", 20.0)):
        for offset, date in enumerate(dates):
            rows.append({"date": date, "symbol": symbol, "close": base + offset})
    frame = pd.DataFrame(rows)
    return MultiStockTradingEnv(
        frame,
        EnvironmentConfig(
            initial_cash=1_000.0,
            n=5,
            max_steps=len(dates) - 1,
            action_mode="shares",
        ),
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


def test_weight_action_integerization_and_rewards():
    dates = pd.date_range("2025-01-01", periods=2, freq="D")
    frame = pd.DataFrame(
        {
            "date": [dates[0], dates[0], dates[1], dates[1]],
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "close": [10.0, 20.0, 11.0, 21.0],
        }
    )
    env = MultiStockTradingEnv(
        frame,
        EnvironmentConfig(
            initial_cash=1_000.0,
            max_steps=1,
            action_mode="weights",
            tau=0.2,
            lambda_turnover=0.0,
            friction=FrictionConfig(commission_rate=0.0, min_commission=0.0, slippage_bps=0.0),
        ),
    )
    try:
        env.reset()
        action = np.array([0.7, 0.3], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert terminated
        assert not truncated
        prev_weights = np.full(len(env.symbols), 1.0 / len(env.symbols))
        clamped = apply_tau_limit(prev_weights, action.astype(np.float64), env.config.tau)
        price_map = {sym: 10.0 if sym == "AAA" else 20.0 for sym in env.symbols}
        holdings_map, _, _ = integerize_allocations(
            {sym: weight for sym, weight in zip(env.symbols, clamped)},
            price_map,
            nav=1_000.0,
            symbols=env.symbols,
            lot_size=1,
        )
        expected_holdings = [int(holdings_map[sym]) for sym in env.symbols]
        assert env.holdings.tolist() == expected_holdings
        gross_value = info["gross_value"]
        expected_nav = info["cash"] + sum(h * p for h, p in zip(env.holdings, [11.0, 21.0]))
        assert info["portfolio_value"] == pytest.approx(expected_nav)
        assert gross_value == pytest.approx(expected_nav)
        expected_log_return = np.log(expected_nav / 1_000.0)
        assert info["log_return"] == pytest.approx(expected_log_return)
        assert reward == pytest.approx(expected_log_return)
        weights = info["weights"]
        expected_weights = [(h * p) / expected_nav for h, p in zip(env.holdings, [11.0, 21.0])]
        assert weights[0] == pytest.approx(expected_weights[0])
        assert weights[1] == pytest.approx(expected_weights[1])
    finally:
        env.close()


def test_evaluate_transaction_costs_symmetry():
    friction = FrictionConfig(commission_rate=0.0025, slippage_bps=5.0)
    buy_notionals = [1_000.0, 2_000.0]
    buy_prices = [10.0, 20.0]
    sell_notionals = [1_500.0]
    sell_prices = [15.0]

    commission, slippage, total = evaluate_transaction_costs(
        buy_notionals=buy_notionals,
        buy_prices=buy_prices,
        sell_notionals=sell_notionals,
        sell_prices=sell_prices,
        friction=friction,
    )

    total_notional = sum(buy_notionals) + sum(sell_notionals)
    expected_commission = total_notional * friction.commission_rate
    expected_slippage = total_notional * (friction.slippage_bps / 10_000.0)

    assert commission == pytest.approx(expected_commission)
    assert slippage == pytest.approx(expected_slippage)
    assert total == pytest.approx(total_notional)


def test_price_map_exposes_symbol_prices():
    env = _two_symbol_env()
    try:
        price_lookup = env.price_map(0)
        assert set(price_lookup.keys()) == {"AAA", "BBB"}
        assert price_lookup["AAA"] == pytest.approx(10.0)
        assert price_lookup["BBB"] == pytest.approx(20.0)
    finally:
        env.close()
