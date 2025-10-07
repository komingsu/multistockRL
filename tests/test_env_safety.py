from __future__ import annotations

import numpy as np
import pandas as pd

from src.env.multi_stock_env import EnvironmentConfig, MultiStockTradingEnv, SafetyConfig


def _make_frame():
    # Two symbols, 5 days, with turbulence columns
    rows = []
    dates = pd.date_range('2024-01-01', periods=5, freq='B')
    for d in dates:
        rows.append({"date": d.strftime('%Y-%m-%d'), "symbol": "AAA", "close": 100.0})
        rows.append({"date": d.strftime('%Y-%m-%d'), "symbol": "BBB", "close": 100.0})
    df = pd.DataFrame(rows)
    # Inject turbulence: crisis on last two days
    q = pd.DataFrame({
        'date': dates,
        'turb_port': [0.0, 0.2, 0.5, 0.96, 0.995],
        'turb_sys': [0.0, 0.1, 0.4, 0.97, 0.996],
    })
    q['date'] = pd.to_datetime(q['date'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.merge(q, on='date', how='left')
    return df


def test_safety_gmax_and_buy_freeze():
    df = _make_frame()
    cfg = EnvironmentConfig(
        initial_cash=1000.0,
        action_mode='weights',
        tau=0.5,
        safety=SafetyConfig(enabled=True, gmax_crisis=0.1, per_asset_cap=0.2, buy_freeze_crisis=True),
    )
    env = MultiStockTradingEnv(df, cfg)
    obs, info = env.reset()
    # Step to crisis day (index 3)
    for _ in range(3):
        # Target all-in on AAA (assets-only logits; cash inferred)
        action = np.array([10.0, 0.0], dtype=np.float32)
        obs, reward, term, trunc, info = env.step(action)
    # Now in caution/crisis, check g_max effect on weights sum
    assert info.get('g_max') <= 1.0
    # Next step should be crisis: buy-freeze prevents increases
    prev_hold = env.holdings.copy()
    action = np.array([10.0, 0.0], dtype=np.float32)
    obs, reward, term, trunc, info = env.step(action)
    # holdings should not increase due to buy-freeze in crisis
    assert np.all(env.holdings <= prev_hold)
    assert info.get('regime') in ('caution', 'crisis')
