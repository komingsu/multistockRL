from __future__ import annotations

import pandas as pd

from src.data.feature_engineering import build_feature_frame


def test_symbol_is_text_and_zero_padded():
    daily = pd.DataFrame([
        {"symbol": 5930, "date": "2024-01-01", "open": 1, "high": 2, "low": 1, "close": 1.5, "volume": 10, "value": 15},
        {"symbol": "000020", "date": "2024-01-01", "open": 2, "high": 3, "low": 2, "close": 2.5, "volume": 20, "value": 50},
        {"symbol": 5930, "date": "2024-01-02", "open": 1.6, "high": 1.8, "low": 1.4, "close": 1.7, "volume": 12, "value": 18},
        {"symbol": "000020", "date": "2024-01-02", "open": 2.4, "high": 2.6, "low": 2.2, "close": 2.45, "volume": 22, "value": 54},
    ])
    sm = pd.DataFrame({
        "symbol": [5930, "000020"],
        "name": ["Samsung Elec", "Dongwha"],
        "market": ["KOSPI", "KOSPI"],
        "industry": ["IT", "Pharma"],
        "market_cap": [1_000_000_000, 10_000_000],
    })

    df = build_feature_frame(daily, sm)
    assert df["symbol"].dtype == object
    assert set(df["symbol"]) == {"005930", "000020"}

