from __future__ import annotations

import pandas as pd

from src.data.feature_engineering import build_feature_frame
from src.data.schema import get_schema, validate_dataframe


def test_build_feature_frame_emits_required_columns():
    # Minimal two symbols, 5 days
    rows = []
    for sym in ["AAA", "BBB"]:
        for i, px in enumerate([10, 11, 12, 11, 13]):
            rows.append({
                "symbol": sym,
                "date": f"2024-01-0{i+1}",
                "open": px,
                "high": px + 1,
                "low": px - 1,
                "close": px,
                "volume": 1000 + i,
                "value": (1000 + i) * px,
            })
    daily = pd.DataFrame(rows)
    sm = pd.DataFrame({
        "symbol": ["AAA", "BBB"],
        "name": ["A", "B"],
        "market": ["KOSPI", "KOSDAQ"],
        "industry": ["X", "Y"],
        "market_cap": [1_000_000, 2_000_000],
    })

    df = build_feature_frame(daily, sm)
    # Verify no schema violations (allowing NaNs for flow columns)
    errors = validate_dataframe(df, strict=False)
    assert errors == [], f"Schema violations: {errors}"
    # Primary key uniqueness
    pkey_cols = ("symbol", "date")
    assert int(df.duplicated(subset=list(pkey_cols)).sum()) == 0

