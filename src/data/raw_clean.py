"""Reproduce the raw cleaning step from the notebook '일봉 원천지표 받아오기'.

Rules (from the notebook):
- Identify investor ratio columns (prsn_*, frgn_*, orgn_* with buy/sell/net for val/vol).
- Drop symbols that have no investor ratio data at all (all ratios NaN for that symbol).
- Keep only dates where all remaining symbols have complete investor ratios (no NaNs) across the ratio set.
- Preserve all other columns but filter rows to the kept symbols/dates.

This function assumes a frame that already contains the investor ratio columns
and basic OHLCV + metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


RATIO_COLS: list[str] = [
    "prsn_buy_val_ratio",
    "prsn_sell_val_ratio",
    "prsn_net_val_ratio",
    "prsn_buy_vol_ratio",
    "prsn_sell_vol_ratio",
    "prsn_net_vol_ratio",
    "frgn_buy_val_ratio",
    "frgn_sell_val_ratio",
    "frgn_net_val_ratio",
    "frgn_buy_vol_ratio",
    "frgn_sell_vol_ratio",
    "frgn_net_vol_ratio",
    "orgn_buy_val_ratio",
    "orgn_sell_val_ratio",
    "orgn_net_val_ratio",
    "orgn_buy_vol_ratio",
    "orgn_sell_vol_ratio",
    "orgn_net_vol_ratio",
]


def build_daily_raw_clean(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    # Ensure required keys
    if "symbol" not in df.columns or "date" not in df.columns:
        raise KeyError("Input frame must contain 'symbol' and 'date' columns")
    # Restrict to rows where at least one ratio exists per symbol
    has_any = (
        df.groupby("symbol")[RATIO_COLS]
        .apply(lambda x: x.notna().any().any())
        .rename("has_any")
    )
    keep_symbols = has_any[has_any].index.tolist()
    clean = df[df["symbol"].isin(keep_symbols)].copy()

    # Keep only dates where all remaining symbols have complete ratios (no NaNs)
    complete_row = clean[RATIO_COLS].notna().all(axis=1)
    clean["__complete"] = complete_row
    n_syms = clean["symbol"].nunique()
    date_ok = clean.groupby("date")["__complete"].sum().rename("ok_cnt")
    keep_dates = date_ok[date_ok == n_syms].index
    out = (
        clean[clean["date"].isin(keep_dates)]
        .drop(columns=["__complete"])
        .sort_values(["date", "symbol"])  # notebook sorts date->symbol
        .reset_index(drop=True)
    )
    return out


def write_daily_raw_clean(input_csv: str | Path, dest_parquet: str | Path, dest_csv: str | Path) -> tuple[Path, Path]:
    src = Path(input_csv)
    df = pd.read_csv(src, dtype={"symbol": str}, parse_dates=["date"], low_memory=False)
    out = build_daily_raw_clean(df)
    p = Path(dest_parquet)
    c = Path(dest_csv)
    p.parent.mkdir(parents=True, exist_ok=True)
    c.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(p, index=False)
    out.to_csv(c, index=False, encoding="utf-8-sig")
    return p, c


__all__ = ["build_daily_raw_clean", "write_daily_raw_clean"]

