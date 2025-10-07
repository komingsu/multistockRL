"""Feature engineering to produce schema-compatible processed CSV for training.

This module transforms raw daily OHLCV and symbol master into
`data/proc/daily_with_indicators.csv` with exactly the columns expected by
`src/data/schema.py`.

Notes
- Flow ratio columns (prsn_*, frgn_*, orgn_*) are not available from the basic
  daily OHLCV source; we include them as present with NaN values to satisfy the
  schema and keep compatibility. When such data becomes available, this module
  can be extended to populate them.
- All prices are assumed adjusted (adj_flag=True) when KIS API is used with
  `fid_org_adj_prc=1`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


SCHEMA_FLOW_COLUMNS: Tuple[str, ...] = (
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
)


def _ensure_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume", "value"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    # Treat symbol as text; preserve leading zeros for numeric codes (KRX uses 6-digit codes)
    sym = out["symbol"].astype(str).str.strip()
    mask_num = sym.str.fullmatch(r"\d+")
    sym_padded = sym.where(~mask_num, sym.str.zfill(6))
    out["symbol"] = sym_padded
    return out


def _rolling_mean(series: pd.Series, window: int) -> pd.Series:
    # Match notebook default: full window required (min_periods=window)
    return series.rolling(window).mean()


def _true_range(high: pd.Series, low: pd.Series, prev_close: pd.Series) -> pd.Series:
    a = high - low
    b = (high - prev_close).abs()
    c = (low - prev_close).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)


def build_feature_frame(
    daily: pd.DataFrame,
    symbol_master: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return schema-compatible dataframe with indicators and metadata.

    Parameters
    ----------
    daily: raw daily OHLCV with columns [symbol, date, open, high, low, close, volume, value]
    symbol_master: optional metadata with columns [symbol, name, market, industry, market_cap, ...]
    """

    base = _ensure_types(daily).sort_values(["symbol", "date"]).copy()
    # Previous close (keep if supplied)
    if "prdy_close" not in base.columns:
        base["prdy_close"] = base.groupby("symbol")["close"].shift(1)
    # Change metrics (keep if already provided to match legacy outputs)
    if "change" not in base.columns:
        base["change"] = base["close"] - base["prdy_close"]
    if "change_rate" not in base.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            base["change_rate"] = base["close"] / base["prdy_close"] - 1.0
    # Range and gap
    with np.errstate(divide="ignore", invalid="ignore"):
        base["range_pct"] = (base["high"] - base["low"]) / base["close"]
    with np.errstate(divide="ignore", invalid="ignore"):
        base["gap_pct"] = base["open"] / base["prdy_close"] - 1.0

    # Moving averages
    for w in (5, 10, 20):
        base[f"ma_{w}"] = base.groupby("symbol")["close"].transform(lambda s: _rolling_mean(s, w))
    # Distance to MAs
    base["dist_ma5"] = base["close"] / base["ma_5"] - 1.0
    base["dist_ma10"] = base["close"] / base["ma_10"] - 1.0
    base["dist_ma20"] = base["close"] / base["ma_20"] - 1.0
    # Slopes (1-day differences of MAs)
    base["ma10_slope"] = base.groupby("symbol")["ma_10"].diff(1)
    base["ma20_slope"] = base.groupby("symbol")["ma_20"].diff(1)

    # ATR
    prev_close = base["prdy_close"] if "prdy_close" in base.columns else base.groupby("symbol")["close"].shift(1)
    tr = _true_range(base["high"], base["low"], prev_close)
    # Compute ATR per symbol
    base["__tr"] = tr
    # Match notebook: simple rolling mean with default min_periods (== window)
    base["atr_5"] = base.groupby("symbol")["__tr"].transform(lambda s: s.rolling(5).mean())
    base["atr_14"] = base.groupby("symbol")["__tr"].transform(lambda s: s.rolling(14).mean())
    base = base.drop(columns=["__tr"])  # cleanup

    # Realized volatility of log returns
    with np.errstate(divide="ignore", invalid="ignore"):
        base["__log_ret"] = np.log(base["close"] / base["prdy_close"]).replace([np.inf, -np.inf], np.nan)
    # Match notebook: rolling std with default min_periods (== window)
    base["rv_10"] = base.groupby("symbol")["__log_ret"].transform(lambda s: s.rolling(10).std())
    base["rv_20"] = base.groupby("symbol")["__log_ret"].transform(lambda s: s.rolling(20).std())
    base = base.drop(columns=["__log_ret"])  # cleanup

    # Flags
    base["adj_flag"] = True

    # Merge meta
    if symbol_master is not None and not symbol_master.empty:
        sm = symbol_master.copy()
        keep_cols = [
            c
            for c in [
                "symbol",
                "name",
                "market",
                "industry",
                "market_cap",
            ]
            if c in sm.columns
        ]
        sm = sm[keep_cols].drop_duplicates("symbol")
        base = base.merge(sm, on="symbol", how="left")

    # Ensure required columns exist (flow ratios as NaN placeholders)
    for col in SCHEMA_FLOW_COLUMNS:
        if col not in base.columns:
            base[col] = np.nan

    # Order and select schema columns
    ordered_cols: List[str] = [
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "value",
        "adj_flag",
        "name",
        "market",
        "industry",
        "market_cap",
        "prdy_close",
        "change",
        "change_rate",
        *SCHEMA_FLOW_COLUMNS,
        "ma_5",
        "ma_10",
        "ma_20",
        "dist_ma5",
        "dist_ma10",
        "dist_ma20",
        "ma10_slope",
        "ma20_slope",
        "atr_5",
        "atr_14",
        "rv_10",
        "rv_20",
        "range_pct",
        "gap_pct",
    ]
    # Keep only available columns, preserve order
    cols = [c for c in ordered_cols if c in base.columns]
    out = base[cols].copy()

    # Primary key uniqueness (symbol, date)
    out = out.dropna(subset=["symbol", "date"]).drop_duplicates(["symbol", "date"]).sort_values(["symbol", "date"])  # type: ignore

    return out


def write_processed_csv(daily_parquet: str | Path, symbol_master_parquet: str | Path | None, dest_csv: str | Path) -> Path:
    """Load raw parquet(s), build features, and write schema-compatible CSV."""

    daily = pd.read_parquet(daily_parquet)
    sm = pd.read_parquet(symbol_master_parquet) if symbol_master_parquet else None
    df = build_feature_frame(daily, sm)
    dest = Path(dest_csv)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return dest


__all__ = ["build_feature_frame", "write_processed_csv"]
