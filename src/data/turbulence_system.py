"""Build a system-wide daily close frame from KIS master + daily API.

Computes the system returns frame (e.g., KOSPI constituents) for turbulence
calculation. This module relies on KIS credentials available in the
environment and the helper in src/data/ingest.py for per-symbol daily fetches.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import time
import pandas as pd

from .ingest import _resolve_kis_auth_from_env, get_daily_candle_kis
import FinanceDataReader as fdr  # type: ignore


@dataclass(slots=True)
class SystemUniverseConfig:
    market_filter: str = "KOSPI"  # substring match on market column when present
    min_coverage: float = 0.90
    request_pause: float = 0.08  # seconds between API calls


def load_symbols_from_master(master_csv: str | Path, *, market_filter: str = "KOSPI") -> List[str]:
    df = pd.read_csv(master_csv, dtype=str, low_memory=False)
    cols = set(df.columns)
    if {"시장구분", "단축코드"}.issubset(cols):
        base = df[df["시장구분"].str.contains(market_filter, case=False, na=False)]
        syms = base["단축코드"].astype(str).str.zfill(6).unique().tolist()
    elif {"mkt_nm", "shrn_iscd"}.issubset(cols):
        base = df[df["mkt_nm"].str.contains(market_filter, case=False, na=False)]
        syms = base["shrn_iscd"].astype(str).str.zfill(6).unique().tolist()
    elif "symbol" in cols:
        # Fallback schema: assume already filtered
        syms = df["symbol"].astype(str).str.zfill(6).unique().tolist()
    else:
        raise RuntimeError("Unsupported master schema; expected columns like 시장구분/단축코드 or mkt_nm/shrn_iscd")
    if not syms:
        raise RuntimeError("No symbols found for the requested market filter")
    return syms


def build_system_frame_from_kis_master(
    master_csv: str | Path,
    *,
    start: str,
    end: str,
    env: str = "real",
    config: SystemUniverseConfig | None = None,
    max_symbols: Optional[int] = None,
) -> pd.DataFrame:
    cfg = config or SystemUniverseConfig()
    symbols = load_symbols_from_master(master_csv, market_filter=cfg.market_filter)
    if max_symbols is not None:
        symbols = symbols[: int(max_symbols)]
    auth = _resolve_kis_auth_from_env(env)
    rows: list[pd.DataFrame] = []
    for i, sym in enumerate(symbols, 1):
        df = get_daily_candle_kis(sym, start, end, auth)
        if not df.empty:
            rows.append(df[["symbol", "date", "close"]].copy())
        if cfg.request_pause:
            time.sleep(cfg.request_pause)
    if not rows:
        raise RuntimeError("No daily data fetched from KIS; check credentials/date range")
    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    return out


__all__ = ["SystemUniverseConfig", "load_symbols_from_master", "build_system_frame_from_kis_master"]
def build_system_frame_from_fdr(
    symbols: Iterable[str],
    *,
    start: str,
    end: str,
    max_symbols: Optional[int] = None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for i, sym in enumerate(symbols, 1):
        if max_symbols is not None and i > int(max_symbols):
            break
        try:
            df = fdr.DataReader(str(sym), start, end)
            if not df.empty:
                df = df.reset_index().rename(columns={"Close": "close", df.index.name or "Date": "date"})
                part = df[["date", "close"]].copy()
                part["symbol"] = str(sym).zfill(6)
                part["date"] = pd.to_datetime(part["date"]).dt.strftime("%Y-%m-%d")
                rows.append(part)
        except Exception:
            continue
    if not rows:
        raise RuntimeError("FDR fallback did not return any data")
    out = pd.concat(rows, ignore_index=True)
    return out[["symbol", "date", "close"]]

__all__ += ["build_system_frame_from_fdr"]
