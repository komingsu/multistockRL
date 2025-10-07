"""Data ingestion utilities to build the processed dataset used by training.

This module adapts legacy collection notebooks/scripts (see sample_code2/) into
lightweight, scriptable functions that fit the project's architecture:

- Fetch symbol master via FinanceDataReader (no credentials required).
- Optionally fetch daily OHLCV via KIS API (requires env credentials).
- Load cached raw files from data/raw if present instead of refetching.

The end goal is to emit a single, schema‑compatible CSV at
`data/proc/daily_with_indicators.csv` after feature engineering is applied in
`src/data/feature_engineering.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional

import os
import time

import pandas as pd


# ------------------------------
# Symbol master via FDR
# ------------------------------


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")


def get_symbol_master_fdr() -> pd.DataFrame:
    """Return symbol master dataframe using FinanceDataReader KRX listing.

    Columns (when available): symbol, name, market, sector, industry,
    market_cap, shares, per, pbr, eps, bps, is_kospi200, is_kosdaq150
    """

    try:
        import FinanceDataReader as fdr  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dep
        raise RuntimeError(
            "FinanceDataReader is required to build symbol master. Install via `pip install finance-datareader`."
        ) from exc

    krx = fdr.StockListing("KRX")
    rename_map = {
        "Code": "symbol",
        "Name": "name",
        "Market": "market",
        "Sector": "sector",
        "Industry": "industry",
        "Marcap": "market_cap",
        "Stocks": "shares",
        "PER": "per",
        "PBR": "pbr",
        "EPS": "eps",
        "BPS": "bps",
    }
    for k, v in rename_map.items():
        if k in krx.columns:
            krx = krx.rename(columns={k: v})

    cols = [
        c
        for c in [
            "symbol",
            "name",
            "market",
            "sector",
            "industry",
            "market_cap",
            "shares",
            "per",
            "pbr",
            "eps",
            "bps",
        ]
        if c in krx.columns
    ]
    df = krx[cols].dropna(subset=["symbol"]).drop_duplicates("symbol")
    # Preserve leading zeros for numeric codes (KRX uses 6-digit codes)
    sym = df["symbol"].astype(str).str.strip()
    mask_num = sym.str.fullmatch(r"\d+")
    df["symbol"] = sym.where(~mask_num, sym.str.zfill(6))
    for c in ["market_cap", "shares", "per", "pbr", "eps", "bps"]:
        if c in df.columns:
            df[c] = _safe_numeric(df[c])

    # Index membership flags (best effort)
    for idx_name, colflag in [("KOSPI200", "is_kospi200"), ("KOSDAQ150", "is_kosdaq150")]:
        try:
            idx_df = fdr.StockListing(idx_name)
            sym_col = "Code" if "Code" in idx_df.columns else "Symbol"
            syms = set(idx_df[sym_col].astype(str).str.zfill(6))
            df[colflag] = df["symbol"].astype(str).isin(syms)
        except Exception:  # pragma: no cover - not critical for unit tests
            df[colflag] = False

    if "market" in df.columns:
        df["market"] = df["market"].astype(str)

    return df


def save_symbol_master(dest_dir: str | Path = "data/raw/kis/symbol_master") -> Path:
    """Save symbol master parquet into the given directory, dated by YYYYMMDD."""

    outdir = Path(dest_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = get_symbol_master_fdr()
    today = datetime.now().strftime("%Y%m%d")
    path = outdir / f"{today}.parquet"
    df.to_parquet(path, index=False)
    return path


# ------------------------------
# KIS daily OHLCV (optional)
# ------------------------------


@dataclass(slots=True)
class KisAuth:
    appkey: str
    appsecret: str
    access_token: str
    env: str = "real"  # or "mock"


def _resolve_kis_auth_from_env(env: str = "real") -> KisAuth:
    """Load KIS credentials from environment.

    Required env vars:
      KIS_API_KEY, KIS_API_SECRET, KIS_ACCESS_TOKEN
    For mock env, use KIS_API_KEY_MOCK, KIS_API_SECRET_MOCK, KIS_ACCESS_TOKEN_MOCK
    """

    if env not in {"real", "mock"}:
        raise ValueError("env must be 'real' or 'mock'")
    suffix = "" if env == "real" else "_MOCK"
    key = os.getenv(f"KIS_API_KEY{suffix}")
    sec = os.getenv(f"KIS_API_SECRET{suffix}")
    tok = os.getenv(f"KIS_ACCESS_TOKEN{suffix}")
    if not (key and sec and tok):  # attempt client-credentials token fetch
        if not (key and sec):  # pragma: no cover
            raise RuntimeError(
                "Missing KIS credentials in environment. Set KIS_API_KEY(_MOCK), KIS_API_SECRET(_MOCK)."
            )
        try:
            import requests  # lazy import
            base = _kis_base_url(env)
            url = f"{base}/oauth2/tokenP"
            resp = requests.post(url, json={"grant_type": "client_credentials", "appkey": key, "appsecret": sec}, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            tok = data.get("access_token")
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Failed to obtain KIS access token; set KIS_ACCESS_TOKEN(_MOCK) explicitly.") from exc
    return KisAuth(appkey=key, appsecret=sec, access_token=tok, env=env)


def _kis_base_url(env: str) -> str:
    return "https://openapi.koreainvestment.com:9443" if env == "real" else "https://openapivts.koreainvestment.com:29443"


def get_daily_candle_kis(symbol: str, start_date: str, end_date: str, auth: KisAuth) -> pd.DataFrame:
    """Fetch daily OHLCV for a symbol using KIS API.

    Returns empty dataframe on API anomalies for resilience.
    """

    import requests  # lazy import to avoid test dependency

    url = f"{_kis_base_url(auth.env)}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
    tr_id = "FHKST03010100" if auth.env == "real" else "VTKST03010100"
    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {auth.access_token}",
        "appkey": auth.appkey,
        "appsecret": auth.appsecret,
        "tr_id": tr_id,
    }
    params = {
        "fid_cond_mrkt_div_code": "J",
        "fid_input_iscd": symbol,
        "fid_org_adj_prc": "1",
        "fid_period_div_code": "D",
        "fid_input_date_1": start_date,
        "fid_input_date_2": end_date,
    }

    try:
        res = requests.get(url, headers=headers, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
    except Exception:  # pragma: no cover - network dependent
        return pd.DataFrame()

    if not isinstance(data, dict) or "output2" not in data:
        return pd.DataFrame()

    df = pd.DataFrame(data["output2"]).rename(
        columns={
            "stck_bsop_date": "date",
            "stck_oprc": "open",
            "stck_hgpr": "high",
            "stck_lwpr": "low",
            "stck_clpr": "close",
            "acml_vol": "volume",
            "acml_tr_pbmn": "value",
        }
    )
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date")
    # Preserve leading zeros
    s = str(symbol)
    df["symbol"] = s.zfill(6) if s.isdigit() else s
    return df


def collect_daily_ohlcv(
    symbols: Iterable[str],
    *,
    start_date: str,
    end_date: str,
    env: str = "real",
    rate_limit_sleep: float = 0.2,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Collect daily OHLCV for many symbols via KIS.

    Requires KIS credentials in environment. Use moderate rate limiting.
    """

    auth = _resolve_kis_auth_from_env(env)
    rows: List[pd.DataFrame] = []
    tmp = [str(s).strip() for s in symbols]
    symbols = [s.zfill(6) if s.isdigit() else s for s in tmp]
    for i, sym in enumerate(symbols, 1):
        frame = get_daily_candle_kis(str(sym), start_date, end_date, auth)
        if not frame.empty:
            rows.append(frame)
        if show_progress and (i % 50 == 0):  # lightweight progress
            print(f"[KIS] Collected {i}/{len(symbols)} symbols")
        if rate_limit_sleep:
            time.sleep(rate_limit_sleep)
    if not rows:
        return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume", "value"])  # type: ignore
    return pd.concat(rows, ignore_index=True)


def load_or_collect_daily(
    *,
    cache_dir: str | Path = "data/raw/kis/daily",
    lookback_days: int = 365,
    env: str = "real",
    symbols: Optional[Iterable[str]] = None,
) -> Path:
    """Return path to a cached parquet of daily OHLCV, collecting if missing.

    If `symbols` is None, loads symbol master of today and uses all symbols.
    """

    outdir = Path(cache_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")
    outfile = outdir / f"{today}.parquet"
    if outfile.exists():
        return outfile

    # Discover universe
    if symbols is None:
        sym_path = save_symbol_master()  # also ensures existence
        sym_df = pd.read_parquet(sym_path)
        universe = sym_df["symbol"].astype(str).tolist()
    else:
        universe = [str(s) for s in symbols]

    end = datetime.now()
    start = end - timedelta(days=max(1, int(lookback_days)))
    df = collect_daily_ohlcv(
        universe,
        start_date=start.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"),
        env=env,
    )
    df.to_parquet(outfile, index=False)
    return outfile


__all__ = [
    "get_symbol_master_fdr",
    "save_symbol_master",
    "collect_daily_ohlcv",
    "load_or_collect_daily",
]
