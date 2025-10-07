"""Investor flow collection and ratio computation (KIS).

This module ports the notebook logic (일봉 원천지표 받아오기) for
daily per-stock investor aggregates and derives group-internal ratios.

Endpoints
- TR: FHPTJ04160001 (daily investor trades by stock)
- URL: /uapi/domestic-stock/v1/quotations/inquire-investor

Notes
- Requires KIS credentials in environment (see src/data/ingest.py helper).
- Pagination is controlled by response headers 'tr_cont'/'Tr_Cont'. When value
  is 'M', pass 'tr_cont=N' on the next request to continue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import time
import pandas as pd
import numpy as np
import requests

from .ingest import _resolve_kis_auth_from_env, _kis_base_url


TR_STOCK_INV_DAILY = "FHPTJ04160001"


def _headers(tr_id: str, *, appkey: str, appsecret: str, access_token: str) -> dict:
    return {
        "content-type": "application/json; charset=utf-8",
        "authorization": f"Bearer {access_token}",
        "appkey": appkey,
        "appsecret": appsecret,
        "tr_id": tr_id,
        "custtype": "P",
    }


def _kis_get_with_headers(url: str, tr_id: str, params: dict, *, appkey: str, appsecret: str, access_token: str, retry: int = 3, sleep: float = 0.35) -> Tuple[dict, dict]:
    last_err: Optional[Exception] = None
    for _ in range(retry):
        try:
            r = requests.get(url, headers=_headers(tr_id, appkey=appkey, appsecret=appsecret, access_token=access_token), params=params, timeout=15)
            if r.ok:
                return r.json(), r.headers
            last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:  # pragma: no cover - network dependent
            last_err = e
        time.sleep(sleep)
    if last_err:
        raise last_err
    raise RuntimeError("Unknown KIS request error")


INV_DAILY_KEEP = {
    "stck_bsop_date": "date",
    "acml_vol": "total_volume",
    "acml_tr_pbmn": "total_value_mn",
    # 개인
    "prsn_shnu_vol": "prsn_buy_vol",
    "prsn_seln_vol": "prsn_sell_vol",
    "prsn_shnu_tr_pbmn": "prsn_buy_val_mn",
    "prsn_seln_tr_pbmn": "prsn_sell_val_mn",
    # 외국인
    "frgn_shnu_vol": "frgn_buy_vol",
    "frgn_seln_vol": "frgn_sell_vol",
    "frgn_shnu_tr_pbmn": "frgn_buy_val_mn",
    "frgn_seln_tr_pbmn": "frgn_sell_val_mn",
    # 기관
    "orgn_shnu_vol": "orgn_buy_vol",
    "orgn_seln_vol": "orgn_sell_vol",
    "orgn_shnu_tr_pbmn": "orgn_buy_val_mn",
    "orgn_seln_tr_pbmn": "orgn_sell_val_mn",
}

INV_DAILY_NUM = [
    "total_volume",
    "total_value_mn",
    "prsn_buy_vol",
    "prsn_sell_vol",
    "prsn_buy_val_mn",
    "prsn_sell_val_mn",
    "frgn_buy_vol",
    "frgn_sell_vol",
    "frgn_buy_val_mn",
    "frgn_sell_val_mn",
    "orgn_buy_vol",
    "orgn_sell_vol",
    "orgn_buy_val_mn",
    "orgn_sell_val_mn",
]


def _safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    try:
        return float(a) / float(b) if (a is not None and b not in (None, 0)) else None
    except Exception:
        return None


def compute_group_ratios_strict(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in INV_DAILY_NUM:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out["buy_val_sum"] = out[["prsn_buy_val_mn", "frgn_buy_val_mn", "orgn_buy_val_mn"]].sum(axis=1, min_count=1)
    out["sell_val_sum"] = out[["prsn_sell_val_mn", "frgn_sell_val_mn", "orgn_sell_val_mn"]].sum(axis=1, min_count=1)
    out["val_den"] = out["buy_val_sum"] + out["sell_val_sum"]

    out["buy_vol_sum"] = out[["prsn_buy_vol", "frgn_buy_vol", "orgn_buy_vol"]].sum(axis=1, min_count=1)
    out["sell_vol_sum"] = out[["prsn_sell_vol", "frgn_sell_vol", "orgn_sell_vol"]].sum(axis=1, min_count=1)
    out["vol_den"] = out["buy_vol_sum"] + out["sell_vol_sum"]

    keep = ["symbol", "date"]
    for grp in ["prsn", "frgn", "orgn"]:
        out[f"{grp}_buy_val_ratio"] = out.apply(lambda r: _safe_div(r[f"{grp}_buy_val_mn"], r["buy_val_sum"]), axis=1)
        out[f"{grp}_sell_val_ratio"] = out.apply(lambda r: _safe_div(r[f"{grp}_sell_val_mn"], r["sell_val_sum"]), axis=1)
        out[f"{grp}_net_val_ratio"] = out.apply(lambda r: _safe_div((r[f"{grp}_buy_val_mn"] - r[f"{grp}_sell_val_mn"]), r["val_den"]), axis=1)
        out[f"{grp}_buy_vol_ratio"] = out.apply(lambda r: _safe_div(r[f"{grp}_buy_vol"], r["buy_vol_sum"]), axis=1)
        out[f"{grp}_sell_vol_ratio"] = out.apply(lambda r: _safe_div(r[f"{grp}_sell_vol"], r["sell_vol_sum"]), axis=1)
        out[f"{grp}_net_vol_ratio"] = out.apply(lambda r: _safe_div((r[f"{grp}_buy_vol"] - r[f"{grp}_sell_vol"]), r["vol_den"]), axis=1)
        keep += [
            f"{grp}_buy_val_ratio",
            f"{grp}_sell_val_ratio",
            f"{grp}_net_val_ratio",
            f"{grp}_buy_vol_ratio",
            f"{grp}_sell_vol_ratio",
            f"{grp}_net_vol_ratio",
        ]

    out = out[keep].copy()
    return out


def fetch_investor_trade_by_stock_daily(symbol: str, start: str, end: str, *, env: str = "real", pause: float = 0.35, max_loops: int = 100) -> pd.DataFrame:
    auth = _resolve_kis_auth_from_env(env)
    url = f"{_kis_base_url(auth.env)}/uapi/domestic-stock/v1/quotations/inquire-investor"

    def _one_anchor(anchor: str):
        params = {
            "fid_cond_mrkt_div_code": "J",
            "fid_input_iscd": str(symbol).zfill(6),
            "fid_input_date_1": start,
            "fid_input_date_2": anchor,
            "fid_etc_cls_code": "",
        }
        batch_rows = []
        while True:
            js, hdr = _kis_get_with_headers(url, TR_STOCK_INV_DAILY, params, appkey=auth.appkey, appsecret=auth.appsecret, access_token=auth.access_token)
            rows = js.get("output2") or []
            batch_rows.extend(rows)
            trc = (hdr.get("tr_cont") or hdr.get("Tr_Cont") or "").strip()
            if trc == "M":
                params["tr_cont"] = "N"
            else:
                break
            if len(batch_rows) > 20000:
                break
            time.sleep(pause)
        return batch_rows

    all_rows = []
    anchor = end
    loops = 0
    start_d = pd.to_datetime(start, format="%Y%m%d").date()
    while loops < max_loops:
        loops += 1
        rows = _one_anchor(anchor)
        if not rows:
            break
        all_rows.extend(rows)
        tmp = pd.DataFrame(rows)
        if "stck_bsop_date" not in tmp.columns or tmp.empty:
            break
        tmp["date"] = pd.to_datetime(tmp["stck_bsop_date"], format="%Y%m%d").dt.date
        min_d = tmp["date"].min()
        if min_d <= start_d:
            break
        anchor = (pd.Timestamp(min_d) - pd.Timedelta(days=1)).strftime("%Y%m%d")
        time.sleep(pause)

    if not all_rows:
        return pd.DataFrame(columns=["symbol", "date"])

    df = pd.DataFrame(all_rows).rename(columns=INV_DAILY_KEEP)[list(INV_DAILY_KEEP.values())]
    df.insert(0, "symbol", str(symbol).zfill(6))
    for c in INV_DAILY_NUM:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"]).reset_index(drop=True)
    return df


def collect_investor_ratios_for_universe(symbols: Iterable[str], start: str, end: str, *, env: str = "real") -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for s in symbols:
        raw = fetch_investor_trade_by_stock_daily(s, start=start, end=end, env=env)
        if raw.empty:
            continue
        ratios = compute_group_ratios_strict(raw)
        frames.append(ratios)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["symbol", "date"])  # type: ignore


__all__ = [
    "compute_group_ratios_strict",
    "fetch_investor_trade_by_stock_daily",
    "collect_investor_ratios_for_universe",
]

