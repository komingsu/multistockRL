"""Universe selection and portfolio weighting based on engineered features.

This is a refactor of sample_code2/run_score_quant.py into importable functions
that return dataframes and can be used by a CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


@dataclass(slots=True)
class SelectionConfig:
    top_n: int = 50
    sector_top_k: int = 5
    sector_cap: Optional[int] = None  # e.g., 10 to cap per-sector in final list
    liquidity_cutoff_pct: float = 0.20
    exclude_konex: bool = True
    use_index_if_available: bool = True
    weighting: str = "inv_vol_liq"  # equal | inv_vol | inv_vol_liq
    max_weight_cap: float = 0.05


FACTOR_COLS = {
    "momentum": "ret_60d",
    "volatility": "volatility_20d",
    "liquidity": "value_traded",
    "size": "log_mcap",
    "value_per": "per",
    "value_pbr": "pbr",
}


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _zscore(s: pd.Series) -> pd.Series:
    s = _safe_numeric(s)
    mu = s.mean()
    sigma = s.std(ddof=0)
    if not np.isfinite(sigma) or sigma == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sigma


def _build_sector_key(cs: pd.DataFrame) -> pd.Series:
    # sector → industry → market → symbol
    s = cs.get("sector")
    if s is None or s.isna().all():
        s = cs.get("industry")
    if s is None or s.isna().all():
        s = cs.get("market")
    if s is None:
        return cs["symbol"].astype(str)
    s = s.copy()
    mask_na = s.isna() | (s.astype(str).str.strip() == "")
    s.loc[mask_na] = cs.loc[mask_na, "symbol"].astype(str)
    return s.astype(str)


def _apply_universe(cs: pd.DataFrame, cfg: SelectionConfig) -> pd.DataFrame:
    base = cs.copy()

    if cfg.use_index_if_available and ("is_kospi200" in base.columns or "is_kosdaq150" in base.columns):
        mask = False
        if "is_kospi200" in base.columns:
            mask = mask | base["is_kospi200"].fillna(False)
        if "is_kosdaq150" in base.columns:
            mask = mask | base["is_kosdaq150"].fillna(False)
        chosen = base[mask].copy()
        if len(chosen) >= cfg.top_n:
            base = chosen

    if "market_cap" in base.columns and len(base) > cfg.top_n:
        q = base["market_cap"].quantile(0.50)
        base = base[base["market_cap"] >= q].copy()

    if "value_traded" in base.columns and len(base) > cfg.top_n:
        ql = base["value_traded"].quantile(cfg.liquidity_cutoff_pct)
        base = base[base["value_traded"] >= ql].copy()

    if cfg.exclude_konex and "market" in base.columns:
        base = base[base["market"] != "KONEX"].copy()

    return base


def _compute_score(cs: pd.DataFrame) -> pd.DataFrame:
    need = []
    for k in ["momentum", "volatility", "liquidity", "size"]:
        if FACTOR_COLS[k] in cs.columns:
            need.append(FACTOR_COLS[k])
    if len(need) < 3:
        raise ValueError("Insufficient factor columns present for scoring")

    sector_key = _build_sector_key(cs)
    z: dict[str, pd.Series] = {}
    if FACTOR_COLS["momentum"] in cs.columns:
        z["mom"] = cs.groupby(sector_key)[FACTOR_COLS["momentum"]].transform(_zscore)
    if FACTOR_COLS["liquidity"] in cs.columns:
        z["liq"] = cs.groupby(sector_key)[FACTOR_COLS["liquidity"]].transform(_zscore)
    if FACTOR_COLS["size"] in cs.columns:
        z["size"] = cs.groupby(sector_key)[FACTOR_COLS["size"]].transform(_zscore)
    if FACTOR_COLS["volatility"] in cs.columns:
        z["lvol"] = -cs.groupby(sector_key)[FACTOR_COLS["volatility"]].transform(_zscore)
    if FACTOR_COLS.get("value_per") in cs.columns:
        z["val_per"] = -cs.groupby(sector_key)[FACTOR_COLS["value_per"]].transform(_zscore)
    if FACTOR_COLS.get("value_pbr") in cs.columns:
        z["val_pbr"] = -cs.groupby(sector_key)[FACTOR_COLS["value_pbr"]].transform(_zscore)

    zdf = pd.DataFrame(z, index=cs.index).fillna(0.0)
    weights = {"mom": 0.30, "lvol": 0.25, "liq": 0.20, "size": 0.15, "val_per": 0.05, "val_pbr": 0.05}
    use_keys = [k for k in weights.keys() if k in zdf.columns]
    w_sum = sum(weights[k] for k in use_keys)
    for k in use_keys:
        zdf[k] = zdf[k] * (weights[k] / w_sum)

    out = cs.copy()
    out["sector_key"] = sector_key.values
    out["score"] = zdf[use_keys].sum(axis=1)
    return out


def _sector_top_k(scored: pd.DataFrame, k: int) -> pd.DataFrame:
    df = scored.copy()
    if "sector_key" not in df.columns:
        df["sector_key"] = _build_sector_key(df)
    return (
        df.sort_values(["sector_key", "score"], ascending=[True, False])
        .groupby("sector_key", group_keys=False)
        .head(k)
        .copy()
    )


def _assign_weights(df_top: pd.DataFrame, method: str, max_cap: float) -> pd.DataFrame:
    out = df_top.copy()
    if method == "equal":
        out["weight"] = 1.0 / len(out)
        return out

    vol_col = "volatility_60d" if "volatility_60d" in out.columns else "volatility_20d"
    vol = _safe_numeric(out.get(vol_col, np.nan)).replace(0, np.nan)

    if method in ("inv_vol", "inv_vol_liq"):
        inv = 1.0 / vol
        inv = inv.fillna(inv.median())
        raw = inv
        if method == "inv_vol_liq":
            liq = _safe_numeric(out.get("value_traded", np.nan))
            # Normalise liquidity 0..1 for stability
            std = liq.std(ddof=0)
            if np.isfinite(std) and std > 0:
                liq_z = (liq - liq.mean()) / std
                liq_scale = (liq_z - np.nanmin(liq_z))
                if np.nanmax(liq_scale) > 0:
                    liq_scale = liq_scale / np.nanmax(liq_scale)
            else:
                liq_scale = pd.Series(0.0, index=liq.index)
            raw = inv * (1 + liq_scale)

        w = raw / raw.sum()
        if max_cap:
            w = w.clip(upper=max_cap)  # cap then renormalise
            w = w / w.sum()
        out["weight"] = w.values
        return out

    out["weight"] = 1.0 / len(out)
    return out


def latest_cross_section(features: pd.DataFrame) -> pd.DataFrame:
    dmax = pd.to_datetime(features["date"]).max()
    return features[features["date"] == dmax].copy()


def select_portfolio(features: pd.DataFrame, cfg: SelectionConfig | None = None) -> pd.DataFrame:
    """Return a top‑N selection dataframe with weights and scores."""

    cfg = cfg or SelectionConfig()
    cs = latest_cross_section(features)
    base = _apply_universe(cs, cfg)
    scored = _compute_score(base)
    picks = _sector_top_k(scored, cfg.sector_top_k)

    ranked = picks.sort_values("score", ascending=False).copy()
    if len(ranked) < cfg.top_n:
        need = cfg.top_n - len(ranked)
        remain = scored.loc[~scored["symbol"].isin(ranked["symbol"])].sort_values("score", ascending=False)
        ranked = pd.concat([ranked, remain.head(need)], ignore_index=True)

    # Optional final sector cap
    if cfg.sector_cap and cfg.sector_cap > 0:
        if "sector_key" not in ranked.columns:
            ranked["sector_key"] = _build_sector_key(ranked)
        final_idx: list[int] = []
        counts: dict[str, int] = {}
        for idx, row in ranked.iterrows():
            sec = str(row["sector_key"]) if "sector_key" in row else "UNKNOWN"
            counts.setdefault(sec, 0)
            if counts[sec] < cfg.sector_cap:
                final_idx.append(idx)
                counts[sec] += 1
            if len(final_idx) >= cfg.top_n:
                break
        ranked = ranked.loc[final_idx].copy()

    port = ranked.sort_values("score", ascending=False).head(cfg.top_n).copy()
    port["rank"] = np.arange(1, len(port) + 1)
    port = _assign_weights(port, cfg.weighting, cfg.max_weight_cap)
    return port


def save_selection_csv(df: pd.DataFrame, outdir: str | Path, label: str = "top50") -> tuple[Path, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # Use latest date in df as label prefix
    dmax = pd.to_datetime(df["date"]).max()
    stamp = dmax.strftime("%Y%m%d") if hasattr(dmax, "strftime") else str(dmax)
    p_path = outdir / f"{stamp}_{label}.parquet"
    c_path = outdir / f"{stamp}_{label}.csv"
    df.to_parquet(p_path, index=False)
    df.to_csv(c_path, index=False, encoding="utf-8-sig")
    return p_path, c_path


__all__ = [
    "SelectionConfig",
    "select_portfolio",
    "save_selection_csv",
    "latest_cross_section",
]

