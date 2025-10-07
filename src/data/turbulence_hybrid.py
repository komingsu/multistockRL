"""Hybrid turbulence signal generation (portfolio + system) with regimes.

Implements:
- Returns matrices (all symbols, subset) from processed CSV
- Mahalanobis distance with Ledoit–Wolf shrinkage over rolling windows
- Chi-square CDF scaling and EMA smoothing
- Hysteresis-based regime classification and g_max mapping

Outputs are saved to `data/proc/factors/turbulence_hybrid.csv`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import chi2
from sklearn.covariance import LedoitWolf


@dataclass(slots=True)
class TurbulenceConfig:
    window: int = 180
    window_max: int = 252
    winsor_pct: float = 0.01
    ema_lambda: float = 0.2
    k_pca: int = 10  # not used explicitly; shrinkage handles high-dim
    med_on: float = 0.95
    med_off: float = 0.92
    hi_on: float = 0.99
    hi_off: float = 0.96


def _winsorize(arr: NDArray[np.float64], p: float) -> NDArray[np.float64]:
    lo = np.nanquantile(arr, p)
    hi = np.nanquantile(arr, 1 - p)
    return np.clip(arr, lo, hi)


def _ema(series: pd.Series, lam: float) -> pd.Series:
    alpha = float(lam)
    return series.ewm(alpha=alpha, adjust=False).mean()


def _compute_rolling_mahalanobis(df_ret: pd.DataFrame, window: int, winsor_pct: float) -> pd.Series:
    """Return chi-square CDF scaled turbulence series per date for a wide return panel.

    df_ret: index=date, columns=symbol, values=log-return
    """
    dates = df_ret.index.to_list()
    cols = list(df_ret.columns)
    p = len(cols)
    values = df_ret.values.astype(float)
    scores: list[float] = []
    for t in range(len(dates)):
        if t == 0:
            scores.append(np.nan)
            continue
        start = max(0, t - window)
        ref = values[start:t, :]
        x = values[t, :]
        if ref.shape[0] < max(20, min(window // 3, window)):
            scores.append(np.nan)
            continue
        # Drop columns with any NaN within the reference window for stability
        valid_cols = np.all(np.isfinite(ref), axis=0)
        if valid_cols.sum() < 5:
            scores.append(np.nan)
            continue
        ref = ref[:, valid_cols]
        x = x[valid_cols]
        ref_w = np.apply_along_axis(lambda a: _winsorize(a, winsor_pct), 0, ref)
        mu = np.nanmean(ref_w, axis=0)
        X = ref_w - mu
        lw = LedoitWolf(store_precision=True, assume_centered=True)
        lw.fit(X)
        prec = lw.precision_
        z = (x - mu)
        z = np.where(np.isfinite(z), z, 0.0)
        try:
            d2 = float(z @ prec @ z)
        except Exception:
            d2 = np.nan
        # Map to tail probability via chi-square CDF (df=p)
        q = 1.0 - float(chi2.cdf(d2, df=p)) if np.isfinite(d2) else np.nan
        scores.append(q)
    return pd.Series(scores, index=df_ret.index)


def _returns_wide(frame: pd.DataFrame, *, symbol_col: str = "symbol", date_col: str = "date") -> pd.DataFrame:
    f = frame.copy()
    f[date_col] = pd.to_datetime(f[date_col])
    f[symbol_col] = f[symbol_col].astype(str)
    if "prdy_close" in f.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            f["log_ret"] = np.log(f["close"] / f["prdy_close"])  # D uses D-1 info
    else:
        f = f.sort_values([symbol_col, date_col])
        f["prdy_close"] = f.groupby(symbol_col)["close"].shift(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            f["log_ret"] = np.log(f["close"] / f["prdy_close"])  # safe
    f = f.dropna(subset=["log_ret"]).copy()
    wide = f.pivot_table(index=date_col, columns=symbol_col, values="log_ret", aggfunc="first").sort_index()
    return wide


def build_hybrid_signals(
    processed_csv: str | Path,
    *,
    subset_symbols: Optional[Iterable[str]] = None,
    config: TurbulenceConfig | None = None,
) -> pd.DataFrame:
    cfg = config or TurbulenceConfig()
    df = pd.read_csv(processed_csv, low_memory=False)
    # All-universe returns
    R_all = _returns_wide(df)
    # Subset returns (47/50 names)
    if subset_symbols is None:
        R_sub = R_all  # fallback to all
    else:
        subset = [str(s) for s in subset_symbols]
        keep = [c for c in R_all.columns if c in subset]
        R_sub = R_all[keep] if keep else R_all
    # Rolling Mahalanobis -> chi-square tail probability
    q_port = _compute_rolling_mahalanobis(R_sub, window=min(cfg.window_max, cfg.window), winsor_pct=cfg.winsor_pct)
    q_sys = _compute_rolling_mahalanobis(R_all, window=min(cfg.window_max, cfg.window), winsor_pct=cfg.winsor_pct)
    q_port_s = _ema(q_port.fillna(method="ffill").fillna(0.0), cfg.ema_lambda)
    q_sys_s = _ema(q_sys .fillna(method="ffill").fillna(0.0), cfg.ema_lambda)
    q_any = pd.concat([q_port_s, q_sys_s], axis=1).max(axis=1)

    # Hysteresis regime
    regime = []
    state = "normal"
    for val in q_any.fillna(0.0):
        if state == "normal":
            if val >= cfg.hi_on:
                state = "crisis"
            elif val >= cfg.med_on:
                state = "caution"
        elif state == "caution":
            if val >= cfg.hi_on:
                state = "crisis"
            elif val < cfg.med_off:
                state = "normal"
        elif state == "crisis":
            if val < cfg.hi_off:
                state = "caution"
        regime.append(state)

    out = pd.DataFrame({
        "date": R_all.index,
        "q_port": q_port_s.values,
        "q_sys": q_sys_s.values,
        "q_any": q_any.values,
        "regime": regime,
    })
    return out


def build_hybrid_signals_from_frame(
    frame: pd.DataFrame,
    *,
    subset_symbols: Optional[Iterable[str]] = None,
    config: TurbulenceConfig | None = None,
) -> pd.DataFrame:
    cfg = config or TurbulenceConfig()
    # All-universe returns
    R_all = _returns_wide(frame)
    # Subset returns (47/50 names)
    if subset_symbols is None:
        R_sub = R_all  # fallback to all
    else:
        subset = [str(s) for s in subset_symbols]
        keep = [c for c in R_all.columns if c in subset]
        R_sub = R_all[keep] if keep else R_all
    # Rolling Mahalanobis -> chi-square tail probability
    q_port = _compute_rolling_mahalanobis(R_sub, window=min(cfg.window_max, cfg.window), winsor_pct=cfg.winsor_pct)
    q_sys = _compute_rolling_mahalanobis(R_all, window=min(cfg.window_max, cfg.window), winsor_pct=cfg.winsor_pct)
    q_port_s = _ema(q_port.fillna(method="ffill").fillna(0.0), cfg.ema_lambda)
    q_sys_s = _ema(q_sys .fillna(method="ffill").fillna(0.0), cfg.ema_lambda)
    q_any = pd.concat([q_port_s, q_sys_s], axis=1).max(axis=1)
    regime = []
    state = "normal"
    for val in q_any.fillna(0.0):
        if state == "normal":
            if val >= cfg.hi_on:
                state = "crisis"
            elif val >= cfg.med_on:
                state = "caution"
        elif state == "caution":
            if val >= cfg.hi_on:
                state = "crisis"
            elif val < cfg.med_off:
                state = "normal"
        elif state == "crisis":
            if val < cfg.hi_off:
                state = "caution"
        regime.append(state)
    out = pd.DataFrame({
        "date": R_all.index,
        "q_port": q_port_s.values,
        "q_sys": q_sys_s.values,
        "q_any": q_any.values,
        "regime": regime,
    })
    return out


def build_hybrid_signals_from_frames(
    portfolio_frame: pd.DataFrame,
    system_frame: pd.DataFrame,
    *,
    subset_symbols: Optional[Iterable[str]] = None,
    config: TurbulenceConfig | None = None,
) -> pd.DataFrame:
    cfg = config or TurbulenceConfig()
    R_all = _returns_wide(system_frame)
    if subset_symbols is None:
        R_sub = _returns_wide(portfolio_frame)
    else:
        R_sub_full = _returns_wide(portfolio_frame)
        subset = [str(s) for s in subset_symbols]
        keep = [c for c in R_sub_full.columns if c in subset]
        R_sub = R_sub_full[keep] if keep else R_sub_full
    q_port = _compute_rolling_mahalanobis(R_sub, window=min(cfg.window_max, cfg.window), winsor_pct=cfg.winsor_pct)
    q_sys = _compute_rolling_mahalanobis(R_all, window=min(cfg.window_max, cfg.window), winsor_pct=cfg.winsor_pct)
    q_port_s = _ema(q_port.ffill().fillna(0.0), cfg.ema_lambda)
    q_sys_s = _ema(q_sys.ffill().fillna(0.0), cfg.ema_lambda)
    # Align indices
    idx = q_port_s.index.intersection(q_sys_s.index)
    q_port_s = q_port_s.reindex(idx)
    q_sys_s = q_sys_s.reindex(idx)
    q_any = pd.concat([q_port_s, q_sys_s], axis=1).max(axis=1)
    regime = []
    state = "normal"
    for val in q_any.fillna(0.0):
        if state == "normal":
            if val >= cfg.hi_on:
                state = "crisis"
            elif val >= cfg.med_on:
                state = "caution"
        elif state == "caution":
            if val >= cfg.hi_on:
                state = "crisis"
            elif val < cfg.med_off:
                state = "normal"
        elif state == "crisis":
            if val < cfg.hi_off:
                state = "caution"
        regime.append(state)
    out = pd.DataFrame({
        "date": idx,
        "q_port": q_port_s.values,
        "q_sys": q_sys_s.values,
        "q_any": q_any.values,
        "regime": regime,
    })
    return out


def write_hybrid_signals(
    processed_csv: str | Path,
    *,
    subset_symbols: Optional[Iterable[str]] = None,
    config: TurbulenceConfig | None = None,
    dest: str | Path = "data/proc/factors/turbulence_hybrid.csv",
) -> Path:
    df = build_hybrid_signals(processed_csv, subset_symbols=subset_symbols, config=config)
    path = Path(dest)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def write_returns_snapshots(
    processed_csv: str | Path,
    *,
    subset_symbols: Optional[Iterable[str]] = None,
    dest_dir: str | Path = "data/proc/returns",
) -> Tuple[Path, Path]:
    """Write wide daily log-return panels for all symbols and subset.

    Returns paths to (all_returns.csv, subset_returns.csv).
    """
    df = pd.read_csv(processed_csv, low_memory=False)
    R_all = _returns_wide(df)
    if subset_symbols is None:
        R_sub = R_all
    else:
        subset = [str(s) for s in subset_symbols]
        keep = [c for c in R_all.columns if c in subset]
        R_sub = R_all[keep] if keep else R_all
    outdir = Path(dest_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    p_all = outdir / "all_returns.csv"
    p_sub = outdir / "subset_returns.csv"
    R_all.to_csv(p_all, index=True)
    R_sub.to_csv(p_sub, index=True)
    return p_all, p_sub


__all__ = [
    "TurbulenceConfig",
    "build_hybrid_signals",
    "build_hybrid_signals_from_frame",
    "build_hybrid_signals_from_frames",
    "write_hybrid_signals",
    "write_returns_snapshots",
]
