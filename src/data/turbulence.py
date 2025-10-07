"""Turbulence metrics derived from cross‑sectional return dispersion.

This module prepares daily turbulence proxies for later use in policy design.
To keep the training dataset schema unchanged, outputs are stored separately
under `data/proc/factors/turbulence.csv`.

Definitions (simple proxies):
- turbulence_all: cross‑sectional standard deviation of 1‑day log returns across
  the full universe on each date.
- turbulence_subset: same metric computed only for a provided subset (e.g., the
  selected 50 names).

More sophisticated variants (e.g., Mahalanobis distance using rolling covariance)
can be added later while preserving the interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def _log_ret(df: pd.DataFrame) -> pd.Series:
    prev = df.groupby("symbol")["close"].shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(df["close"] / prev)
    return r.replace([np.inf, -np.inf], np.nan)


def compute_turbulence_simple(
    features: pd.DataFrame,
    *,
    subset_symbols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Return a dataframe with columns [date, turbulence_all, turbulence_subset]."""

    df = features[["symbol", "date", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["log_ret"] = _log_ret(df)

    # All-universe dispersion per date
    agg_all = (
        df.groupby("date")["log_ret"].std(ddof=0).rename("turbulence_all").reset_index()
    )

    if subset_symbols is not None:
        subset = df[df["symbol"].astype(str).isin({str(s) for s in subset_symbols})]
        agg_sub = (
            subset.groupby("date")["log_ret"].std(ddof=0).rename("turbulence_subset").reset_index()
        )
    else:
        agg_sub = agg_all.rename(columns={"turbulence_all": "turbulence_subset"})

    out = pd.merge(agg_all, agg_sub, on="date", how="outer").sort_values("date")
    return out


def write_turbulence_csv(df: pd.DataFrame, dest: str | Path = "data/proc/factors/turbulence.csv") -> Path:
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest_path, index=False)
    return dest_path


__all__ = ["compute_turbulence_simple", "write_turbulence_csv"]

