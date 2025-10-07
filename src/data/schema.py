"""Schema specification and validation helpers for processed tabular data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from pandas.api import types as ptypes


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    """Expected shape of a column in the processed CSV."""

    name: str
    kinds: tuple[str, ...]
    allow_null: bool
    description: str
    min_value: float | None = None
    max_value: float | None = None

    def accepts(self, series: pd.Series) -> bool:
        """Return True when the series dtype matches an allowed pandas kind."""

        for kind in self.kinds:
            checker = _TYPE_CHECKERS.get(kind)
            if checker is None:
                continue
            if checker(series):
                return True
        return False


def _is_string(series: pd.Series) -> bool:
    return ptypes.is_string_dtype(series.dtype) or ptypes.is_object_dtype(series.dtype)


_TYPE_CHECKERS: dict[str, callable[[pd.Series], bool]] = {
    "int": lambda s: ptypes.is_integer_dtype(s.dtype),
    "float": lambda s: ptypes.is_float_dtype(s.dtype),
    "numeric": lambda s: ptypes.is_numeric_dtype(s.dtype),
    "string": _is_string,
    "bool": lambda s: ptypes.is_bool_dtype(s.dtype),
    "datetime": lambda s: ptypes.is_datetime64_any_dtype(s.dtype),
}

_PRIMARY_KEY = ("symbol", "date")


def _build_schema() -> list[ColumnSpec]:
    specs: list[ColumnSpec] = [
        ColumnSpec("symbol", ("int", "string"), False, "Instrument identifier."),
        ColumnSpec("date", ("string", "datetime"), False, "Trading session date."),
        ColumnSpec("name", ("string",), True, "Human readable instrument name."),
        ColumnSpec("market", ("string",), False, "Exchange or venue code."),
        ColumnSpec("industry", ("string",), True, "GICS-like industry sector."),
        ColumnSpec("adj_flag", ("bool",), False, "Vendor adjustment flag."),
        ColumnSpec("open", ("int", "float"), False, "Open price.", 0.0, None),
        ColumnSpec("high", ("int", "float"), False, "High price.", 0.0, None),
        ColumnSpec("low", ("int", "float"), False, "Low price.", 0.0, None),
        ColumnSpec("close", ("int", "float"), False, "Close price.", 0.0, None),
        ColumnSpec("prdy_close", ("float", "int"), True, "Previous close price.", 0.0, None),
        ColumnSpec("volume", ("int",), False, "Traded volume.", 0.0, None),
        ColumnSpec("value", ("int",), False, "Traded value.", 0.0, None),
        ColumnSpec("market_cap", ("float", "int"), True, "Market capitalisation.", 0.0, None),
        ColumnSpec("change", ("float", "int"), True, "Price change vs prior close."),
        ColumnSpec("change_rate", ("float", "int"), True, "Price change rate vs prior close."),
        ColumnSpec("range_pct", ("float",), True, "Intra-day price range ratio.", 0.0, None),
        ColumnSpec("gap_pct", ("float",), True, "Open gap vs prior close."),
        ColumnSpec("ma_5", ("float",), True, "5 day moving average."),
        ColumnSpec("ma_10", ("float",), True, "10 day moving average."),
        ColumnSpec("ma_20", ("float",), True, "20 day moving average."),
        ColumnSpec("dist_ma5", ("float",), True, "Distance to 5 day MA."),
        ColumnSpec("dist_ma10", ("float",), True, "Distance to 10 day MA."),
        ColumnSpec("dist_ma20", ("float",), True, "Distance to 20 day MA."),
        ColumnSpec("ma10_slope", ("float",), True, "Slope of 10 day MA."),
        ColumnSpec("ma20_slope", ("float",), True, "Slope of 20 day MA."),
        ColumnSpec("atr_5", ("float",), True, "Average true range over 5 days.", 0.0, None),
        ColumnSpec("atr_14", ("float",), True, "Average true range over 14 days.", 0.0, None),
        ColumnSpec("rv_10", ("float",), True, "Realised volatility (10 sessions).", 0.0, None),
        ColumnSpec("rv_20", ("float",), True, "Realised volatility (20 sessions).", 0.0, None),
    ]

    ratio_columns = [
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
    for column in ratio_columns:
        specs.append(
            ColumnSpec(
                column,
                ("float", "int"),
                True,
                "Normalised order-flow ratio.",
                -10.0,
                10.0,
            )
        )

    return specs


SCHEMA: tuple[ColumnSpec, ...] = tuple(_build_schema())
PRIMARY_KEY: tuple[str, str] = _PRIMARY_KEY


def get_schema() -> Sequence[ColumnSpec]:
    """Return the immutable schema specification."""

    return SCHEMA


def validate_dataframe(df: pd.DataFrame, *, strict: bool = True) -> list[str]:
    """Validate a dataframe against the expected schema.

    Returns a list of human-readable error messages. An empty list indicates
    the frame matches the schema.
    """

    errors: list[str] = []
    expected_names = {spec.name for spec in SCHEMA}

    missing = [name for name in expected_names if name not in df.columns]
    if missing:
        errors.append(f"Missing columns: {', '.join(sorted(missing))}")

    if strict:
        extras = sorted(set(df.columns) - expected_names)
        if extras:
            errors.append(f"Unexpected columns present: {', '.join(extras)}")

    for spec in SCHEMA:
        if spec.name not in df.columns:
            continue
        series = df[spec.name]
        if not spec.accepts(series):
            errors.append(
                f"Column '{spec.name}' has dtype {series.dtype} which does not match {spec.kinds}"
            )

        if not spec.allow_null and series.isna().any():
            null_count = int(series.isna().sum())
            errors.append(f"Column '{spec.name}' contains {null_count} null values")

        needs_range_check = spec.min_value is not None or spec.max_value is not None
        numeric_series = None
        if needs_range_check:
            numeric_series = pd.to_numeric(series.dropna(), errors="coerce")
            invalid_numeric = int(numeric_series.isna().sum())
            if invalid_numeric:
                errors.append(
                    f"Column '{spec.name}' contains {invalid_numeric} non-numeric values"
                )
            numeric_series = numeric_series.dropna()

        if spec.min_value is not None and numeric_series is not None and not numeric_series.empty:
            below = numeric_series < spec.min_value
            if below.any():
                errors.append(
                    f"Column '{spec.name}' has {int(below.sum())} values below {spec.min_value}"
                )
        if spec.max_value is not None and numeric_series is not None and not numeric_series.empty:
            above = numeric_series > spec.max_value
            if above.any():
                errors.append(
                    f"Column '{spec.name}' has {int(above.sum())} values above {spec.max_value}"
                )

    if set(_PRIMARY_KEY).issubset(df.columns):
        duplicated = df.duplicated(subset=list(_PRIMARY_KEY)).sum()
        if duplicated:
            errors.append(f"Primary key ({', '.join(_PRIMARY_KEY)}) has {duplicated} duplicates")

    if "date" in df.columns:
        try:
            pd.to_datetime(df["date"], errors="raise")
        except (ValueError, TypeError) as exc:
            errors.append(f"Column 'date' is not fully parseable to datetime: {exc}")

    return errors


def validate_csv(path: str | Path, *, strict: bool = True, read_kwargs: dict | None = None) -> list[str]:
    """Validate a CSV file on disk and return any schema violations."""

    read_kwargs = dict(read_kwargs or {})
    df = pd.read_csv(path, **read_kwargs)
    return validate_dataframe(df, strict=strict)


__all__ = [
    "ColumnSpec",
    "PRIMARY_KEY",
    "SCHEMA",
    "get_schema",
    "validate_dataframe",
    "validate_csv",
]
