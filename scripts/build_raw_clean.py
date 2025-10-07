#!/usr/bin/env python
"""Replicate raw cleaning step from the legacy notebook.

Usage:
  python -m scripts.build_raw_clean \
    --input data/proc/daily_raw_merged.csv \
    --out-parquet data/proc/daily_raw_clean.parquet \
    --out-csv data/proc/daily_raw_clean.csv

Note: the input must already include investor ratio columns as produced by the
legacy ingestion (prsn_*, frgn_*, orgn_*). This script only applies the row/date
filtering rules to match the notebook output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.raw_clean import write_daily_raw_clean


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build daily_raw_clean from merged raw CSV with investor ratios.")
    p.add_argument("--input", required=True, help="Input CSV path including investor ratio columns.")
    p.add_argument("--out-parquet", default="data/proc/daily_raw_clean.parquet", help="Output parquet path.")
    p.add_argument("--out-csv", default="data/proc/daily_raw_clean.csv", help="Output CSV path.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    write_daily_raw_clean(args.input, args.out_parquet, args.out_csv)
    print(f"✅ Saved: {args.out_parquet} and {args.out_csv}")


if __name__ == "__main__":
    main()

