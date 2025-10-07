#!/usr/bin/env python
"""Data build CLI: collect raw, build features, selection, and turbulence.

This command produces training‑compatible processed CSV without adding extra
columns beyond the current schema. Optional artifacts are saved in dedicated
folders under `data/proc/`.

Examples
- Build everything from cached/raw sources (no network):
    python -m scripts.build_data --offline --no-selection --no-turbulence

- End‑to‑end (requires KIS credentials for collection):
    python -m scripts.build_data --collect --features --selection --turbulence
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.feature_engineering import write_processed_csv
from src.data.ingest import load_or_collect_daily, save_symbol_master
from src.data.selection import SelectionConfig, save_selection_csv, select_portfolio
from src.data.turbulence import compute_turbulence_simple, write_turbulence_csv
from src.data.turbulence_system import build_system_frame_from_fdr as _fdr_sys


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build training dataset and optional artifacts.")
    p.add_argument("--collect", action="store_true", help="Collect raw OHLCV via KIS (requires credentials).")
    p.add_argument("--end2end", action="store_true", help="End-to-end build: collect daily, investor ratios, raw-clean, and processed CSV.")
    p.add_argument("--offline", action="store_true", help="Do not collect; use cached raw files if present.")
    p.add_argument("--features", action="store_true", help="Build features and write processed CSV.")
    p.add_argument("--selection", action="store_true", help="Compute top‑N selection and save CSV/parquet.")
    p.add_argument("--turbulence", action="store_true", help="Compute turbulence factors and save CSV.")
    p.add_argument("--turbulence-hybrid", action="store_true", help="Compute hybrid turbulence (portfolio + system) and save CSV.")
    p.add_argument("--subset-selection", default=None, help="Optional selection CSV (topN) to define subset symbols for portfolio turbulence.")
    p.add_argument("--kis-master", default=None, help="Optional KIS master CSV to build system universe for turbulence (e.g., KOSPI only).")
    p.add_argument("--sys-max-symbols", type=int, default=None, help="Optional cap on number of symbols to fetch for system turbulence (for quick runs).")
    p.add_argument("--out-csv", default="data/proc/daily_with_indicators.csv", help="Processed CSV output path.")
    p.add_argument("--selection-out", default="data/proc/selection", help="Folder to store selection outputs.")
    p.add_argument("--symbols-parquet", default=None, help="Optional symbol master parquet path (use latest if omitted).")
    p.add_argument("--daily-parquet", default=None, help="Optional daily OHLCV parquet path (use latest if omitted).")
    p.add_argument("--raw-clean-csv", default=None, help="Optional raw-clean CSV (with investor ratios). If provided, features are built from this CSV directly.")
    p.add_argument("--top-n", type=int, default=50, help="Selection list size.")
    p.add_argument("--env", default="real", help="KIS environment: real or mock (for collection).")
    p.add_argument("--start", default=None, help="Start date YYYYMMDD (for investor collection).")
    p.add_argument("--end", default=None, help="End date YYYYMMDD (for investor collection).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Resolve raw inputs unless raw-clean is provided (raw-clean path can skip daily/symbol master)
    sym_path: Optional[Path] = None
    daily_path: Optional[Path] = None
    if not args.raw_clean_csv or args.end2end:
        if args.symbols_parquet:
            sym_path = Path(args.symbols_parquet)
        else:
            sym_path = save_symbol_master()

        if args.daily_parquet:
            daily_path = Path(args.daily_parquet)
        else:
            if args.collect or args.end2end:
                daily_path = load_or_collect_daily()
            else:
                cache_dir = Path("data/raw/kis/daily")
                if not cache_dir.exists():
                    raise FileNotFoundError("No cached daily parquet found; run with --collect first or provide --daily-parquet")
                candidates = sorted(cache_dir.glob("*.parquet"))
                if not candidates:
                    raise FileNotFoundError("No cached daily parquet found in data/raw/kis/daily")
                daily_path = candidates[-1]

    # End-to-end path: build investor ratios + raw-clean, then features
    if args.end2end:
        from src.data.raw_clean import build_daily_raw_clean
        from src.data.investor import collect_investor_ratios_for_universe
        daily = pd.read_parquet(daily_path)
        symbols = sorted(daily["symbol"].astype(str).str.zfill(6).unique().tolist())
        dmin = args.start or pd.to_datetime(daily["date"]).min().strftime("%Y%m%d")
        dmax = args.end or pd.to_datetime(daily["date"]).max().strftime("%Y%m%d")
        print(f"Collecting investor ratios for {len(symbols)} symbols, {dmin}..{dmax} ({args.env})")
        inv = collect_investor_ratios_for_universe(symbols, start=dmin, end=dmax, env=args.env)
        # Merge daily + symbol master meta
        sm = pd.read_parquet(sym_path) if sym_path else None
        base = daily.copy()
        if sm is not None and not sm.empty:
            keep_cols = [c for c in ["symbol","name","market","industry","market_cap"] if c in sm.columns]
            meta = sm[keep_cols].drop_duplicates("symbol")
            base = base.merge(meta, on="symbol", how="left")
        # Align types
        base["symbol"] = base["symbol"].astype(str).str.zfill(6)
        base["date"] = pd.to_datetime(base["date"]).dt.date
        merged = base.merge(inv, on=["symbol","date"], how="left").sort_values(["date","symbol"])  # like notebook
        raw_clean = build_daily_raw_clean(merged)
        # Build features from raw-clean
        from src.data.feature_engineering import build_feature_frame
        df = build_feature_frame(raw_clean, None)
        features = df
        # Embed hybrid turbulence (two signals) into final CSV
        from src.data.turbulence_hybrid import build_hybrid_signals_from_frames, TurbulenceConfig
        from src.data.turbulence_system import build_system_frame_from_kis_master, build_system_frame_from_fdr, SystemUniverseConfig
        from src.data.selection import SelectionConfig as _SelCfg, select_portfolio as _select
        subset = None
        if args.subset_selection:
            scsv = pd.read_csv(args.subset_selection)
            subset = scsv.get("symbol", pd.Series(dtype=str)).astype(str).tolist()
        if subset is None or len(subset) == 0:
            # Default: compute a quick top-N selection
            try:
                sel_df = _select(features, _SelCfg(top_n=args.top_n))
                subset = sel_df["symbol"].astype(str).tolist()
                print(f"[hybrid] Using computed subset of size {len(subset)} for turb_port (selection module)")
            except Exception as e:
                # Fallback: last-date cross-section by market_cap/value/volume
                try:
                    f = features.copy()
                    f["date"] = pd.to_datetime(f["date"]) 
                    last = f[f["date"] == f["date"].max()].copy()
                    for col in ["market_cap","value","volume"]:
                        if col in last.columns:
                            last[col] = pd.to_numeric(last[col], errors='coerce')
                    if "market_cap" in last.columns and last["market_cap"].notna().any():
                        pick = last.sort_values("market_cap", ascending=False).head(args.top_n)
                    elif "value" in last.columns and last["value"].notna().any():
                        pick = last.sort_values("value", ascending=False).head(args.top_n)
                    else:
                        pick = last.sort_values("volume", ascending=False).head(args.top_n)
                    subset = pick["symbol"].astype(str).tolist()
                    print(f"[hybrid] Using fallback subset of size {len(subset)} (last-date ranking)")
                except Exception as e2:  # pragma: no cover
                    print(f"[hybrid] Subset selection failed ({e}, {e2}); falling back to all symbols")
                    subset = None
        # Build system returns frame (prefer KIS master scope if provided)
        if args.kis_master:
            sys_frame = build_system_frame_from_kis_master(
                args.kis_master,
                start=dmin,
                end=dmax,
                env=args.env,
                config=SystemUniverseConfig(market_filter="KOSPI"),
                max_symbols=args.sys_max_symbols,
            )
        else:
            sys_frame = base.copy()
            sys_frame["date"] = pd.to_datetime(sys_frame["date"])  # ensure parse
        hybrid = build_hybrid_signals_from_frames(features, sys_frame, subset_symbols=subset, config=TurbulenceConfig())
        # Merge on date; add two columns only (turb_port, turb_sys)
        tmp = features.copy()
        tmp["date"] = pd.to_datetime(tmp["date"]).dt.normalize()
        hybrid2 = hybrid.copy()
        hybrid2["date"] = pd.to_datetime(hybrid2["date"]).dt.normalize()
        merged_final = tmp.merge(hybrid2[["date","q_port","q_sys"]], on="date", how="left")
        merged_final = merged_final.rename(columns={"q_port":"turb_port","q_sys":"turb_sys"})
        merged_final["date"] = merged_final["date"].dt.strftime("%Y-%m-%d")
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        merged_final.to_csv(args.out_csv, index=False)
        print(f"✅ Final CSV written with turbulence: {args.out_csv}")
        features = merged_final
    else:
        # Build features and write processed CSV
        if args.raw_clean_csv:
            # Build directly from raw-clean CSV
            from src.data.feature_engineering import build_feature_frame
            raw_clean = pd.read_csv(args.raw_clean_csv, low_memory=False)
            # Convert date column to string for consistent downstream IO
            if "date" in raw_clean.columns:
                raw_clean["date"] = pd.to_datetime(raw_clean["date"]).dt.strftime("%Y-%m-%d")
            df = build_feature_frame(raw_clean, None)
            Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
            features = df
        else:
            if args.features or (not args.selection and not args.turbulence):
                write_processed_csv(daily_path, sym_path, args.out_csv)
                print(f"✅ Processed CSV written: {args.out_csv}")
            # Load feature frame for downstream artifacts
            features = pd.read_csv(args.out_csv, low_memory=False)

        # Always embed hybrid turbulence in final CSV (no extra factor files)
        from src.data.turbulence_hybrid import build_hybrid_signals_from_frame, build_hybrid_signals_from_frames, TurbulenceConfig
        from src.data.turbulence_system import build_system_frame_from_kis_master, SystemUniverseConfig
        from src.data.selection import SelectionConfig as _SelCfg, select_portfolio as _select
        subset = None
        if args.subset_selection:
            scsv = pd.read_csv(args.subset_selection)
            subset = scsv.get("symbol", pd.Series(dtype=str)).astype(str).tolist()
        if subset is None or len(subset) == 0:
            try:
                sel_df = _select(features, _SelCfg(top_n=args.top_n))
                subset = sel_df["symbol"].astype(str).tolist()
                print(f"[hybrid] Using computed subset of size {len(subset)} for turb_port (selection module)")
            except Exception as e:
                try:
                    f = features.copy()
                    f["date"] = pd.to_datetime(f["date"]) 
                    last = f[f["date"] == f["date"].max()].copy()
                    for col in ["market_cap","value","volume"]:
                        if col in last.columns:
                            last[col] = pd.to_numeric(last[col], errors='coerce')
                    if "market_cap" in last.columns and last["market_cap"].notna().any():
                        pick = last.sort_values("market_cap", ascending=False).head(args.top_n)
                    elif "value" in last.columns and last["value"].notna().any():
                        pick = last.sort_values("value", ascending=False).head(args.top_n)
                    else:
                        pick = last.sort_values("volume", ascending=False).head(args.top_n)
                    subset = pick["symbol"].astype(str).tolist()
                    print(f"[hybrid] Using fallback subset of size {len(subset)} (last-date ranking)")
                except Exception as e2:  # pragma: no cover
                    print(f"[hybrid] Subset selection failed ({e}, {e2}); falling back to all symbols")
                    subset = None
        # Build system frame from KIS master if provided
        if args.kis_master:
            f = features.copy(); f["date"] = pd.to_datetime(f["date"]) 
            dmin = f["date"].min().strftime("%Y%m%d"); dmax = f["date"].max().strftime("%Y%m%d")
            try:
                sys_frame = build_system_frame_from_kis_master(
                    args.kis_master,
                    start=dmin,
                    end=dmax,
                    env=args.env,
                    config=SystemUniverseConfig(market_filter="KOSPI"),
                    max_symbols=args.sys_max_symbols,
                )
            except Exception as e:
                # Fallback to FDR price source
                syms = pd.read_csv(args.kis_master, dtype=str).get('symbol').astype(str).tolist()
                sys_frame = _fdr_sys(syms, start=dmin, end=dmax, max_symbols=args.sys_max_symbols)
            hybrid = build_hybrid_signals_from_frames(features, sys_frame, subset_symbols=subset, config=TurbulenceConfig())
        else:
            hybrid = build_hybrid_signals_from_frame(features, subset_symbols=subset, config=TurbulenceConfig())
        tmp = features.copy(); tmp["date"] = pd.to_datetime(tmp["date"]).dt.normalize()
        hybrid2 = hybrid.copy(); hybrid2["date"] = pd.to_datetime(hybrid2["date"]).dt.normalize()
        merged_final = tmp.merge(hybrid2[["date","q_port","q_sys"]], on="date", how="left")
        merged_final = merged_final.rename(columns={"q_port":"turb_port","q_sys":"turb_sys"})
        merged_final["date"] = merged_final["date"].dt.strftime("%Y-%m-%d")
        merged_final.to_csv(args.out_csv, index=False)
        features = merged_final

    # Selection
    if args.selection:
        cfg = SelectionConfig(top_n=args.top_n)
        sel = select_portfolio(features, cfg)
        p_path, c_path = save_selection_csv(sel, args.selection_out, label=f"top{args.top_n}")
        print(f"✅ Selection saved: {p_path}, {c_path}")

    # Turbulence factors (hybrid: subset vs all)
    if args.turbulence:
        subset = None
        if args.selection:
            subset = sel["symbol"].astype(str).tolist()  # type: ignore[name-defined]
        tb = compute_turbulence_simple(features, subset_symbols=subset)
        out = write_turbulence_csv(tb)
        print(f"✅ Turbulence CSV saved: {out}")
    # hybrid turbulence already embedded into final CSV; no extra files


if __name__ == "__main__":
    main()
