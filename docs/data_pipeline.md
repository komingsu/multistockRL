# Data Pipeline Strategy

## Train/Val/Test (TVT) Policy
- Split by calendar date ranges defined in config, avoiding data leakage across future periods.
- Default split proposal:
  - Train: 2020-09-28 ~ 2024-12-31
  - Validation: 2025-01-02 ~ 2025-06-30
  - Test: 2025-07-01 ~ 2025-09-26
- Sliding evaluation: windowed preprocessing ensures each batch only sees statistics from its lookback period.

## Windowed Preprocessing
- Each sample generated from a rolling window of length `window.size` (default 60 trading days).
- Target horizon `window.horizon` (default 1 day ahead close return).
- Per-window normalization: compute mean/std inside the window for each numeric channel. This keeps price levels, flows, and engineered signals comparable across symbols despite stock-specific scales.
- Feature groups:
  - Price-derived: `open`, `high`, `low`, `close`, `volume`, `value`, `market_cap`, `change`, `change_rate`, `range_pct`, `gap_pct`.
  - Flow ratios: personal/foreign/institutional buy/sell/net ratios.
  - Technical indicators: moving averages, slopes, ATR, RV, distance metrics.
- Engineered features (per-symbol, pre-window):
  - `close_log_return_lag_{1,5,10}` — trailing log returns capturing daily, weekly, and bi-weekly momentum.
  - `volume_pct_change_lag_1` — one-day relative change in turnover to highlight liquidity shocks.
- Categorical columns (`symbol`, `market`, `industry`) transformed via embedding indices; `industry` missing values mapped to token `UNKNOWN_INDUSTRY`.
- Trading gap mask: windows carry a parallel mask vector with `0` where day gaps exceed one session (e.g., holidays, halts) so downstream batching can dampen those steps instead of imputing values.

## Config-Driven Workflow
- `configs/base.yaml` declares dataset path, column names, TVT ranges, window size, preprocessing directives (lag specs, masks), and normalization options.
- Loader reads config, validates schema, fills categorical defaults, applies lag transforms per symbol, and materializes symbol panel sorted by date.
- Splitter produces iterators for train/val/test according to config ranges without writing intermediate files.

## Data Loader Requirements
1. Assert schema (columns, dtypes) and sort by (`symbol`, `date`).
2. Fill `industry` NA with `UNKNOWN_INDUSTRY`; record original missing mask for diagnostics.
3. Optionally filter symbols or dates per config include/exclude lists.
4. Emit data structure: dictionary keyed by split -> list of `WindowBatch` objects containing inputs/targets, metadata (start/end date, symbol list), and window masks.
5. Provide hook for on-the-fly transformations (log scaling, differencing) inside window generator.

## Outstanding Questions
- Target definition: default to next-day log return; adaptable through config (e.g., multi-day horizon, classification labels).
- Additional engineered signals: extend lag specs to other columns (e.g., net flows) once use-cases emerge.
- Handling non-trading days across symbols: current mask flags gaps; decide whether to drop or weight down those timesteps in batching.
- Feature selection per experiment: support config overrides to drop/append engineered features without touching code.

