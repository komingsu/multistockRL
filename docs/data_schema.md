# Tabular Data Schema

## Dataset Overview
- Source file: `data/proc/daily_with_indicators.csv`.
- Primary key: (`symbol`, `date`).
- Each row captures a single trading day for a single equity with aligned price, flow, and technical indicators.
- Target column for supervised objectives: `close` (transformed via log-return by default).

## Column Groups

### Identifiers & Metadata
| Column | Type | Nulls | Notes |
| --- | --- | --- | --- |
| symbol | integer (cast to string downstream) | disallow | Numeric code unique per instrument.|
| date | string ISO-8601 (parsed to datetime) | disallow | Trading session date. |
| name | string | allow | Human-readable instrument name; optional. |
| market | string | disallow | Exchange/venue identifier. |
| industry | string | allow | Filled with `UNKNOWN_INDUSTRY` when missing. |
| adj_flag | boolean | disallow | Adjustment flag from data vendor. |

### Price & Liquidity
| Column | Type | Nulls | Notes |
| --- | --- | --- | --- |
| open | integer | disallow | Opening price (local currency). |
| high | integer | disallow | High price of session. |
| low | integer | disallow | Low price of session. |
| close | integer | disallow | Closing price of session. |
| prdy_close | float | allow | Prior-day close when available. |
| volume | integer | disallow | Trading volume. |
| value | integer | disallow | Trading value. |
| market_cap | float | allow | Market capitalization. |
| change | float | allow | Absolute change vs prior close. |
| change_rate | float | allow | Percent change vs prior close. |
| range_pct | float | allow | (High - Low) / Close. |
| gap_pct | float | allow | (Open - Prior Close) / Prior Close. |

### Order Flow Ratios
All ratios are expressed as float percentages in [-1, 1] and may contain missing values when the venue does not report the component.
| Column | Type | Nulls | Notes |
| --- | --- | --- | --- |
| prsn_buy_val_ratio | float | allow | Retail buy value ratio. |
| prsn_sell_val_ratio | float | allow | Retail sell value ratio. |
| prsn_net_val_ratio | float | allow | Retail net value ratio. |
| prsn_buy_vol_ratio | float | allow | Retail buy volume ratio. |
| prsn_sell_vol_ratio | float | allow | Retail sell volume ratio. |
| prsn_net_vol_ratio | float | allow | Retail net volume ratio. |
| frgn_buy_val_ratio | float | allow | Foreign buy value ratio. |
| frgn_sell_val_ratio | float | allow | Foreign sell value ratio. |
| frgn_net_val_ratio | float | allow | Foreign net value ratio. |
| frgn_buy_vol_ratio | float | allow | Foreign buy volume ratio. |
| frgn_sell_vol_ratio | float | allow | Foreign sell volume ratio. |
| frgn_net_vol_ratio | float | allow | Foreign net volume ratio. |
| orgn_buy_val_ratio | float | allow | Institutional buy value ratio. |
| orgn_sell_val_ratio | float | allow | Institutional sell value ratio. |
| orgn_net_val_ratio | float | allow | Institutional net value ratio. |
| orgn_buy_vol_ratio | float | allow | Institutional buy volume ratio. |
| orgn_sell_vol_ratio | float | allow | Institutional sell volume ratio. |
| orgn_net_vol_ratio | float | allow | Institutional net volume ratio. |

### Technical Indicators
| Column | Type | Nulls | Notes |
| --- | --- | --- | --- |
| ma_5 | float | allow | 5-day moving average of close. |
| ma_10 | float | allow | 10-day moving average of close. |
| ma_20 | float | allow | 20-day moving average of close. |
| dist_ma5 | float | allow | (Close - MA5) / MA5. |
| dist_ma10 | float | allow | (Close - MA10) / MA10. |
| dist_ma20 | float | allow | (Close - MA20) / MA20. |
| ma10_slope | float | allow | Slope of MA10 over trailing window. |
| ma20_slope | float | allow | Slope of MA20 over trailing window. |
| atr_5 | float | allow | Average True Range over 5 sessions. |
| atr_14 | float | allow | Average True Range over 14 sessions. |
| rv_10 | float | allow | Realized volatility over 10 sessions. |
| rv_20 | float | allow | Realized volatility over 20 sessions. |

## Validation Rules
1. Columns listed above must exist with matching casing (strict schema).
2. (`symbol`, `date`) pairs must be unique after `date` is parsed as datetime.
3. Numeric fields that allow nulls should use NaN rather than empty strings.
4. Categorical gaps are filled with the configured tokens (e.g., `UNKNOWN_INDUSTRY`).
5. Percent/ratio fields should clip to [-10, 10] prior to model ingestion to guard against vendor outliers.

## Integration Notes
- The schema drives validation utilities in `src/data/schema.py`.
- Automated checks live under `tests/test_data_schema.py` and gate CSV ingestion in CI.
- Notebook `notebooks/data_validation.ipynb` offers exploratory validation and summary statistics for new data drops.

- Sliding window preprocessing appends engineered columns (e.g.,  `close_log_return_lag_*`, `volume_pct_change_lag_1`) and a trading-gap mask at runtime; these stay out of the raw CSV but are documented in `docs/data_pipeline.md`. 
