# Multi-Stock Environment Overview

## Core Class
- `MultiStockTradingEnv` (`src/env/multi_stock_env.py`) implements a Gymnasium `Env` with discrete buy/hold/sell actions per symbol.
- Environments are constructed through `build_environment()` / `make_env_factory()` in `src/utils/builders.py`, which pass split-aware `pandas.DataFrame` inputs and resolve `EnvironmentConfig` defaults.

## Episode Lifecycle
1. `reset()` positions the agent at the earliest date in the split with `initial_cash` and zero holdings.
2. Each `step()` advances exactly one trading day; the episode terminates after the final date (or `environment.max_steps` if configured).
3. No intermediate rewards are emitted. On termination the reward equals net portfolio growth: `final_portfolio_value - initial_cash`, scaled by `reward_scaling`.
4. The environment tracks `asset_history` and `date_history` so downstream callbacks can reconstruct per-day traces.

## Spaces
- **Action space**: `MultiDiscrete` with dimension equal to the number of symbols. Each entry supports values `0 .. 2n`, which translate to share trades in `[-n, +n]` via `trade_units = action - n`. Negative values trigger sells limited by current holdings (no shorting).
- **Observation space**: `Box` of length `1 + 2 * num_symbols`, containing normalised cash, per-symbol z-scored prices, and holdings scaled by `n`.

## Transaction Policy
- Sells realise proceeds `price * qty` and then pay commission + slippage derived from `FrictionConfig` (buys incur no fees).
- Commission is computed via `compute_commission(*notionals*)`; slippage defaults to `slippage_bps` unless a custom function is provided.
- Buys deduct only the trade notional (`price * qty`), respecting available cash.

## Stress Hooks
- The optional `environment.stress` config toggles stress testing; when `enabled`, it scales commission/slippage via `commission_multiplier`/`slippage_multiplier`.
- Stress overrides are also surfaced by the evaluation CLI (`scripts/evaluate.py --stress-*`).

## Info Dictionary
Each `step()` returns an `info` dict with:
- `cash`: current cash balance.
- `holdings`: vector of integer share counts per symbol.
- `prices`: latest close prices aligned to holdings.
- `portfolio_value`: `cash + holdings * prices` (also surfaced as `nav`).
- `nav_return`: single-step return vs. the prior portfolio value.
- `turnover`: trade notional divided by the prior portfolio value.
- `drawdown`: peak-to-trough drawdown relative to the running high watermark.
- `reward_unscaled`: terminal reward in currency units (0.0 for intermediate steps).
- On terminal steps additional keys are added: `episode_final`, `final_reward`, and `final_portfolio_value`.

## Evaluation Support
- Evaluation callbacks consume `date_history`, `cash`, and `holdings` to emit per-day CSV traces with columns for date, cash, per-symbol share counts, portfolio value, and day-over-day value change.
- `MultiStockTradingEnv` is deterministic given a fixed dataset; random seeds only affect future extensions (e.g., stochastic price perturbations).

## Known Gaps
- Provide helper utilities for stress scenarios (e.g., slippage shocks, turbulence masks) once the base trading policy is finalised.
- Add dedicated evaluation/report scripts that consume the emitted traces and metrics.
