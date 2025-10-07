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

## Action & Execution
- Default `action_mode` is **`weights`**: the policy emits non-negative allocation weights (plus implicit cash). We clamp the proposed weights via a τ limiter (half-L1 move ≤ τ), convert them to integer share counts with `integerize_allocations`, then execute trades.
- A legacy `action_mode: shares` keeps the previous `MultiDiscrete` interface; this remains available for ablation tests.
- Execution honors minimum lot size (default 1 share), respects available cash/leverage, and records the realized `{weights, cash_fraction}` per step.

## Transaction Policy
- Buys and sells compute one-way costs through `evaluate_transaction_costs` (wrapping `compute_commission`/`compute_slippage`), ensuring both legs pay the configured commission + slippage.
- Trade costs are debited *after* applying price moves (post-return gross NAV), keeping cost and turnover ratios aligned with the reward.
- Default friction applies a 0.5% commission and 0.5% slippage per leg (≈1% round-trip) and the charge hits immediately on both buy and sell fills.

## Reward
- Each step emits `log(V_t / V_{t-1}) - λ * turnover_t`, where `turnover_t` uses post-return gross NAV. `λ` defaults to 0.02 but can be tuned per experiment.
- Cumulative reward therefore tracks geometric return (log wealth) adjusted for turnover penalties.

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
- `log_return`: natural log of the single-step return.
- `turnover`: trade notional divided by the prior portfolio value.
- `drawdown`: peak-to-trough drawdown relative to the running high watermark.
- `commission_cost` / `slippage_cost`: one-way aggregate transaction costs applied this step.
- `cost_ratio`: transaction-cost share of the post-return NAV (`cost / nav_gross`).
- `reward_unscaled`: terminal reward in currency units (0.0 for intermediate steps).
- `weights`: realized fractional allocation (post-trade, post-return) per symbol.
- `cash_fraction`: realized cash share of gross NAV.
- `gross_value`: post-return gross portfolio value prior to cost deductions.
- On terminal steps additional keys are added: `episode_final`, `final_reward`, and `final_portfolio_value`.

## Portfolio Overrides & Integerization
- `MultiStockTradingEnv` now exposes `resolve_portfolio_start_index(state)` and `price_map(index)` so callers can derive deterministic price vectors for override conversion.
- Overrides **must** enumerate every tradable symbol. `validate_holdings_payload` enforces coverage and integer share counts (including explicit zeros for flat positions).
- The evaluation/inference helpers provide `_prepare_portfolio_override` which supports `allocations` (weights) when `--integerize` is passed to `scripts/inference.py`. We convert weights to integer shares via `integerize_allocations`, log the executed weights per step, and persist final `{shares, weights, cash}` payloads next to trace CSVs.
- Missing VecNormalize statistics now raise immediately: checkpoints trained with `training.vecnormalize.enabled = true` emit `vecnormalize.pkl`, and inference refuses to run without the matching scaler.

## Evaluation Support
- Evaluation callbacks consume `date_history`, `cash`, and `holdings` to emit per-day CSV traces with columns for date, cash, per-symbol share counts, portfolio value, and day-over-day value change.
- `MultiStockTradingEnv` is deterministic given a fixed dataset; random seeds only affect future extensions (e.g., stochastic price perturbations).

## Known Gaps
- Provide helper utilities for stress scenarios (e.g., slippage shocks, turbulence masks) once the base trading policy is finalised.
- Add dedicated evaluation/report scripts that consume the emitted traces and metrics.
