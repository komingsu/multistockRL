# Architecture Overview

## High-Level Diagram
1. **Data Layer** -> loads processed CSV panels, validates schema, and provides globally normalised price features.
2. **Environment Layer** -> `src/env/multi_stock_env.py` simulates portfolio cash/holdings, commission frictions, and discrete share actions.
3. **Agent Layer** -> `src/agents/` exposes factories for Stable-Baselines3 policies and custom components.
4. **Training Pipeline** -> `src/pipelines/training.py` composes data, environment, agent, logging, evaluation, and checkpoint storage.
5. **Evaluation Layer** -> validation callbacks and scripts replay deterministic episodes and summarise metrics.
6. **Experiment Utilities** -> config loaders, experiment logger, metric writers, and analysis helpers.

## Module Responsibilities
- `src/data/loader.py`: read CSV, enforce schema, build symbol panels, provide globally z-scored price tensors, expose split-aware frames.
- `src/data/features.py`: optional extra indicators, lag transforms, and masks (unused in the discrete baseline but available).
- `src/env/multi_stock_env.py`: gymnasium environment with discrete share trades (`[-n,...,0,...,+n]` per stock), globally normalised state inputs, and terminal rewards based on portfolio growth.
- `src/env/frictions.py`: reusable commission helpers (fee schedules, proportional slippage models).
- `src/agents/factory.py`: instantiate SB3 algorithms with config-driven hyper-parameters.
- `src/pipelines/training.py`: orchestrate PPO training, episode reward logging, checkpointing, and validation sweeps.
- `src/utils/config.py`: load/merge YAML configs, surface helper accessors.
- `src/loggers/experiment.py`: unified experiment logger writing configs, scalar metrics, and human-readable events.

## Data Handling
- Daily bars keyed by `date`, `symbol`, OHLCV, plus indicator columns.
- Loader caches per-split data frames and computes global per-symbol z-scores for the entire split to feed the environment.
- Observation tensors expose `[cash_norm, price_zscores..., holdings_norm...]`; rewards always use raw balance changes (no normalisation).

## Environment Mechanics
- Episodes cover the full train or validation window: the agent starts with KRW 10,000,000,000 cash, zero holdings, and trades until the final date.
- State comprises normalised cash, globally z-scored per-symbol prices, and holdings scaled by the trade lot `n`.
- Actions form a `MultiDiscrete` vector allowing sells/holds/buys up to `n` shares per stock; selling is bounded by current holdings (no shorting).
- Intermediate rewards are zero; the terminal reward equals the increase in portfolio value (`final_value - initial_cash`), optionally scaled by `reward_scaling`.
- Commission and slippage are applied only on sells (buys incur no fees); rates come from `friction.commission_rate` and `friction.slippage_bps`.

## Configuration Strategy
- YAML files under `configs/` define data splits, environment parameters (`environment.n`, `environment.initial_cash`, `reward_scaling`, `max_steps`), and agent hyper-parameters.
- Each run writes its resolved config snapshot into `artifacts/<run>/config.yaml` through the experiment logger.
- Base, debug-smoke, and full-training presets are provided; adjust trade lot size or epoch count per experiment.

## Logging & Artifacts
- Experiment logger stores metrics (with `train_reward`/`valid_reward` as primaries) and event logs in each run directory under `artifacts/`.
- Project epochs are configurable via `training.project_epoch_unit`/`project_epoch_value`; evaluation and checkpointing fire on that cadence while metrics capture `total_env_steps`, `episodes_completed`, `updates_applied`, `wall_time`, nav/turnover/drawdown, and algorithm metadata (`algorithm`, `n_steps` or `updates_per_step`).
- `scripts/evaluate.py` runs deterministic evaluations on saved checkpoints (with optional stress multipliers) and writes JSON/CSV summaries plus per-episode traces.
- `scripts/report.py` aggregates completed runs into metric summaries and daily NAV tables under `reports/`.
- Each project epoch writes per-step portfolio traces to `validation_traces/project_epoch_<k>_episode_<n>.csv`, capturing date, cash, per-symbol share counts, portfolio value, and the day-over-day value change; final evaluation episodes mirror the same schema under `final_eval_traces/`.
- Checkpoints are saved under `artifacts/<run>/checkpoints/` (with best-validation copies under `checkpoints/best/`).
- Validation callbacks record final portfolio rewards every epoch without feeding gradients back into training.

## Experiment Workflow
1. Pick a config or override fragment and set the desired trade lot (`environment.n`).
2. Launch `python -m scripts.train --config configs/<name>.yaml --run-name <alias>`.
3. Monitor console or `artifacts/<run>/metrics.csv` for terminal train/validation rewards and secondary indicators.
4. Iterate on hyper-parameters or extend reward/cost modelling as needed before longer production runs.

## Extensibility Touchpoints
- Add indicators by extending loader/feature transforms and referencing them in configs.
- Plug alternate agents by updating `src/agents/factory.py` registry.
- Adjust reward shaping by subclassing the environment or augmenting the terminal reward calculation.
- Ensemble or evaluation scripts can consume saved checkpoints thanks to the consistent logging surface.

## Current Implementation Status
- **Discrete multi-stock RL pipeline**: Implemented via `MultiStockTradingEnv` and PPO baseline; project epochs drive logging/checkpointing.
- **Episode horizon**: Episodes span the full available date range per split unless `environment.max_steps` overrides.
- **Terminal reward**: Net portfolio growth (`final_value - initial_cash`) scaled by `reward_scaling`; no intermediate rewards.
- **Evaluation traces**: Validation and final eval callbacks emit per-day CSV traces with holdings, cash, portfolio value, and daily deltas.
- **Transaction costs**: Commission and slippage apply on sells only (buys are fee-free) via `compute_commission`/`compute_slippage` inside `MultiStockTradingEnv`.

## Outstanding Gaps
- Provide deterministic evaluation/report scripts beyond the callback outputs (CLI or notebook).
- Expand agent factory defaults for SAC/DDPG/A2C/TD3 with tested hyper-parameters.
- Add stress and scenario hooks (e.g., slippage shocks) once reporting is in place.
