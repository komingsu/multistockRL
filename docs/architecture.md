# Architecture Overview

## High‑Level Diagram
1. Data Layer → loads processed CSV panels, validates schema, provides split‑aware frames.
2. Environment Layer → `src/env/multi_stock_env.py` simulates portfolio cash/holdings with weight actions, τ‑limiter, integerized execution, and realistic frictions.
3. Agent Layer → `src/agents/` exposes SB3 policies via config‑driven factories.
4. Training Pipeline → `src/pipelines/training.py` composes env/agent/logging/eval/checkpointing with patience/rollback.
5. Evaluation Layer → deterministic rollouts, trace/plot emitters, and summary metrics.
6. Experiment Utilities → config loader, logger, VecNormalize helpers, and builders.

## Module Responsibilities
- `src/utils/config.py`: config dataclass + YAML loader; split/date helpers.
- `src/utils/builders.py`: factories bridging config → dataset/env/agent/logger/monitors.
- `src/env/multi_stock_env.py`: Gymnasium env with action_mode `weights|shares`; weights normalized to simplex, τ‑limited, integerized, executed with one‑way costs; per‑step log‑return − λ·turnover reward; D→D+1 alignment.
- `src/env/costs.py` + `src/env/frictions.py`: commission/slippage primitives and aggregation.
- `src/utils/portfolio.py`: `apply_tau_limit`, integerization, and payload validation.
- `src/pipelines/training.py`: project epochs, monitoring, patience‑gated evaluation, rollback/early stop, checkpoint IO.
- `src/pipelines/evaluation.py`: deterministic evaluation, trace emission, optional valuation plots, VecNormalize enforcement.
- `src/loggers/experiment.py`: run‑scoped CSV/YAML logging of metrics, configs, and events.

## Data Handling
- Processed CSV at `data/proc/daily_with_indicators.csv` keyed by (`symbol`, `date`), with OHLCV, indicators, and metadata matching `src/data/schema.py`.
- Build pipeline:
  - `scripts/build_data.py` orchestrates raw collection (optional KIS), feature engineering, selection, and turbulence factors.
  - `src/data/ingest.py` fetches symbol master (FDR) and optional KIS daily OHLCV.
  - `src/data/feature_engineering.py` transforms raw → schema‑compatible indicators.
  - `src/data/selection.py` produces top‑N selection/weights for analysis (not required by training).
  - `src/data/turbulence.py` writes separate turbulence factors CSV (kept out of training CSV to preserve schema).
- Split‑aware frames feed the env directly; observations include prices and optional time feature; normalization via VecNormalize is recommended and enforced at eval when enabled.

## Environment Mechanics
- Episodes traverse the split window; start with configured `initial_cash` and zero holdings.
- Weights mode: policy proposes non‑negative weights (normalized to simplex), τ limiter caps half‑L1 movement, integerization converts weights to shares, then buys/sells execute with costs. Cash becomes the residual after execution (also tracked as `cash_fraction`).
- Shares mode (legacy): `MultiDiscrete` buy/hold/sell actions; retained for ablations.
- Reward per step: `log(V_t/V_{t-1}) − λ·turnover_t`; logs `nav_return`, `log_return`, `turnover`, `drawdown`, cost ratios, realized `weights`, `cash_fraction`.
- Strict time alignment: decisions use D close to trade at D+1; no lookahead.

## Configuration Strategy
- YAML under `configs/` defines data/splits/env/agents/training/logging; env exposes `action_mode`, `tau`, `lambda_turnover`, `max_leverage`, and friction settings; training exposes VecNormalize settings and patience/rollback knobs.
- Each run writes a resolved `config.yaml` under its run directory in `artifacts/`.

## Logging & Artifacts
- Metrics in `metrics.csv` and events in `*.log`; counters include project epoch, total env steps, episodes, updates, nav/turnover/drawdown.
- Deterministic evaluation runs each epoch; only new validation peaks update “best” and get valuation plots. Rollback and early‑stop managed by patience.
- Checkpoints live under `checkpoints/step_xxxxxx/model.pt` with `vecnormalize.pkl` beside them when enabled.
- `scripts/evaluate.py` and `scripts/report.py` provide standalone evaluation and reporting utilities.

## Experiment Workflow
1. Choose a config (debug/full) and run `python -m scripts.train --config configs/<name>.yaml --run-name <alias>`.
2. Monitor `artifacts/<run>/metrics.csv` and logs; validation improvements gate “best” updates.
3. Use `scripts/evaluate.py` for deterministic rollouts of checkpoints; traces and plots appear under the run directory when improvements occur.

## Extensibility Touchpoints
- Add features via data transforms and configure observation composition.
- Expand algorithms by registering in `src/agents/factory.py` and wiring configs.
- Add risk constraints via projection/penalty layers before τ/integerization.
- Enhance evaluation with PSR/DSR, bootstrap CIs, and purged/CPCV selection.

## Current Implementation Status
- Weight‑based env with τ limiter and integerized execution — implemented.
- Costs and turnover‑aware log‑return reward — implemented.
- Deterministic evaluation with patience/rollback, best‑checkpoint saving, and plots on new peaks — implemented.
- VecNormalize save/load and strict parity checks at eval/inference — implemented.
- PPO baseline + training CLI with project epochs — implemented.
- Data build CLI and feature engineering integrated; outputs remain backward‑compatible with the training schema.

## Outstanding Gaps
- Projection with cash asset (simplex long‑only) — planned (no shorting).
- Multi‑algo baselines (SAC/TD3/TQC/RecurrentPPO) — planned.
- Selection/validation: Purged K‑Fold + embargo; CPCV — planned.
- Significance metrics (PSR/DSR) and block/bootstrap CIs — planned.
