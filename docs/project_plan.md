# Project Plan

## Goal
Build a multi-asset reinforcement learning trader that operates on daily bars, models transaction costs and slippage, and can be iterated quickly for performance tuning.

## Scope and Assumptions
- Single developer workload, no external services beyond local files.
- Data lives under `data/proc/` and already includes technical indicators.
- Further preprocessing, model swaps, additional indicators, ensemble logic, and training strategies must remain easy to plug in.
- Logging flows through `src/loggers.ExperimentLogger` into run-scoped directories (CSV + YAML snapshots).

## Guiding Principles
1. Keep modules short and readable; prefer pure functions where possible.
2. Hide experiment hyperparameters inside versioned config files.
3. Encapsulate brokerage frictions (fees, slippage, position limits) inside the environment layer.
4. Favor deterministic preprocessing so experiments are reproducible.
5. Provide lightweight hooks for evaluation and reporting without full-scale dashboards.

## Workstreams
1. **Data Pipeline**
   - Validate column schema, missing values, and symbol coverage.
   - Implement preprocessing transforms (normalization, feature windows, masks) with reusable routines.
2. **Environment**
   - Rewrite gym environment to handle vectorized multi-stock positions, order sizing, and friction models.
   - Add scenario randomization hooks (fee shocks, slippage sampling, trading halts).
3. **Agent & Policy**
   - Implement modular policy factory that can return different SB3 (and future custom) agents.
   - Support curriculum components (warm start, pretraining, offline rollouts) as optional steps.
4. **Training Orchestrator**
   - CLI script that loads data, rebuilds/resumes envs, trains agents, evaluates on holdout periods, and saves artifacts.
   - Emit structured .log files with end-of-run summaries, metrics.csv telemetry, and torch checkpoints bundling config + model state for resumability.
5. **Evaluation & Analytics**
   - Generate plots/tables offline (Matplotlib, Pandas) and save to `reports/`.
   - Include Monte-Carlo stress tests and bootstrapped performance intervals.
6. **Experiment Management**
   - Store configs under `configs/` (YAML/JSON).
   - Provide helper to stamp run metadata and log file paths.

## Near-Term Milestones
- [x] **M0** (2025-10-03): Confirm data schema and friction parameters (fees, slippage, position limits).
- [x] **M1**: Deliver rewritten `MultiStockTradingEnv` with deterministic step logic, friction hooks scaffolded, and unit coverage.
- [ ] **M2**: Produce baseline PPO training run end-to-end with saved checkpoints and evaluation report.
- [ ] **M3**: Add hyperparameter sweep harness and ensemble evaluation utilities.
- [ ] **M4**: Document tuning heuristics and failure recovery playbook.

## Deliverables
- Modular source tree under `src/` with unit-testable components.
- Config files for at least PPO, SAC, and a placeholder custom policy.
- Automated scripts: `train.py`, `evaluate.py`, `make_dataset.py`.
- Documentation set: architecture overview, checklist, agent diary.
- Logging conventions with sample metric exports under `artifacts/`.

## Risks & Mitigations
- **Data drift or gaps**: include schema checks and timeline assertions pre-training.
- **Instability from slippage modeling**: keep random seeds logged; add clamping on position deltas.
- **Experiment debt**: update agent diary after each session; prune stale configs.
- **Runtime failures**: wrap training loop with try/except that writes errors to log and safe checkpoints.

## Next Actions
1. Extend reporting to compute additional analytics (Sharpe, volatility, downside risk) on evaluation outputs.
2. Capture git commit metadata and runtime context in each run directory.
3. Prepare hyperparameter sweep utilities for PPO/SAC and document recommended search spaces.
4. Implement optional scenario injectors (turbulence halts, price shocks) leveraging the new stress hooks.




