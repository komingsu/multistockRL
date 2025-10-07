# Project Checklist

## Foundation
- [x] Draft initial project plan and capture architecture milestones.
- [x] Publish baseline architecture overview.
- [ ] Confirm Python environment via `requirements.txt`.
- [x] Create `src/` package skeleton (data, envs, agents, pipelines, utils).
- [x] Add `configs/base.yaml` capturing default hyperparameters, TVT splits, and feature groups.
- [x] Set up logging helper and default `.log` destination folder.
- [x] Define lightweight logging utilities for metrics + experiment runs aligned with config system.

## Environment Hardening (P0)
- [x] Switch to weight‑based actions with τ limiter and integerized execution.
- [x] Unified one‑way commission + slippage costs on both legs.
- [x] Reward: per‑step `log(V_t/V_{t-1}) − λ·turnover` with turnover from post‑return gross NAV.
- [x] Deterministic D→D+1 alignment; forbid lookahead; tests added.
- [ ] Projection: logits→simplex (cash asset) and optional ℓ1‑ball with `max_leverage`.
- [ ] Property tests: projection invariants (sum‑to‑1/leverage bound) and 100+ randomized τ scenarios.

## Data
- [x] Document schema for `data/proc/*.csv`.
- [x] Implement loader with schema validation and missing-data handling.
- [x] Publish data pipeline and preprocessing strategy documentation.
- [x] Implement sliding window dataset with per-window normalization.
- [x] Log dataset window counts for train/validation/test sanity checks.
- [x] Extend preprocessing transforms (lag features, masks, categorical gaps).
- [ ] Cache processed datasets into `artifacts/datasets/`.
- [x] Add automated schema validation on new CSV drops before training.
- [x] Build validation notebook and pytest checks for schema enforcement.
 - [x] Integrate data build CLI (`scripts/build_data.py`) with symbol master, features, selection, and turbulence (separate CSV).

## Environment
- [x] Rewrite multi-stock gym environment with commission/slippage logic scaffolded.
- [x] Implement frictions module for reusable fee/slippage math.
- [x] Add deterministic unit tests for key environment edge cases.
- [x] Expose evaluation hooks (per-day traces emitted by callbacks).
- [x] Apply sell-side-only commission and integrate slippage into `MultiStockTradingEnv`.
- [x] Emit NAV/turnover/drawdown diagnostics via `info` for richer logging.
- [x] Document environment API signatures and current limitations.
 - [x] Add `action_mode: weights|shares`, τ, `lambda_turnover`, and leverage knobs.

## Agents
- [x] Build agent factory supporting PPO/SAC/DDPG/A2C/TD3 instantiation.
- [ ] Define vetted default policy kwargs and action-noise settings for off-policy agents.
- [ ] Implement custom callbacks for risk monitoring beyond metrics logging.
- [x] Provide baseline training config for PPO with validation split.
 - [ ] Add `--algo` switch in training CLI and configs for SAC/TD3/TQC/RecurrentPPO.

## Training Pipeline
- [x] Create CLI entrypoint for training with config argument.
- [x] Implement checkpointing and resume logic.
- [x] Log enriched metrics (NAV, turnover, drawdown) alongside reward each project epoch.
- [x] Store best model snapshot per run.
- [x] Introduce project-epoch cadence for logging, evaluation, and checkpointing.
- [x] Add patience-based rollback and early stopping.
- [x] Enforce VecNormalize parity on evaluation/inference; save stats next to checkpoints.
 - [ ] Apply TimeFeatureWrapper to observations (episode progress) where valid.

## Evaluation
- [x] Emit per-episode CSV traces (validation + final eval) with holdings/cash/value/deltas.
- [x] Build standalone evaluation/report script for deterministic rollouts.
- [x] Generate performance report (returns, sharpe, drawdown, turnover).
- [x] Add stress testing hooks (commission/slippage multipliers).
- [x] Produce summary CSV/JSON artifacts for offline analysis.
 - [x] Add PSR/DSR and block/bootstrap CIs to evaluation summary.
 - [ ] Selection: implement Purged K‑Fold + embargo utilities; consider CPCV.

## Experiment Management
- [x] Standardize run directory structure under `artifacts/`.
- [x] Record resolved configs with each run.
- [ ] Capture git commit hash alongside run metadata.
- [x] Maintain agent diary after every major change.
- [ ] Review checklist weekly and reprioritize.
- [ ] Fill project milestones and keep them current as deliverables ship.

## Robustness & Deployment
- [ ] EMA (slow policy) for inference; toggle in config.
- [ ] Optional SWA over last N checkpoints; A/B compare.
- [ ] Snapshot/policy ensembles across seeds/epochs; re‑project to simplex.

## Backlog (Later Improvements)
- [ ] Model selection: Purged K‑Fold with embargo; evaluate CPCV when infra stabilizes.
- [ ] Acceptance gating: require PSR/DSR thresholds for “best” model updates (configurable).
- [ ] Hyper‑parameter sweeps (PB2/PBT) after selection metrics are in place.
- [ ] Feature work: volatility targeting, additional indicators, PIT alignment audits.
- [ ] Reporting: include CI bands in aggregated reports; add per‑episode significance flags.
