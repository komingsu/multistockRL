# Agent Operating Guide

## Mission Context
- **Goal**: maintain and improve a weight-based multi-asset PPO trader over daily bars, with realistic frictions (0.5 % commission + 0.5 % slippage) and deterministic evaluation hooks.
- **Highlights**  
  - Actions: continuous allocation weights passed through a τ-limiter (default 0.08) and integerized execution inside `MultiStockTradingEnv`.  
  - Reward: per-step `log(V_t/V_{t-1}) - λ·turnover` (default λ = 0.03) accumulated per episode for training logs.  
  - Evaluation: averages three deterministic episodes, records `{PnL, percent return}` only when hitting a new validation peak, and enforces patience-based rollback / early stopping.

## Daily Workflow
| Phase | Actions |
| --- | --- |
| **Before** | Read the latest diary entry (e.g., `agent_diary/YYYY-MM-DD.md`) and update the plan tool. Check `git status`, confirm configs/data referenced today exist. |
| **During** | Keep the plan in sync, rely on logging (no stray prints), validate assumptions with targeted probes, and record key design decisions in the diary. |
| **After** | Run relevant smoke tests (`python -m pytest`) and always run a 2‑epoch debug train (`python -m scripts.train --config configs/debug_two_epoch.yaml --run-name <alias>`), capture artefacts/plots, update project milestones, summarise outcomes in the diary, and stage/commit if stable. |

Reference: `docs/agent_checklist.md` for the detailed tick list.

## Core Commands
- **Tests**: `python -m pytest` (CI bar; finishes in ≈15 s).  
- **Smoke PPO run (debug)**: `python -m scripts.train --config configs/debug_smoke.yaml --run-name <alias>`  
  - Uses 5 epochs, eval every epoch, patience/rollback toggled, continuous weights.  
- **Full PPO run**: `python -m scripts.train --config configs/full_training.yaml --run-name <alias>`  
- **Deterministic evaluation**: `python -m scripts.evaluate --config <yaml> --checkpoint <artifact>/checkpoints/step_xxxxxx/model.pt`  
  - Includes percent-return metrics; traces/plots appear only for runs that improved the validation peak.

*Tip*: SB3 still checks CUDA; we explicitly set `device: cpu` in configs to avoid GPU usage and warnings.

## Configuration Notes
- YAML under `configs/` controls data splits, environment, and agent hyperparameters.
  - `environment.action_mode: weights`, `tau`, `lambda_turnover`, and friction parameters must stay consistent across base/debug/full profiles.
  - `training.vecnormalize.enabled: true` ensures VecNormalize stats are captured beside checkpoints; mismatched stats refuse evaluation.
  - Early stopping knobs live under `training`: `patience`, `rollback_patience`, `early_stop_patience`.
- Always commit config changes alongside diary updates explaining the rationale.

## Logging & Artifacts
- Every run creates `artifacts/<timestamp>_<run>/` containing:
  - `config.yaml`, `metrics.csv`, `<date>_<run>.log` (human-readable event stream).  
  - `validation_traces/` & valuation plots **only** when validation PnL hits a new peak.  
  - `checkpoints/` (per-epoch) and `checkpoints/best/` for best models; VecNormalize stats saved as `vecnormalize.pkl`.  
  - `final_eval_traces/` holding the concluding evaluation rollouts.
- Inspect logs for lines such as `Validation @ project epoch ... return=...` to monitor percent returns and patience behaviour (`Rolled back ...`, `Early stopping triggered ...`).

## Code Touchpoints
- **Environment**: `src/env/multi_stock_env.py` houses the weight-mode execution and log-return reward logic.  
- **Transaction costs**: unified helper `src/env/costs.py`.  
- **Portfolio utilities**: `src/utils/portfolio.py` (`apply_tau_limit`, `integerize_allocations`, validation).  
- **Training pipeline**: `src/pipelines/training.py` orchestrates PPO, cumulative reward logging, evaluation patience, and rollout of best checkpoints on regression.  
- **Evaluation CLI**: `src/pipelines/evaluation.py` generates traces/plots and aggregates episode summaries.  
- **Experiment logger**: `src/loggers/experiment.py` writes metrics and event timelines per run.

## When Enhancing or Debugging
- Use the debug config first; it surfaces evaluation gating quickly (look for “Saved evaluation valuation plot” only on improving epochs).
- If validation stalls: inspect `validation_traces/` around the last improvement, check turnover/cost ratios in `metrics.csv`, and adjust `tau`, `λ`, or PPO hyperparameters.
- Any new feature (indicators, alternative agents) must be accompanied by tests and diary notes.

## Documentation & Project Roadmap
- Architectural and project expectations live in `docs/architecture.md` and `docs/project_plan.md`.  
- `agent_diary/YYYY-MM-DD.md` holds session outcomes and next steps; keep it current before ending a work block.

## Merging & Releases
- Branch: `chore/2025-10-07-diary-plan` consolidates the latest environment/evaluation changes.
- On merge readiness:
  1. Ensure `python -m pytest` passes.
  2. Package the smoke run artefacts if they support the change rationale.
  3. Summarize modifications in the diary and commit with an informative message.
  4. Open a PR targeting `main`, referencing the diary outcomes and expected impact.
