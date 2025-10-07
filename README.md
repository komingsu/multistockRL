# multistockRL — Weight‑Based Multi‑Asset RL Trader

This repository contains a daily‑bar, multi‑asset trading agent built on Stable‑Baselines3 with realistic frictions, τ‑limited allocation changes, integerized execution, and deterministic evaluation hooks.

Key properties
- Action mode: continuous allocation weights with τ‑limiter and integerized execution inside the environment.
- Costs: one‑way commission + slippage applied on both legs; reward subtracts an L1 turnover penalty.
- Evaluation: deterministic multi‑episode rollouts with patience‑gated improvement tracking, rollback, and early stopping.
- Normalization: VecNormalize supported and enforced for eval/inference when enabled.

Quick start
- Train (debug): `python -m scripts.train --config configs/debug_smoke.yaml --run-name debug_weights`
- Evaluate a checkpoint: `python -m scripts.evaluate --config <yaml> --checkpoint <run>/checkpoints/step_xxxxxx/model.pt`

Repository structure (annotated)

```
.
├── Agents.md                      # Agent operating guide and daily workflow
├── agent_diary/                   # Daily logs, decisions, and outcomes
├── artifacts/                     # Run directories (configs, metrics, checkpoints, traces)
├── configs/                       # YAML configs controlling data/env/agents/training
├── data/                          # Local data (processed CSVs, metadata)
├── docs/                          # Architecture, plans, checklists, API docs
├── notebooks/                     # Exploratory notebooks (data validation, etc.)
├── requirements.txt               # Python dependencies
├── sample_code/
│   ├── env.py                     # Example environment sketch (# demo only)
│   └── models.py                  # Example policy sketch (# demo only)
├── scripts/
│   ├── train.py                   # CLI: training entrypoint (# wraps pipelines.training)
│   ├── evaluate.py                # CLI: deterministic evaluation of checkpoints
│   └── report.py                  # CLI: summarize run metrics and traces
├── src/                           # Source package
│   ├── __init__.py                # Package marker
│   ├── agents/
│   │   ├── __init__.py            # Package exports
│   │   └── factory.py             # Build SB3 agent specs from config
│   ├── env/
│   │   ├── __init__.py            # Public env exports
│   │   ├── adapters.py            # Windowed data adapter (# optional)
│   │   ├── costs.py               # Commission/slippage aggregation helpers
│   │   ├── frictions.py           # Friction config + cost primitives
│   │   ├── multi_stock_env.py     # Core trading env (weights/τ/integerization/reward)
│   │   └── rewards.py             # Reward shaping utilities (# optional)
│   ├── loggers/
│   │   ├── __init__.py            # Package exports
│   │   └── experiment.py          # CSV/YAML logger for metrics/events/config snapshots
│   ├── pipelines/
│   │   ├── __init__.py            # Pipeline exports
│   │   ├── checkpointing.py       # Save/load model + metadata (Torch dicts)
│   │   ├── evaluation.py          # Deterministic rollouts, trace/plot emitters
│   │   ├── inference.py           # Inference wrapper + CLI helpers
│   │   ├── normalization.py       # VecNormalize save/load/clone helpers
│   │   └── training.py            # Training loop, patience/rollback, logging
│   └── utils/
│       ├── __init__.py            # Utility exports
│       ├── builders.py            # Factories to wire config->dataset/env/agent/logger
│       ├── config.py              # Config dataclass + YAML loader
│       └── portfolio.py           # τ limiter, integerization, payload validation
└── tests/                         # Pytest suites (env, portfolio, pipeline, etc.)
```

Notes on .py files
- Lines above include inline `#` one‑liners describing each file’s purpose.
- For deeper interfaces and examples, see `docs/architecture.md` and `docs/environment_api.md`.

