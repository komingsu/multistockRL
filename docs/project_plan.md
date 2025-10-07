# Project Plan

## Goal
Build a weight‑based, multi‑asset trader on daily bars with realistic frictions (commission + slippage), τ‑limited allocation changes, integerized execution, and deterministic evaluation with patience/rollback.

## Scope and Assumptions
- Single‑developer workflow; local files only.
- Processed data under `data/` with indicators available; additional transforms remain plug‑in.
- All behavior is config‑driven; logging flows through `src/loggers/experiment.py` into run‑scoped `artifacts/`.

## Guiding Principles
1. Action mapping in env; agents remain algorithmic plug‑ins.
2. Deterministic evaluation; improvements only recorded on new validation peaks.
3. Reproducibility: versioned configs, VecNormalize stats saved/loaded, CPU default.
4. Simple, testable modules; focused unit/property tests for τ limiter, costs, alignment.

## Workstreams (Status)
1. Data Pipeline — done (schema, loader, windowing, preprocessing docs)
2. Environment — done (weights mode + τ limiter + integerization; unified costs; turnover‑aware log‑return reward; D→D+1 alignment)
3. Training Orchestrator — done (project epochs; patience‑gated eval; rollback/early‑stop; checkpoints + VecNormalize)
4. Evaluation & Analytics — in progress (deterministic rollouts, traces/plots; add PSR/DSR + bootstrap CIs)
5. Baselines — in progress (PPO baseline done; add SAC/TD3/TQC/RecurrentPPO via `--algo`; default wrappers: VecNormalize + TimeFeatureWrapper)
6. Risk/Projection — next (simplex/ℓ1‑ball projection with cash asset; leverage bounds)
7. Selection/Validation — next (Purged K‑Fold + embargo; CPCV later)
8. Averaging/Ensembles — next (EMA/SWA for inference; snapshot ensembles)

## Milestones
- [x] M0 (2025‑10‑03): Confirm data schema and friction parameters.
- [x] M1: Weight‑based env with τ limiter, integerization, log‑return – λ·turnover reward; unit tests.
- [x] M2: Baseline PPO end‑to‑end with checkpoints, deterministic eval, validation‑peak gating, and rollback.
- [ ] M3: Multi‑algo baseline switch (SAC/TD3/TQC/RecurrentPPO) behind config/CLI; enable TimeFeatureWrapper by default.
- [ ] M4: Selection with Purged K‑Fold + embargo; add PSR/DSR + block/bootstrap CIs to reports.
- [ ] M5: EMA “slow policy” for inference and optional SWA over final checkpoints.
- [ ] M6: Simplex/ℓ1‑ball projection (cash asset, leverage cap) + tests.
- [ ] M7: Hyperparameter sweep harness and ensemble utilities.

## Deliverables
- Modular source tree under `src/` with unit‑testable components.
- Configs for PPO (done); SAC/TD3/TQC/RPPO (pending), with VecNormalize + TimeFeatureWrapper defaults.
- Scripts: `train.py`, `evaluate.py`, `report.py` (done).
- Docs: architecture, environment API, plan/checklist, agent diary (ongoing).
- Artifacts: metrics/logs/checkpoints, traces, and valuation plots for new validation peaks.

## Risks & Mitigations
- Data drift/gaps → schema checks and split assertions pre‑training.
- Cost/τ instability → property tests for τ limiter and projection invariants.
- Eval leakage/selection bias → Purged K‑Fold with embargo; CPCV later.
- Reproducibility → enforce VecNormalize parity at eval/inference.

## Next Actions (Sequenced)
1. P0: Projection layer (simplex + ℓ1‑ball w/ leverage); unit/property tests; integrate before τ.
2. P1: Baseline switch: add SAC/TD3/TQC/RPPO via `agents.factory` and config `--algo`.
3. P2: Metrics: implement PSR/DSR + block/bootstrap CIs; persist into `evaluation` outputs.
4. P2: Splits: Purged K‑Fold + embargo utilities; wire into model selection.
5. P3: EMA/SWA toggles; inference uses EMA by default; document.
6. P4: Feature set & vol targeting; audit PIT alignment and survivorship.





