"""Shared evaluation helpers and deterministic rollout utilities."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv

from src.pipelines.checkpointing import load_checkpoint
from src.pipelines.normalization import (
    load_vecnormalize_stats,
    stats_path_for_checkpoint,
    vecnormalize_enabled,
)
from src.utils.builders import build_agent_spec, build_dataset, make_env_factory
from src.utils.config import Config
from src.utils.portfolio import integerize_allocations, validate_holdings_payload
from src.utils.metrics import (
    CI,
    compute_dsr,
    compute_psr,
    compute_sharpe,
    moving_block_bootstrap_ci,
)


def build_trace_row(
    date_value,
    cash_value,
    holdings,
    portfolio_value,
    delta_value,
    symbols,
    *,
    nav_return: Optional[float] = None,
    log_return: Optional[float] = None,
    turnover: Optional[float] = None,
    drawdown: Optional[float] = None,
    weights: Optional[Sequence[float]] = None,
    cash_fraction: Optional[float] = None,
    q_port: Optional[float] = None,
    q_sys: Optional[float] = None,
    regime: Optional[str] = None,
    g_max: Optional[float] = None,
    safety_weight_throttle: Optional[bool] = None,
    safety_quantity_scale: Optional[float] = None,
    buy_freeze_hits: Optional[int] = None,
    cap_hit_ratio: Optional[float] = None,
) -> Dict[str, float]:
    """Construct a CSV-friendly row describing portfolio state."""

    if hasattr(date_value, "isoformat"):
        date_str = date_value.isoformat()
    else:
        date_str = str(date_value)
    row: Dict[str, float] = {
        "date": date_str,
        "cash": float(cash_value),
        "portfolio_value": float(portfolio_value),
        "portfolio_value_change": float(delta_value),
        "nav_return": float(nav_return if nav_return is not None else 0.0),
        "log_return": float(log_return if log_return is not None else 0.0),
        "turnover": float(turnover if turnover is not None else 0.0),
        "drawdown": float(drawdown if drawdown is not None else 0.0),
    }
    for symbol, shares in zip(symbols, holdings):
        row[str(symbol)] = int(shares)
    if weights is not None:
        for symbol, weight in zip(symbols, weights):
            row[f"weight_{symbol}"] = float(weight)
    if cash_fraction is not None:
        row["cash_fraction"] = float(cash_fraction)
    if q_port is not None:
        row["q_port"] = float(q_port)
    if q_sys is not None:
        row["q_sys"] = float(q_sys)
    if g_max is not None:
        row["g_max"] = float(g_max)
    if regime is not None:
        row["regime"] = str(regime)
    if safety_weight_throttle is not None:
        row["safety_weight_throttle"] = bool(safety_weight_throttle)
    if safety_quantity_scale is not None:
        row["safety_quantity_scale"] = float(safety_quantity_scale)
    if buy_freeze_hits is not None:
        row["buy_freeze_hits"] = int(buy_freeze_hits)
    if cap_hit_ratio is not None:
        row["cap_hit_ratio"] = float(cap_hit_ratio)
    return row


def _prepare_portfolio_override(
    env,
    base_state: Optional[Dict[str, Any]],
    *,
    integerize: bool,
    lot_size: int,
) -> tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    if base_state is None:
        return None, {}
    state = copy.deepcopy(base_state)
    metadata: Dict[str, Any] = {}

    if integerize:
        allocations = (
            state.pop("allocations", None)
            or state.pop("weights", None)
            or state.pop("target_weights", None)
        )
        if allocations is None:
            raise ValueError(
                "integerize=True requires 'allocations' (or 'weights') in the portfolio_state."
            )
        nav_value = (
            state.get("nav")
            or state.get("portfolio_value")
            or state.get("initial_nav")
        )
        if nav_value is None:
            raise ValueError(
                "integerize=True requires a 'nav' or 'portfolio_value' field in the portfolio_state."
            )
        start_idx = env.resolve_portfolio_start_index(state)
        price_map = env.price_map(start_idx)
        holdings, residual_cash, exec_weights = integerize_allocations(
            allocations,
            price_map,
            nav=float(nav_value),
            symbols=env.symbols,
            lot_size=lot_size,
        )
        state["holdings"] = holdings
        state["cash"] = residual_cash
        metadata["initial_executed_weights"] = exec_weights
        metadata["initial_residual_cash"] = residual_cash
        metadata["nav"] = float(nav_value)
    elif "holdings" not in state:
        raise ValueError("Portfolio state must include 'holdings' when integerize is False.")

    validate_holdings_payload(env.symbols, state["holdings"])
    state["holdings"] = {str(sym): int(state["holdings"][sym]) for sym in env.symbols}
    if "cash" not in state:
        raise ValueError("Portfolio state must include 'cash'.")
    state["cash"] = float(state["cash"])

    return state, metadata


def evaluate_full_episodes(
    model,
    vec_env,
    episodes: int,
    *,
    deterministic: bool,
    trace_writer=None,
    portfolio_state: Optional[Dict[str, Any]] = None,
    integerize: bool = False,
    positions_writer: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    lot_size: int = 1,
) -> np.ndarray:
    """Roll out multiple full episodes and optionally emit per-step traces."""

    rewards: List[float] = []
    base_state = copy.deepcopy(portfolio_state) if portfolio_state is not None else None

    for episode_idx in range(episodes):
        base_env = vec_env.envs[0]
        env = getattr(base_env, "unwrapped", base_env)
        override_state, metadata = _prepare_portfolio_override(
            env,
            base_state,
            integerize=integerize,
            lot_size=lot_size,
        )
        if override_state is not None:
            vec_env.env_method("set_portfolio_state", override_state, indices=[0])

        obs = vec_env.reset()
        traces: List[Dict[str, float]] = []

        initial_prices = np.asarray(env.price_matrix[env.current_idx], dtype=np.float64)
        initial_holdings = np.asarray(env.holdings, dtype=np.float64)
        initial_nav = float(env.initial_cash)
        if initial_nav > 0:
            initial_weights = (initial_holdings * initial_prices) / initial_nav
            initial_cash_fraction = float(env.cash) / initial_nav
        else:
            initial_weights = None
            initial_cash_fraction = None

        initial_row = build_trace_row(
            env.date_history[0],
            env.cash,
            env.holdings,
            env.initial_cash,
            0.0,
            env.symbols,
            nav_return=0.0,
            log_return=0.0,
            turnover=0.0,
            drawdown=0.0,
            weights=initial_weights.tolist() if initial_weights is not None else None,
            cash_fraction=initial_cash_fraction,
            q_port=getattr(env, "_turb_map", {}).get(env.date_history[0], (np.nan, np.nan))[0] if hasattr(env, "_turb_map") else None,
            q_sys=getattr(env, "_turb_map", {}).get(env.date_history[0], (np.nan, np.nan))[1] if hasattr(env, "_turb_map") else None,
            regime=getattr(env, "_regime_state", None),
            g_max=getattr(env, "_current_gmax", None),
        )
        traces.append(initial_row)
        prev_value = env.initial_cash
        done = False
        last_info: Optional[Dict[str, Any]] = None

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, rewards_vec, dones, infos = vec_env.step(action)
            info = infos[0]
            last_info = info
            portfolio_value = float(info.get("portfolio_value", 0.0))
            delta = portfolio_value - prev_value
            prev_value = portfolio_value
            nav_return = info.get("nav_return")
            log_return = info.get("log_return")
            turnover = info.get("turnover")
            drawdown = info.get("drawdown")
            date_value = info.get("date", env.date_history[-1])

            prices_step = np.asarray(
                info.get("prices", env.price_matrix[env.current_idx]),
                dtype=np.float64,
            )
            holdings_step = np.asarray(
                info.get("holdings", env.holdings),
                dtype=np.float64,
            )
            weights_payload = info.get("weights")
            if weights_payload is not None:
                weights_step = np.asarray(weights_payload, dtype=np.float64)
            else:
                weights_step = (holdings_step * prices_step) / portfolio_value if portfolio_value > 0 else None
            cash_fraction = info.get("cash_fraction")
            if cash_fraction is None and portfolio_value > 0:
                cash_fraction = float(info.get("cash", env.cash)) / portfolio_value

            trace_row = build_trace_row(
                date_value,
                info.get("cash", env.cash),
                holdings_step,
                portfolio_value,
                delta,
                env.symbols,
                nav_return=nav_return,
                log_return=log_return,
                turnover=turnover,
                drawdown=drawdown,
                weights=weights_step.tolist() if weights_step is not None else None,
                cash_fraction=cash_fraction,
                q_port=info.get("q_port"),
                q_sys=info.get("q_sys"),
                regime=info.get("regime"),
                g_max=info.get("g_max"),
                safety_weight_throttle=info.get("safety_weight_throttle"),
                safety_quantity_scale=info.get("safety_quantity_scale"),
                buy_freeze_hits=info.get("buy_freeze_hits"),
                cap_hit_ratio=info.get("cap_hit_ratio"),
            )
            traces.append(trace_row)
            done = bool(dones[0])

        final_reward = traces[-1]["portfolio_value"] - env.initial_cash if traces else 0.0
        rewards.append(final_reward)
        if trace_writer is not None:
            trace_writer(episode_idx, traces, env)

        if positions_writer is not None:
            if last_info is None:
                last_info = {
                    "holdings": env.holdings,
                    "prices": env.price_matrix[env.current_idx],
                    "cash": env.cash,
                    "portfolio_value": env.asset_history[-1],
                }
            final_holdings = np.asarray(last_info.get("holdings", env.holdings), dtype=np.int64)
            final_prices = np.asarray(
                last_info.get("prices", env.price_matrix[env.current_idx]),
                dtype=np.float64,
            )
            final_cash = float(last_info.get("cash", env.cash))
            final_nav = float(last_info.get("portfolio_value", env.asset_history[-1]))
            if final_nav > 0:
                executed_weights = {
                    symbol: float(price * qty) / final_nav
                    for symbol, price, qty in zip(env.symbols, final_prices, final_holdings)
                }
            else:
                executed_weights = {symbol: 0.0 for symbol in env.symbols}
            summary_payload = {
                "episode": episode_idx + 1,
                "holdings": {symbol: int(qty) for symbol, qty in zip(env.symbols, final_holdings)},
                "weights": executed_weights,
                "cash_end": final_cash,
            }
            if metadata:
                summary_payload["integerize"] = metadata
            positions_writer(episode_idx, summary_payload)

    return np.asarray(rewards, dtype=float)



def _render_valuation_curves(
    trace_frames: List[pd.DataFrame],
    output_dir: Path,
    *,
    file_prefix: str = "episode",
) -> List[Path]:
    """Render NAV curves per episode, annotating the peak value."""

    if not trace_frames:
        return []

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "matplotlib is required to render valuation curves. Install it or disable plotting."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    valuation_paths: List[Path] = []
    for episode_idx, frame in enumerate(trace_frames, start=1):
        if frame.empty or "portfolio_value" not in frame:
            continue
        data = frame.reset_index(drop=True).copy()
        data["date"] = pd.to_datetime(data["date"])

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(data["date"], data["portfolio_value"], label="NAV", color="#1f77b4")
        # Annotate turbulence & throttling
        if "q_sys" in data.columns:
            try:
                ax2 = ax.twinx()
                ax2.plot(data["date"], data["q_sys"], color="#ff7f0e", alpha=0.4, label="turb_sys")
                ax2.set_ylim(0.0, 1.05)
                ax2.set_ylabel("turbulence")
            except Exception:
                pass
        # Highlight throttling days
        if "buy_freeze_hits" in data.columns or "safety_quantity_scale" in data.columns:
            for i, row in data.iterrows():
                mark = False
                if "buy_freeze_hits" in data.columns and float(row.get("buy_freeze_hits", 0)) > 0:
                    mark = True
                if "safety_quantity_scale" in data.columns and float(row.get("safety_quantity_scale", 1.0)) < 0.999:
                    mark = True
                if mark:
                    ax.axvline(pd.to_datetime(row["date"]), color="#d62728", alpha=0.15, linestyle=":")

        peak_row = data["portfolio_value"].idxmax()
        if peak_row is not None and not pd.isna(peak_row):
            peak_value = float(data.loc[peak_row, "portfolio_value"])
            peak_date = pd.to_datetime(data.loc[peak_row, "date"]).to_pydatetime()
            ax.scatter(
                [peak_date],
                [peak_value],
                color="#d62728",
                marker="o",
                label="Peak NAV",
            )
            ax.annotate(
                f"Peak {peak_value:,.2f}\n{peak_date.date()}",
                xy=(peak_date, peak_value),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=9,
                color="#d62728",
                bbox={"boxstyle": "round,pad=0.3", "fc": "#ffe8e8", "ec": "#d62728"},
            )

        ax.set_title(f"Episode {episode_idx} NAV")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.autofmt_xdate()
        fig.tight_layout()

        file_path = output_dir / f"{file_prefix}_{episode_idx:02d}_valuation.png"
        fig.savefig(file_path, dpi=150)
        plt.close(fig)
        valuation_paths.append(file_path)

    return valuation_paths


@dataclass(slots=True)
class EvaluationResult:
    """Container describing a deterministic evaluation run."""

    summary: Dict[str, float]
    episode_summaries: pd.DataFrame
    episode_frames: List[pd.DataFrame]
    trace_paths: List[Path]
    position_records: List[Dict[str, Any]]
    position_paths: List[Path]
    valuation_paths: List[Path]
    timesteps: int


def run_evaluation(
    config: Config,
    checkpoint_path: Path,
    *,
    eval_split: str = "validation",
    episodes: int = 1,
    deterministic: bool = True,
    output_dir: Optional[Path] = None,
    stress_overrides: Optional[Dict[str, Any]] = None,
    portfolio_state: Optional[Dict[str, Any]] = None,
    render_nav_curves: bool = False,
    integerize: bool = False,
    lot_size: int = 1,
) -> EvaluationResult:
    """Load a checkpoint and run deterministic evaluation episodes."""

    config_eval = copy.deepcopy(config)
    if stress_overrides:
        stress = dict(config_eval.environment.get("stress", {}))
        stress.update(stress_overrides)
        stress.setdefault("enabled", True)
        config_eval.environment["stress"] = stress

    dataset = build_dataset(config_eval)
    eval_factory = make_env_factory(
        config_eval,
        split=eval_split,
        dataset=dataset,
        max_steps_override=None,
    )
    eval_vec_env = DummyVecEnv([eval_factory])

    training_cfg = config_eval.training_settings() if hasattr(config_eval, "training_settings") else {}
    vecnormalize_cfg = training_cfg.get("vecnormalize")
    expects_normalizer = vecnormalize_enabled(vecnormalize_cfg)
    stats_path = stats_path_for_checkpoint(checkpoint_path)
    stats_present = stats_path.exists()
    if expects_normalizer:
        eval_vec_env = load_vecnormalize_stats(stats_path, eval_vec_env, training=False)
    elif stats_present:
        raise RuntimeError(
            "VecNormalize stats found beside checkpoint but config.vecnormalize.enabled is false; "
            "the checkpoint was trained with normalization and requires matching stats."
        )

    agent_spec = build_agent_spec(config_eval)
    model = agent_spec.instantiate(eval_vec_env)
    timesteps = load_checkpoint(model, Path(checkpoint_path))

    trace_frames: List[pd.DataFrame] = []
    trace_paths: List[Path] = []
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    position_records: List[Dict[str, Any]] = []
    position_paths: List[Path] = []

    def _trace_writer(ep_index: int, traces: List[Dict[str, float]], env) -> None:
        frame = pd.DataFrame(traces)
        trace_frames.append(frame)
        if output_dir is not None:
            file_path = output_dir / f"episode_{ep_index + 1:02d}.csv"
            frame.to_csv(file_path, index=False)
            trace_paths.append(file_path)

    def _positions_writer(ep_index: int, payload: Dict[str, Any]) -> None:
        position_records.append(payload)
        if output_dir is not None:
            file_path = output_dir / f"episode_{ep_index + 1:02d}_positions.json"
            with file_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            position_paths.append(file_path)

    rewards = evaluate_full_episodes(
        model,
        eval_vec_env,
        episodes,
        deterministic=deterministic,
        trace_writer=_trace_writer,
        portfolio_state=portfolio_state,
        integerize=integerize,
        positions_writer=_positions_writer,
        lot_size=lot_size,
    )

    eval_vec_env.close()

    if not trace_frames:
        trace_frames = []

    valuation_paths: List[Path] = []
    if render_nav_curves and trace_frames:
        chart_root = output_dir if output_dir is not None else checkpoint_path.parent
        chart_dir = chart_root / "valuation_plots"
        valuation_paths = _render_valuation_curves(trace_frames, chart_dir)

    episode_records: List[Dict[str, float]] = []
    final_navs: List[float] = []
    total_returns: List[float] = []
    max_drawdowns: List[float] = []
    avg_turnovers: List[float] = []
    all_nav_returns: List[float] = []

    for idx, frame in enumerate(trace_frames):
        start_value = float(frame["portfolio_value"].iloc[0]) if not frame.empty else 0.0
        end_value = float(frame["portfolio_value"].iloc[-1]) if not frame.empty else 0.0
        final_navs.append(end_value)
        total_return = (end_value - start_value) / start_value if start_value else 0.0
        total_returns.append(total_return)
        max_drawdown = float(frame["drawdown"].max()) if "drawdown" in frame else 0.0
        max_drawdowns.append(max_drawdown)
        avg_turnover = float(frame.get("turnover", pd.Series([0.0])).mean()) if not frame.empty else 0.0
        avg_turnovers.append(avg_turnover)
        position_path = position_paths[idx] if idx < len(position_paths) else None
        episode_records.append(
            {
                "episode": idx + 1,
                "final_nav": end_value,
                "total_return": total_return,
                "max_drawdown": max_drawdown,
                "avg_turnover": avg_turnover,
                "final_cash": float(frame["cash"].iloc[-1]) if "cash" in frame else 0.0,
                "position_json": str(position_path) if position_path is not None else "",
            }
        )
        if "nav_return" in frame:
            nav_ret = frame["nav_return"].astype(float).to_numpy()
            # First row is initial (0); include all to preserve length
            all_nav_returns.extend(nav_ret.tolist())

    episode_summaries = pd.DataFrame(episode_records)

    summary: Dict[str, float] = {
        "episodes": float(episodes),
        "mean_final_reward": float(rewards.mean()) if rewards.size else 0.0,
        "std_final_reward": float(rewards.std()) if rewards.size else 0.0,
        "checkpoint_timesteps": float(timesteps),
    }
    if final_navs:
        summary.update(
            {
                "mean_final_nav": float(np.mean(final_navs)),
                "std_final_nav": float(np.std(final_navs)),
                "mean_total_return": float(np.mean(total_returns)),
                "worst_total_return": float(np.min(total_returns)),
                "mean_max_drawdown": float(np.mean(max_drawdowns)),
                "max_drawdown": float(np.max(max_drawdowns)),
                "mean_turnover": float(np.mean(avg_turnovers)),
            }
        )

    # Statistical reporting (Sharpe, PSR/DSR, bootstrap CIs)
    if all_nav_returns:
        arr = np.asarray(all_nav_returns, dtype=float)
        sharpe = compute_sharpe(arr, annualize=True, periods=252)
        psr0 = compute_psr(arr, s_benchmark=0.0, annualize=True, periods=252)
        # Trials default: number of episodes as a conservative proxy (can be overridden in training integration)
        dsr = compute_dsr(arr, trials=max(1, int(episodes)), corr=0.0, annualize=True, periods=252)
        sr_ci = moving_block_bootstrap_ci(arr, stat_fn=lambda x: compute_sharpe(x, annualize=True), block_len=5, reps=400, alpha=0.05, random_state=123)
        tr_ci = moving_block_bootstrap_ci(arr, stat_fn=lambda x: float(np.nanmean(x)), block_len=5, reps=400, alpha=0.05, random_state=321)
        summary.update(
            {
                "sharpe": float(sharpe),
                "psr0": float(psr0),
                "dsr": float(dsr),
                "sharpe_ci_low": float(sr_ci.low),
                "sharpe_ci_high": float(sr_ci.high),
                "mean_return_ci_low": float(tr_ci.low),
                "mean_return_ci_high": float(tr_ci.high),
            }
        )

    return EvaluationResult(
        summary=summary,
        episode_summaries=episode_summaries,
        episode_frames=trace_frames,
        trace_paths=trace_paths,
        position_records=position_records,
        position_paths=position_paths,
        valuation_paths=valuation_paths,
        timesteps=timesteps,
    )
