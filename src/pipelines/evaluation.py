"""Shared evaluation helpers and deterministic rollout utilities."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv

from src.pipelines.checkpointing import load_checkpoint
from src.utils.builders import build_agent_spec, build_dataset, make_env_factory
from src.utils.config import Config


def build_trace_row(
    date_value,
    cash_value,
    holdings,
    portfolio_value,
    delta_value,
    symbols,
    *,
    nav_return: Optional[float] = None,
    turnover: Optional[float] = None,
    drawdown: Optional[float] = None,
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
        "turnover": float(turnover if turnover is not None else 0.0),
        "drawdown": float(drawdown if drawdown is not None else 0.0),
    }
    for symbol, shares in zip(symbols, holdings):
        row[str(symbol)] = int(shares)
    return row



def evaluate_full_episodes(
    model,
    vec_env,
    episodes: int,
    *,
    deterministic: bool,
    trace_writer=None,
    portfolio_state: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Roll out multiple full episodes and optionally emit per-step traces."""

    rewards: List[float] = []
    for episode_idx in range(episodes):
        if portfolio_state is not None:
            vec_env.env_method(
                "set_portfolio_state",
                copy.deepcopy(portfolio_state),
                indices=[0],
            )
        obs = vec_env.reset()
        base_env = vec_env.envs[0]
        env = getattr(base_env, "unwrapped", base_env)
        traces: List[Dict[str, float]] = []
        initial_row = build_trace_row(
            env.date_history[0],
            env.cash,
            env.holdings,
            env.initial_cash,
            0.0,
            env.symbols,
            nav_return=0.0,
            turnover=0.0,
            drawdown=0.0,
        )
        traces.append(initial_row)
        prev_value = env.initial_cash
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, rewards_vec, dones, infos = vec_env.step(action)
            info = infos[0]
            portfolio_value = float(info.get("portfolio_value", 0.0))
            delta = portfolio_value - prev_value
            prev_value = portfolio_value
            nav_return = info.get("nav_return")
            turnover = info.get("turnover")
            drawdown = info.get("drawdown")
            date_value = info.get("date", env.date_history[-1])
            trace_row = build_trace_row(
                date_value,
                info.get("cash", env.cash),
                info.get("holdings", env.holdings),
                portfolio_value,
                delta,
                env.symbols,
                nav_return=nav_return,
                turnover=turnover,
                drawdown=drawdown,
            )
            traces.append(trace_row)
            done = bool(dones[0])
        final_reward = traces[-1]["portfolio_value"] - env.initial_cash if traces else 0.0
        rewards.append(final_reward)
        if trace_writer is not None:
            trace_writer(episode_idx, traces, env)
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

    agent_spec = build_agent_spec(config_eval)
    model = agent_spec.instantiate(eval_vec_env)
    timesteps = load_checkpoint(model, Path(checkpoint_path))

    trace_frames: List[pd.DataFrame] = []
    trace_paths: List[Path] = []
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    def _trace_writer(ep_index: int, traces: List[Dict[str, float]], env) -> None:
        frame = pd.DataFrame(traces)
        trace_frames.append(frame)
        if output_dir is not None:
            file_path = output_dir / f"episode_{ep_index + 1:02d}.csv"
            frame.to_csv(file_path, index=False)
            trace_paths.append(file_path)

    rewards = evaluate_full_episodes(
        model,
        eval_vec_env,
        episodes,
        deterministic=deterministic,
        trace_writer=_trace_writer,
        portfolio_state=portfolio_state,
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
        episode_records.append(
            {
                "episode": idx + 1,
                "final_nav": end_value,
                "total_return": total_return,
                "max_drawdown": max_drawdown,
                "avg_turnover": avg_turnover,
                "final_cash": float(frame["cash"].iloc[-1]) if "cash" in frame else 0.0,
            }
        )

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

    return EvaluationResult(
        summary=summary,
        episode_summaries=episode_summaries,
        episode_frames=trace_frames,
        trace_paths=trace_paths,
        valuation_paths=valuation_paths,
        timesteps=timesteps,
    )
