"""Training pipeline tying together config builders and Stable-Baselines agents."""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv

from src.pipelines.checkpointing import load_checkpoint, save_checkpoint
from src.pipelines.evaluation import (
    _render_valuation_curves,
    build_trace_row,
    evaluate_full_episodes,
)
from src.utils.builders import (
    build_agent_spec,
    build_dataset,
    build_logger,
    make_env_factory,
)
from src.utils.config import Config, load_config


def _clean_metrics(raw: Dict[str, Any]) -> Dict[str, float]:
    cleaned: Dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, (int, float)):
            cleaned[key] = float(value)
        elif isinstance(value, np.ndarray) and value.shape == ():
            cleaned[key] = float(value)
    return cleaned



class MonitoringCallback(BaseCallback):
    """Push SB3 training metrics into the experiment logger."""

    def __init__(self, experiment_logger, log_interval: int) -> None:
        super().__init__()
        self.experiment_logger = experiment_logger
        self.log_interval = max(1, int(log_interval))

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_interval == 0:
            metrics = _clean_metrics(dict(self.model.logger.name_to_value))
            if metrics:
                metrics["num_timesteps"] = float(self.num_timesteps)
                self.experiment_logger.log_metrics(self.num_timesteps, metrics)
        return True

    def _on_training_end(self) -> None:
        metrics = _clean_metrics(dict(self.model.logger.name_to_value))
        if metrics:
            metrics["num_timesteps"] = float(self.num_timesteps)
            self.experiment_logger.log_metrics(self.num_timesteps, metrics)





@dataclass(slots=True)
class ProjectCounters:
    total_env_steps: int = 0
    episodes_completed: int = 0
    updates_applied: int = 0
    wall_time: float = 0.0
    nav: Optional[float] = None
    turnover: Optional[float] = None
    drawdown: Optional[float] = None


@dataclass(slots=True)
class ProjectEpochConfig:
    unit: str
    value: float
    log_every: int
    eval_every: Optional[int]
    checkpoint_every: Optional[int]


class ProjectEpochTracker:
    """Stateful helper tracking project epochs and unified counters."""

    def __init__(self, config: ProjectEpochConfig) -> None:
        unit = config.unit.lower().strip()
        if unit not in {"steps", "episodes", "wall_time"}:
            raise ValueError(
                f"Unsupported project epoch unit '{config.unit}'. Expected one of steps, episodes, wall_time."
            )
        value = float(config.value)
        if value <= 0:
            raise ValueError("project_epoch_value must be positive")
        self.unit = unit
        self.value = value
        self.log_every = max(1, int(config.log_every))
        self.eval_every = None if config.eval_every is None else max(1, int(config.eval_every))
        self.checkpoint_every = (
            None if config.checkpoint_every is None else max(1, int(config.checkpoint_every))
        )
        self.counters = ProjectCounters()
        self._start_time = time.perf_counter()
        self._project_epochs = 0
        self._last_log_epoch = 0
        self._last_eval_epoch = 0
        self._last_checkpoint_epoch = 0

    @property
    def project_epochs_completed(self) -> int:
        return self._project_epochs

    @property
    def last_logged_epoch(self) -> int:
        return self._last_log_epoch

    def record_step(self, env_steps: int, *, updates_applied: Optional[int] = None, nav: Optional[float] = None, turnover: Optional[float] = None, drawdown: Optional[float] = None) -> None:
        env_steps_int = int(env_steps)
        if env_steps_int > self.counters.total_env_steps:
            self.counters.total_env_steps = env_steps_int
        if updates_applied is not None:
            self.counters.updates_applied = max(
                self.counters.updates_applied, int(updates_applied)
            )
        if nav is not None:
            self.counters.nav = float(nav)
        if turnover is not None:
            self.counters.turnover = float(turnover)
        if drawdown is not None:
            self.counters.drawdown = float(drawdown)
        self.counters.wall_time = max(0.0, time.perf_counter() - self._start_time)
        self._refresh_epochs()

    def register_episode(
        self,
        *,
        nav: Optional[float] = None,
        turnover: Optional[float] = None,
        drawdown: Optional[float] = None,
    ) -> None:
        self.counters.episodes_completed += 1
        if nav is not None:
            self.counters.nav = float(nav)
        if turnover is not None:
            self.counters.turnover = float(turnover)
        if drawdown is not None:
            self.counters.drawdown = float(drawdown)
        self._refresh_epochs()

    def summary_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "project_epochs_completed": self.project_epochs_completed,
            "total_env_steps": self.counters.total_env_steps,
            "episodes_completed": self.counters.episodes_completed,
            "updates_applied": self.counters.updates_applied,
            "wall_time": self.counters.wall_time,
        }
        if self.counters.nav is not None:
            payload["nav"] = self.counters.nav
        if self.counters.turnover is not None:
            payload["turnover"] = self.counters.turnover
        if self.counters.drawdown is not None:
            payload["drawdown"] = self.counters.drawdown
        return payload

    def metrics_payload(self) -> Dict[str, float]:
        payload: Dict[str, float] = {
            "project_epoch": float(self.project_epochs_completed),
            "total_env_steps": float(self.counters.total_env_steps),
            "episodes_completed": float(self.counters.episodes_completed),
            "updates_applied": float(self.counters.updates_applied),
            "wall_time": float(self.counters.wall_time),
        }
        if self.counters.nav is not None:
            payload["nav"] = float(self.counters.nav)
        if self.counters.turnover is not None:
            payload["turnover"] = float(self.counters.turnover)
        if self.counters.drawdown is not None:
            payload["drawdown"] = float(self.counters.drawdown)
        return payload

    def consume_log_trigger(self) -> Optional[int]:
        epoch = self._consume_pending(self._last_log_epoch, self.log_every)
        if epoch is not None:
            self._last_log_epoch = epoch
        return epoch

    def consume_eval_trigger(self) -> Optional[int]:
        if self.eval_every is None:
            return None
        epoch = self._consume_pending(self._last_eval_epoch, self.eval_every)
        if epoch is not None:
            self._last_eval_epoch = epoch
        return epoch

    def consume_checkpoint_trigger(self) -> Optional[int]:
        if self.checkpoint_every is None:
            return None
        epoch = self._consume_pending(self._last_checkpoint_epoch, self.checkpoint_every)
        if epoch is not None:
            self._last_checkpoint_epoch = epoch
        return epoch

    def _basis_value(self) -> float:
        if self.unit == "steps":
            return float(self.counters.total_env_steps)
        if self.unit == "episodes":
            return float(self.counters.episodes_completed)
        return float(self.counters.wall_time)

    def _refresh_epochs(self) -> None:
        basis = self._basis_value()
        new_epochs = int(basis // self.value)
        if new_epochs > self._project_epochs:
            self._project_epochs = new_epochs

    def _consume_pending(self, last_epoch: int, every: int) -> Optional[int]:
        if every <= 0:
            return None
        pending = self._project_epochs - last_epoch
        if pending < every:
            return None
        return last_epoch + every


class EpisodeRewardLogger(BaseCallback):
    """Log terminal portfolio rewards and update unified counters per episode."""

    def __init__(
        self,
        experiment_logger,
        *,
        tag: str,
        summary_state: dict[str, Any],
        tracker: Optional[ProjectEpochTracker] = None,
    ) -> None:
        super().__init__()
        self.experiment_logger = experiment_logger
        self.tag = tag
        self.summary_state = summary_state
        self.tracker = tracker
        self._logged_steps: set[int] = set()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if not info or not info.get("episode_final"):
                continue
            reward_value = float(info.get("final_reward", 0.0))
            step_key = int(self.num_timesteps)
            if step_key in self._logged_steps:
                continue
            self._logged_steps.add(step_key)
            self.summary_state["last_train_reward"] = reward_value
            self.experiment_logger.log_metrics(step_key, {self.tag: reward_value})
            self.experiment_logger.log_event(
                f"Episode completed with {self.tag}={reward_value:.4f} at step {step_key}"
            )
            if self.tracker is not None:
                nav_candidates = (
                    info.get("nav"),
                    info.get("final_portfolio_value"),
                    info.get("portfolio_value"),
                )
                nav_value = next(
                    (float(candidate) for candidate in nav_candidates if candidate is not None),
                    None,
                )
                turnover_value = info.get("turnover")
                drawdown_value = info.get("drawdown", info.get("max_drawdown"))
                if turnover_value is not None:
                    try:
                        turnover_value = float(turnover_value)
                    except (TypeError, ValueError):
                        turnover_value = None
                if drawdown_value is not None:
                    try:
                        drawdown_value = float(drawdown_value)
                    except (TypeError, ValueError):
                        drawdown_value = None
                self.tracker.register_episode(
                    nav=nav_value,
                    turnover=turnover_value,
                    drawdown=drawdown_value,
                )
                self.summary_state.update(self.tracker.summary_payload())
            else:
                self.summary_state["episodes_completed"] = (
                    self.summary_state.get("episodes_completed", 0) + 1
                )
        return True


class ProjectEpochProgressCallback(BaseCallback):
    """Emit project cadence metrics and counters on schedule."""

    def __init__(
        self,
        experiment_logger,
        *,
        tracker: ProjectEpochTracker,
        summary_state: dict[str, Any],
    ) -> None:
        super().__init__()
        self.experiment_logger = experiment_logger
        self.tracker = tracker
        self.summary_state = summary_state
        self._has_logged = False

    def _current_updates(self) -> Optional[int]:
        candidate = getattr(self.model, "_n_updates", None)
        if candidate is None:
            raw = self.model.logger.name_to_value.get("train/n_updates")
            if raw is not None:
                try:
                    candidate = int(raw)
                except (TypeError, ValueError):
                    candidate = None
        return None if candidate is None else int(candidate)

    def _format_summary(self, metrics: Dict[str, Any]) -> str:
        focus_keys = ["train_reward", "valid_reward", "total_env_steps", "episodes_completed"]
        parts: list[str] = []
        for key in focus_keys:
            if key not in metrics:
                continue
            value = metrics[key]
            if isinstance(value, float):
                parts.append(f"{key}={value:.4f}")
            else:
                parts.append(f"{key}={value}")
        if not parts:
            parts = [f"{key}={value}" for key, value in sorted(metrics.items()) if key != "project_epoch"]
        return ", ".join(parts) if parts else "no tracked metrics"

    def _gather_metrics(self) -> Dict[str, Any]:
        metrics = _clean_metrics(dict(self.model.logger.name_to_value))
        metrics.update(self.tracker.metrics_payload())
        if "last_train_reward" in self.summary_state:
            metrics.setdefault("train_reward", self.summary_state.get("last_train_reward"))
        if "last_valid_reward" in self.summary_state and self.summary_state.get("last_valid_reward") is not None:
            metrics.setdefault("valid_reward", self.summary_state["last_valid_reward"])
        return metrics

    def _emit_metrics(self, epoch_index: int) -> None:
        metrics = self._gather_metrics()
        metrics["project_epoch"] = float(epoch_index)
        step = int(self.num_timesteps)
        self.summary_state.update(self.tracker.summary_payload())
        self.summary_state["project_epochs_completed"] = max(
            epoch_index, self.summary_state.get("project_epochs_completed", 0)
        )
        self.experiment_logger.log_metrics(step, metrics)
        summary = self._format_summary(metrics)
        self.experiment_logger.log_event(
            f"Project epoch {epoch_index} recorded at step {step} ({summary})"
        )
        self._has_logged = True

    def _emit_pending(self) -> None:
        while True:
            epoch_index = self.tracker.consume_log_trigger()
            if epoch_index is None:
                break
            self._emit_metrics(epoch_index)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        info = infos[0] if infos else {}
        nav = None
        turnover = None
        drawdown = None
        if info:
            nav = info.get("nav")
            if nav is None:
                nav = info.get("portfolio_value")
            turnover = info.get("turnover")
            drawdown = info.get("drawdown")
            if drawdown is None:
                drawdown = info.get("max_drawdown")
        self.tracker.record_step(
            int(self.num_timesteps),
            updates_applied=self._current_updates(),
            nav=nav,
            turnover=turnover,
            drawdown=drawdown,
        )
        self._emit_pending()
        return True

    def _on_training_end(self) -> None:
        self.tracker.record_step(
            int(self.num_timesteps),
            updates_applied=self._current_updates(),
            nav=self.tracker.counters.nav,
            turnover=self.tracker.counters.turnover,
            drawdown=self.tracker.counters.drawdown,
        )
        self._emit_pending()
        if not self._has_logged:
            self._emit_metrics(self.tracker.project_epochs_completed)
        elif self.tracker.project_epochs_completed > self.tracker.last_logged_epoch:
            self._emit_metrics(self.tracker.project_epochs_completed)


class ProjectCheckpointCallback(BaseCallback):
    """Persist checkpoints whenever a project epoch boundary is reached."""

    def __init__(
        self,
        *,
        tracker: ProjectEpochTracker,
        saver,
        experiment_logger,
        summary_state: dict[str, Any],
    ) -> None:
        super().__init__()
        self.tracker = tracker
        self.saver = saver
        self.experiment_logger = experiment_logger
        self.summary_state = summary_state

    def _on_step(self) -> bool:
        while True:
            epoch_index = self.tracker.consume_checkpoint_trigger()
            if epoch_index is None:
                break
            step_key = int(self.num_timesteps)
            self.summary_state.update(self.tracker.summary_payload())
            self.summary_state["last_checkpoint_project_epoch"] = epoch_index
            self.summary_state["last_checkpoint"] = step_key
            self.saver(self.model, step_key, project_epoch=epoch_index)
            self.experiment_logger.log_event(
                f"Checkpoint saved at project epoch {epoch_index} (step {step_key})"
            )
        return True


class ProjectEvaluationCallback(BaseCallback):
    """Run policy evaluation according to the project epoch cadence."""

    def __init__(
        self,
        eval_env,
        experiment_logger,
        *,
        tracker: ProjectEpochTracker,
        eval_episodes: int,
        best_dir: Optional[Path],
        summary_state: Dict[str, Any],
        run_dir: Path,
        saver,
    ) -> None:
        super().__init__()
        self.eval_env = eval_env
        self.experiment_logger = experiment_logger
        self.tracker = tracker
        self.eval_episodes = max(1, int(eval_episodes))
        self.summary_state = summary_state
        self.run_dir = Path(run_dir)
        self.best_dir = Path(best_dir) if best_dir is not None else None
        if self.best_dir is not None:
            self.best_dir.mkdir(parents=True, exist_ok=True)
        self.best_mean_reward: Optional[float] = None
        self.best_model_path: Optional[Path] = None
        self.latest_result: Optional[Dict[str, float]] = None
        self._saver = saver

    def _on_step(self) -> bool:
        while True:
            epoch_index = self.tracker.consume_eval_trigger()
            if epoch_index is None:
                break
            self._run_evaluation(epoch_index)
        return True

    def _run_evaluation(self, project_epoch: int) -> None:
        trace_dir = self.run_dir / "validation_traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_frames: list[pd.DataFrame] = []

        def trace_writer(ep_index: int, traces: list[dict[str, float]], env) -> None:
            file_path = trace_dir / f"project_epoch_{project_epoch:04d}_episode_{ep_index + 1}.csv"
            frame = pd.DataFrame(traces)
            frame.to_csv(file_path, index=False)
            trace_frames.append(frame)

        rewards = evaluate_full_episodes(
            self.model,
            self.eval_env,
            self.eval_episodes,
            deterministic=True,
            trace_writer=trace_writer,
        )

        valuation_paths: list[Path] = []
        if trace_frames:
            valuation_paths = _render_valuation_curves(
                trace_frames,
                trace_dir,
                file_prefix=f"project_epoch_{project_epoch:04d}_episode",
            )

        mean_reward = float(rewards.mean()) if rewards.size else 0.0
        std_reward = float(rewards.std()) if rewards.size else 0.0
        payload = {
            "project_epoch": float(project_epoch),
            "valid_reward": float(mean_reward),
            "eval_std_reward": float(std_reward),
        }
        payload.update(self.tracker.metrics_payload())
        payload["project_epoch"] = float(project_epoch)
        step_key = int(self.num_timesteps)
        self.latest_result = payload
        self.summary_state["last_valid_reward"] = float(mean_reward)
        self.summary_state["last_eval_epoch"] = project_epoch
        self.summary_state.update(self.tracker.summary_payload())
        self.experiment_logger.log_metrics(step_key, payload)
        self.experiment_logger.log_event(
            f"Validation @ project epoch {project_epoch}: reward={mean_reward:.4f}, std={std_reward:.4f}"
        )
        if valuation_paths:
            for plot_path in valuation_paths:
                self.experiment_logger.log_event(
                    f"Saved evaluation valuation plot -> {plot_path.name}"
                )

        is_best = self.best_mean_reward is None or mean_reward > self.best_mean_reward
        if is_best and self.best_dir is not None:
            self.best_mean_reward = mean_reward
            best_path = self.best_dir / f"best_model_epoch_{project_epoch}.zip"
            self.model.save(str(best_path))
            self.best_model_path = best_path
            self.experiment_logger.log_event(
                f"New best model saved at {best_path.name} (reward {mean_reward:.4f})"
            )

        checkpoint_path = self._saver(
            self.model,
            step_key,
            project_epoch=project_epoch,
        )
        rel_checkpoint = checkpoint_path.relative_to(self.run_dir)
        self.experiment_logger.log_event(
            f"Evaluation checkpoint saved -> {rel_checkpoint}"
        )


def _resolve_updates_per_step(train_freq: Any, gradient_steps: Any) -> Optional[float]:
    if gradient_steps is None:
        return None
    try:
        grad = float(gradient_steps)
    except (TypeError, ValueError):
        return None
    candidate_freq = None
    if isinstance(train_freq, (list, tuple)) and train_freq:
        freq_value = train_freq[0]
        freq_unit = train_freq[1] if len(train_freq) > 1 else "step"
        if isinstance(freq_unit, str) and freq_unit.lower() == "step":
            try:
                candidate_freq = float(freq_value)
            except (TypeError, ValueError):
                candidate_freq = None
    elif isinstance(train_freq, (int, float)):
        candidate_freq = float(train_freq)
    elif hasattr(train_freq, "frequency") and hasattr(train_freq, "unit"):
        freq_unit = getattr(train_freq, "unit")
        if isinstance(freq_unit, str) and freq_unit.lower() == "step":
            try:
                candidate_freq = float(getattr(train_freq, "frequency"))
            except (TypeError, ValueError):
                candidate_freq = None
    if candidate_freq is None or candidate_freq <= 0:
        return None
    return grad / candidate_freq


def _extract_algo_metadata(agent_spec) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {"algorithm": agent_spec.name}
    kwargs = dict(agent_spec.kwargs)
    if "n_steps" in kwargs:
        try:
            metadata["n_steps"] = int(kwargs["n_steps"])
        except (TypeError, ValueError):
            metadata["n_steps"] = kwargs["n_steps"]
    else:
        updates_per_step = _resolve_updates_per_step(
            kwargs.get("train_freq"), kwargs.get("gradient_steps")
        )
        if updates_per_step is not None:
            metadata["updates_per_step"] = updates_per_step
    return metadata


@dataclass(slots=True)
class TrainingResult:
    train_timesteps: int
    train_epochs: int
    train_reward: Optional[float]
    valid_reward: Optional[float]
    eval_std_reward: Optional[float]
    checkpoint_dir: Path
    best_model_path: Optional[Path]
    run_directory: Path





def _resolve_project_interval(
    cfg: Dict[str, Any],
    *,
    project_unit: str,
    project_value: float,
    primary_key: str,
    legacy_epoch_key: Optional[str],
    legacy_step_key: Optional[str],
    default_epochs: Optional[int],
) -> Optional[int]:
    def _sanitize(raw: Optional[float]) -> Optional[int]:
        if raw is None:
            return None
        raw_float = float(raw)
        if raw_float <= 0:
            return None
        return max(1, int(math.ceil(raw_float)))

    if primary_key in cfg:
        return _sanitize(cfg.get(primary_key))
    if legacy_epoch_key and legacy_epoch_key in cfg:
        return _sanitize(cfg.get(legacy_epoch_key))
    if (
        legacy_step_key
        and legacy_step_key in cfg
        and project_unit == "steps"
        and project_value > 0
    ):
        steps_value = float(cfg.get(legacy_step_key))
        if steps_value > 0:
            return _sanitize(steps_value / project_value)
    return _sanitize(default_epochs)


def run_training(
    config: Config,
    *,
    run_name: str = "dev",
    total_timesteps: Optional[int] = None,
    eval_split: str = "validation",
    resume_path: Optional[Path] = None,
) -> TrainingResult:
    dataset = build_dataset(config)
    train_split = "train"
    train_frame = dataset.frame_for_split(train_split)
    if train_frame.empty:
        raise ValueError("Training split produced zero samples; check data and config splits.")
    date_col = config.data["date_column"]
    train_dates = np.sort(train_frame[date_col].astype("datetime64[ns]").unique())
    if train_dates.shape[0] < 2:
        raise ValueError("Training split requires at least two distinct dates")
    window_size = int(config.window.get("size", 60))
    window_start_index = min(window_size - 1, train_dates.shape[0] - 1)
    episode_steps = max(1, train_dates.shape[0] - 1 - window_start_index)
    env_settings = config.environment_settings()
    if env_settings.get("max_steps") is not None:
        max_steps_override = int(env_settings.get("max_steps"))
        if max_steps_override <= 0:
            raise ValueError("environment.max_steps must be positive when provided")
        episode_steps = min(episode_steps, max_steps_override)
    epoch_length = episode_steps

    training_cfg = config.training_settings()

    if total_timesteps is not None:
        timesteps = int(total_timesteps)
    elif "total_epochs" in training_cfg:
        total_epochs = float(training_cfg["total_epochs"])
        if total_epochs <= 0:
            raise ValueError("training.total_epochs must be positive")
        timesteps = int(math.ceil(total_epochs * epoch_length))
    elif "total_timesteps" in training_cfg:
        timesteps = int(training_cfg["total_timesteps"])
    else:
        timesteps = epoch_length

    if timesteps <= 0:
        raise ValueError("Total timesteps must be positive")

    planned_epochs = max(1, int(math.ceil(timesteps / epoch_length)))

    project_unit = str(training_cfg.get("project_epoch_unit", "steps")).lower()
    if project_unit not in {"steps", "episodes", "wall_time"}:
        raise ValueError(
            "training.project_epoch_unit must be one of {steps, episodes, wall_time}"
        )
    if "project_epoch_value" in training_cfg:
        project_value = float(training_cfg["project_epoch_value"])
    elif project_unit == "steps":
        project_value = float(epoch_length)
    elif project_unit == "episodes":
        project_value = 1.0
    else:
        project_value = float(training_cfg.get("project_epoch_wall_time_seconds", 60.0))
    if project_value <= 0:
        raise ValueError("training.project_epoch_value must be positive")

    planned_project_epochs: Optional[int] = None
    if project_unit == "steps":
        planned_project_epochs = max(1, int(math.ceil(timesteps / project_value)))

    if planned_project_epochs is not None and planned_project_epochs >= 10:
        default_log_epochs = max(1, planned_project_epochs // 10)
    else:
        default_log_epochs = 1
    if planned_project_epochs is not None and planned_project_epochs >= 4:
        default_eval_epochs = max(1, planned_project_epochs // 4)
        default_ckpt_epochs = max(1, planned_project_epochs // 4)
    else:
        default_eval_epochs = 1
        default_ckpt_epochs = 1

    log_interval_project_epochs = _resolve_project_interval(
        training_cfg,
        project_unit=project_unit,
        project_value=project_value,
        primary_key="log_interval_project_epochs",
        legacy_epoch_key="log_interval_epochs",
        legacy_step_key="log_interval",
        default_epochs=default_log_epochs,
    )
    if log_interval_project_epochs is None:
        log_interval_project_epochs = 1

    eval_interval_project_epochs = _resolve_project_interval(
        training_cfg,
        project_unit=project_unit,
        project_value=project_value,
        primary_key="eval_interval_project_epochs",
        legacy_epoch_key="eval_interval_epochs",
        legacy_step_key="eval_freq",
        default_epochs=default_eval_epochs,
    )

    checkpoint_interval_project_epochs = _resolve_project_interval(
        training_cfg,
        project_unit=project_unit,
        project_value=project_value,
        primary_key="checkpoint_interval_project_epochs",
        legacy_epoch_key="checkpoint_interval_epochs",
        legacy_step_key="checkpoint_freq",
        default_epochs=default_ckpt_epochs,
    )

    eval_episodes = max(1, int(training_cfg.get("eval_episodes", 1)))

    if "log_interval_steps" in training_cfg:
        log_interval_steps = max(1, int(training_cfg["log_interval_steps"]))
    elif "log_interval" in training_cfg:
        log_interval_steps = max(1, int(training_cfg["log_interval"]))
    else:
        log_interval_steps = max(1, epoch_length // 4)

    logger = build_logger(config, run_name)
    run_dir = logger.as_run_directory()
    checkpoint_dir = run_dir / "checkpoints"

    if timesteps < epoch_length:
        logger.log_event(
            f"Total timesteps ({timesteps}) smaller than episode length ({epoch_length}); adjusting to run full episodes."
        )
        timesteps = epoch_length
        planned_epochs = max(1, int(math.ceil(timesteps / epoch_length)))
        if project_unit == "steps" and project_value > 0:
            planned_project_epochs = max(1, int(math.ceil(timesteps / project_value)))

    env_factory = make_env_factory(config, split=train_split, dataset=dataset)
    train_vec_env = DummyVecEnv([env_factory])

    agent_spec = build_agent_spec(config)
    model = agent_spec.instantiate(train_vec_env)

    start_timesteps = 0
    if resume_path is not None:
        start_timesteps = load_checkpoint(model, Path(resume_path), experiment_logger=logger)

    tracker_config = ProjectEpochConfig(
        unit=project_unit,
        value=project_value,
        log_every=log_interval_project_epochs,
        eval_every=eval_interval_project_epochs,
        checkpoint_every=checkpoint_interval_project_epochs,
    )
    tracker = ProjectEpochTracker(tracker_config)
    tracker.record_step(start_timesteps, updates_applied=getattr(model, "_n_updates", None))

    summary_state: Dict[str, Any] = {
        "alias": run_name,
        "start_timesteps": start_timesteps,
        "episode_steps": epoch_length,
        "planned_epochs": planned_epochs,
        "project_epoch_unit": project_unit,
        "project_epoch_value": project_value,
        "planned_project_epochs": planned_project_epochs,
        "planned_timesteps": timesteps,
        "log_interval_project_epochs": log_interval_project_epochs,
        "eval_interval_project_epochs": eval_interval_project_epochs,
        "checkpoint_interval_project_epochs": checkpoint_interval_project_epochs,
    }
    summary_state.update(tracker.summary_payload())
    algo_metadata = _extract_algo_metadata(agent_spec)
    summary_state.update(algo_metadata)
    summary_state["policy"] = agent_spec.policy

    metadata_parts = ", ".join(f"{k}={v}" for k, v in algo_metadata.items())
    logger.log_event(
        f"Initialising training for {run_name}: algorithm={agent_spec.name}, total_timesteps={timesteps}, "
        f"project_epoch_unit={project_unit}, project_epoch_value={project_value}, "
        f"planned_project_epochs={planned_project_epochs if planned_project_epochs is not None else 'unknown'}"
    )
    logger.log_event(f"Algorithm metadata -> {metadata_parts}")

    def saver(model_ref, current_timesteps: int, *, project_epoch: Optional[int] = None) -> Path:
        summary_state.update(tracker.summary_payload())
        summary_state["last_checkpoint"] = current_timesteps
        if project_epoch is not None:
            summary_state["last_checkpoint_project_epoch"] = project_epoch
        return save_checkpoint(model_ref, config, checkpoint_dir, current_timesteps, summary_state)

    callbacks: list[BaseCallback] = [
        EpisodeRewardLogger(
            logger,
            tag="train_reward",
            summary_state=summary_state,
            tracker=tracker,
        ),
        ProjectEpochProgressCallback(
            logger,
            tracker=tracker,
            summary_state=summary_state,
        ),
        MonitoringCallback(logger, log_interval=log_interval_steps),
    ]

    if tracker.checkpoint_every is not None:
        callbacks.append(
            ProjectCheckpointCallback(
                tracker=tracker,
                saver=saver,
                experiment_logger=logger,
                summary_state=summary_state,
            )
        )

    eval_vec_env = None
    eval_callback: Optional[ProjectEvaluationCallback] = None
    best_model_path: Optional[Path] = None
    if (
        tracker.eval_every is not None
        and eval_split in config.splits
        and eval_episodes > 0
    ):
        eval_factory = make_env_factory(
            config,
            split=eval_split,
            dataset=dataset,
            max_steps_override=None,
        )
        eval_vec_env = DummyVecEnv([eval_factory])
        best_dir = checkpoint_dir / "best"
        eval_callback = ProjectEvaluationCallback(
            eval_vec_env,
            logger,
            tracker=tracker,
            eval_episodes=eval_episodes,
            best_dir=best_dir,
            summary_state=summary_state,
            run_dir=run_dir,
            saver=saver,
        )
        callbacks.append(eval_callback)

    callback_chain: BaseCallback | CallbackList
    callback_chain = callbacks[0] if len(callbacks) == 1 else CallbackList(callbacks)

    model.learn(total_timesteps=timesteps, callback=callback_chain, progress_bar=False)

    tracker.record_step(int(model.num_timesteps), updates_applied=getattr(model, "_n_updates", None))
    summary_state.update(tracker.summary_payload())

    eval_mean: Optional[float] = None
    eval_std: Optional[float] = None
    if eval_vec_env is not None:
        final_trace_dir = run_dir / "final_eval_traces"

        def trace_writer(ep_index: int, traces: list[dict[str, float]], env) -> None:
            final_trace_dir.mkdir(parents=True, exist_ok=True)
            file_path = final_trace_dir / f"episode_{ep_index + 1:02d}.csv"
            pd.DataFrame(traces).to_csv(file_path, index=False)

        final_rewards = evaluate_full_episodes(
            model,
            eval_vec_env,
            eval_episodes,
            deterministic=True,
            trace_writer=trace_writer,
        )
        eval_mean = float(final_rewards.mean()) if final_rewards.size else 0.0
        eval_std = float(final_rewards.std()) if final_rewards.size else 0.0
        final_payload = {
            "project_epoch": float(tracker.project_epochs_completed),
            "valid_reward": float(eval_mean),
            "eval_std_reward": float(eval_std),
        }
        final_payload.update(tracker.metrics_payload())
        final_payload["project_epoch"] = float(tracker.project_epochs_completed)
        logger.log_metrics(
            tracker.counters.total_env_steps,
            final_payload,
        )
        logger.log_event(
            f"Final validation reward={eval_mean:.4f}, std={eval_std:.4f}"
        )
        summary_state["last_valid_reward"] = float(eval_mean)
        if eval_callback and eval_callback.best_model_path is not None:
            best_model_path = eval_callback.best_model_path
        eval_vec_env.close()

    final_step = tracker.counters.total_env_steps or timesteps
    final_checkpoint = saver(model, final_step, project_epoch=tracker.project_epochs_completed)

    completed_epochs = max(1, int(math.ceil(final_step / epoch_length)))
    planned_timesteps_value = int(summary_state.get("planned_timesteps", timesteps))
    planned_epochs_value = int(summary_state.get("planned_epochs", planned_epochs))
    step_overrun = max(0, final_step - planned_timesteps_value)
    epoch_overrun = max(0, completed_epochs - planned_epochs_value)
    if step_overrun > 0:
        rollout_multiple = getattr(model, "n_steps", None)
        if rollout_multiple:
            logger.log_event(
                f"Actual training steps {final_step} exceeded planned {planned_timesteps_value} by {step_overrun} to honor rollout multiple of {rollout_multiple}."
            )
        else:
            logger.log_event(
                f"Actual training steps {final_step} exceeded planned {planned_timesteps_value} by {step_overrun} due to rollout rounding."
            )
    summary_state.update(
        {
            "run_completed": True,
            "train_timesteps": final_step,
            "train_epochs": completed_epochs,
            "train_timesteps_overrun": step_overrun,
            "train_epochs_overrun": epoch_overrun,
            "train_reward": summary_state.get("last_train_reward"),
            "valid_reward": None if eval_mean is None else float(eval_mean),
            "eval_std_reward": None if eval_std is None else float(eval_std),
            "final_checkpoint": str(final_checkpoint),
            "total_env_steps": tracker.counters.total_env_steps,
            "episodes_completed": tracker.counters.episodes_completed,
            "updates_applied": tracker.counters.updates_applied,
            "project_epochs_completed": tracker.project_epochs_completed,
            "wall_time": tracker.counters.wall_time,
            "date": datetime.utcnow().date().isoformat(),
        }
    )
    logger.finalize_summary(summary_state)

    train_vec_env.close()
    logger.log_event("Training complete")
    logger.close()

    return TrainingResult(
        train_timesteps=final_step,
        train_epochs=completed_epochs,
        train_reward=summary_state.get("last_train_reward"),
        valid_reward=float(eval_mean) if eval_mean is not None else None,
        eval_std_reward=float(eval_std) if eval_std is not None else None,
        checkpoint_dir=checkpoint_dir,
        best_model_path=best_model_path,
        run_directory=run_dir,
    )

def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO agent on sliding-window dataset.")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config file")
    parser.add_argument("--run-name", type=str, default="ppo_dev", help="Run name for logging")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total training timesteps")
    parser.add_argument("--resume", type=str, default=None, help="Path to model.pt checkpoint to resume from")
    args = parser.parse_args()

    config = load_config(args.config)
    run_training(
        config,
        run_name=args.run_name,
        total_timesteps=args.timesteps,
        resume_path=Path(args.resume) if args.resume else None,
    )


if __name__ == "__main__":
    main()
