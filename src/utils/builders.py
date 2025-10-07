"""Factory helpers bridging configs to runtime objects."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from stable_baselines3.common.monitor import Monitor

from .config import Config

_NO_OVERRIDE = object()


def build_dataset(config: Config):
    from src.data.loader import SlidingWindowDataset

    return SlidingWindowDataset(config)


def build_environment_config(config: Config, *, max_steps_override=_NO_OVERRIDE) -> "EnvironmentConfig":
    from src.env.frictions import FrictionConfig
    from src.env.multi_stock_env import ActionProjectionConfig, EnvironmentConfig, SafetyConfig

    settings = config.environment_settings()
    if max_steps_override is not _NO_OVERRIDE:
        settings["max_steps"] = max_steps_override
    friction_payload = settings.pop("friction", {}) or {}
    projection_payload = settings.pop("projection", {}) or {}
    safety_payload = settings.pop("safety", {}) or {}
    settings.pop("reward", None)
    friction_cfg = FrictionConfig(**friction_payload)
    projection_cfg = ActionProjectionConfig(**projection_payload)
    safety_cfg = SafetyConfig(**safety_payload) if safety_payload is not None else SafetyConfig()
    return EnvironmentConfig(friction=friction_cfg, projection=projection_cfg, safety=safety_cfg, **settings)


def build_environment(config: Config, market_frame: Any, *, max_steps_override=_NO_OVERRIDE) -> "MultiStockTradingEnv":
    from src.env.multi_stock_env import MultiStockTradingEnv

    env_cfg = build_environment_config(config, max_steps_override=max_steps_override)
    return MultiStockTradingEnv(data=market_frame, config=env_cfg)


def build_agent_spec(config: Config, name: str | None = None) -> "AgentSpec":
    from src.agents.factory import build_agent_spec as _builder

    return _builder(config, name)


def build_logger(config: Config, run_name: str) -> "ExperimentLogger":
    from src.loggers import ExperimentLogger

    logging_settings = config.logging_settings()
    metrics = logging_settings.get("metrics", [])
    flush_interval = int(logging_settings.get("flush_interval", 100))
    root_dir = Path(logging_settings.get("run_dir", "artifacts/runs"))
    destination = logging_settings.get("mode") or logging_settings.get("destination", "file")
    timestamp_format = logging_settings.get("timestamp_format", "%Y-%m-%d %H:%M:%S")
    logger = ExperimentLogger(
        root_dir=root_dir,
        run_name=run_name,
        metrics=metrics,
        flush_interval=flush_interval,
        destination=destination,
        timestamp_format=timestamp_format,
    )
    logger.log_config(config)
    return logger


def make_env_factory(config: Config, split: str, dataset: Any | None = None, *, max_steps_override=_NO_OVERRIDE) -> Callable[[], Any]:
    def _factory() -> Any:
        local_dataset = dataset or build_dataset(config)
        frame = local_dataset.frame_for_split(split)
        env = build_environment(config, frame, max_steps_override=max_steps_override)
        if getattr(env.config, "time_feature_wrapper", False):
            from src.env.wrappers import EpisodeProgressWrapper

            env = EpisodeProgressWrapper(env, normalize=True)
        return Monitor(env)

    return _factory
