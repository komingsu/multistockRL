
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.builders import (
    build_agent_spec,
    build_environment,
    build_environment_config,
    build_logger,
)
from src.utils.config import load_config


def test_environment_builder_extracts_settings():
    config = load_config("configs/base.yaml")
    env_cfg = build_environment_config(config)
    assert env_cfg.initial_cash == 10_000_000_000.0
    assert env_cfg.friction.commission_rate == 0.00025

    frame = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02"],
            "symbol": ["AAA", "AAA"],
            "close": [100.0, 101.0],
        }
    )
    env = build_environment(config, frame)
    try:
        observation, info = env.reset()
        assert observation.dtype == np.float32
        assert info["portfolio_value"] == env.initial_cash
    finally:
        env.close()


def test_agent_spec_uses_default_policy():
    config = load_config("configs/base.yaml")
    spec = build_agent_spec(config)
    assert spec.name == config.agents["default"].upper()
    assert spec.policy == "MlpPolicy"


def test_logger_writes_config(tmp_path):
    config = load_config("configs/base.yaml")
    config.logging["run_dir"] = str(tmp_path / "runs")
    logger = build_logger(config, run_name="smoke")
    assert logger.as_run_directory().exists()
    logger.log_metrics(0, {"reward": 0.0})
    logger.flush()
    assert logger.metrics_path.exists()
    assert logger.config_path.exists()
    logger.close()
