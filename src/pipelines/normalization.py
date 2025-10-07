"""Helpers for managing VecNormalize statistics across train/eval pipelines."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from stable_baselines3.common.vec_env import VecNormalize


def maybe_wrap_vecnormalize(
    env,
    settings: Optional[Dict[str, Any]],
    *,
    training: bool,
) -> Tuple[Any, Optional[VecNormalize]]:
    """Optionally wrap an environment with VecNormalize based on settings."""

    if not settings or not settings.get("enabled", False):
        return env, None

    norm_obs = bool(settings.get("norm_obs", True))
    norm_reward = bool(settings.get("norm_reward", False))
    clip_obs = float(settings.get("clip_obs", 10.0))
    clip_reward = float(settings.get("clip_reward", 10.0))
    gamma = float(settings.get("gamma", 0.99))
    epsilon = float(settings.get("epsilon", 1e-8))

    vec_env = VecNormalize(
        env,
        norm_obs=norm_obs,
        norm_reward=norm_reward,
        clip_obs=clip_obs,
        clip_reward=clip_reward,
        gamma=gamma,
        epsilon=epsilon,
    )
    vec_env.training = training
    if not training:
        vec_env.norm_reward = False
    return vec_env, vec_env


def clone_vecnormalize_stats(
    source: VecNormalize,
    env,
    *,
    training: bool,
) -> VecNormalize:
    """Clone statistics from ``source`` onto a new VecNormalize wrapper."""

    clone = VecNormalize(
        env,
        norm_obs=source.norm_obs,
        norm_reward=source.norm_reward,
        clip_obs=source.clip_obs,
        clip_reward=source.clip_reward,
        gamma=source.gamma,
        epsilon=source.epsilon,
    )
    clone.obs_rms = copy.deepcopy(source.obs_rms)
    clone.ret_rms = copy.deepcopy(source.ret_rms)
    clone.training = training
    if not training:
        clone.norm_reward = False
    return clone


def save_vecnormalize_stats(vec_env, path: Path) -> None:
    """Persist VecNormalize statistics beside a checkpoint directory."""

    if isinstance(vec_env, VecNormalize):
        path.parent.mkdir(parents=True, exist_ok=True)
        vec_env.save(str(path))


def stats_path_for_checkpoint(checkpoint_path: Path) -> Path:
    """Return the VecNormalize stats path next to a checkpoint."""

    return checkpoint_path.parent / "vecnormalize.pkl"


def load_vecnormalize_stats(
    stats_path: Path,
    env,
    *,
    training: bool,
) -> VecNormalize:
    """Load VecNormalize statistics and attach them onto an environment."""

    if not stats_path.exists():
        raise FileNotFoundError(f"VecNormalize stats missing at {stats_path}")
    vec_env = VecNormalize.load(str(stats_path), env)
    vec_env.training = training
    if not training:
        vec_env.norm_reward = False
    return vec_env


def vecnormalize_enabled(settings: Optional[Dict[str, Any]]) -> bool:
    """Return True when VecNormalize is explicitly enabled."""

    return bool(settings and settings.get("enabled", False))


__all__ = [
    "maybe_wrap_vecnormalize",
    "clone_vecnormalize_stats",
    "save_vecnormalize_stats",
    "load_vecnormalize_stats",
    "stats_path_for_checkpoint",
    "vecnormalize_enabled",
]
