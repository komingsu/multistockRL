"""Checkpoint helpers shared across training and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch

from src.utils.config import Config


def checkpoint_payload(model, config: Config, timesteps: int, summary: Dict[str, Any]) -> Dict[str, Any]:
    """Snapshot model weights, optimizer state, and config metadata."""

    policy_state = model.policy.state_dict()
    optimizer_state = (
        model.policy.optimizer.state_dict() if hasattr(model.policy, "optimizer") else {}
    )
    config_payload = {
        "data": config.data,
        "splits": config.splits,
        "window": config.window,
        "loader": config.loader,
        "environment": config.environment,
        "agents": config.agents,
        "logging": config.logging,
        "preprocessing": config.preprocessing,
        "training": config.training,
    }
    payload = {
        "timesteps": timesteps,
        "policy_state_dict": policy_state,
        "optimizer_state_dict": optimizer_state,
        "config": config_payload,
        "summary": summary,
    }
    return payload


def save_checkpoint(
    model, config: Config, checkpoint_dir: Path, timesteps: int, summary: Dict[str, Any]
) -> Path:
    """Persist a checkpoint file containing weights and metadata."""

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    step_dir = checkpoint_dir / f"step_{timesteps:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = step_dir / "model.pt"
    payload = checkpoint_payload(model, config, timesteps, summary)
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    model, checkpoint_path: Path, *, experiment_logger: Optional[Any] = None
) -> int:
    """Load model weights/optimizer state from a checkpoint and return timesteps."""

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    policy_state = payload.get("policy_state_dict", {})
    optimizer_state = payload.get("optimizer_state_dict")
    model.policy.load_state_dict(policy_state)
    if optimizer_state and hasattr(model.policy, "optimizer"):
        model.policy.optimizer.load_state_dict(optimizer_state)
    timesteps = int(payload.get("timesteps", 0))
    if experiment_logger is not None:
        experiment_logger.log_event(
            f"Resumed from checkpoint {checkpoint_path} (timesteps={timesteps})"
        )
    return timesteps
