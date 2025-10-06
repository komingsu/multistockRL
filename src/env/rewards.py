"""Reward shaping utilities for trading environments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RewardConfig:
    """Weights for reward shaping components."""

    pnl_weight: float = 1.0
    cost_weight: float = 1.0
    turnover_penalty: float = 0.1
    drawdown_penalty: float = 0.5
    inventory_penalty: float = 0.05

@dataclass(slots=True)
class RewardContext:
    """Snapshot of portfolio state used to compute step rewards."""

    nav_return: float
    cost_ratio: float
    turnover: float
    drawdown: float
    inventory: float


def compute_reward(context: RewardContext, config: RewardConfig) -> float:
    """Combine reward shaping components into a scalar."""

    pnl_term = config.pnl_weight * context.nav_return
    cost_term = config.cost_weight * context.cost_ratio
    turnover_term = config.turnover_penalty * context.turnover
    drawdown_term = config.drawdown_penalty * context.drawdown
    inventory_term = config.inventory_penalty * context.inventory
    return pnl_term - (cost_term + turnover_term + drawdown_term + inventory_term)


__all__ = [
    "RewardConfig",
    "RewardContext",
    "compute_reward",
]
