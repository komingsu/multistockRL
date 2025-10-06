"""Environment package exposing trading environment scaffolds."""

from .adapters import WindowedDataAdapter
from .multi_stock_env import EnvironmentConfig, MultiStockTradingEnv
from .frictions import FrictionConfig, compute_commission, compute_slippage
from .rewards import RewardConfig, RewardContext, compute_reward

__all__ = [
    "EnvironmentConfig",
    "FrictionConfig",
    "RewardConfig",
    "RewardContext",
    "MultiStockTradingEnv",
    "WindowedDataAdapter",
    "compute_commission",
    "compute_slippage",
    "compute_reward",
]
