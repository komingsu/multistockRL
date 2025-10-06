"""Commission and slippage modelling helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray


SlippageFn = Callable[[NDArray[np.float64], NDArray[np.float64]], float]


@dataclass(slots=True)
class FrictionConfig:
    """Configuration for commission and slippage costs."""

    commission_rate: float = 0.00025
    min_commission: float = 0.0
    slippage_bps: float = 1.0
    custom_slippage: SlippageFn | None = None


def compute_commission(trade_notional: NDArray[np.float64], config: FrictionConfig) -> float:
    """Return commission cost for absolute trade notionals."""

    absolute_notional = np.abs(trade_notional)
    gross_cost = float(absolute_notional.sum() * config.commission_rate)
    return max(gross_cost, config.min_commission)


def compute_slippage(
    trade_notional: NDArray[np.float64],
    reference_prices: NDArray[np.float64] | None,
    config: FrictionConfig,
) -> float:
    """Estimate slippage impact in currency units."""

    absolute_notional = np.abs(trade_notional)
    if absolute_notional.size == 0:
        return 0.0
    if config.custom_slippage is not None:
        if reference_prices is None:
            raise ValueError("custom slippage requires reference prices")
        return float(config.custom_slippage(absolute_notional, reference_prices))
    bps = config.slippage_bps / 10_000.0
    return float(absolute_notional.sum() * bps)


__all__ = [
    "FrictionConfig",
    "compute_commission",
    "compute_slippage",
]
