"""Shared transaction cost helpers for trading environments."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .frictions import FrictionConfig, compute_commission, compute_slippage


def _to_array(values: Iterable[float] | Sequence[float]) -> NDArray[np.float64]:
    """Convert an iterable of notionals/prices into a float64 NumPy array."""

    if isinstance(values, np.ndarray):
        return values.astype(np.float64, copy=False)
    seq = list(values)
    if not seq:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(seq, dtype=np.float64)


def evaluate_transaction_costs(
    *,
    buy_notionals: Iterable[float] | Sequence[float],
    buy_prices: Iterable[float] | Sequence[float],
    sell_notionals: Iterable[float] | Sequence[float],
    sell_prices: Iterable[float] | Sequence[float],
    friction: FrictionConfig,
) -> Tuple[float, float, float]:
    """Compute commission/slippage across buy and sell legs and total turnover.

    Returns a tuple ``(commission_cost, slippage_cost, total_notional)`` where
    the total notional captures absolute traded value across both sides.
    """

    buy_notional_arr = _to_array(buy_notionals)
    buy_price_arr = _to_array(buy_prices)
    sell_notional_arr = _to_array(sell_notionals)
    sell_price_arr = _to_array(sell_prices)

    commission_cost = 0.0
    slippage_cost = 0.0

    if buy_notional_arr.size:
        commission_cost += compute_commission(buy_notional_arr, friction)
        slippage_cost += compute_slippage(buy_notional_arr, buy_price_arr, friction)
    if sell_notional_arr.size:
        commission_cost += compute_commission(sell_notional_arr, friction)
        slippage_cost += compute_slippage(sell_notional_arr, sell_price_arr, friction)

    total_notional = float(buy_notional_arr.sum() + sell_notional_arr.sum())
    return commission_cost, slippage_cost, total_notional


__all__ = ["evaluate_transaction_costs"]
