"""Portfolio utilities for allocation transforms and validation."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_weight_array(weights: ArrayLike) -> NDArray[np.float64]:
    """Convert arbitrary array-like weights into a float64 NumPy array."""

    arr = np.asarray(weights, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("Weights must be a 1-D array.")
    if arr.size == 0:
        raise ValueError("Weights array cannot be empty.")
    if not np.isfinite(arr).all():
        raise ValueError("Weights must be finite.")
    return arr


def apply_tau_limit(
    prev_weights: ArrayLike,
    target_weights: ArrayLike,
    tau: float,
    *,
    atol: float = 1e-8,
) -> NDArray[np.float64]:
    """Clamp the L1 move between consecutive allocations to ``tau``.

    The limiter mirrors the softmax transition guard described in the design
    notes: we scale the proposed delta if the half-L1 move exceeds ``tau``.
    """

    if tau < 0:
        raise ValueError("tau must be non-negative.")
    prev_arr = _as_weight_array(prev_weights)
    target_arr = _as_weight_array(target_weights)
    if prev_arr.shape != target_arr.shape:
        raise ValueError("prev_weights and target_weights must share the same shape.")

    def _normalise(arr: NDArray[np.float64]) -> NDArray[np.float64]:
        """Normalise non-negative weights, falling back to equal weights if degenerate."""

        clipped = np.clip(arr.astype(np.float64, copy=True), 0.0, None)
        total = float(clipped.sum())
        if not np.isfinite(total) or total <= atol:
            return np.full(clipped.shape, 1.0 / clipped.size, dtype=np.float64)
        return clipped / total

    prev_arr = _normalise(prev_arr)
    target_arr = _normalise(target_arr)

    delta = target_arr - prev_arr
    half_l1 = 0.5 * np.abs(delta).sum()
    if half_l1 <= tau + atol:
        result = target_arr
    else:
        if tau == 0:
            scale = 0.0
        else:
            scale = min(1.0, tau / max(half_l1, atol))
        result = prev_arr + scale * delta

    # Guard against negative drift / numerical decay by re-normalising.
    result = np.clip(result, 0.0, 1.0)
    normaliser = result.sum()
    if not np.isfinite(normaliser) or normaliser <= 0:
        # Degenerate case: fall back to equally weighted allocation.
        result = np.full_like(result, 1.0 / result.size)
    else:
        result = result / normaliser

    return result



def _normalise_allocation_weights(
    weights: Mapping[str, float],
    symbols: Sequence[str],
) -> Dict[str, float]:
    adjusted: Dict[str, float] = {}
    total = 0.0
    for symbol in symbols:
        value = float(weights.get(symbol, 0.0))
        if value < 0:
            raise ValueError(f"Allocation weight for '{symbol}' must be non-negative.")
        adjusted[symbol] = value
        total += value
    if total <= 0:
        raise ValueError("Allocation weights must contain positive mass.")
    for symbol in adjusted:
        adjusted[symbol] /= total
    return adjusted


def integerize_allocations(
    weights: Mapping[str, float],
    prices: Mapping[str, float],
    *,
    nav: float,
    symbols: Sequence[str],
    lot_size: int = 1,
) -> Tuple[Dict[str, int], float, Dict[str, float]]:
    """Convert target weights into integer share counts and residual cash.

    Returns a tuple ``(holdings, residual_cash, executed_weights)``.
    """

    if nav <= 0:
        raise ValueError("nav must be positive for integerization.")
    if lot_size <= 0:
        raise ValueError("lot_size must be positive.")

    normalised = _normalise_allocation_weights(weights, symbols)
    holdings: Dict[str, int] = {}
    residual_cash = float(nav)

    for symbol in sorted(symbols, key=lambda sym: normalised.get(sym, 0.0), reverse=True):
        weight = normalised.get(symbol, 0.0)
        price = float(prices.get(symbol, 0.0))
        if price <= 0:
            raise ValueError(f"Price for symbol '{symbol}' must be positive for integerization.")
        target_notional = nav * weight
        qty = int(target_notional // (price * lot_size)) * lot_size
        qty = max(qty, 0)
        holdings[symbol] = qty
        residual_cash -= qty * price

    # Ensure all symbols are present (even those with zero allocation)
    for symbol in symbols:
        holdings.setdefault(symbol, 0)

    executed_weights: Dict[str, float] = {}
    if nav > 0:
        for symbol in symbols:
            executed_notional = holdings[symbol] * float(prices[symbol])
            executed_weights[symbol] = executed_notional / nav if nav > 0 else 0.0

    residual_cash = max(residual_cash, 0.0)
    return holdings, residual_cash, executed_weights


def validate_holdings_payload(symbols: Sequence[str], holdings: Mapping[str, float]) -> None:
    """Validate explicit holdings cover all symbols with non-negative integers."""

    missing = [symbol for symbol in symbols if symbol not in holdings]
    if missing:
        raise KeyError(
            "Portfolio override missing symbols: " + ", ".join(missing)
        )
    for symbol in symbols:
        qty = holdings[symbol]
        if isinstance(qty, bool):
            raise ValueError(f"Holding for symbol '{symbol}' must be an integer, not boolean.")
        if float(qty) < 0:
            raise ValueError(f"Holding for symbol '{symbol}' must be non-negative.")
        if not float(qty).is_integer():
            raise ValueError(f"Holding for symbol '{symbol}' must be an integer count.")


__all__ = [
    "apply_tau_limit",
    "integerize_allocations",
    "validate_holdings_payload",
]
