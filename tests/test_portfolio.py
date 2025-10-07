from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.portfolio import apply_tau_limit, integerize_allocations, validate_holdings_payload


@st.composite
def paired_vectors(draw, *, min_size: int = 2, max_size: int = 6):
    size = draw(st.integers(min_value=min_size, max_value=max_size))

    prev_raw = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=10.0, allow_infinity=False, allow_nan=False),
            min_size=size,
            max_size=size,
        )
    )
    target_raw = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=10.0, allow_infinity=False, allow_nan=False),
            min_size=size,
            max_size=size,
        )
    )
    if all(value == 0 for value in prev_raw):
        prev_raw[0] = 1.0
    if all(value == 0 for value in target_raw):
        target_raw[0] = 1.0

    prev = np.asarray(prev_raw, dtype=np.float64)
    target = np.asarray(target_raw, dtype=np.float64)
    prev /= prev.sum()
    target /= target.sum()
    return prev, target


@settings(deadline=None, max_examples=200)
@given(
    vectors=paired_vectors(),
    tau=st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False),
)
def test_tau_limit_enforces_l1_bound(vectors: tuple[np.ndarray, np.ndarray], tau: float):
    prev, target = vectors
    result = apply_tau_limit(prev, target, tau)
    move = 0.5 * np.abs(result - prev).sum()
    assert move <= tau + 1e-6
    assert np.isclose(result.sum(), 1.0, atol=1e-8)
    assert np.all(result >= -1e-9)


@settings(deadline=None, max_examples=100)
@given(vectors=paired_vectors())
def test_tau_limit_passes_through_when_within_budget(vectors: tuple[np.ndarray, np.ndarray]):
    prev, target = vectors
    move = 0.5 * np.abs(target - prev).sum()
    tau = move + 1e-4
    result = apply_tau_limit(prev, target, tau)
    assert np.allclose(result, target, atol=1e-7)


def test_tau_limit_zero_tau_returns_previous_allocation():
    prev = np.array([0.6, 0.4], dtype=np.float64)
    target = np.array([0.1, 0.9], dtype=np.float64)
    result = apply_tau_limit(prev, target, 0.0)
    assert np.allclose(result, prev)


def test_tau_limit_handles_zero_mass_previous_weights():
    prev = np.zeros(3, dtype=np.float64)
    target = np.array([0.5, 0.3, 0.2], dtype=np.float64)
    result = apply_tau_limit(prev, target, tau=0.25)
    assert np.isfinite(result).all()
    assert np.isclose(result.sum(), 1.0, atol=1e-8)
    assert np.all(result >= -1e-9)


def test_integerize_allocations_rounds_down_and_tracks_cash():
    weights = {"AAA": 0.6, "BBB": 0.4}
    prices = {"AAA": 10.0, "BBB": 20.0}
    nav = 100.0
    symbols = ["AAA", "BBB"]

    holdings, cash, executed = integerize_allocations(
        weights,
        prices,
        nav=nav,
        symbols=symbols,
    )

    assert holdings == {"AAA": 6, "BBB": 2}
    assert cash == pytest.approx(0.0)
    assert executed["AAA"] == pytest.approx(0.6)
    assert executed["BBB"] == pytest.approx(0.4)


def test_integerize_allocations_residual_cash_preserved():
    weights = {"AAA": 0.5, "BBB": 0.5}
    prices = {"AAA": 33.0, "BBB": 47.0}
    nav = 100.0
    symbols = ["AAA", "BBB"]

    holdings, cash, executed = integerize_allocations(
        weights,
        prices,
        nav=nav,
        symbols=symbols,
    )

    total_notional = holdings["AAA"] * prices["AAA"] + holdings["BBB"] * prices["BBB"]
    assert total_notional <= nav + 1e-6
    assert cash == pytest.approx(nav - total_notional)
    assert sum(executed.values()) == pytest.approx(total_notional / nav if nav > 0 else 0.0)


def test_validate_holdings_payload_catches_missing_symbol():
    symbols = ["AAA", "BBB"]
    with pytest.raises(KeyError):
        validate_holdings_payload(symbols, {"AAA": 1})
