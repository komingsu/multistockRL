from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.metrics import compute_sharpe, compute_psr, compute_dsr, moving_block_bootstrap_ci


def test_psr_monotonic_with_sharpe():
    rng = np.random.default_rng(42)
    # Two series with different means
    r1 = rng.normal(loc=0.0/252, scale=0.02, size=252)
    r2 = rng.normal(loc=0.0005, scale=0.02, size=252)
    sr1 = compute_sharpe(r1, annualize=True)
    sr2 = compute_sharpe(r2, annualize=True)
    psr1 = compute_psr(r1)
    psr2 = compute_psr(r2)
    assert sr2 > sr1
    assert psr2 >= psr1


def test_dsr_leq_psr_when_trials_gt_one():
    rng = np.random.default_rng(7)
    r = rng.normal(loc=0.0005, scale=0.02, size=252)
    psr = compute_psr(r)
    dsr = compute_dsr(r, trials=10)
    assert dsr <= psr + 1e-9


def test_moving_block_bootstrap_ci_bounds_stat():
    rng = np.random.default_rng(123)
    r = rng.normal(loc=0.0, scale=0.02, size=252)
    stat = lambda x: float(np.nanmean(x))
    ci = moving_block_bootstrap_ci(r, stat_fn=stat, block_len=5, reps=200, alpha=0.1, random_state=1)
    m = stat(r)
    assert ci.low <= m <= ci.high
