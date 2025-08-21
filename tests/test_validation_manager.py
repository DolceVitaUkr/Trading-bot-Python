import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from tradingbot.core.validationmanager import (
    BacktestEngine,
    ValidationManager,
    cvar,
    maxdrawdown,
    sharperatio,
    walkforwardsplits,
)


def test_metrics_math():
    returns = [0.01, 0.02, -0.01]
    equity = np.cumsum(returns)
    assert sharperatio(returns, freq=1) == pytest.approx(0.43643578047, rel=1e-6)
    assert maxdrawdown(equity) == pytest.approx(-1 / 3, rel=1e-6)
    tail = [-0.02, -0.05, -0.03, 0.04]
    assert cvar(tail, 0.05) == pytest.approx(0.05)


def test_split_purge_embargo():
    data = pd.DataFrame({"x": range(30)})
    splits = walkforwardsplits(data, n_splits=2, purge=2, embargo=2)
    assert len(splits) == 2
    t0, v0 = splits[0]
    t1, v1 = splits[1]
    assert max(t0) == 9
    assert min(v0) - max(t0) - 1 == 2  # purge gap
    assert min(v1) - max(v0) - 1 == 4  # embargo + purge gap


def test_gating_logic(tmp_path):
    vm = ValidationManager(log_dir=tmp_path)
    good_metrics = {"sharpe": 2.5, "max_dd": 0.1}
    ok, reasons = vm.gating(good_metrics, n_trades=600)
    assert ok and reasons == []

    bad_metrics = {"sharpe": 1.0, "max_dd": 0.2}
    ok, reasons = vm.gating(bad_metrics, n_trades=100)
    assert not ok
    assert any("trades" in r for r in reasons)
    assert any("sharpe" in r for r in reasons)
    assert any("max_dd" in r for r in reasons)


def test_backtest_deterministic(tmp_path):
    orders = [
        {"timestamp": "2020-01-01", "side": "buy", "price": 100, "qty": 1},
        {"timestamp": "2020-01-02", "side": "sell", "price": 110, "qty": 1},
    ]
    engine = BacktestEngine(seed=42, taker_fee=0, spread=0, slippage_vol_coeff=0, latency_jitter_ms=10)
    result = engine.run(orders)
    assert result["equity"][-1] == pytest.approx(9.9865529881)

    engine2 = BacktestEngine(seed=42, taker_fee=0, spread=0, slippage_vol_coeff=0, latency_jitter_ms=10)
    result2 = engine2.run(orders)
    assert result["equity"][-1] == result2["equity"][-1]

