import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from tradingbot.core.risk_manager import RiskManager


def test_learning_phase_uses_fixed_notional():
    rm = RiskManager(balance=500.0)
    size = rm.compute_size(price=100.0, stop_loss=90.0)
    assert size == 10.0


def test_growth_phase_applies_tiers_and_weights():
    rm = RiskManager(balance=2000.0)
    # base risk 0.5% -> 2000*0.005=10 risk amount -> /0.1 = 100 notional
    # strong signal weighting (1.5) -> 150
    size = rm.compute_size(price=100.0, stop_loss=90.0, signal_strength="strong")
    assert size == pytest.approx(150.0)


def test_drawdown_reduces_risk():
    rm = RiskManager(balance=2000.0)
    rm.peak_equity = 4000.0  # 50% drawdown -> floor risk 0.25%
    size = rm.compute_size(price=100.0, stop_loss=90.0)
    assert size == pytest.approx(50.0)


def test_breach_check_flags_exposure_and_daily_loss():
    rm = RiskManager(balance=1000.0, max_daily_loss_pct=0.1, max_exposure_pct=0.5)
    rm.register_open(600.0)  # exposure > 50% of balance
    rm.balance = 850.0       # 15% loss from starting equity
    breaches = rm.risk_breach_check()
    assert "DAILY_LOSS" in breaches
    assert "EXPOSURE" in breaches
