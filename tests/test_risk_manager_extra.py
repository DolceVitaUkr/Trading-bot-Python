import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from tradingbot.core.risk_manager import RiskManager


def test_compute_size_respects_risk():
    rm = RiskManager(balance=1000.0, risk_per_trade=0.02)
    notional = rm.compute_size(price=100.0, stop_loss=90.0)
    assert notional == pytest.approx(200.0)


def test_breach_check_flags_exposure_and_daily_loss():
    rm = RiskManager(balance=1000.0, max_daily_loss_pct=0.1, max_exposure_pct=0.5)
    rm.register_open(600.0)  # exposure > 50% of balance
    rm.balance = 850.0       # 15% loss from starting equity
    breaches = rm.risk_breach_check()
    assert "DAILY_LOSS" in breaches
    assert "EXPOSURE" in breaches
