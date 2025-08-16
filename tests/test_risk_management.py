import pytest
from modules.risk_management import RiskManager, RiskViolationError


@pytest.fixture
def risk_manager():
    """Returns a RiskManager instance with default test settings."""
    return RiskManager(
        account_balance=10000,
        per_pair_cap_pct=0.25,
        portfolio_cap_pct=0.50,
        base_risk_per_trade_pct=0.01,
        min_rr=2.0,
        atr_mult_sl=1.5,
        atr_mult_tp=3.0,
    )


def test_compute_sl_tp_from_atr(risk_manager):
    """
    Tests the compute_sl_tp_from_atr method.
    """
    # Long position
    sl, tp = risk_manager.compute_sl_tp_from_atr("long", 100, 2)
    assert sl == 97.0  # 100 - 1.5 * 2
    assert tp == 106.0  # 100 + 2.0 * (100 - 97)

    # Short position
    sl, tp = risk_manager.compute_sl_tp_from_atr("short", 100, 2)
    assert sl == 103.0  # 100 + 1.5 * 2
    assert tp == 94.0  # 100 - 2.0 * (103 - 100)


def test_size_position_usd_capped(risk_manager):
    """
    Tests the size_position_usd_capped method.
    """
    # No cap hit
    assert risk_manager.size_position_usd_capped("BTC/USDT", 1000) == 1000

    # Per-pair cap hit
    assert risk_manager.size_position_usd_capped(
        "BTC/USDT", 3000) == 2500  # 10000 * 0.25

    # Portfolio cap hit
    eth_position = risk_manager.calculate_position_size(
        "ETH/USDT", "long", 3000, 2900, risk_percent=0.01
    )
    risk_manager.register_open_position("ETH/USDT", eth_position)

    assert risk_manager.size_position_usd_capped("BTC/USDT", 3000) == 2500


def test_calculate_position_size(risk_manager):
    """
    Tests the calculate_position_size method.
    """
    pos = risk_manager.calculate_position_size(
        "BTC/USDT", "long", 50000, 49000, risk_percent=0.01)
    assert pos.symbol == "BTC/USDT"
    assert pos.side == "long"
    assert pos.entry_price == 50000
    assert pos.stop_loss == 49000
    assert pos.risk_percent == 0.01
    assert pos.dollar_risk == 100  # 10000 * 0.01


def test_update_equity(risk_manager):
    """
    Tests the update_equity method.
    """
    risk_manager.update_equity(10500)
    assert risk_manager.peak_equity == 10500
    risk_manager.update_equity(9500)
    assert risk_manager.current_equity == 9500
    assert risk_manager.peak_equity == 10500


def test_max_drawdown_violation(risk_manager):
    """
    Tests that a RiskViolationError is raised when the max drawdown is exceeded.
    """
    risk_manager.max_drawdown_limit = 0.1
    risk_manager.update_equity(11000)
    with pytest.raises(RiskViolationError):
        risk_manager.update_equity(9899)  # 11000 * (1-0.10009)

def test_zero_atr(risk_manager):
    """
    Tests that compute_sl_tp_from_atr handles zero ATR correctly.
    """
    sl, tp = risk_manager.compute_sl_tp_from_atr("long", 100, 0)
    assert sl == 100
    assert tp == 100

from modules.risk_management import PositionRisk
def test_portfolio_at_limit(risk_manager):
    """
    Tests that size_position_usd_capped handles the case where the portfolio is
    already at its limit.
    """
    pos = PositionRisk("ETH/USDT", "long", 3000, 1.66, 0.01, 2.0, 2900, 3200, 100)
    risk_manager.register_open_position("ETH/USDT", pos)
    with pytest.raises(RiskViolationError):
        risk_manager.size_position_usd_capped("BTC/USDT", 1000)

def test_zero_sl_distance(risk_manager):
    """
    Tests that calculate_position_size handles a zero stop-loss distance
    correctly.
    """
    with pytest.raises(RiskViolationError):
        risk_manager.calculate_position_size("BTC/USDT", "long", 50000, 50000)

def test_position_tracking(risk_manager):
    """
    Tests that register_open_position and unregister_position track
    positions correctly.
    """
    assert risk_manager.portfolio_exposure_usd() == 0
    pos1 = PositionRisk("BTC/USDT", "long", 50000, 0.04, 0.01, 2.0, 49000, 52000, 100)
    risk_manager.register_open_position("BTC/USDT", pos1)
    assert risk_manager.portfolio_exposure_usd() == 2000

    pos2 = PositionRisk("ETH/USDT", "long", 3000, 0.33, 0.01, 2.0, 2900, 3200, 100)
    risk_manager.register_open_position("ETH/USDT", pos2)
    assert risk_manager.portfolio_exposure_usd() == 2990 # 2000 + 990

    risk_manager.unregister_position("BTC/USDT")
    assert risk_manager.portfolio_exposure_usd() == 990
