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
    # Long position
    sl, tp = risk_manager.compute_sl_tp_from_atr("long", 100, 2)
    assert sl == 97.0  # 100 - 1.5 * 2
    assert tp == 106.0 # 100 + 2.0 * (100 - 97)

    # Short position
    sl, tp = risk_manager.compute_sl_tp_from_atr("short", 100, 2)
    assert sl == 103.0 # 100 + 1.5 * 2
    assert tp == 94.0  # 100 - 2.0 * (103 - 100)

def test_size_position_usd_capped(risk_manager):
    # No cap hit
    assert risk_manager.size_position_usd_capped("BTC/USDT", 1000) == 1000

    # Per-pair cap hit
    assert risk_manager.size_position_usd_capped("BTC/USDT", 3000) == 2500  # 10000 * 0.25

    # Portfolio cap hit
    eth_position = risk_manager.calculate_position_size(
        "ETH/USDT", "long", 3000, 2900, risk_percent=0.01
    )
    risk_manager.register_open_position("ETH/USDT", eth_position)

    # With the ETH position open, the portfolio exposure is $2500.
    # The portfolio cap is $5000 (50% of 10000).
    # Remaining capacity is $5000 - $2500 = $2500.
    # The per-pair cap for BTC/USDT is $2500.
    # The effective cap is the minimum of remaining portfolio capacity and per-pair cap.
    # So, the next position is capped at $2500.
    assert risk_manager.size_position_usd_capped("BTC/USDT", 3000) == 2500

def test_calculate_position_size(risk_manager):
    pos = risk_manager.calculate_position_size("BTC/USDT", "long", 50000, 49000, risk_percent=0.01)
    assert pos.symbol == "BTC/USDT"
    assert pos.side == "long"
    assert pos.entry_price == 50000
    assert pos.stop_loss == 49000
    assert pos.risk_percent == 0.01
    assert pos.dollar_risk == 100 # 10000 * 0.01

def test_update_equity(risk_manager):
    risk_manager.update_equity(10500)
    assert risk_manager.peak_equity == 10500
    risk_manager.update_equity(9500)
    assert risk_manager.current_equity == 9500
    assert risk_manager.peak_equity == 10500

def test_max_drawdown_violation(risk_manager):
    risk_manager.max_drawdown_limit = 0.1
    risk_manager.update_equity(11000)
    with pytest.raises(RiskViolationError):
        risk_manager.update_equity(9899) # 11000 * (1-0.10009)
