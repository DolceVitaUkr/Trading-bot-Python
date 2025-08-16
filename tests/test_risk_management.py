import pytest
from datetime import datetime, timedelta, timezone
from modules.risk_management import RiskManager
import config

@pytest.fixture
def risk_manager():
    """Returns a RiskManager instance with default test settings."""
    config.KPI_TARGETS['daily_loss_limit'] = 0.03
    config.PER_TRADE_RISK_PERCENT = 0.01
    # Set high caps to isolate other logic in most tests
    return RiskManager(
        account_balance=10000,
        per_pair_cap_pct=1.0,  # 100%
        portfolio_cap_pct=1.0  # 100%
    )

class TestNewRiskManager:
    def test_initial_state(self, risk_manager):
        assert risk_manager.equity == 10000
        assert risk_manager.daily_start_equity == 10000
        assert risk_manager.consecutive_losses == 0
        assert risk_manager.in_cooldown_until is None
        assert risk_manager.is_trade_allowed() is True

    def test_daily_loss_limit(self, risk_manager):
        # Simulate a loss that hits the 3% limit
        pnl = -301
        new_equity = 9699
        risk_manager.record_trade_closure(pnl, new_equity)

        assert risk_manager.is_trade_allowed() is False

    def test_daily_loss_limit_reset(self, risk_manager):
        # Hit the limit
        risk_manager.record_trade_closure(-301, 9699)
        assert risk_manager.is_trade_allowed() is False

        # Simulate time passing to the next day
        risk_manager.last_trade_day = datetime.now(timezone.utc).date() - timedelta(days=1)

        # The check should reset the daily limit
        assert risk_manager.is_trade_allowed() is True
        assert risk_manager.daily_start_equity == 9699

    def test_consecutive_loss_cooldown_3(self, risk_manager):
        # Use smaller losses to avoid hitting the daily limit
        risk_manager.record_trade_closure(-10, 9990)
        risk_manager.record_trade_closure(-10, 9980)
        risk_manager.record_trade_closure(-10, 9970)

        assert risk_manager.consecutive_losses == 3
        assert risk_manager.in_cooldown_until is not None
        assert risk_manager.is_trade_allowed() is False, "Trading should be paused during cooldown"

        # Check that cooldown lifts after time
        risk_manager.in_cooldown_until = datetime.now(timezone.utc) - timedelta(minutes=1)
        assert risk_manager.is_trade_allowed() is True, "Trading should be allowed after cooldown"

    def test_consecutive_loss_cooldown_5_session_stop(self, risk_manager):
        for _ in range(5):
            risk_manager.record_trade_closure(-100, risk_manager.equity - 100)

        assert risk_manager.consecutive_losses == 5
        assert risk_manager.in_cooldown_until.year == 9999 # Check for session stop sentinel
        assert risk_manager.is_trade_allowed() is False

    def test_loss_streak_reset_by_win(self, risk_manager):
        risk_manager.record_trade_closure(-100, 9900)
        risk_manager.record_trade_closure(-100, 9800)
        assert risk_manager.consecutive_losses == 2

        risk_manager.record_trade_closure(50, 9850)
        assert risk_manager.consecutive_losses == 0
        assert risk_manager.in_cooldown_until is None
        assert risk_manager.is_trade_allowed() is True

    def test_calculate_position_size(self, risk_manager):
        entry_price = 50000
        sl_price = 49500

        # Risking 1% of 10k equity = $100
        # Price risk per unit = 50000 - 49500 = $500
        # Expected quantity = 100 / 500 = 0.2
        quantity, _ = risk_manager.calculate_position_size("BTCUSDT", entry_price, sl_price)
        assert quantity == pytest.approx(0.2)

    def test_calculate_position_size_below_minimum(self, risk_manager):
        risk_manager.equity = 1000
        config.MIN_TRADE_AMOUNT_USD = 500
        entry_price = 50000

        # Let's make it fail
        # Required position value is MIN_TRADE_AMOUNT_USD
        # Let's say risk is small
        sl_price = 40000
        # Price risk = $10000
        # Dollar risk = 1% of 1k = $10
        # Quantity = 10 / 10000 = 0.001
        # Position value = 0.001 * 50000 = $50
        # This is less than MIN_TRADE_AMOUNT_USD = $500

        quantity, _ = risk_manager.calculate_position_size("BTCUSDT", entry_price, sl_price)
        assert quantity is None
