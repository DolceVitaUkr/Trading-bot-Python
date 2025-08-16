import pytest
from datetime import datetime, timedelta, timezone
from modules.risk_management import RiskManager
import config

@pytest.fixture
def sizing_policy():
    """Provides a default sizing policy for tests."""
    return {
      "global": {},
      "leverage_tiers": [],
      "asset_caps": {
        "SPOT":   {"max_leverage": 1.0},
        "PERP":   {"max_leverage": 3.0, "liq_buffer_pct": 0.20},
      }
    }

@pytest.fixture
def risk_manager(sizing_policy):
    """Returns a RiskManager instance with default test settings."""
    config.KPI_TARGETS['daily_loss_limit'] = 0.03
    return RiskManager(
        account_balance=10000,
        sizing_policy=sizing_policy,
        notifier=None
    )

class TestRiskManager:
    def test_initial_state(self, risk_manager):
        assert risk_manager.equity == 10000
        # A proposal with default/safe values
        proposal = {"leverage": 1.0, "sl_distance": 100}
        allowed, _ = risk_manager.allow(proposal, "SPOT", 50000)
        assert allowed is True

    def test_allow_false_on_daily_loss_limit(self, risk_manager):
        risk_manager.record_trade_closure(pnl=-301, new_equity=9699)
        proposal = {"leverage": 1.0, "sl_distance": 100}
        allowed, reason = risk_manager.allow(proposal, "SPOT", 50000)
        assert allowed is False
        assert "Daily loss limit" in reason

    def test_allow_false_on_cooldown(self, risk_manager):
        for _ in range(3):
            risk_manager.record_trade_closure(-10, risk_manager.equity - 10)

        proposal = {"leverage": 1.0, "sl_distance": 100}
        allowed, reason = risk_manager.allow(proposal, "SPOT", 50000)
        assert allowed is False
        assert "In cooldown" in reason

    def test_allow_leverage_check(self, risk_manager):
        proposal_spot_bad = {"leverage": 1.1, "sl_distance": 10}
        allowed, reason = risk_manager.allow(proposal_spot_bad, "SPOT", 50000)
        assert allowed is False
        assert "exceeds asset class cap" in reason

        # sl_distance must be >= 0.20 * 50000 = 10000 to pass liq buffer check
        proposal_perp_ok = {"leverage": 3.0, "sl_distance": 10000}
        allowed, _ = risk_manager.allow(proposal_perp_ok, "PERP", 50000)
        assert allowed is True

        proposal_perp_bad = {"leverage": 3.1, "sl_distance": 10}
        allowed, reason = risk_manager.allow(proposal_perp_bad, "PERP", 50000)
        assert allowed is False
        assert "exceeds asset class cap" in reason

    def test_allow_liq_buffer_check(self, risk_manager):
        price = 100
        # Required buffer is 20% (0.20)

        # SL is 21% away (21 / 100), so it should be allowed
        proposal_ok = {"leverage": 1.0, "sl_distance": 21}
        allowed, _ = risk_manager.allow(proposal_ok, "PERP", price)
        assert allowed is True

        # SL is 19% away (19 / 100), should be rejected
        proposal_bad = {"leverage": 1.0, "sl_distance": 19}
        allowed, reason = risk_manager.allow(proposal_bad, "PERP", price)
        assert allowed is False
        assert "below required liquidation buffer" in reason

        # Check should not apply to non-PERP assets
        allowed, _ = risk_manager.allow(proposal_bad, "SPOT", price)
        assert allowed is True

    def test_trade_closure_and_cooldown_logic_remains(self, risk_manager):
        """Verifies that the original cooldown logic is still functional."""
        risk_manager.record_trade_closure(-10, 9990)
        risk_manager.record_trade_closure(-10, 9980)
        assert risk_manager.consecutive_losses == 2

        risk_manager.record_trade_closure(50, 10030)
        assert risk_manager.consecutive_losses == 0

        for _ in range(5):
            risk_manager.record_trade_closure(-10, risk_manager.equity - 10)

        assert risk_manager.consecutive_losses == 5
        assert risk_manager.in_cooldown_until.year == 9999
        allowed, _ = risk_manager.allow({}, "SPOT", 1)
        assert not allowed
