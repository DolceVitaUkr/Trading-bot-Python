import pytest
from unittest.mock import MagicMock
from modules.risk_management import RiskManager
from modules.Kill_Switch import KillSwitch
import config

@pytest.fixture
def sizing_policy():
    """Provides a default sizing policy for tests."""
    return {
      "global": {},
      "leverage_tiers": [],
      "asset_caps": {
        "SPOT":   {"max_leverage": 1.0},
        "PERP":   {"max_leverage": 3.0, "liq_buffer_pct": 0.20, "funding_rate_threshold": -0.0002},
      }
    }

@pytest.fixture
def risk_manager(sizing_policy):
    """Returns a RiskManager instance with default test settings for original tests."""
    config.KPI_TARGETS['daily_loss_limit'] = 0.03
    # For original tests, we don't need the new dependencies
    mock_kill_switch = MagicMock(spec=KillSwitch)
    mock_kill_switch.is_active.return_value = False
    mock_data_provider = MagicMock()
    return RiskManager(
        account_balance=10000,
        sizing_policy=sizing_policy,
        kill_switch=mock_kill_switch,
        data_provider=mock_data_provider,
        notifier=None
    )

class TestRiskManagerOriginal:
    def test_initial_state(self, risk_manager):
        assert risk_manager.equity == 10000
        proposal = {"leverage": 1.0, "sl_distance": 100}
        allowed, _ = risk_manager.allow(proposal, "SPOT", "BTCUSDT", "buy", 50000)
        assert allowed is True

    def test_allow_false_on_daily_loss_limit(self, risk_manager):
        risk_manager.record_trade_closure(pnl=-301, new_equity=9699)
        proposal = {"leverage": 1.0, "sl_distance": 100}
        allowed, reason = risk_manager.allow(proposal, "SPOT", "BTCUSDT", "buy", 50000)
        assert allowed is False
        assert "Daily loss limit" in reason

# --- New Test Class for Filters ---

@pytest.fixture
def mock_kill_switch():
    ks = MagicMock(spec=KillSwitch)
    ks.is_active.return_value = False
    # Store active switches in the mock for assertion
    ks.active_kill_switches = {}

    def activate_side_effect(asset_class, reason):
        ks.active_kill_switches[asset_class] = reason
    ks.activate.side_effect = activate_side_effect

    # Configure is_active to check the dictionary
    ks.is_active.side_effect = lambda asset: asset in ks.active_kill_switches

    return ks

@pytest.fixture
def mock_data_provider():
    dp = MagicMock()
    dp.get_funding_rate.return_value = 0.0001 # Default positive funding
    return dp

@pytest.fixture
def risk_manager_for_filters(sizing_policy, mock_kill_switch, mock_data_provider):
    """Returns a RiskManager instance with mocked dependencies for filter tests."""
    return RiskManager(
        account_balance=10000,
        sizing_policy=sizing_policy,
        kill_switch=mock_kill_switch,
        data_provider=mock_data_provider,
        notifier=None
    )

class TestRiskManagerFilters:

    def test_kill_switch_filter_rejects(self, risk_manager_for_filters, mock_kill_switch):
        """Test that a trade is rejected if the kill switch is active."""
        mock_kill_switch.activate("PERP", "Daily DD Breach")

        proposal = {"leverage": 1.0, "sl_distance": 100}
        allowed, reason = risk_manager_for_filters.allow(proposal, "PERP", "ETHUSDT", "buy", 3000)

        assert not allowed
        assert "Kill switch active" in reason

        # A trade on SPOT should still be allowed
        allowed_spot, _ = risk_manager_for_filters.allow(proposal, "SPOT", "BTCUSDT", "buy", 50000)
        assert allowed_spot

    def test_funding_filter_rejects_long_with_high_negative_rate(self, risk_manager_for_filters, mock_data_provider):
        """Test that a long trade is rejected if funding rate is too negative."""
        mock_data_provider.get_funding_rate.return_value = -0.0003 # Below threshold of -0.0002

        # sl_distance must be > 600 (20% of 3000) to pass the liq buffer check
        proposal = {"leverage": 1.0, "sl_distance": 601}
        allowed, reason = risk_manager_for_filters.allow(proposal, "PERP", "ETHUSDT", "buy", 3000)

        assert not allowed
        assert "High negative funding rate" in reason

    def test_funding_filter_allows_long_with_positive_rate(self, risk_manager_for_filters, mock_data_provider):
        """Test that a long trade is allowed with positive funding."""
        mock_data_provider.get_funding_rate.return_value = 0.0001
        proposal = {"leverage": 1.0, "sl_distance": 601}
        allowed, _ = risk_manager_for_filters.allow(proposal, "PERP", "ETHUSDT", "buy", 3000)
        assert allowed

    def test_funding_filter_rejects_short_with_high_positive_rate(self, risk_manager_for_filters, mock_data_provider):
        """Test that a short trade is rejected if funding rate is too positive."""
        mock_data_provider.get_funding_rate.return_value = 0.0003 # Above threshold

        proposal = {"leverage": 1.0, "sl_distance": 601}
        allowed, reason = risk_manager_for_filters.allow(proposal, "PERP", "ETHUSDT", "sell", 3000)

        assert not allowed
        assert "High positive funding rate" in reason

    def test_funding_filter_not_applied_to_spot(self, risk_manager_for_filters, mock_data_provider):
        """Test that the funding filter is not applied to SPOT assets."""
        mock_data_provider.get_funding_rate.return_value = -0.0005 # Very high negative rate
        proposal = {"leverage": 1.0, "sl_distance": 100}
        allowed, _ = risk_manager_for_filters.allow(proposal, "SPOT", "BTCUSDT", "buy", 50000)
        assert allowed
