import pytest
from unittest.mock import MagicMock
from modules.Kill_Switch import KillSwitch
from modules.Portfolio_Manager import PortfolioManager

@pytest.fixture
def mock_portfolio_manager():
    """Fixture to create a mock PortfolioManager."""
    pm = MagicMock(spec=PortfolioManager)
    pm.get_all_asset_classes.return_value = ["SPOT", "PERP"]
    return pm

@pytest.fixture
def kill_switch(mock_portfolio_manager):
    """Fixture to create a KillSwitch instance with a mock PortfolioManager."""
    config = {
        "daily_drawdown_limit": 0.05,
        "monthly_drawdown_limit": 0.15,
        "max_slippage_events": 3,
        "max_api_errors": 10
    }
    return KillSwitch(config, mock_portfolio_manager)

# --- Test Cases ---

def test_initial_state(kill_switch):
    """Test that the kill switch is not active initially."""
    assert kill_switch.is_active("SPOT") is False
    assert kill_switch.is_active("PERP") is False
    assert len(kill_switch.active_kill_switches) == 0

def test_activate_and_reset(kill_switch):
    """Test activating and resetting the kill switch."""
    kill_switch.activate("SPOT", "Manual activation")
    assert kill_switch.is_active("SPOT") is True
    assert kill_switch.is_active("PERP") is False

    kill_switch.reset("SPOT")
    assert kill_switch.is_active("SPOT") is False

def test_global_kill_switch(kill_switch):
    """Test that a global 'ALL' kill switch applies to all asset classes."""
    kill_switch.activate("ALL", "Global shutdown")
    assert kill_switch.is_active("SPOT") is True
    assert kill_switch.is_active("PERP") is True
    assert kill_switch.is_active("FOREX") is True

def test_trigger_by_api_errors(kill_switch):
    """Test triggering based on high API error counts."""
    api_errors = {"PERP": 11, "SPOT": 5}
    kill_switch.check_api_errors(api_errors)

    assert kill_switch.is_active("PERP") is True
    assert kill_switch.is_active("SPOT") is False

def test_trigger_by_slippage_per_asset_class(kill_switch):
    """Test that slippage events are counted per asset class."""
    slippage_events = [
        {'asset_class': 'PERP', 'slippage_pct': 0.5},
        {'asset_class': 'SPOT', 'slippage_pct': 0.6},
        {'asset_class': 'PERP', 'slippage_pct': 0.7},
        {'asset_class': 'PERP', 'slippage_pct': 0.8}, # 3rd PERP event
    ]
    kill_switch.check_slippage(slippage_events)

    assert kill_switch.is_active("PERP") is True
    assert kill_switch.is_active("SPOT") is False

def test_trigger_by_daily_drawdown(kill_switch, mock_portfolio_manager):
    """Test the daily drawdown trigger."""
    # Equity drops from 10000 to 9400 (6% drop)
    equity_history = [9800, 9900, 10000, 9400]
    mock_portfolio_manager.get_equity_history.return_value = equity_history

    kill_switch.check_drawdowns()

    # It should have been called for both SPOT and PERP, so we check the mock
    mock_portfolio_manager.get_equity_history.assert_any_call("SPOT", days=30)
    mock_portfolio_manager.get_equity_history.assert_any_call("PERP", days=30)

    # Assuming the mock returns the same history for both, both should be deactivated
    assert kill_switch.is_active("SPOT") is True
    assert kill_switch.is_active("PERP") is True
    assert "Daily drawdown limit breached" in kill_switch.active_kill_switches["SPOT"]

def test_trigger_by_monthly_drawdown(kill_switch, mock_portfolio_manager):
    """Test the monthly drawdown trigger."""
    # Equity drops from peak of 12000 to 10000 (16.67% drop)
    equity_history = [10000, 11000, 12000, 11500, 11000, 10500, 10000]
    mock_portfolio_manager.get_equity_history.return_value = equity_history

    kill_switch.check_drawdowns()

    assert kill_switch.is_active("SPOT") is True
    assert "Monthly drawdown limit breached" in kill_switch.active_kill_switches["SPOT"]

def test_no_drawdown_if_equity_recovers(kill_switch, mock_portfolio_manager):
    """Test that drawdown is not triggered if equity is stable or growing."""
    equity_history = [9800, 9900, 10000, 10100]
    mock_portfolio_manager.get_equity_history.return_value = equity_history

    kill_switch.check_drawdowns()

    assert kill_switch.is_active("SPOT") is False
    assert kill_switch.is_active("PERP") is False
