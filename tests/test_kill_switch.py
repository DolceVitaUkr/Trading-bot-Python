import unittest
from unittest.mock import Mock
from modules.Kill_Switch import KillSwitch

class TestKillSwitch(unittest.TestCase):

    def setUp(self):
        """Set up a new KillSwitch instance for each test."""
        config = {
            "daily_drawdown_limit": 0.05,
            "monthly_drawdown_limit": 0.15,
            "max_slippage_events": 3,
            "max_api_errors": 10
        }
        # Mock PortfolioManager, as KillSwitch depends on it
        self.mock_portfolio_manager = Mock()
        self.kill_switch = KillSwitch(config, self.mock_portfolio_manager)

    def test_initial_state(self):
        """Test that the kill switch is not active initially."""
        self.assertFalse(self.kill_switch.is_active("SPOT"))
        self.assertFalse(self.kill_switch.is_active("PERP"))
        self.assertEqual(len(self.kill_switch.active_kill_switches), 0)

    def test_activate_and_is_active(self):
        """Test activating the kill switch for a specific asset class."""
        reason = "Test activation"
        self.kill_switch.activate("SPOT", reason)

        self.assertTrue(self.kill_switch.is_active("SPOT"))
        self.assertFalse(self.kill_switch.is_active("PERP"))
        self.assertEqual(self.kill_switch.active_kill_switches["SPOT"], reason)

    def test_reset(self):
        """Test resetting an active kill switch."""
        self.kill_switch.activate("SPOT", "Test")
        self.assertTrue(self.kill_switch.is_active("SPOT"))

        self.kill_switch.reset("SPOT")
        self.assertFalse(self.kill_switch.is_active("SPOT"))
        self.assertNotIn("SPOT", self.kill_switch.active_kill_switches)

    def test_trigger_by_slippage(self):
        """Test that the kill switch is triggered by excessive slippage events."""
        slippage_events = ["event1", "event2", "event3"]
        self.kill_switch.check_slippage(slippage_events)

        # The stub activates for "ALL" asset classes on slippage
        self.assertTrue(self.kill_switch.is_active("ALL"))

    def test_no_trigger_by_slippage_if_below_threshold(self):
        """Test that the kill switch is not triggered if slippage is below threshold."""
        slippage_events = ["event1", "event2"]
        self.kill_switch.check_slippage(slippage_events)
        self.assertFalse(self.kill_switch.is_active("ALL"))

    def test_trigger_by_api_errors(self):
        """Test that the kill switch is triggered by excessive API errors."""
        api_errors = {"PERP": 11, "SPOT": 5}
        self.kill_switch.check_api_errors(api_errors)

        self.assertTrue(self.kill_switch.is_active("PERP"))
        self.assertFalse(self.kill_switch.is_active("SPOT"))

if __name__ == '__main__':
    unittest.main()
