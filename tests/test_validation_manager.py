import unittest
from modules.Validation_Manager import ValidationManager

class TestValidationManager(unittest.TestCase):

    def setUp(self):
        """Set up a new ValidationManager instance for each test."""
        config = {"min_trades_for_approval": 500}
        self.validation_manager = ValidationManager(config)
        self.strategy_id = "TestStrategy"
        self.asset_class = "SPOT"

    def test_approve_strategy_sufficient_trades(self):
        """Test that a strategy is approved if it has enough trades."""
        results = {"trades": 501, "sharpe_ratio": 2.0}
        self.validation_manager.approve_strategy(self.strategy_id, self.asset_class, results)

        self.assertTrue(self.validation_manager.is_strategy_approved(self.strategy_id, self.asset_class))
        self.assertNotIn(self.asset_class, self.validation_manager.rejected_strategies)

    def test_reject_strategy_insufficient_trades(self):
        """Test that a strategy is rejected if it has fewer than the minimum required trades."""
        results = {"trades": 499, "sharpe_ratio": 2.0}
        self.validation_manager.approve_strategy(self.strategy_id, self.asset_class, results)

        self.assertFalse(self.validation_manager.is_strategy_approved(self.strategy_id, self.asset_class))
        self.assertIn(self.asset_class, self.validation_manager.rejected_strategies)
        self.assertEqual(len(self.validation_manager.rejected_strategies[self.asset_class]), 1)
        rejection_log = self.validation_manager.rejected_strategies[self.asset_class][0]
        self.assertEqual(rejection_log['strategy_id'], self.strategy_id)
        self.assertIn("Not enough trades", rejection_log['reason'])

    def test_reject_strategy_explicitly(self):
        """Test that a strategy can be explicitly rejected for any reason."""
        results = {"trades": 1000, "sharpe_ratio": 0.5}
        reason = "Low Sharpe Ratio"
        self.validation_manager.reject_strategy(self.strategy_id, self.asset_class, results, reason)

        self.assertFalse(self.validation_manager.is_strategy_approved(self.strategy_id, self.asset_class))
        self.assertIn(self.asset_class, self.validation_manager.rejected_strategies)
        rejection_log = self.validation_manager.rejected_strategies[self.asset_class][0]
        self.assertEqual(rejection_log['reason'], reason)

    def test_approval_is_asset_class_specific(self):
        """Test that strategy approval is specific to an asset class."""
        results = {"trades": 600}
        self.validation_manager.approve_strategy(self.strategy_id, "SPOT", results)

        self.assertTrue(self.validation_manager.is_strategy_approved(self.strategy_id, "SPOT"))
        self.assertFalse(self.validation_manager.is_strategy_approved(self.strategy_id, "PERP"))

if __name__ == '__main__':
    unittest.main()
