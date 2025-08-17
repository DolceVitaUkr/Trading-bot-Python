import unittest
from modules.Validation_Manager import ValidationManager

class TestValidationManager(unittest.TestCase):

    def setUp(self):
        """Set up a new ValidationManager instance for each test."""
        config = {
            "min_trades_for_approval": 500,
            "min_sharpe_ratio": 1.0,
            "min_out_of_sample_sharpe": 0.5
        }
        self.validation_manager = ValidationManager(config)
        # Clear pre-populated strategies for isolated tests
        self.validation_manager.approved_strategies = {}
        self.validation_manager.rejected_strategies = {}
        self.strategy_id = "TestStrategy"
        self.asset_class = "SPOT"

    def test_approve_strategy_passing_all_criteria(self):
        """Test that a strategy is approved if it passes all criteria."""
        mock_backtest = {"trades": 600, "sharpe_ratio": 1.5}
        mock_wfa = {"trades": 200, "out_of_sample_sharpe": 0.8}

        self.validation_manager.run_validation_for_strategy(
            self.strategy_id, self.asset_class, mock_backtest, mock_wfa
        )

        self.assertTrue(self.validation_manager.is_strategy_approved(self.strategy_id, self.asset_class))
        self.assertNotIn(self.asset_class, self.validation_manager.rejected_strategies)

    def test_reject_strategy_insufficient_trades(self):
        """Test that a strategy is rejected for insufficient trades."""
        mock_backtest = {"trades": 499, "sharpe_ratio": 2.0}
        mock_wfa = {"trades": 200, "out_of_sample_sharpe": 1.0}

        self.validation_manager.run_validation_for_strategy(
            self.strategy_id, self.asset_class, mock_backtest, mock_wfa
        )

        self.assertFalse(self.validation_manager.is_strategy_approved(self.strategy_id, self.asset_class))
        self.assertIn(self.asset_class, self.validation_manager.rejected_strategies)
        rejection_log = self.validation_manager.rejected_strategies[self.asset_class][0]
        self.assertEqual(rejection_log['strategy_id'], self.strategy_id)
        self.assertIn("Insufficient trades", rejection_log['reason'])

    def test_reject_strategy_low_sharpe_ratio(self):
        """Test that a strategy is rejected for a low Sharpe ratio."""
        mock_backtest = {"trades": 1000, "sharpe_ratio": 0.5}
        mock_wfa = {"trades": 400, "out_of_sample_sharpe": 1.0}

        self.validation_manager.run_validation_for_strategy(
            self.strategy_id, self.asset_class, mock_backtest, mock_wfa
        )

        self.assertFalse(self.validation_manager.is_strategy_approved(self.strategy_id, self.asset_class))
        self.assertIn(self.asset_class, self.validation_manager.rejected_strategies)
        rejection_log = self.validation_manager.rejected_strategies[self.asset_class][0]
        self.assertIn("Sharpe ratio too low", rejection_log['reason'])

    def test_approval_is_asset_class_specific(self):
        """Test that strategy approval is specific to an asset class."""
        mock_backtest = {"trades": 600, "sharpe_ratio": 1.5}
        mock_wfa = {"trades": 200, "out_of_sample_sharpe": 0.8}

        self.validation_manager.run_validation_for_strategy(
            self.strategy_id, "SPOT", mock_backtest, mock_wfa
        )

        self.assertTrue(self.validation_manager.is_strategy_approved(self.strategy_id, "SPOT"))
        self.assertFalse(self.validation_manager.is_strategy_approved(self.strategy_id, "PERP"))

if __name__ == '__main__':
    unittest.main()
