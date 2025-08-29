"""
Integration tests for Multi-Asset Paper → Pre-Validator pipeline.

Tests the complete validation pipeline end-to-end across all asset types.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

# Import pipeline components
from tradingbot.core.paper_trader import PaperTrader
from tradingbot.core.strategy_manager import StrategyManager
from tradingbot.core.validation_manager import ValidationManager
from tradingbot.core.telegrambot import TelegramNotifier
from tradingbot.core.configmanager import config_manager


class TestValidationPipeline(unittest.TestCase):
    """Integration tests for the complete validation pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_result_path = Path(self.temp_dir) / "test_results"
        self.test_result_path.mkdir(parents=True, exist_ok=True)
        
        # Test parameters
        self.test_asset = "crypto_spot"
        self.test_strategy = "test_strategy"
        self.test_symbols = ["BTCUSDT", "ETHUSDT"]
        self.test_start = "2024-01-01"
        self.test_end = "2024-06-30"
        
        # Initialize managers
        self.paper_trader = PaperTrader()
        self.strategy_manager = StrategyManager()
        self.validation_manager = ValidationManager()
        self.telegram_notifier = TelegramNotifier()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_mock_trade_data(self) -> pd.DataFrame:
        """Create mock trade data for testing."""
        np.random.seed(42)  # For reproducible tests
        
        # Generate 100 mock trades
        num_trades = 100
        dates = pd.date_range(start=self.test_start, end=self.test_end, periods=num_trades)
        
        trades_data = []
        for i, date in enumerate(dates):
            entry_price = 100 + np.random.normal(0, 10)
            exit_price = entry_price * (1 + np.random.normal(0.02, 0.05))  # 2% average return, 5% volatility
            
            trades_data.append({
                'symbol': np.random.choice(self.test_symbols),
                'side': 'buy' if i % 2 == 0 else 'sell',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'quantity': 100,
                'timestamp': date.strftime('%Y-%m-%d'),
                'entry_time': date.strftime('%Y-%m-%d %H:%M:%S'),
                'exit_time': (date + pd.Timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
            })
        
        return pd.DataFrame(trades_data)

    def _create_mock_equity_data(self) -> pd.DataFrame:
        """Create mock equity curve data for testing."""
        np.random.seed(42)
        
        dates = pd.date_range(start=self.test_start, end=self.test_end, freq='D')
        daily_returns = np.random.normal(0.0008, 0.02, len(dates))  # Modest positive returns
        
        equity_values = [10000]  # Starting equity
        for ret in daily_returns[1:]:
            equity_values.append(equity_values[-1] * (1 + ret))
        
        return pd.DataFrame({
            'date': dates,
            'equity': equity_values
        })

    def test_complete_pipeline_crypto_spot(self):
        """Test complete pipeline for crypto spot trading."""
        # Create mock data files
        trades_df = self._create_mock_trade_data()
        equity_df = self._create_mock_equity_data()
        
        trades_file = self.test_result_path / "trades.csv"
        equity_file = self.test_result_path / "equity.csv"
        
        trades_df.to_csv(trades_file, index=False)
        equity_df.to_csv(equity_file, index=False)
        
        # Step 1: Test KPI computation
        kpis = self.strategy_manager.Compute_KPIs(
            asset=self.test_asset,
            result_path=str(self.test_result_path)
        )
        
        # Verify KPI structure
        self.assertIn('sharpe_ratio', kpis)
        self.assertIn('max_drawdown', kpis)
        self.assertIn('profit_factor', kpis)
        self.assertIn('win_rate', kpis)
        self.assertIn('overall_pass', kpis)
        self.assertIsInstance(kpis['sharpe_ratio'], (int, float))
        
        # Step 2: Test baseline generation
        baselines = self.strategy_manager.Generate_Baselines(
            asset=self.test_asset,
            symbols=self.test_symbols,
            start=self.test_start,
            end=self.test_end,
            result_path=str(self.test_result_path)
        )
        
        # Verify baseline structure
        self.assertIsInstance(baselines, dict)
        self.assertGreater(len(baselines), 0)
        
        # Check that baseline files were created
        baseline_file = self.test_result_path / "baselines.json"
        self.assertTrue(baseline_file.exists())
        
        # Step 3: Test robustness checks
        robustness = self.validation_manager.Robustness_Checks(
            asset=self.test_asset,
            strategy_result_path=str(self.test_result_path),
            baseline_results=baselines
        )
        
        # Verify robustness structure
        self.assertIn('overall_assessment', robustness)
        self.assertIn('bootstrap_test', robustness)
        self.assertIn('oos_is_test', robustness)
        self.assertIn('pbo_test', robustness)
        self.assertIn('baseline_comparisons', robustness)
        
        # Step 4: Test compliance checks
        compliance = self.validation_manager.Risk_Compliance_Checks(
            asset=self.test_asset,
            strategy_result_path=str(self.test_result_path)
        )
        
        # Verify compliance structure
        self.assertIn('overall_compliance', compliance)
        self.assertIn('sample_validation', compliance)
        self.assertIn('position_size', compliance)
        self.assertIn('trading_frequency', compliance)
        self.assertIn('drawdown', compliance)
        self.assertIn('leverage', compliance)
        
        # Step 5: Test validation package preparation
        validation_package = self.strategy_manager.Prepare_Validator_Package(
            asset=self.test_asset,
            strategy=self.test_strategy,
            kpis=kpis,
            baselines=baselines,
            robustness=robustness,
            compliance=compliance,
            result_path=str(self.test_result_path)
        )
        
        # Verify validation package structure
        self.assertIn('final_verdict', validation_package)
        self.assertIn('validation_scores', validation_package)
        self.assertIn('metadata', validation_package)
        self.assertIn('performance_metrics', validation_package)
        self.assertIn('risk_assessment', validation_package)
        
        # Check that package files were created
        package_file = self.test_result_path / "validation_package.json"
        summary_file = self.test_result_path / "validation_summary.json"
        self.assertTrue(package_file.exists())
        self.assertTrue(summary_file.exists())
        
        # Step 6: Test Telegram notification (mocked)
        with patch.object(self.telegram_notifier, 'send_message_async') as mock_send:
            notification_result = self.telegram_notifier.Notify_Telegram_Update(
                asset=self.test_asset,
                strategy=self.test_strategy,
                validation_result=validation_package,
                result_path=str(self.test_result_path)
            )
            
            # Verify notification structure
            self.assertIn('message_content', notification_result)
            self.assertIn('timestamp', notification_result)
            self.assertIsInstance(notification_result['message_content'], str)
            self.assertGreater(len(notification_result['message_content']), 0)

    def test_pipeline_crypto_futures(self):
        """Test pipeline for crypto futures with specific rules."""
        # Create test data
        trades_df = self._create_mock_trade_data()
        trades_df['leverage'] = np.random.uniform(1, 10, len(trades_df))  # Add leverage column
        equity_df = self._create_mock_equity_data()
        
        trades_file = self.test_result_path / "trades.csv"
        equity_file = self.test_result_path / "equity.csv"
        
        trades_df.to_csv(trades_file, index=False)
        equity_df.to_csv(equity_file, index=False)
        
        # Test with crypto_futures asset
        kpis = self.strategy_manager.Compute_KPIs(
            asset="crypto_futures",
            result_path=str(self.test_result_path)
        )
        
        # Verify crypto_futures specific rules are applied
        asset_rules = config_manager.get_asset_rules("crypto_futures")
        self.assertIn("stress_tests", asset_rules)
        self.assertIn("leverage_cap", asset_rules)
        self.assertEqual(asset_rules["leverage_cap"], 50)
        
        # Test compliance with leverage rules
        compliance = self.validation_manager.Risk_Compliance_Checks(
            asset="crypto_futures",
            strategy_result_path=str(self.test_result_path)
        )
        
        self.assertIn('leverage', compliance)
        self.assertIn('within_limits', compliance['leverage'])

    def test_pipeline_forex(self):
        """Test pipeline for forex trading."""
        # Create test data
        trades_df = self._create_mock_trade_data()
        equity_df = self._create_mock_equity_data()
        
        trades_file = self.test_result_path / "trades.csv"
        equity_file = self.test_result_path / "equity.csv"
        
        trades_df.to_csv(trades_file, index=False)
        equity_df.to_csv(equity_file, index=False)
        
        # Test forex-specific baselines
        baselines = self.strategy_manager.Generate_Baselines(
            asset="forex",
            symbols=["EURUSD", "GBPUSD"],
            start=self.test_start,
            end=self.test_end,
            result_path=str(self.test_result_path)
        )
        
        # Verify forex-specific baseline strategies
        asset_rules = config_manager.get_asset_rules("forex")
        expected_baselines = asset_rules.get("baselines", [])
        self.assertIn("carry_neutral_sma_50_200", expected_baselines)

    def test_pipeline_forex_options(self):
        """Test pipeline for forex options trading."""
        # Create test data
        trades_df = self._create_mock_trade_data()
        equity_df = self._create_mock_equity_data()
        
        trades_file = self.test_result_path / "trades.csv"
        equity_file = self.test_result_path / "equity.csv"
        
        trades_df.to_csv(trades_file, index=False)
        equity_df.to_csv(equity_file, index=False)
        
        # Test options-specific compliance
        compliance = self.validation_manager.Risk_Compliance_Checks(
            asset="forex_options",
            strategy_result_path=str(self.test_result_path)
        )
        
        # Verify Greeks compliance is checked for options
        self.assertIn('greeks', compliance)
        self.assertIn('delta_within_band', compliance['greeks'])
        self.assertIn('gamma_within_limits', compliance['greeks'])
        self.assertIn('vega_within_limits', compliance['greeks'])

    def test_pipeline_failure_scenarios(self):
        """Test pipeline behavior with various failure scenarios."""
        
        # Test 1: Missing files
        with self.assertRaises(FileNotFoundError):
            self.strategy_manager.Compute_KPIs(
                asset=self.test_asset,
                result_path=str(self.test_result_path)
            )
        
        # Test 2: Empty trades data
        empty_trades = pd.DataFrame(columns=['symbol', 'side', 'entry_price', 'exit_price', 'quantity', 'timestamp'])
        equity_df = self._create_mock_equity_data()
        
        empty_trades.to_csv(self.test_result_path / "trades.csv", index=False)
        equity_df.to_csv(self.test_result_path / "equity.csv", index=False)
        
        kpis = self.strategy_manager.Compute_KPIs(
            asset=self.test_asset,
            result_path=str(self.test_result_path)
        )
        
        # Should handle empty data gracefully
        self.assertEqual(kpis['total_trades'], 0)
        self.assertEqual(kpis['win_rate'], 0)

    def test_asset_specific_thresholds(self):
        """Test that asset-specific thresholds are correctly applied."""
        
        for asset in ["crypto_spot", "crypto_futures", "forex", "forex_options"]:
            asset_rules = config_manager.get_asset_rules(asset)
            thresholds = asset_rules.get("thresholds", {})
            
            # Verify key thresholds exist
            self.assertIn("sharpe", thresholds)
            self.assertIn("max_dd", thresholds)
            self.assertIn("profit_factor", thresholds)
            
            # Verify asset-specific differences
            if asset == "forex_options":
                # Options should have more relaxed thresholds
                self.assertLessEqual(thresholds.get("sharpe", 2.0), 2.0)
                self.assertGreaterEqual(thresholds.get("max_dd", 0.15), 0.15)

    def test_validation_scoring_system(self):
        """Test the validation scoring and weighting system."""
        
        # Create test data with known characteristics
        trades_df = self._create_mock_trade_data()
        equity_df = self._create_mock_equity_data()
        
        trades_file = self.test_result_path / "trades.csv"
        equity_file = self.test_result_path / "equity.csv"
        
        trades_df.to_csv(trades_file, index=False)
        equity_df.to_csv(equity_file, index=False)
        
        # Run pipeline components
        kpis = self.strategy_manager.Compute_KPIs(
            asset=self.test_asset,
            result_path=str(self.test_result_path)
        )
        
        baselines = self.strategy_manager.Generate_Baselines(
            asset=self.test_asset,
            symbols=self.test_symbols,
            start=self.test_start,
            end=self.test_end,
            result_path=str(self.test_result_path)
        )
        
        robustness = self.validation_manager.Robustness_Checks(
            asset=self.test_asset,
            strategy_result_path=str(self.test_result_path),
            baseline_results=baselines
        )
        
        compliance = self.validation_manager.Risk_Compliance_Checks(
            asset=self.test_asset,
            strategy_result_path=str(self.test_result_path)
        )
        
        # Test validation package scoring
        validation_package = self.strategy_manager.Prepare_Validator_Package(
            asset=self.test_asset,
            strategy=self.test_strategy,
            kpis=kpis,
            baselines=baselines,
            robustness=robustness,
            compliance=compliance,
            result_path=str(self.test_result_path)
        )
        
        # Verify scoring components
        scores = validation_package['validation_scores']
        self.assertIn('kpi_score', scores)
        self.assertIn('baseline_score', scores)
        self.assertIn('robustness_score', scores)
        self.assertIn('compliance_score', scores)
        self.assertIn('final_score', scores)
        
        # Verify scores are in valid range [0, 1]
        for score_name, score_value in scores.items():
            self.assertGreaterEqual(score_value, 0.0, f"{score_name} should be >= 0")
            self.assertLessEqual(score_value, 1.0, f"{score_name} should be <= 1")


if __name__ == '__main__':
    unittest.main()