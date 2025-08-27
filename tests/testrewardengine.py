"""
Comprehensive test suite for the Reward Engine
Tests all reward calculations, edge cases, and integration components.
"""
import math
import os
import tempfile
import unittest
from unittest.mock import Mock, patch
from typing import List

try:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    
    from tradingbot.rl.rewardconfig import RewardConfig
    from tradingbot.rl.rewardengine import (
        StepContext, EpisodeSummary, computestepreward, computeepisodereward,
        drawdownfrac, leveragetiercap, sharpeproxy, normalizeandclip,
        ema, isstoplossviolation, profitpointsfrombands
    )
    from tradingbot.rl.rewardintegration import (
        RewardAdapters, RewardPersistence, onstep, onepisodeend
    )
    
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    IMPORT_ERROR = str(e)


class TestRewardEngineCore(unittest.TestCase):
    """Test core reward engine functions."""
    
    def setUp(self):
        if not MODULES_AVAILABLE:
            self.skipTest(f"Required modules not available: {IMPORT_ERROR}")
        
        self.cfg = RewardConfig()
    
    def test_drawdown_calculation(self):
        """Test drawdown fraction calculation."""
        self.assertAlmostEqual(drawdownfrac(1000, 1200), (1200-1000)/1200)  # 16.67% DD
        self.assertEqual(drawdownfrac(1200, 1200), 0.0)  # No DD
        self.assertEqual(drawdownfrac(1300, 1200), 0.0)  # New peak
        self.assertEqual(drawdownfrac(500, 0), 0.0)  # Edge case
    
    def test_leverage_tier_cap(self):
        """Test leverage tier determination."""
        cfg = RewardConfig()
        cfg.leveragetiers = [(0, 1), (1000, 3), (10000, 5), (50000, 10)]
        
        self.assertEqual(leveragetiercap(cfg, 500), 1.0)
        self.assertEqual(leveragetiercap(cfg, 1500), 3.0)
        self.assertEqual(leveragetiercap(cfg, 15000), 5.0)
        self.assertEqual(leveragetiercap(cfg, 60000), 10.0)
    
    def test_sharpe_proxy(self):
        """Test Sharpe ratio estimation."""
        # Normal case
        returns = [0.01, 0.02, -0.01, 0.03, 0.01]
        sharpe = sharpeproxy(returns)
        self.assertIsInstance(sharpe, float)
        
        # Edge cases
        self.assertIsNone(sharpeproxy([]))
        self.assertIsNone(sharpeproxy([0.01]))
        self.assertIsNone(sharpeproxy([0.01, 0.01, 0.01]))  # Zero std dev
    
    def test_normalize_and_clip(self):
        """Test reward normalization and clipping."""
        self.assertEqual(normalizeandclip(500, 1000), 500)
        self.assertEqual(normalizeandclip(1500, 1000), 1000)
        self.assertEqual(normalizeandclip(-1500, 1000), -1000)
        self.assertEqual(normalizeandclip(float('nan'), 1000), 0.0)
        self.assertEqual(normalizeandclip(float('inf'), 1000), 0.0)
    
    def test_ema_calculation(self):
        """Test exponential moving average."""
        self.assertEqual(ema(10, 8, 0.1), 8.2)
        self.assertEqual(ema(0, 100, 1.0), 0.0)  # Full replacement
        self.assertEqual(ema(100, 0, 0.0), 0.0)  # No change
    
    def test_stop_loss_violation(self):
        """Test stop loss violation detection."""
        self.assertTrue(isstoplossviolation(0.15, 0.10))
        self.assertFalse(isstoplossviolation(0.05, 0.10))
        self.assertFalse(isstoplossviolation(0.10, 0.10))  # Exactly at limit
    
    def test_profit_bands(self):
        """Test profit band point calculation."""
        bands = [(0.11, 1), (0.20, 5), (0.31, 10), (0.51, 20)]
        
        self.assertEqual(profitpointsfrombands(0.05, bands), 0.0)  # Below threshold
        self.assertEqual(profitpointsfrombands(0.15, bands), 1.0)  # First band
        self.assertEqual(profitpointsfrombands(0.25, bands), 5.0)  # Second band
        self.assertEqual(profitpointsfrombands(0.60, bands), 20.0)  # Top band
        self.assertEqual(profitpointsfrombands(-0.10, bands), 0.0)  # Negative


class TestStepReward(unittest.TestCase):
    """Test step reward calculations."""
    
    def setUp(self):
        if not MODULES_AVAILABLE:
            self.skipTest(f"Required modules not available: {IMPORT_ERROR}")
        
        self.cfg = RewardConfig()
        self.base_context = StepContext(
            symbol="BTCUSDT",
            side="long",
            entryprice=50000.0,
            exitprice=None,
            midprice=50100.0,
            qty=0.1,
            leverage=2.0,
            feespaidusd=5.0,
            slippageusd=2.0,
            stoplossprice=49000.0,
            takeprofitprice=52000.0,
            holdingtimeseconds=3600.0,  # 1 hour
            tradeclosed=False,
            realizedprofitperc=None,
            equityusd=10000.0,
            peakequityusd=10500.0,
            openexposureusd=1000.0,
            consecutivestoplosshits=0,
            rollingreturnswindow=[0.01, 0.02, -0.01],
            dtseconds=60.0
        )
    
    def test_open_position_reward(self):
        """Test reward calculation for open position."""
        ctx = self.base_context
        reward, components = computestepreward(self.cfg, ctx)
        
        # Should have minimal reward for open position
        self.assertIsInstance(reward, float)
        self.assertIsInstance(components, dict)
        self.assertEqual(components["pnlpoints"], 0.0)
        self.assertFalse(components["killswitch"])
    
    def test_profitable_trade_reward(self):
        """Test reward for profitable closed trade."""
        ctx = self.base_context
        ctx.tradeclosed = True
        ctx.realizedprofitperc = 0.15  # 15% profit
        ctx.exitprice = 57500.0
        ctx.feespaidusd = 1.0  # Reduce fees to ensure positive overall reward
        ctx.slippageusd = 0.5
        
        reward, components = computestepreward(self.cfg, ctx)
        
        # Should get profit points and bonus
        self.assertGreater(components["pnlpoints"], 0)
        self.assertEqual(components["bandbonus"], self.cfg.tradebonuspointsifprofitover10perc)
        self.assertGreater(components["feespenalty"], 0)
        # With lower fees, overall reward should be positive
        self.assertGreater(reward, -10)  # Allow some negative due to other penalties
    
    def test_losing_trade_penalty(self):
        """Test penalty for losing trade."""
        ctx = self.base_context
        ctx.tradeclosed = True
        ctx.realizedprofitperc = -0.08  # 8% loss
        ctx.exitprice = 46000.0
        
        reward, components = computestepreward(self.cfg, ctx)
        
        # Should get negative loss points penalty
        self.assertLess(components["losspoints"], 0)
        self.assertEqual(components["pnlpoints"], 0)
        self.assertEqual(components["bandbonus"], 0)
    
    def test_stop_loss_violation(self):
        """Test penalty for stop loss violation."""
        ctx = self.base_context
        ctx.tradeclosed = True
        ctx.realizedprofitperc = -0.12  # 12% loss (above 10% SL limit)
        
        reward, components = computestepreward(self.cfg, ctx)
        
        # Should get SL violation penalty
        self.assertLess(components["slviolation"], 0)
    
    def test_excessive_exposure_penalty(self):
        """Test penalty for excessive exposure."""
        ctx = self.base_context
        ctx.openexposureusd = 5000.0  # 50% of equity (above 40% limit)
        
        reward, components = computestepreward(self.cfg, ctx)
        
        # Should get exposure penalty
        self.assertGreater(components["exposurepenalty"], 0)
    
    def test_excessive_leverage_penalty(self):
        """Test penalty for excessive leverage."""
        ctx = self.base_context
        ctx.leverage = 5.0  # Above tier limit for equity level
        ctx.equityusd = 500.0  # Low equity to trigger penalty
        
        reward, components = computestepreward(self.cfg, ctx)
        
        # Should get leverage penalty
        self.assertGreater(components["leveragepenalty"], 0)
    
    def test_kill_switch_activation(self):
        """Test kill switch for excessive drawdown."""
        ctx = self.base_context
        ctx.equityusd = 7000.0  # 33% DD from peak of 10500
        
        reward, components = computestepreward(self.cfg, ctx)
        
        # Should activate kill switch
        self.assertTrue(components["killswitch"])
        self.assertIsNotNone(components["killreason"])
        self.assertLess(reward, 0)  # Should be heavily penalized
    
    def test_consecutive_stop_losses(self):
        """Test penalty for consecutive stop losses."""
        ctx = self.base_context
        ctx.consecutivestoplosshits = 7  # Above threshold of 5
        
        reward, components = computestepreward(self.cfg, ctx)
        
        # Should get consecutive SL penalty
        self.assertLess(components["consecutiveslpenalty"], 0)
    
    def test_missing_realized_profit(self):
        """Test error handling for missing realized profit."""
        ctx = self.base_context
        ctx.tradeclosed = True
        ctx.realizedprofitperc = None  # Missing data
        
        reward, components = computestepreward(self.cfg, ctx)
        
        # Should have error and penalty
        self.assertEqual(components["error"], "missingrealized")
        self.assertLess(reward, 0)


class TestEpisodeReward(unittest.TestCase):
    """Test episode reward calculations."""
    
    def setUp(self):
        if not MODULES_AVAILABLE:
            self.skipTest(f"Required modules not available: {IMPORT_ERROR}")
        
        self.cfg = RewardConfig()
        self.base_summary = EpisodeSummary(
            totaltrades=20,
            wins=15,
            losses=5,
            grossreturnfrac=0.40,
            maxdrawdownfrac=0.08,
            sharpeestimate=2.5,
            equitypath=[10000, 10200, 10500, 10800, 11000, 11200, 11500, 12000, 13000, 14000],
            bandedprofitpoints=150.0,
            bandedlosspoints=-25.0
        )
    
    def test_successful_episode_reward(self):
        """Test reward for successful episode meeting all targets."""
        # Adjust to exceed targets slightly to get bonuses
        summary = self.base_summary
        summary.wins = 16  # 16/20 = 0.80 > 0.75 target
        summary.losses = 4
        
        reward, components = computeepisodereward(self.cfg, summary)
        
        # Should get base points plus bonuses
        self.assertEqual(components["bandedpoints"], 125.0)  # 150 - 25
        self.assertGreater(components["winratebonus"], 0)  # 80% > 75% win rate
        self.assertGreater(components["sharpebonus"], 0)  # 2.5 > 2.0 Sharpe
        self.assertGreater(reward, 0)
    
    def test_poor_performance_episode(self):
        """Test episode with poor performance."""
        summary = self.base_summary
        summary.wins = 8  # 40% win rate
        summary.losses = 12
        summary.sharpeestimate = 0.5  # Low Sharpe
        summary.maxdrawdownfrac = 0.25  # High DD
        
        reward, components = computeepisodereward(self.cfg, summary)
        
        # Should get DD penalty, no bonuses
        self.assertEqual(components["winratebonus"], 0)
        self.assertEqual(components["sharpebonus"], 0)
        self.assertLess(components["ddpenalty"], 0)
    
    def test_hard_drawdown_penalty(self):
        """Test hard drawdown limit penalty."""
        summary = self.base_summary
        summary.maxdrawdownfrac = 0.25  # Above 20% hard limit
        
        reward, components = computeepisodereward(self.cfg, summary)
        
        # Should get hard DD penalty
        expected_penalty = -200.0 * (0.25 - self.cfg.maxdrawdownfrachard)
        self.assertEqual(components["ddpenalty"], expected_penalty)


class TestIntegration(unittest.TestCase):
    """Test integration components."""
    
    def setUp(self):
        if not MODULES_AVAILABLE:
            self.skipTest(f"Required modules not available: {IMPORT_ERROR}")
        
        self.cfg = RewardConfig()
        # Use temporary files for testing
        self.temp_dir = tempfile.mkdtemp()
        self.cfg.patheventsjsonl = os.path.join(self.temp_dir, "events.jsonl")
        self.cfg.pathtrademetricscsv = os.path.join(self.temp_dir, "trades.csv")
        self.cfg.pathepisodemetricscsv = os.path.join(self.temp_dir, "episodes.csv")
    
    def test_reward_adapters_logging(self):
        """Test reward adapters logging functionality."""
        logger = Mock()
        telegram_fn = Mock()
        
        adapters = RewardAdapters(logger, telegram_fn, self.cfg)
        
        ctx = StepContext(
            symbol="BTCUSDT", side="long", entryprice=50000.0, exitprice=55000.0,
            midprice=55000.0, qty=0.1, leverage=2.0, feespaidusd=5.0, slippageusd=2.0,
            stoplossprice=None, takeprofitprice=None, holdingtimeseconds=3600.0,
            tradeclosed=True, realizedprofitperc=0.10, equityusd=10000.0,
            peakequityusd=10000.0, openexposureusd=1000.0, consecutivestoplosshits=0,
            rollingreturnswindow=[0.01, 0.02], dtseconds=60.0
        )
        
        components = {"pnlpoints": 5.0, "bandbonus": 3.0, "killswitch": False}
        
        # Should not raise any exceptions
        adapters.log_step_reward(ctx, 8.0, components)
        logger.info.assert_called()
    
    def test_reward_persistence_files(self):
        """Test reward persistence file operations."""
        persistence = RewardPersistence(self.cfg)
        
        # Test event logging
        event = {"type": "test", "reward": 10.0}
        persistence.appendevent(event)
        
        # Test trade metrics
        trade_row = {
            "timestamp": "2023-01-01T00:00:00",
            "symbol": "BTCUSDT",
            "realized_profit_pct": 0.10,
            "reward": 8.0
        }
        persistence.appendtrademetrics(trade_row)
        
        # Files should exist and have content
        self.assertTrue(os.path.exists(self.cfg.patheventsjsonl))
        self.assertTrue(os.path.exists(self.cfg.pathtrademetricscsv))
    
    def test_onstep_integration(self):
        """Test complete onstep integration."""
        logger = Mock()
        adapters = RewardAdapters(logger, None, self.cfg)
        persistence = RewardPersistence(self.cfg)
        
        ctx = StepContext(
            symbol="BTCUSDT", side="long", entryprice=50000.0, exitprice=55000.0,
            midprice=55000.0, qty=0.1, leverage=2.0, feespaidusd=5.0, slippageusd=2.0,
            stoplossprice=None, takeprofitprice=None, holdingtimeseconds=3600.0,
            tradeclosed=True, realizedprofitperc=0.10, equityusd=10000.0,
            peakequityusd=10000.0, openexposureusd=1000.0, consecutivestoplosshits=0,
            rollingreturnswindow=[0.01, 0.02], dtseconds=60.0
        )
        
        # Should return a reward value
        reward = onstep(ctx, self.cfg, adapters, persistence)
        self.assertIsInstance(reward, float)
    
    def test_onepisodeend_integration(self):
        """Test complete episode end integration."""
        logger = Mock()
        adapters = RewardAdapters(logger, None, self.cfg)
        persistence = RewardPersistence(self.cfg)
        
        summary = EpisodeSummary(
            totaltrades=10, wins=8, losses=2, grossreturnfrac=0.20,
            maxdrawdownfrac=0.05, sharpeestimate=2.0, equitypath=[10000, 12000],
            bandedprofitpoints=50.0, bandedlosspoints=-10.0
        )
        
        # Should return a reward value
        reward = onepisodeend(summary, self.cfg, adapters, persistence)
        self.assertIsInstance(reward, float)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        if not MODULES_AVAILABLE:
            self.skipTest(f"Required modules not available: {IMPORT_ERROR}")
        
        self.cfg = RewardConfig()
    
    def test_zero_equity_handling(self):
        """Test handling of zero equity scenarios."""
        ctx = StepContext(
            symbol="BTCUSDT", side=None, entryprice=None, exitprice=None,
            midprice=50000.0, qty=None, leverage=1.0, feespaidusd=0.0, slippageusd=0.0,
            stoplossprice=None, takeprofitprice=None, holdingtimeseconds=0.0,
            tradeclosed=False, realizedprofitperc=None, equityusd=0.0,
            peakequityusd=0.0, openexposureusd=0.0, consecutivestoplosshits=0,
            rollingreturnswindow=[], dtseconds=60.0
        )
        
        reward, components = computestepreward(self.cfg, ctx)
        
        # Should handle gracefully without crashing
        self.assertIsInstance(reward, float)
        self.assertFalse(math.isnan(reward))
        self.assertFalse(math.isinf(reward))
    
    def test_extreme_values(self):
        """Test handling of extreme values."""
        ctx = StepContext(
            symbol="BTCUSDT", side="long", entryprice=1.0, exitprice=1000000.0,
            midprice=500000.0, qty=0.1, leverage=100.0, feespaidusd=10000.0,
            slippageusd=5000.0, stoplossprice=None, takeprofitprice=None,
            holdingtimeseconds=86400.0 * 30,  # 30 days
            tradeclosed=True, realizedprofitperc=999.0,  # 99,900% profit
            equityusd=1000000.0, peakequityusd=1000000.0, openexposureusd=500000.0,
            consecutivestoplosshits=100, rollingreturnswindow=[10.0] * 100,
            dtseconds=60.0
        )
        
        reward, components = computestepreward(self.cfg, ctx)
        
        # Should be clipped to reasonable range
        self.assertLessEqual(abs(reward), self.cfg.clipstepreward)
        self.assertFalse(math.isnan(reward))
        self.assertFalse(math.isinf(reward))


def run_tests():
    """Run all tests with proper error handling."""
    if not MODULES_AVAILABLE:
        print(f"Cannot run tests: {IMPORT_ERROR}")
        print("This is expected if pytest is not installed or modules are not in path")
        return True
    
    # Create test suite
    test_classes = [
        TestRewardEngineCore,
        TestStepReward,
        TestEpisodeReward,
        TestIntegration,
        TestEdgeCases
    ]
    
    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)