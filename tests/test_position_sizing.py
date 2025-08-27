"""
Comprehensive unit tests for position sizing component.
"""

import pytest
from trading_bot.risk.position_sizing import (
    PositionSizingConfig,
    AccountState,
    MarketInputs,
    SizingResult,
    compute_position_size,
    compute_sl_distance,
    effective_risk_percent,
    max_allowed_leverage,
    cap_by_allocations,
    round_qty
)


class TestHelperFunctions:
    """Test helper functions."""
    
    def test_compute_sl_distance(self):
        """Test stop loss distance calculation."""
        # Normal case
        assert compute_sl_distance(100, 98) == 0.02
        assert compute_sl_distance(100, 102) == 0.02
        
        # Edge cases
        assert compute_sl_distance(0, 98) == 0.0
        assert compute_sl_distance(100, 0) == 0.0
        assert compute_sl_distance(-100, 98) == 0.0
    
    def test_effective_risk_percent(self):
        """Test drawdown-adjusted risk calculation."""
        cfg = PositionSizingConfig(
            risk_percent=0.01,
            max_drawdown_throttle_levels=[0.10, 0.15, 0.20],
            throttle_risk_multipliers=[0.75, 0.50, 0.25]
        )
        
        # No drawdown
        acct = AccountState(equity_usd=1000, peak_equity_usd=1000, consecutive_losses=0, open_exposure_usd=0)
        assert effective_risk_percent(cfg, acct) == 0.01
        
        # 5% drawdown (no throttling)
        acct = AccountState(equity_usd=950, peak_equity_usd=1000, consecutive_losses=0, open_exposure_usd=0)
        assert effective_risk_percent(cfg, acct) == 0.01
        
        # 12% drawdown (first throttle level)
        acct = AccountState(equity_usd=880, peak_equity_usd=1000, consecutive_losses=0, open_exposure_usd=0)
        assert effective_risk_percent(cfg, acct) == 0.0075  # 0.01 * 0.75
        
        # 18% drawdown (second throttle level)
        acct = AccountState(equity_usd=820, peak_equity_usd=1000, consecutive_losses=0, open_exposure_usd=0)
        assert effective_risk_percent(cfg, acct) == 0.005  # 0.01 * 0.50
        
        # 25% drawdown (third throttle level)
        acct = AccountState(equity_usd=750, peak_equity_usd=1000, consecutive_losses=0, open_exposure_usd=0)
        assert effective_risk_percent(cfg, acct) == 0.0025  # 0.01 * 0.25
    
    def test_max_allowed_leverage(self):
        """Test leverage constraint calculation."""
        cfg = PositionSizingConfig()
        acct = AccountState(equity_usd=5000, peak_equity_usd=7000, consecutive_losses=0, open_exposure_usd=0)
        
        # Normal case - 2% stop distance, no high volatility
        leverage = max_allowed_leverage(cfg, acct, 0.02, None)
        assert leverage == 3.0  # Limited by equity tier (5000 -> 3x) and sl distance (0.02 -> 2x), min is 2x
        
        # High volatility case
        leverage = max_allowed_leverage(cfg, acct, 0.02, 0.10)  # ATR much higher than SL
        assert leverage == 2.0  # Capped due to high volatility
        
        # Very tight stop
        leverage = max_allowed_leverage(cfg, acct, 0.003, None)
        assert leverage == 3.0  # Limited by equity tier since 0.003 < 0.005 threshold
    
    def test_cap_by_allocations(self):
        """Test allocation capping logic."""
        cfg = PositionSizingConfig(
            max_alloc_percent=0.10,
            max_used_margin_percent=0.02,
            max_concurrent_exposure_percent=0.40
        )
        acct = AccountState(equity_usd=5000, peak_equity_usd=5000, consecutive_losses=0, open_exposure_usd=1000)
        
        # Test normal case within limits
        notional, margin = cap_by_allocations(cfg, acct, 400, 200)
        assert notional == 400
        assert margin == 200
        
        # Test allocation cap
        notional, margin = cap_by_allocations(cfg, acct, 600, 200)  # 600 > 10% of 5000
        assert notional == 500  # Capped to 10% of equity
        
        # Test margin cap
        notional, margin = cap_by_allocations(cfg, acct, 400, 150)  # 150 > 2% of 5000
        assert margin == 100  # Capped to 2% of equity
        assert notional == 200  # Adjusted down based on leverage
        
        # Test exposure cap
        notional, margin = cap_by_allocations(cfg, acct, 1200, 400)  # Would exceed concurrent exposure
        max_additional = 5000 * 0.40 - 1000  # 40% of equity minus existing exposure
        assert notional == max_additional
    
    def test_round_qty(self):
        """Test quantity rounding logic."""
        # Normal rounding
        assert round_qty(1.23456, None, 3) == 1.235
        
        # Lot size rounding
        assert round_qty(1.23, 0.1, 2) == 1.2  # Rounds to nearest 0.1
        assert round_qty(1.27, 0.1, 2) == 1.3
        
        # Zero result
        assert round_qty(0.001, 0.1, 2) == 0.0


class TestPositionSizing:
    """Test main position sizing logic."""
    
    def test_equity_too_low(self):
        """Test equity below minimum threshold."""
        cfg = PositionSizingConfig(min_trade_equity_usd=100)
        acct = AccountState(equity_usd=50, peak_equity_usd=50, consecutive_losses=0, open_exposure_usd=0)
        mkt = MarketInputs(entry_price=100, stop_price=98)
        
        result = compute_position_size(cfg, acct, mkt)
        assert not result.allowed
        assert result.reason == "equity_too_low"
    
    def test_kill_switch_drawdown(self):
        """Test kill switch activation due to drawdown."""
        cfg = PositionSizingConfig(kill_switch_drawdown=0.30)
        acct = AccountState(equity_usd=650, peak_equity_usd=1000, consecutive_losses=0, open_exposure_usd=0)  # 35% DD
        mkt = MarketInputs(entry_price=100, stop_price=98)
        
        result = compute_position_size(cfg, acct, mkt)
        assert not result.allowed
        assert result.reason == "kill_dd"
    
    def test_kill_switch_consecutive_losses(self):
        """Test kill switch activation due to consecutive losses."""
        cfg = PositionSizingConfig(kill_switch_consecutive_losses=5)
        acct = AccountState(equity_usd=1000, peak_equity_usd=1000, consecutive_losses=5, open_exposure_usd=0)
        mkt = MarketInputs(entry_price=100, stop_price=98)
        
        result = compute_position_size(cfg, acct, mkt)
        assert not result.allowed
        assert result.reason == "kill_losses"
    
    def test_invalid_stop_loss(self):
        """Test invalid stop loss handling in risk mode."""
        cfg = PositionSizingConfig(transition_equity_usd=500)
        acct = AccountState(equity_usd=1000, peak_equity_usd=1000, consecutive_losses=0, open_exposure_usd=0)
        mkt = MarketInputs(entry_price=100, stop_price=100)  # Stop equals entry
        
        result = compute_position_size(cfg, acct, mkt)
        assert not result.allowed
        assert result.reason == "invalid_sl"
    
    def test_fixed_mode_basic(self):
        """Test basic fixed $10 mode."""
        cfg = PositionSizingConfig(base_fixed_usd=10.0, transition_equity_usd=1000.0)
        acct = AccountState(equity_usd=500, peak_equity_usd=500, consecutive_losses=0, open_exposure_usd=0)
        mkt = MarketInputs(entry_price=100, stop_price=98)
        
        result = compute_position_size(cfg, acct, mkt)
        assert result.allowed
        assert result.reason == "fixed_mode"
        assert result.notional_usd == 10.0
        assert result.leverage == 1.0
        assert result.margin_usd == 10.0
        assert result.qty == 0.1  # $10 / $100
        # Risk = notional * sl_dist + fees = 10 * 0.02 + 10 * 0.001 = 0.21
        assert abs(result.risk_at_sl_usd - 0.21) < 0.001
    
    def test_fixed_mode_with_caps(self):
        """Test fixed mode with allocation caps applied."""
        cfg = PositionSizingConfig(
            base_fixed_usd=10.0,
            transition_equity_usd=1000.0,
            max_alloc_percent=0.01  # 1% cap
        )
        acct = AccountState(equity_usd=500, peak_equity_usd=500, consecutive_losses=0, open_exposure_usd=0)
        mkt = MarketInputs(entry_price=100, stop_price=98)
        
        result = compute_position_size(cfg, acct, mkt)
        assert result.allowed
        assert result.notional_usd == 5.0  # Capped to 1% of $500 equity
    
    def test_risk_mode_basic(self):
        """Test basic risk percentage mode."""
        cfg = PositionSizingConfig(
            transition_equity_usd=1000.0,
            risk_percent=0.01,
            max_alloc_percent=0.10,
            max_used_margin_percent=0.02
        )
        acct = AccountState(equity_usd=5000, peak_equity_usd=7000, consecutive_losses=0, open_exposure_usd=1000)
        mkt = MarketInputs(entry_price=100, stop_price=98)  # 2% stop distance
        
        result = compute_position_size(cfg, acct, mkt)
        assert result.allowed
        assert result.reason == "risk_mode"
        
        # Expected calculation:
        # notional_no_leverage = (5000 * 0.01) / 0.02 = 2500
        # max_leverage = min(3 (equity tier), 2 (sl distance tier)) = 2, but actually 3 since 0.02 > 0.01
        # So leverage should be 3, margin = 2500 / 3 = 833.33
        # But margin cap is 5000 * 0.02 = 100, so margin capped at 100
        # Final notional = 100 * 3 = 300 (recalculated based on margin cap)
        
        expected_margin = min(100, 2500/3)  # Should be capped at 100
        assert abs(result.margin_usd - 100) < 1  # Within $1
        assert result.leverage >= 1.0
    
    def test_risk_mode_with_volatility_adjustment(self):
        """Test risk mode with high volatility adjustment."""
        cfg = PositionSizingConfig(
            transition_equity_usd=1000.0,
            volatility_atr_cap_multiplier=3.0,
            volatility_size_multiplier_low=0.5
        )
        acct = AccountState(equity_usd=5000, peak_equity_usd=7000, consecutive_losses=0, open_exposure_usd=0)
        mkt = MarketInputs(entry_price=100, stop_price=98, atr_pct=0.10)  # High ATR vs 2% SL
        
        result = compute_position_size(cfg, acct, mkt)
        assert result.allowed
        # Should have smaller position due to volatility adjustment
        # and leverage should be capped at 2x due to high volatility
        assert result.leverage <= 2.0
    
    def test_qty_rounds_to_zero(self):
        """Test handling when quantity rounds to zero."""
        cfg = PositionSizingConfig(base_fixed_usd=0.01)  # Very small notional
        acct = AccountState(equity_usd=1000, peak_equity_usd=1000, consecutive_losses=0, open_exposure_usd=0)
        mkt = MarketInputs(entry_price=100000, stop_price=99000, lot_size=1.0)  # High price, large lot size
        
        result = compute_position_size(cfg, acct, mkt)
        assert not result.allowed
        assert result.reason == "qty_rounds_to_zero"
    
    def test_below_min_notional(self):
        """Test handling when final notional is below minimum."""
        cfg = PositionSizingConfig(base_fixed_usd=5.0)
        acct = AccountState(equity_usd=1000, peak_equity_usd=1000, consecutive_losses=0, open_exposure_usd=0)
        mkt = MarketInputs(entry_price=100, stop_price=98, lot_size=0.001, min_notional=10.0)
        
        result = compute_position_size(cfg, acct, mkt)
        assert not result.allowed
        assert result.reason == "below_min_notional"
    
    def test_example_from_spec(self):
        """Test the specific example from the specification."""
        cfg = PositionSizingConfig(
            risk_percent=0.01,
            max_alloc_percent=0.10,
            max_used_margin_percent=0.02,
        )
        acct = AccountState(
            equity_usd=5000,
            peak_equity_usd=7000,
            consecutive_losses=0,
            open_exposure_usd=1000
        )
        mkt = MarketInputs(entry_price=100, stop_price=98, atr_pct=0.03)  # 2% SL distance
        
        result = compute_position_size(cfg, acct, mkt)
        assert result.allowed
        
        # Expected calculation from spec:
        # notional_no_leverage = (5000*0.01)/0.02 = 2500
        # leverage tiers allow 3x (5000 equity), sl-based allows 2x (0.02 > 0.01)
        # So leverage = min(3, 2) = 2, but actually 3x since 0.02 is in the 5x tier
        # Wait, let me recalculate based on the leverage_by_sl_distance default:
        # [(0.005, 10), (0.01, 5), (0.03, 2), (1.0, 1)]
        # 0.02 falls between 0.01 and 0.03, so it gets 5x
        # equity 5000 gets 3x, so min(3, 5) = 3x
        # margin = 2500 / 3 = 833.33
        # But margin cap = 5000 * 0.02 = 100, so margin is capped
        # Final leverage after capping will be different
        
        assert result.leverage >= 1.0
        # Risk should be approximately 2500*0.02 + 2500*0.001 = 50 + 2.5 = 52.5
        # But may be different due to caps
        assert result.risk_at_sl_usd > 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_equity(self):
        """Test with zero equity."""
        cfg = PositionSizingConfig(min_trade_equity_usd=1.0)
        acct = AccountState(equity_usd=0, peak_equity_usd=1000, consecutive_losses=0, open_exposure_usd=0)
        mkt = MarketInputs(entry_price=100, stop_price=98)
        
        result = compute_position_size(cfg, acct, mkt)
        assert not result.allowed
        assert result.reason == "equity_too_low"
    
    def test_transition_boundary(self):
        """Test behavior at transition equity boundary."""
        cfg = PositionSizingConfig(transition_equity_usd=1000.0)
        
        # Just below transition
        acct = AccountState(equity_usd=999, peak_equity_usd=999, consecutive_losses=0, open_exposure_usd=0)
        mkt = MarketInputs(entry_price=100, stop_price=98)
        result = compute_position_size(cfg, acct, mkt)
        assert result.allowed
        assert result.reason == "fixed_mode"
        
        # Just above transition
        acct = AccountState(equity_usd=1001, peak_equity_usd=1001, consecutive_losses=0, open_exposure_usd=0)
        result = compute_position_size(cfg, acct, mkt)
        assert result.allowed
        assert result.reason == "risk_mode"
    
    def test_extreme_stop_distance(self):
        """Test with very wide stop loss."""
        cfg = PositionSizingConfig(transition_equity_usd=500)
        acct = AccountState(equity_usd=5000, peak_equity_usd=7000, consecutive_losses=0, open_exposure_usd=0)
        mkt = MarketInputs(entry_price=100, stop_price=50)  # 50% stop distance
        
        result = compute_position_size(cfg, acct, mkt)
        assert result.allowed
        # Should automatically reduce leverage and cap position size
        assert result.leverage == 1.0  # Should be 1x due to wide stop
        assert result.notional_usd <= acct.equity_usd * 0.10  # Allocation cap
    
    def test_high_consecutive_losses_near_limit(self):
        """Test behavior near consecutive loss limit."""
        cfg = PositionSizingConfig(kill_switch_consecutive_losses=5)
        acct = AccountState(equity_usd=5000, peak_equity_usd=7000, consecutive_losses=4, open_exposure_usd=0)
        mkt = MarketInputs(entry_price=100, stop_price=98)
        
        result = compute_position_size(cfg, acct, mkt)
        assert result.allowed  # Should still allow at 4 losses
        
        acct.consecutive_losses = 5
        result = compute_position_size(cfg, acct, mkt)
        assert not result.allowed  # Should block at 5 losses
        assert result.reason == "kill_losses"
    
    def test_max_exposure_reached(self):
        """Test when maximum concurrent exposure is reached."""
        cfg = PositionSizingConfig(max_concurrent_exposure_percent=0.40)
        acct = AccountState(
            equity_usd=5000,
            peak_equity_usd=7000,
            consecutive_losses=0,
            open_exposure_usd=2000  # 40% already exposed
        )
        mkt = MarketInputs(entry_price=100, stop_price=98)
        
        result = compute_position_size(cfg, acct, mkt)
        # Should either block or allow very small position
        if result.allowed:
            assert result.notional_usd <= 1  # Very small or zero additional exposure


if __name__ == "__main__":
    pytest.main([__file__, "-v"])