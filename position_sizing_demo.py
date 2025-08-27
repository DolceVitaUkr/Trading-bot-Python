#!/usr/bin/env python3
"""
Demo runner for world-class position sizing component.

This demonstrates the position sizing algorithm with various scenarios:
- Fixed $10 mode for small accounts
- Risk percentage mode for larger accounts  
- Drawdown protection and kill switches
- Leverage constraints and volatility adjustments
- Allocation caps and edge case handling
"""

from trading_bot.risk.position_sizing import (
    PositionSizingConfig,
    AccountState,
    MarketInputs,
    compute_position_size
)


def print_result(scenario_name: str, result):
    """Print position sizing result in a formatted way."""
    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*60}")
    
    if result.allowed:
        print(f"Position ALLOWED ({result.reason})")
        print(f"   Notional:     ${result.notional_usd:,.2f}")
        print(f"   Margin:       ${result.margin_usd:,.2f}")
        print(f"   Leverage:     {result.leverage:.2f}x")
        print(f"   Quantity:     {result.qty:.6f}")
        print(f"   Risk at SL:   ${result.risk_at_sl_usd:,.2f}")
    else:
        print(f"Position BLOCKED: {result.reason}")
        print(f"   Notional:     ${result.notional_usd:,.2f}")
        print(f"   Margin:       ${result.margin_usd:,.2f}")
        print(f"   Quantity:     {result.qty:.6f}")


def demo_position_sizing():
    """Run comprehensive position sizing demonstrations."""
    
    print("WORLD-CLASS POSITION SIZING COMPONENT DEMO")
    print("=" * 60)
    
    # Configuration
    config = PositionSizingConfig(
        base_fixed_usd=10.0,
        transition_equity_usd=1000.0,
        risk_percent=0.01,
        max_alloc_percent=0.10,
        max_used_margin_percent=0.02,
        max_concurrent_exposure_percent=0.40,
        kill_switch_drawdown=0.30,
        kill_switch_consecutive_losses=5
    )
    
    # Scenario 1: Small Account - Fixed $10 Mode
    print_result(
        "Small Account - Fixed $10 Mode",
        compute_position_size(
            config,
            AccountState(equity_usd=500, peak_equity_usd=500, consecutive_losses=0, open_exposure_usd=0),
            MarketInputs(entry_price=100.0, stop_price=98.0)
        )
    )
    
    # Scenario 2: Medium Account - Risk Percentage Mode
    print_result(
        "Medium Account - Risk Percentage Mode", 
        compute_position_size(
            config,
            AccountState(equity_usd=5000, peak_equity_usd=7000, consecutive_losses=0, open_exposure_usd=1000),
            MarketInputs(entry_price=100.0, stop_price=98.0, atr_pct=0.03)
        )
    )
    
    # Scenario 3: High Volatility Environment
    print_result(
        "High Volatility - Size Reduction",
        compute_position_size(
            config,
            AccountState(equity_usd=10000, peak_equity_usd=12000, consecutive_losses=0, open_exposure_usd=0),
            MarketInputs(entry_price=100.0, stop_price=98.0, atr_pct=0.15)  # Very high ATR
        )
    )
    
    # Scenario 4: Drawdown Protection - Throttled Risk
    print_result(
        "Drawdown Protection - 12% Drawdown", 
        compute_position_size(
            config,
            AccountState(equity_usd=4400, peak_equity_usd=5000, consecutive_losses=0, open_exposure_usd=0),
            MarketInputs(entry_price=100.0, stop_price=98.0)
        )
    )
    
    # Scenario 5: Wide Stop Loss - Leverage Reduction
    print_result(
        "Wide Stop Loss - Auto Leverage Reduction",
        compute_position_size(
            config,
            AccountState(equity_usd=10000, peak_equity_usd=12000, consecutive_losses=0, open_exposure_usd=0),
            MarketInputs(entry_price=100.0, stop_price=80.0)  # 20% stop distance
        )
    )
    
    # Scenario 6: Kill Switch - Excessive Drawdown
    print_result(
        "Kill Switch - Excessive Drawdown (35%)",
        compute_position_size(
            config,
            AccountState(equity_usd=3250, peak_equity_usd=5000, consecutive_losses=0, open_exposure_usd=0),
            MarketInputs(entry_price=100.0, stop_price=98.0)
        )
    )
    
    # Scenario 7: Kill Switch - Consecutive Losses
    print_result(
        "Kill Switch - Too Many Consecutive Losses",
        compute_position_size(
            config,
            AccountState(equity_usd=5000, peak_equity_usd=7000, consecutive_losses=5, open_exposure_usd=0),
            MarketInputs(entry_price=100.0, stop_price=98.0)
        )
    )
    
    # Scenario 8: Edge Case - Invalid Stop Loss
    print_result(
        "Edge Case - Invalid Stop Loss",
        compute_position_size(
            config,
            AccountState(equity_usd=5000, peak_equity_usd=7000, consecutive_losses=0, open_exposure_usd=0),
            MarketInputs(entry_price=100.0, stop_price=100.0)  # Stop equals entry
        )
    )
    
    # Scenario 9: Large Account with High Leverage
    print_result(
        "Large Account - High Leverage Available",
        compute_position_size(
            config,
            AccountState(equity_usd=100000, peak_equity_usd=120000, consecutive_losses=0, open_exposure_usd=10000),
            MarketInputs(entry_price=100.0, stop_price=99.5)  # Tight 0.5% stop
        )
    )
    
    # Scenario 10: Allocation Cap Test
    print_result(
        "Allocation Cap - Position Capped by Max Allocation",
        compute_position_size(
            config,
            AccountState(equity_usd=2000, peak_equity_usd=2000, consecutive_losses=0, open_exposure_usd=0),
            MarketInputs(entry_price=100.0, stop_price=95.0)  # 5% stop, would normally be large position
        )
    )
    
    print(f"\n{'='*60}")
    print("DEMO COMPLETE - Position sizing component ready for production!")
    print("Key Features Demonstrated:")
    print("   * Fixed $10 mode -> Risk percentage mode transition")
    print("   * Drawdown-based risk throttling")
    print("   * Volatility-aware size adjustments") 
    print("   * Leverage constraints by equity and stop distance")
    print("   * Allocation and margin usage caps")
    print("   * Kill switches for drawdown and consecutive losses")
    print("   * Edge case handling and input validation")
    print(f"{'='*60}")


if __name__ == "__main__":
    demo_position_sizing()