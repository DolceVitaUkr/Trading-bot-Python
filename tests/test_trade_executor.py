import pytest
from unittest.mock import MagicMock
from modules.trade_executor import TradeExecutor
from modules.Strategy_Manager import Decision

@pytest.fixture
def sizing_policy():
    """Provides a default sizing policy for tests."""
    return {
      "global": {
        "slippage_bps": 2, # 2 bps = 0.02%
        "fee_bps": 10      # 10 bps = 0.1%
      }
    }

@pytest.fixture
def trade_executor(sizing_policy):
    """Fixture for a TradeExecutor instance."""
    # Mock the exchange API to avoid actual API calls
    mock_exchange = MagicMock()
    return TradeExecutor(
        sizing_policy=sizing_policy,
        simulation_mode=True,
        exchange=mock_exchange
    )

@pytest.fixture
def sample_decision():
    """Provides a sample decision object for a long trade."""
    return Decision(
        signal="buy",
        sl=49000.0,
        tp=51000.0,
        meta={"symbol": "BTC/USDT"}
    )

class TestTradeExecutor:
    def test_execute_buy_order_at_sl(self, trade_executor, sample_decision):
        """
        Tests a simulated buy order that hits the stop loss, and verifies
        net P&L calculation.
        """
        entry_price = 50000.0
        size_usd = 1000.0
        leverage = 2.0

        # --- Manual Calculations for Verification ---
        notional_size = size_usd * leverage # 2000
        quantity = notional_size / entry_price # 2000 / 50000 = 0.04

        slippage_bps = trade_executor.global_policy.get("slippage_bps", 0)
        effective_entry = entry_price * (1 + slippage_bps / 10000) # 50000 * 1.0002 = 50010
        exit_price = sample_decision.sl # 49000

        pnl_per_unit = exit_price - effective_entry # 49000 - 50010 = -1010
        gross_pnl = pnl_per_unit * quantity # -1010 * 0.04 = -40.4

        fee_bps = trade_executor.global_policy.get("fee_bps", 0)
        fee_rate = fee_bps / 10000 # 0.001
        entry_fees = (quantity * effective_entry) * fee_rate # 0.04 * 50010 * 0.001 = 2.0004
        exit_fees = (quantity * exit_price) * fee_rate # 0.04 * 49000 * 0.001 = 1.96
        total_fees = entry_fees + exit_fees # 3.9604

        net_pnl = gross_pnl - total_fees # -40.4 - 3.9604 = -44.3604

        receipt = trade_executor.execute_order(
            decision=sample_decision,
            size_usd=size_usd,
            leverage=leverage,
            price=entry_price
        )

        assert receipt["status"] == "closed"
        assert receipt["pnl_gross_usd"] == pytest.approx(gross_pnl)
        assert receipt["fees_usd"] == pytest.approx(total_fees)
        assert receipt["pnl_net_usd"] == pytest.approx(net_pnl)
        assert receipt["effective_leverage"] == leverage

    def test_execute_sell_order_at_sl(self, trade_executor):
        """
        Tests a simulated sell (short) order that hits the stop loss.
        """
        entry_price = 100.0
        size_usd = 500.0
        leverage = 5.0

        sell_decision = Decision(
            signal="sell",
            sl=102.0, # SL for a short is above entry
            tp=98.0,
            meta={"symbol": "ETH/USDT"}
        )

        # --- Manual Calculations for Verification ---
        notional_size = size_usd * leverage # 2500
        quantity = notional_size / entry_price # 2500 / 100 = 25

        slippage_bps = trade_executor.global_policy.get("slippage_bps", 0)
        effective_entry = entry_price * (1 - slippage_bps / 10000) # 100 * 0.9998 = 99.98
        exit_price = sell_decision.sl # 102.0

        pnl_per_unit = effective_entry - exit_price # 99.98 - 102.0 = -2.02
        gross_pnl = pnl_per_unit * quantity # -2.02 * 25 = -50.5

        fee_bps = trade_executor.global_policy.get("fee_bps", 0)
        fee_rate = fee_bps / 10000
        entry_fees = (quantity * effective_entry) * fee_rate # 25 * 99.98 * 0.001 = 2.4995
        exit_fees = (quantity * exit_price) * fee_rate # 25 * 102.0 * 0.001 = 2.55
        total_fees = entry_fees + exit_fees # 5.0495

        net_pnl = gross_pnl - total_fees # -50.5 - 5.0495 = -55.5495

        receipt = trade_executor.execute_order(
            decision=sell_decision,
            size_usd=size_usd,
            leverage=leverage,
            price=entry_price
        )

        assert receipt["status"] == "closed"
        assert receipt["pnl_net_usd"] == pytest.approx(net_pnl)
        assert receipt["fees_usd"] == pytest.approx(total_fees)
