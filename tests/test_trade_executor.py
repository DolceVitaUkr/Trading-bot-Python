import pytest
from unittest.mock import MagicMock
from modules.trade_executor import TradeExecutor
from modules.exchange import ExchangeAPI

@pytest.fixture
def mock_exchange():
    exchange = MagicMock(spec=ExchangeAPI)
    exchange.get_balance.return_value = 10000.0
    exchange.get_price.return_value = 50000.0
    exchange.get_min_cost.return_value = 10.0
    exchange.get_amount_precision.return_value = 6
    exchange.get_price_precision.return_value = 2
    exchange._resolve_symbol.return_value = "BTC/USDT"

    # Mock the internal simulation state and methods
    exchange._sim_cash_usd = 10000.0
    exchange._sim_positions = {}

    def _simulate_order(symbol, side, order_type, quantity, price, **kwargs):
        if side == "buy":
            exchange._sim_positions[symbol] = {
                "side": "long",
                "quantity": quantity,
                "entry_price": price,
            }
            return {"status": "open", "quantity": quantity, "entry_price": price}
        elif side == "sell":
            pos = exchange._sim_positions.pop(symbol, None)
            if pos:
                pnl = (price - pos["entry_price"]) * pos["quantity"]
                return {"status": "closed", "pnl": pnl, "quantity": pos["quantity"]}
            return {"status": "closed", "pnl": 0}

    exchange._simulate_order.side_effect = _simulate_order
    return exchange

@pytest.fixture
def trade_executor(mock_exchange):
    notifier = MagicMock()
    executor = TradeExecutor(simulation_mode=True, notifier=notifier)
    executor.exchange = mock_exchange
    return executor

def test_execute_order_open(trade_executor):
    result = trade_executor.execute_order("BTC/USDT", "buy", quantity=0.001)
    assert result["status"] == "open"
    assert result["symbol"] == "BTC/USDT"
    assert result["side"] == "buy"
    assert result["quantity"] == 0.001
    assert result["entry_price"] == 50000.0
    trade_executor.notifier.send_message_sync.assert_called_once()

def test_close_position(trade_executor):
    # First, open a position
    trade_executor.execute_order("BTC/USDT", "buy", quantity=0.001)

    # Now, close it
    close_result = trade_executor.close_position("BTC/USDT", price=51000)
    assert close_result["status"] == "closed"
    assert close_result["pnl"] == (51000 - 50000) * 0.001

def test_get_balance(trade_executor):
    assert trade_executor.get_balance() == 10000.0
