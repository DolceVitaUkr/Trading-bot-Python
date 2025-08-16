import pytest
from unittest.mock import MagicMock
from modules.trade_executor import TradeExecutor
from modules.exchange import ExchangeAPI


@pytest.fixture
def mock_exchange():
    """
    Fixture for a mocked exchange instance.
    """
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
        pos = exchange._sim_positions.get(symbol)

        if side == "buy":
            if pos:
                new_qty = pos["quantity"] + quantity
                avg_entry = (pos["entry_price"] * pos["quantity"] + price * quantity) / new_qty
                pos["quantity"] = new_qty
                pos["entry_price"] = avg_entry
                return {"status": "open_increased", "quantity": new_qty, "entry_price": avg_entry}
            else:
                exchange._sim_positions[symbol] = {"side": "long", "quantity": quantity, "entry_price": price}
                return {"status": "open", "quantity": quantity, "entry_price": price}
        elif side == "sell":
            if not pos:
                return {"status": "noop"}

            if quantity >= pos["quantity"]:
                pnl = (price - pos["entry_price"]) * pos["quantity"]
                exchange._sim_positions.pop(symbol, None)
                return {"status": "closed", "pnl": pnl, "quantity": pos["quantity"]}
            else:
                pnl = (price - pos["entry_price"]) * quantity
                pos["quantity"] -= quantity
                return {"status": "closed_partial", "pnl": pnl, "quantity": quantity}

    exchange._simulate_order.side_effect = _simulate_order
    return exchange


@pytest.fixture
def trade_executor(mock_exchange):
    """
    Fixture for a TradeExecutor instance with a mocked exchange.
    """
    notifier = MagicMock()
    executor = TradeExecutor(simulation_mode=True, notifier=notifier)
    executor.exchange = mock_exchange
    return executor


def test_execute_order_open(trade_executor):
    """
    Tests the execute_order method for opening a position.
    """
    result = trade_executor.execute_order("BTC/USDT", "buy", quantity=0.001)
    assert result["status"] == "open"
    assert result["symbol"] == "BTC/USDT"
    assert result["side"] == "buy"
    assert result["quantity"] == 0.001
    assert result["entry_price"] == 50000.0


def test_close_position(trade_executor):
    """
    Tests the close_position method.
    """
    # First, open a position
    trade_executor.execute_order("BTC/USDT", "buy", quantity=0.001)

    # Now, close it
    close_result = trade_executor.close_position("BTC/USDT", price=51000)
    assert close_result["status"] == "closed"
    assert close_result["pnl"] == (51000 - 50000) * 0.001


def test_get_balance(trade_executor):
    """
    Tests the get_balance method.
    """
    assert trade_executor.get_balance() == 10000.0

def test_execute_order_reduce_only(trade_executor):
    """
    Tests the execute_order method for a reduce_only order.
    """
    # First, open a position
    trade_executor.execute_order("BTC/USDT", "buy", quantity=0.001)

    # Now, close it with a reduce_only order
    result = trade_executor.execute_order(
        "BTC/USDT", "sell", quantity=0.001, reduce_only=True
    )
    assert result["status"] == "closed"

def test_execute_order_increase_position(trade_executor):
    """
    Tests the execute_order method for increasing a position.
    """
    # First, open a position
    trade_executor.execute_order("BTC/USDT", "buy", quantity=0.001)

    # Now, increase it
    result = trade_executor.execute_order("BTC/USDT", "buy", quantity=0.001)
    assert result["status"] == "open_increased"
    assert result["quantity"] == 0.002

def test_close_position_partial(trade_executor):
    """
    Tests the close_position method for a partial close.
    """
    # First, open a position
    trade_executor.execute_order("BTC/USDT", "buy", quantity=0.002)

    # Now, partially close it
    close_result = trade_executor.close_position(
        "BTC/USDT", price=51000, quantity=0.001
    )
    assert close_result["status"] == "closed_partial"
    assert close_result["pnl"] == (51000 - 50000) * 0.001

def test_close_non_existent_position(trade_executor):
    """
    Tests the close_position method for a non-existent position.
    """
    close_result = trade_executor.close_position("BTC/USDT", price=51000)
    assert close_result["status"] == "noop"
