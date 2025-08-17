import pytest
from unittest.mock import AsyncMock, MagicMock

from modules.brokers.ibkr.Place_IBKR_Order import IBKROrderPlacer, TrainingOnlyError, FundsTransferError
from modules.brokers.ibkr.Connect_IBKR_API import IBKRConnectionManager
from ib_insync import Contract
import config

pytestmark = pytest.mark.asyncio

@pytest.fixture
def mock_connection_manager():
    """Fixture to mock the IBKRConnectionManager."""
    manager = MagicMock(spec=IBKRConnectionManager)
    manager.get_tws_client = AsyncMock()
    return manager

@pytest.fixture
def order_placer(mock_connection_manager):
    """Fixture to create an IBKROrderPlacer instance."""
    return IBKROrderPlacer(mock_connection_manager)

async def test_place_order_in_training_mode_raises_error(order_placer):
    """Test that placing an order in TRAINING_MODE raises TrainingOnlyError."""
    config.TRAINING_MODE = True
    contract = Contract(symbol="EUR", currency="USD", secType="CASH")

    with pytest.raises(TrainingOnlyError, match="Order placement is disabled."):
        await order_placer.place_market_order(contract, "BUY", 10000)

    with pytest.raises(TrainingOnlyError):
        await order_placer.place_limit_order(contract, "SELL", 10000, 1.1)

async def test_place_order_in_live_mode_returns_stub(order_placer):
    """Test that placing an order with TRAINING_MODE=False returns the stub response."""
    config.TRAINING_MODE = False
    contract = Contract(symbol="EUR", currency="USD", secType="CASH")

    response = await order_placer.place_market_order(contract, "BUY", 10000)

    assert response["status"] == "SUBMITTED_STUB"
    assert "Live order placement is disabled" in response["reason"]

async def test_transfer_funds_disabled_by_policy_raises_error(order_placer):
    """Test that transfer_funds raises an error if ALLOW_FUNDS_TRANSFER is False."""
    config.ALLOW_FUNDS_TRANSFER = False

    with pytest.raises(FundsTransferError, match="Fund transfers are disabled by policy."):
        await order_placer.transfer_funds(100, "USD", "some_account")

async def test_transfer_funds_enabled_but_not_implemented_raises_error(order_placer):
    """Test that transfer_funds raises an error even if enabled, because it's not implemented."""
    config.ALLOW_FUNDS_TRANSFER = True

    # It should still raise FundsTransferError because the function is not implemented.
    with pytest.raises(FundsTransferError, match="Fund transfer functionality is not implemented."):
        await order_placer.transfer_funds(100, "USD", "some_account")
