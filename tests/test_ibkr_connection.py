import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from modules.brokers.ibkr.Connect_IBKR_API import IBKRConnectionManager, ConnectionError
import config

# Use pytest-asyncio for async tests
pytestmark = pytest.mark.asyncio

@pytest.fixture
def mock_ib_insync():
    """Fixture to mock the ib_insync.IB class."""
    with patch('modules.brokers.ibkr.Connect_IBKR_API.IB') as mock_ib:
        instance = mock_ib.return_value
        instance.connectAsync = AsyncMock()
        instance.disconnect = MagicMock()
        instance.isConnected = MagicMock(return_value=False)
        instance.managedAccounts = MagicMock(return_value=["DU1234567"]) # Paper account
        instance.client.isReadOnly = MagicMock(return_value=True)
        yield instance

@pytest.fixture
def mock_aiohttp_session():
    """Fixture to mock aiohttp.ClientSession."""
    with patch('modules.brokers.ibkr.Connect_IBKR_API.aiohttp.ClientSession') as mock_session:
        instance = mock_session.return_value
        # Mock the async context manager (__aenter__ and __aexit__)
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={"connected": True})
        mock_response.raise_for_status = MagicMock()

        instance.get = MagicMock(return_value=mock_response)
        yield instance

async def test_connect_tws_success(mock_ib_insync):
    """Test successful connection to TWS."""
    manager = IBKRConnectionManager()

    await manager.connect_tws()

    mock_ib_insync.connectAsync.assert_called_once()
    assert manager.is_paper_account is True

async def test_connect_tws_live_account_in_paper_mode_fails(mock_ib_insync):
    """Test that connecting to a live account in paper mode raises an error."""
    config.IBKR_API_MODE = "paper"
    mock_ib_insync.managedAccounts.return_value = ["U1234567"] # Live account

    manager = IBKRConnectionManager()

    with pytest.raises(ConnectionError, match="Connected to live account in paper mode."):
        await manager.connect_tws()

async def test_connect_tws_training_mode_sets_readonly(mock_ib_insync):
    """Test that TRAINING_MODE=True sets the readonly flag."""
    config.TRAINING_MODE = True
    manager = IBKRConnectionManager()

    await manager.connect_tws()

    # Check if readonly=True was passed to connectAsync
    call_args = mock_ib_insync.connectAsync.call_args
    assert call_args.kwargs.get("readonly") is True

async def test_connect_tws_live_mode_unsets_readonly(mock_ib_insync):
    """Test that TRAINING_MODE=False unsets the readonly flag."""
    config.TRAINING_MODE = False
    manager = IBKRConnectionManager()

    await manager.connect_tws()

    call_args = mock_ib_insync.connectAsync.call_args
    assert call_args.kwargs.get("readonly") is False

async def test_connect_web_api_success(mock_aiohttp_session):
    """Test successful connection to the Web API gateway."""
    manager = IBKRConnectionManager()
    await manager.connect_web_api()

    mock_aiohttp_session.get.assert_called_once()
    assert "iserver/auth/status" in mock_aiohttp_session.get.call_args[0][0]
    assert manager.web_session is not None
