import pytest
import pandas as pd
from unittest.mock import AsyncMock, MagicMock, patch

from modules.brokers.ibkr.Fetch_IBKR_MarketData import IBKRMarketDataFetcher
from modules.brokers.ibkr.Connect_IBKR_API import IBKRConnectionManager
from ib_insync import Contract

pytestmark = pytest.mark.asyncio

@pytest.fixture
def mock_connection_manager():
    """Fixture to mock the IBKRConnectionManager."""
    manager = MagicMock(spec=IBKRConnectionManager)
    # Mock the IB client that the manager would return
    mock_ib_client = AsyncMock()
    mock_ib_client.reqHeadTimeStampAsync = AsyncMock()
    mock_ib_client.reqHistoricalDataAsync = AsyncMock(return_value=[]) # Return empty list by default
    manager.get_tws_client = AsyncMock(return_value=mock_ib_client)
    return manager

@patch('utils.rate_limiter.ibkr_rate_limiter.wait_for_historical_slot', new_callable=AsyncMock)
async def test_fetch_historical_bars_calls_rate_limiter(mock_wait_slot, mock_connection_manager):
    """Test that fetch_historical_bars awaits the rate limiter."""
    fetcher = IBKRMarketDataFetcher(mock_connection_manager)
    contract = Contract(symbol="EUR", currency="USD", exchange="IDEALPRO", secType="CASH")

    await fetcher.fetch_historical_bars(contract, "1 D", "1 hour")

    mock_wait_slot.assert_called_once()

async def test_market_data_check_success(mock_connection_manager):
    """Test that market_data_check succeeds when API calls are successful."""
    fetcher = IBKRMarketDataFetcher(mock_connection_manager)
    # No exception should be raised
    await fetcher.market_data_check()

async def test_market_data_check_raises_permission_error(mock_connection_manager):
    """Test that market_data_check raises PermissionError on specific error message."""
    mock_ib = await mock_connection_manager.get_tws_client()
    mock_ib.reqHeadTimeStampAsync.side_effect = Exception("No market data permissions for CSH.EUR")

    fetcher = IBKRMarketDataFetcher(mock_connection_manager)

    with pytest.raises(PermissionError, match="Missing Forex market data subscription."):
        await fetcher.market_data_check()

@patch('pandas.read_parquet')
@patch('pathlib.Path.exists', return_value=True)
async def test_fetch_historical_bars_uses_cache(mock_path_exists, mock_read_parquet, mock_connection_manager):
    """Test that fetch_historical_bars reads from cache if file exists."""
    fetcher = IBKRMarketDataFetcher(mock_connection_manager)
    contract = Contract(symbol="EUR", currency="USD", exchange="IDEALPRO", secType="CASH")

    await fetcher.fetch_historical_bars(contract, "1 D", "1 hour")

    mock_read_parquet.assert_called_once()
    # Ensure the actual API call was NOT made
    mock_ib = await mock_connection_manager.get_tws_client()
    mock_ib.reqHistoricalDataAsync.assert_not_called()

@patch('pandas.DataFrame.to_parquet')
@patch('pathlib.Path.exists', return_value=False)
async def test_fetch_historical_bars_fetches_and_saves_to_cache(mock_path_exists, mock_to_parquet, mock_connection_manager):
    """Test that fetch_historical_bars fetches new data and saves it to cache."""
    mock_ib = await mock_connection_manager.get_tws_client()
    # Create some dummy bar data
    dummy_bar = MagicMock()
    dummy_bar.date = "1672531200" # Unix timestamp
    dummy_bar.open = 1.05
    dummy_bar.high = 1.06
    dummy_bar.low = 1.04
    dummy_bar.close = 1.055
    dummy_bar.volume = 1000
    mock_ib.reqHistoricalDataAsync.return_value = [dummy_bar]

    # Mock the util.df function to return a valid DataFrame
    with patch('ib_insync.util.df', return_value=pd.DataFrame([vars(dummy_bar)])) as mock_util_df:
        fetcher = IBKRMarketDataFetcher(mock_connection_manager)
        contract = Contract(symbol="EUR", currency="USD", exchange="IDEALPRO", secType="CASH")

        df = await fetcher.fetch_historical_bars(contract, "1 D", "1 hour")

        mock_ib.reqHistoricalDataAsync.assert_called_once()
        mock_to_parquet.assert_called_once()
        assert not df.empty
        assert 'close' in df.columns
