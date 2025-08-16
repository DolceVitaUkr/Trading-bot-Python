import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from modules.Strategy_Manager import StrategyManager, Decision
from modules.Validation_Manager import ValidationManager
from modules.News_Agent import NewsAgent
from modules.Portfolio_Manager import PortfolioManager
from modules.data_manager import DataManager

@pytest.fixture
def mock_dependencies():
    """Fixture to create mock dependencies for StrategyManager."""
    data_provider = MagicMock(spec=DataManager)
    validation_manager = MagicMock(spec=ValidationManager)
    news_agent = MagicMock(spec=NewsAgent)
    portfolio_manager = MagicMock(spec=PortfolioManager)

    # Setup default return values for mocks
    df = pd.DataFrame({'high': [1]*200, 'low': [1]*200, 'close': [1]*200})
    data_provider.load_historical_data.return_value = df
    validation_manager.is_strategy_approved.return_value = True
    news_agent.is_high_impact_event_imminent.return_value = (False, "")
    news_agent.get_news_bias.return_value = 'neutral'
    # Configure the mock with the method instead of setting return_value on a non-existent attribute
    portfolio_manager.get_open_positions = MagicMock(return_value=[])

    return {
        "data_provider": data_provider,
        "validation_manager": validation_manager,
        "news_agent": news_agent,
        "portfolio_manager": portfolio_manager
    }

@pytest.fixture
def strategy_manager_for_filters(mock_dependencies):
    """Fixture to create a StrategyManager with mocked dependencies."""
    sm = StrategyManager(**mock_dependencies)
    # Mock the internal strategy call to produce a consistent signal
    sm.strategy_map['TrendFollowStrategy'].check_entry_conditions = MagicMock(return_value={"side": "buy", "sl": 0.9, "tp": 1.1})
    return sm

def test_trade_succeeds_with_all_filters_passing(strategy_manager_for_filters):
    """Test that a decision is made when all filters pass."""
    context = {"regime": "Trend", "asset_class": "SPOT", "adx": 30}
    decision = strategy_manager_for_filters.decide("BTCUSDT", context)
    assert isinstance(decision, Decision)
    assert decision.signal == "buy"

def test_liquidity_filter_rejects(strategy_manager_for_filters, mock_dependencies):
    """Test that a trade is rejected for low liquidity."""
    # This requires a 'get_daily_volume' method on the data_provider mock.
    # We will mock this directly on the instance.
    strategy_manager_for_filters.data_provider.get_daily_volume = MagicMock(return_value=4_000_000) # Below $5M threshold

    context = {"regime": "Trend", "asset_class": "SPOT", "adx": 30}

    # To make the test cleaner, let's patch the logger to check the message
    with patch('modules.Strategy_Manager.logger') as mock_logger:
        decision = strategy_manager_for_filters.decide("BTCUSDT", context)
        assert decision is None
        mock_logger.warning.assert_called_with(
            "Trade rejected for BTCUSDT: Low liquidity (Volume: $4,000,000)",
            extra={'action': 'reject', 'symbol': 'BTCUSDT', 'reason': 'Low liquidity (Volume: $4,000,000)'}
        )

def test_validation_filter_rejects(strategy_manager_for_filters, mock_dependencies):
    """Test that a trade is rejected if the strategy is not approved."""
    mock_dependencies["validation_manager"].is_strategy_approved.return_value = False
    context = {"regime": "Trend", "asset_class": "SPOT", "adx": 30}
    decision = strategy_manager_for_filters.decide("BTCUSDT", context)
    assert decision is None

def test_news_event_filter_rejects(strategy_manager_for_filters, mock_dependencies):
    """Test that a trade is rejected during a high-impact news event."""
    mock_dependencies["news_agent"].is_high_impact_event_imminent.return_value = (True, "FOMC")
    context = {"regime": "Trend", "asset_class": "SPOT", "adx": 30}
    decision = strategy_manager_for_filters.decide("BTCUSDT", context)
    assert decision is None

def test_news_sentiment_filter_rejects_conflicting_signal(strategy_manager_for_filters, mock_dependencies):
    """Test that a long signal is rejected with strong negative sentiment."""
    mock_dependencies["news_agent"].get_news_bias.return_value = 'short'
    context = {"regime": "Trend", "asset_class": "SPOT", "adx": 30}
    decision = strategy_manager_for_filters.decide("BTCUSDT", context)
    assert decision is None

def test_correlation_filter_rejects(strategy_manager_for_filters, mock_dependencies):
    """Test that a trade is rejected due to correlation."""
    # Setup: We already have ETH and SOL open, and we try to open BTC.
    open_positions = ['ETHUSDT', 'SOLUSDT']
    strategy_manager_for_filters.portfolio_manager.get_open_positions.return_value = open_positions
    context = {"regime": "Trend", "asset_class": "SPOT", "adx": 30}

    # We need to patch the internal _is_overexposed_by_correlation to use the mocked portfolio manager data
    # A cleaner way is to just mock the method directly.
    strategy_manager_for_filters._is_overexposed_by_correlation = MagicMock(return_value=True)

    decision = strategy_manager_for_filters.decide("BTCUSDT", context)
    assert decision is None
    strategy_manager_for_filters._is_overexposed_by_correlation.assert_called_once()
