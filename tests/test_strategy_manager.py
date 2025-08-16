import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from modules.Strategy_Manager import StrategyManager, Decision
from modules.data_manager import DataManager

@pytest.fixture
def mock_data_provider():
    """Fixture to create a mock DataManager."""
    dm = MagicMock(spec=DataManager)
    df = pd.DataFrame({
        'high': pd.to_numeric([1, 2, 3, 4, 5] * 40),
        'low': pd.to_numeric([1, 2, 3, 4, 5] * 40),
        'close': pd.to_numeric([1, 2, 3, 4, 5] * 40),
        'volume': pd.to_numeric([100, 200, 300, 400, 500] * 40)
    })
    dm.load_historical_data.return_value = df
    return dm

@pytest.fixture
def strategy_manager(mock_data_provider):
    """Fixture to create a StrategyManager with mock dependencies."""
    validation_manager = MagicMock()
    news_agent = MagicMock()
    portfolio_manager = MagicMock()

    with patch('modules.Strategy_Manager.config') as mock_config:
        mock_config.OPTIMIZATION_PARAMETERS = {}
        manager = StrategyManager(
            data_provider=mock_data_provider,
            validation_manager=validation_manager,
            news_agent=news_agent,
            portfolio_manager=portfolio_manager
        )
        # Mock the filters to pass by default in these original tests
        manager.data_provider.get_daily_volume = MagicMock(return_value=10_000_000)
        manager._is_overexposed_by_correlation = MagicMock(return_value=False)
        manager.news_agent.is_high_impact_event_imminent.return_value = (False, "")
        manager.news_agent.get_news_bias.return_value = 'neutral'
        manager.validation_manager.is_strategy_approved.return_value = True
    return manager

def test_decide_no_signal(strategy_manager):
    """Tests that decide returns None when the strategy finds no signal."""
    strategy_manager.strategy_map['TrendFollowStrategy'].check_entry_conditions = MagicMock(return_value={})

    context = {"regime": "Trend", "adx": 30, "asset_class": "SPOT"}
    decision = strategy_manager.decide("BTC/USDT", context)

    assert decision is None

def test_decide_trend_signal(strategy_manager):
    """Tests a successful decision for a Trend signal."""
    mock_signal = {"side": "buy", "sl": 49000, "tp": 51000}
    strategy_manager.strategy_map['TrendFollowStrategy'].check_entry_conditions = MagicMock(return_value=mock_signal)

    context = {"regime": "Trend", "adx": 40, "asset_class": "SPOT"}
    decision = strategy_manager.decide("BTC/USDT", context)

    assert isinstance(decision, Decision)
    assert decision.signal == "buy"
    assert decision.sl == 49000
    assert "signal_score" in decision.meta
    assert "good_setup" in decision.meta
    assert decision.meta["signal_score"] > 0.5

def test_signal_scoring(strategy_manager):
    """Tests the signal scoring logic for different regimes."""
    # Trend regime scoring
    score_low_trend = strategy_manager._calculate_signal_score("Trend", {"adx": 15})
    score_med_trend = strategy_manager._calculate_signal_score("Trend", {"adx": 30})
    score_high_trend = strategy_manager._calculate_signal_score("Trend", {"adx": 50})
    assert 0 < score_low_trend < score_med_trend < score_high_trend
    assert score_med_trend > 0.5

    # Mean Reversion scoring
    score_good_mr = strategy_manager._calculate_signal_score("MeanReversion", {"adx": 10, "bbw_percentile": 0.9})
    score_bad_mr = strategy_manager._calculate_signal_score("MeanReversion", {"adx": 30, "bbw_percentile": 0.5})
    assert score_good_mr > score_bad_mr
    assert score_good_mr > 0.7

def test_good_setup_flag(strategy_manager):
    """Tests that the good_setup flag is set correctly."""
    mock_signal = {"side": "sell", "sl": 1.2, "tp": 1.1}
    strategy_manager.strategy_map['TrendFollowStrategy'].check_entry_conditions = MagicMock(return_value=mock_signal)

    # Context for a high score -> good_setup should be True
    high_score_context = {"regime": "Trend", "adx": 55, "asset_class": "SPOT"}
    decision_good = strategy_manager.decide("EUR/USD", high_score_context)
    assert decision_good.meta["good_setup"]

    # Context for a low score -> good_setup should be False
    low_score_context = {"regime": "Trend", "adx": 20, "asset_class": "SPOT"}
    decision_bad = strategy_manager.decide("EUR/USD", low_score_context)
    assert not decision_bad.meta["good_setup"]
