import pytest
from unittest.mock import MagicMock, AsyncMock, patch, call

from modules.ui import TradingUI
from textual.widgets import Static, Sparkline

@pytest.fixture
def mock_bot():
    """Fixture for a mocked bot instance."""
    bot = MagicMock()
    bot.is_connected = True
    bot.training = False
    bot.trading = True
    bot.last_heartbeat = 1672531200  # Example timestamp
    return bot

@pytest.mark.asyncio
async def test_ui_initialization(mock_bot):
    """Test if the TradingUI initializes correctly."""
    app = TradingUI(bot=mock_bot)
    assert app.bot is not None
    assert app.title == "Textual Trading Bot"

@pytest.mark.asyncio
async def test_ui_live_metrics_update(mock_bot):
    """Test updating live metrics on the UI."""
    app = TradingUI(bot=mock_bot)

    with patch.object(app, 'query_one') as mock_query_one:
        mock_widget = MagicMock()
        mock_query_one.return_value = mock_widget

        metrics = {
            "balance": 1234.56,
            "equity": 5432.10,
            "symbol": "BTC/USDT",
            "timeframe": "15m",
        }
        app._update_live_metrics(metrics)

        mock_query_one.assert_any_call("#wallet-balance", Static)
        mock_query_one.return_value.update.assert_any_call("$1,234.56")
        mock_query_one.assert_any_call("#portfolio-value", Static)
        mock_query_one.return_value.update.assert_any_call("$5,432.10")
        mock_query_one.assert_any_call("#current-symbol", Static)
        mock_query_one.return_value.update.assert_any_call("BTC/USDT")
        mock_query_one.assert_any_call("#current-timeframe", Static)
        mock_query_one.return_value.update.assert_any_call("15m")

@pytest.mark.asyncio
async def test_ui_timeseries_update(mock_bot):
    """Test updating timeseries data on the UI."""
    app = TradingUI(bot=mock_bot)

    with patch.object(app, 'query_one') as mock_query_one:
        mock_sparkline_wallet = MagicMock()
        mock_sparkline_vwallet = MagicMock()
        mock_sparkline_points = MagicMock()

        def query_one_side_effect(selector, widget_type):
            if selector == "#wallet-sparkline":
                return mock_sparkline_wallet
            elif selector == "#vwallet-sparkline":
                return mock_sparkline_vwallet
            elif selector == "#points-sparkline":
                return mock_sparkline_points
            return MagicMock()

        mock_query_one.side_effect = query_one_side_effect

        app._update_timeseries(wallet=100.0, vwallet=200.0, points=50.0)

        assert app._wallet_hist_data[-1] == 100.0
        assert app._vwallet_hist_data[-1] == 200.0
        assert app._points_hist_data[-1] == 50.0

        assert mock_sparkline_wallet.data == app._wallet_hist_data
        assert mock_sparkline_vwallet.data == app._vwallet_hist_data
        assert mock_sparkline_points.data == app._points_hist_data


@pytest.mark.asyncio
async def test_ui_logging(mock_bot):
    """Test logging messages to the UI."""
    app = TradingUI(bot=mock_bot)
    app.call_soon = AsyncMock()

    with patch.object(app, '_loop', MagicMock()), patch.object(app, 'query_one') as mock_query_one:
        mock_log_widget = AsyncMock()
        mock_query_one.return_value = mock_log_widget
        await app.log("Test log message", level="INFO")

    app.call_soon.assert_called_once()


@pytest.mark.asyncio
async def test_button_press_handler(mock_bot):
    """Test that button presses trigger the correct handlers."""
    app = TradingUI(bot=mock_bot)

    mock_handler = MagicMock()
    app.add_action_handler("start_training", mock_handler)

    from textual.widgets import Button
    button = Button("Start Training", id="start_training")

    with patch('asyncio.run') as mock_run:
        app.on_button_pressed(Button.Pressed(button))

    mock_handler.assert_called_once()
