import pytest
from unittest.mock import MagicMock, patch
import threading
import time
from main import TradingUI

@patch('tkinter.Tk')
@patch('tkinter.ttk.Style')
@patch('tkinter.IntVar')
@patch('tkinter.StringVar')
@patch('tkinter.scrolledtext.ScrolledText')
@patch('matplotlib.figure.Figure')
@patch('matplotlib.backends.backend_tkagg.FigureCanvasTkAgg')
def test_ui_methods(MockFigureCanvas, MockFigure, MockScrolledText, MockStringVar, MockIntVar, MockStyle, MockTk):
    # Arrange
    mock_bot = MagicMock()

    # Configure mocks
    def string_var_side_effect(*args, **kwargs):
        return MagicMock()
    MockStringVar.side_effect = string_var_side_effect

    with patch('modules.ui.tk.Spinbox'), patch('modules.ui.ttk.Combobox'):
        ui = TradingUI(bot=mock_bot)

    # Act
    ui.update_live_metrics({
        "balance": 1234.56,
        "equity": 5432.10,
        "symbol": "BTC/USDT",
        "timeframe": "15m",
    })

    # Assert
    ui.balance_var.set.assert_called_with("$1,234.56")
    ui.value_var.set.assert_called_with("$5,432.10")
    ui.symbol_var.set.assert_called_with("BTC/USDT")
    ui.tf_var.set.assert_called_with("15m")

@patch('tkinter.Tk')
@patch('tkinter.ttk.Style')
@patch('tkinter.IntVar')
@patch('tkinter.StringVar')
@patch('tkinter.scrolledtext.ScrolledText')
@patch('matplotlib.figure.Figure')
@patch('matplotlib.backends.backend_tkagg.FigureCanvasTkAgg')
def test_ui_logs(MockFigureCanvas, MockFigure, MockScrolledText, MockStringVar, MockIntVar, MockStyle, MockTk):
    # Arrange
    mock_bot = MagicMock()
    with patch('modules.ui.tk.Spinbox'), patch('modules.ui.ttk.Combobox'):
        ui = TradingUI(bot=mock_bot)

    # Act
    ui.log("This is a test message", level="INFO")

    # Assert
    ui.log_text.insert.assert_called()
    args, kwargs = ui.log_text.insert.call_args
    assert "This is a test message" in args[1]
