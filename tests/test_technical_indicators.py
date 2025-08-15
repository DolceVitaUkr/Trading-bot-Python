import pytest
from modules.technical_indicators import sma, rsi, atr

def test_sma():
    assert sma([1, 2, 3, 4, 5], 3) == 4.0
    assert sma([1, 2, 3, 4, 5], 5) == 3.0
    assert sma([1, 2, 3, 4, 5], 1) == 5.0
    assert sma([5, 4, 3, 2, 1], 3) == 2.0

def test_sma_edge_cases():
    assert sma([], 5) is None
    assert sma([1, 2, 3], 5) is None
    assert sma([1, 2, 3], 0) is None
    assert sma([1, 2, 3], -1) is None

def test_rsi():
    # Test data from online calculator
    close = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28]
    assert rsi(close, 14) == pytest.approx(70.44, abs=0.1)

def test_rsi_edge_cases():
    assert rsi([], 14) is None
    assert rsi([1] * 10, 14) is None
    assert rsi([1] * 20, 0) is None
    assert rsi([1] * 20, -1) is None

def test_atr():
    # Test data from online calculator
    high = [45.21, 45.35, 45.21, 45.12, 44.87, 44.75, 44.62, 44.5, 44.38, 44.25, 44.12, 44.0, 43.88, 43.75, 43.62]
    low = [44.81, 45.06, 44.94, 44.81, 44.62, 44.5, 44.38, 44.25, 44.12, 44.0, 43.88, 43.75, 43.62, 43.5, 43.38]
    close = [45.0, 45.12, 45.0, 44.94, 44.75, 44.62, 44.5, 44.38, 44.25, 44.12, 44.0, 43.88, 43.75, 43.62, 43.5]
    assert atr(high, low, close, 14) == pytest.approx(0.267, abs=0.001)

def test_atr_edge_cases():
    assert atr([], [], [], 14) is None
    assert atr([1]*10, [1]*10, [1]*10, 14) is None
    assert atr([1]*20, [1]*20, [1]*20, 0) is None
    assert atr([1]*20, [1]*20, [1]*20, -1) is None
