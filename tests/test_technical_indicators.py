import pytest
from modules.technical_indicators import sma, rsi, atr


def test_sma():
    """
    Tests the sma function.
    """
    assert sma([1, 2, 3, 4, 5], 3) == 4.0
    assert sma([1, 2, 3, 4, 5], 5) == 3.0
    assert sma([1, 2, 3, 4, 5], 1) == 5.0
    assert sma([5, 4, 3, 2, 1], 3) == 2.0


def test_sma_edge_cases():
    """
    Tests edge cases for the sma function.
    """
    assert sma([], 5) is None
    assert sma([1, 2, 3], 5) is None
    assert sma([1, 2, 3], 0) is None
    assert sma([1, 2, 3], -1) is None


def test_rsi():
    """
    Tests the rsi function.
    """
    # Test data from online calculator
    close = [
        44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84,
        46.08, 45.89, 46.03, 45.61, 46.28, 46.28]
    assert rsi(close, 14) == pytest.approx(70.44, abs=0.1)


def test_rsi_edge_cases():
    """
    Tests edge cases for the rsi function.
    """
    assert rsi([], 14) is None
    assert rsi([1] * 10, 14) is None
    assert rsi([1] * 20, 0) is None
    assert rsi([1] * 20, -1) is None


def test_atr():
    """
    Tests the atr function.
    """
    # Test data from online calculator
    high = [
        45.21, 45.35, 45.21, 45.12, 44.87, 44.75, 44.62, 44.5, 44.38,
        44.25, 44.12, 44.0, 43.88, 43.75, 43.62]
    low = [
        44.81, 45.06, 44.94, 44.81, 44.62, 44.5, 44.38, 44.25, 44.12,
        44.0, 43.88, 43.75, 43.62, 43.5, 43.38]
    close = [
        45.0, 45.12, 45.0, 44.94, 44.75, 44.62, 44.5, 44.38, 44.25,
        44.12, 44.0, 43.88, 43.75, 43.62, 43.5]
    assert atr(high, low, close, 14) == pytest.approx(0.267, abs=0.001)


def test_atr_edge_cases():
    """
    Tests edge cases for the atr function.
    """
    assert atr([], [], [], 14) is None
    assert atr([1] * 10, [1] * 10, [1] * 10, 14) is None
    assert atr([1] * 20, [1] * 20, [1] * 20, 0) is None
    assert atr([1] * 20, [1] * 20, [1] * 20, -1) is None


import pandas as pd
import numpy as np
from modules.technical_indicators import TechnicalIndicators

class TestTechnicalIndicatorsPandas:
    def setup_method(self):
        """Set up a standard pandas Series for testing."""
        self.close_prices = pd.Series([
            100, 102, 105, 103, 106, 108, 110, 109, 112, 115,
            113, 116, 118, 120, 119, 122, 125, 123, 126, 129
        ])
        # Data for percentile test - needs more data points
        self.long_series = pd.Series(
            np.sin(np.linspace(0, 10, 100)) * 10 + 100
        )


    def test_bollinger_bands(self):
        middle, upper, lower = TechnicalIndicators.bollinger_bands(self.close_prices, window=5)
        assert middle is not None
        assert upper is not None
        assert lower is not None
        assert pd.isna(middle.iloc[3])
        assert not pd.isna(middle.iloc[4])
        # Values re-calculated based on pandas with ddof=0
        assert middle.iloc[4] == pytest.approx(103.2)
        assert upper.iloc[4] == pytest.approx(107.47, abs=0.01)
        assert lower.iloc[4] == pytest.approx(98.93, abs=0.01)

    def test_bollinger_band_width(self):
        bbw = TechnicalIndicators.bollinger_band_width(self.close_prices, window=5)
        assert bbw is not None
        assert pd.isna(bbw.iloc[3])
        assert not pd.isna(bbw.iloc[4])
        # (107.47 - 98.93) / 103.2 = 0.0827
        assert bbw.iloc[4] == pytest.approx(0.0827, abs=0.001)

    def test_z_score(self):
        zscore = TechnicalIndicators.z_score(self.close_prices, window=5)
        assert zscore is not None
        assert pd.isna(zscore.iloc[3])
        assert not pd.isna(zscore.iloc[4])
        # (106 - 103.2) / 2.135 = 1.311
        assert zscore.iloc[4] == pytest.approx(1.311, abs=0.001)

    def test_bb_width_percentile(self):
        # Using a longer series for this test
        percentile = TechnicalIndicators.bb_width_percentile(
            self.long_series, bb_window=20, percentile_lookback=30)
        assert percentile is not None
        # NaNs should be (bb_window - 1) + (percentile_lookback - 1) = 19 + 29 = 48
        assert pd.isna(percentile.iloc[47])
        assert not pd.isna(percentile.iloc[48])
        # Percentiles must be between 0 and 100
        assert percentile.dropna().min() >= 0
        assert percentile.dropna().max() <= 100
