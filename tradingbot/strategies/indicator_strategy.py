"""
Technical Indicator-Based Trading Strategy
Uses real technical indicators to generate trading signals.
"""
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import random

from ..core.indicators import sma, ema, rsi, mfi, atr, bollinger_bands, fibonacci_retracement


class IndicatorStrategy:
    """Base class for indicator-based trading strategies."""
    
    def __init__(self, parameters: Dict[str, any]):
        self.parameters = parameters
        self.name = parameters.get('name', 'indicator_strategy')
        self.variant = parameters.get('variant', 'sma_crossover')
        
        # Default parameters
        self.fast_period = parameters.get('fast_period', 10)
        self.slow_period = parameters.get('slow_period', 20)
        self.rsi_period = parameters.get('rsi_period', 14)
        self.rsi_oversold = parameters.get('rsi_oversold', 30)
        self.rsi_overbought = parameters.get('rsi_overbought', 70)
        
        # Risk parameters
        self.stop_loss = parameters.get('stop_loss', 0.01)  # 1%
        self.take_profit = parameters.get('take_profit', 0.02)  # 2%
        
        # Track price history and OHLCV data
        self.price_history: Dict[str, List[float]] = {}
        self.ohlcv_history: Dict[str, pd.DataFrame] = {}
        
    def update_price(self, symbol: str, price: float):
        """Update price history for a symbol."""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append(price)
        
        # Keep last 100 prices
        if len(self.price_history[symbol]) > 100:
            self.price_history[symbol] = self.price_history[symbol][-100:]
    
    def update_ohlcv(self, symbol: str, ohlcv_data: pd.DataFrame):
        """Update OHLCV history for a symbol."""
        self.ohlcv_history[symbol] = ohlcv_data
    
    def get_signal(self, symbol: str, current_price: float) -> Tuple[str, float]:
        """
        Generate trading signal based on technical indicators.
        
        Returns:
            Tuple of (signal, confidence)
            signal: 'BUY', 'SELL', or 'HOLD'
            confidence: 0.0 to 1.0
        """
        # Update price history
        self.update_price(symbol, current_price)
        
        # Need enough data for indicators
        if symbol not in self.price_history or len(self.price_history[symbol]) < self.slow_period + 1:
            return 'HOLD', 0.0
        
        # Convert to pandas series
        prices = pd.Series(self.price_history[symbol])
        
        # Calculate indicators based on variant
        if self.variant == 'sma_crossover':
            return self._sma_crossover_signal(prices)
        elif self.variant == 'rsi_extremes':
            return self._rsi_extremes_signal(prices)
        elif self.variant == 'ema_momentum':
            return self._ema_momentum_signal(prices)
        elif self.variant == 'bollinger_bands':
            return self._bollinger_bands_signal(prices)
        elif self.variant == 'mfi_volume':
            return self._mfi_volume_signal(prices, symbol)
        elif self.variant == 'atr_breakout':
            return self._atr_breakout_signal(prices, symbol)
        elif self.variant == 'fibonacci_retracement':
            return self._fibonacci_signal(prices)
        elif self.variant == 'combined':
            return self._combined_signal(prices)
        else:
            return 'HOLD', 0.0
    
    def _sma_crossover_signal(self, prices: pd.Series) -> Tuple[str, float]:
        """Simple Moving Average crossover strategy."""
        fast_sma = sma(prices, self.fast_period)
        slow_sma = sma(prices, self.slow_period)
        
        # Get recent values
        fast_current = fast_sma.iloc[-1]
        fast_prev = fast_sma.iloc[-2]
        slow_current = slow_sma.iloc[-1]
        slow_prev = slow_sma.iloc[-2]
        
        # Check for crossovers
        if fast_prev <= slow_prev and fast_current > slow_current:
            # Golden cross - BUY signal
            confidence = min((fast_current - slow_current) / slow_current * 100, 1.0)
            return 'BUY', confidence
        elif fast_prev >= slow_prev and fast_current < slow_current:
            # Death cross - SELL signal
            confidence = min((slow_current - fast_current) / slow_current * 100, 1.0)
            return 'SELL', confidence
        
        return 'HOLD', 0.0
    
    def _rsi_extremes_signal(self, prices: pd.Series) -> Tuple[str, float]:
        """RSI overbought/oversold strategy."""
        rsi_values = rsi(prices, self.rsi_period)
        current_rsi = rsi_values.iloc[-1]
        
        if current_rsi < self.rsi_oversold:
            # Oversold - potential BUY
            confidence = (self.rsi_oversold - current_rsi) / self.rsi_oversold
            return 'BUY', confidence
        elif current_rsi > self.rsi_overbought:
            # Overbought - potential SELL
            confidence = (current_rsi - self.rsi_overbought) / (100 - self.rsi_overbought)
            return 'SELL', confidence
        
        return 'HOLD', 0.0
    
    def _ema_momentum_signal(self, prices: pd.Series) -> Tuple[str, float]:
        """EMA-based momentum strategy."""
        fast_ema = ema(prices, self.fast_period)
        slow_ema = ema(prices, self.slow_period)
        
        # Calculate momentum
        momentum = (fast_ema.iloc[-1] - slow_ema.iloc[-1]) / slow_ema.iloc[-1]
        
        if momentum > 0.001:  # 0.1% threshold
            confidence = min(abs(momentum) * 100, 1.0)
            return 'BUY', confidence
        elif momentum < -0.001:
            confidence = min(abs(momentum) * 100, 1.0)
            return 'SELL', confidence
        
        return 'HOLD', 0.0
    
    def _combined_signal(self, prices: pd.Series) -> Tuple[str, float]:
        """Combined signals from multiple indicators."""
        signals = []
        
        # Get individual signals
        sma_signal, sma_conf = self._sma_crossover_signal(prices)
        rsi_signal, rsi_conf = self._rsi_extremes_signal(prices)
        ema_signal, ema_conf = self._ema_momentum_signal(prices)
        
        # Weight the signals
        weights = {'sma': 0.4, 'rsi': 0.3, 'ema': 0.3}
        
        # Calculate combined signal
        buy_score = 0
        sell_score = 0
        
        if sma_signal == 'BUY':
            buy_score += weights['sma'] * sma_conf
        elif sma_signal == 'SELL':
            sell_score += weights['sma'] * sma_conf
            
        if rsi_signal == 'BUY':
            buy_score += weights['rsi'] * rsi_conf
        elif rsi_signal == 'SELL':
            sell_score += weights['rsi'] * rsi_conf
            
        if ema_signal == 'BUY':
            buy_score += weights['ema'] * ema_conf
        elif ema_signal == 'SELL':
            sell_score += weights['ema'] * ema_conf
        
        # Determine final signal
        if buy_score > sell_score and buy_score > 0.3:  # 30% threshold
            return 'BUY', min(buy_score, 1.0)
        elif sell_score > buy_score and sell_score > 0.3:
            return 'SELL', min(sell_score, 1.0)
        
        return 'HOLD', 0.0
    
    def _bollinger_bands_signal(self, prices: pd.Series) -> Tuple[str, float]:
        """Bollinger Bands mean reversion strategy."""
        period = self.parameters.get('bb_period', 20)
        num_std = self.parameters.get('bb_std', 2)
        
        upper, middle, lower = bollinger_bands(prices, period, num_std)
        current_price = prices.iloc[-1]
        
        # Calculate position relative to bands
        bb_width = upper.iloc[-1] - lower.iloc[-1]
        
        if current_price < lower.iloc[-1]:
            # Price below lower band - oversold
            distance = (lower.iloc[-1] - current_price) / bb_width
            confidence = min(distance * 2, 1.0)  # Scale confidence
            return 'BUY', confidence
        elif current_price > upper.iloc[-1]:
            # Price above upper band - overbought
            distance = (current_price - upper.iloc[-1]) / bb_width
            confidence = min(distance * 2, 1.0)
            return 'SELL', confidence
        
        return 'HOLD', 0.0
    
    def _mfi_volume_signal(self, prices: pd.Series, symbol: str) -> Tuple[str, float]:
        """Money Flow Index strategy - requires volume data."""
        # Skip if we don't have real OHLCV data
        if symbol not in self.ohlcv_history:
            return 'HOLD', 0.0
            
        ohlcv = self.ohlcv_history[symbol]
        if len(ohlcv) < self.parameters.get('mfi_period', 14):
            return 'HOLD', 0.0
        
        high = ohlcv['high']
        low = ohlcv['low'] 
        close = ohlcv['close']
        volume = ohlcv['volume']
        
        mfi_values = mfi(high, low, prices, volume, self.parameters.get('mfi_period', 14))
        current_mfi = mfi_values.iloc[-1]
        
        mfi_oversold = self.parameters.get('mfi_oversold', 20)
        mfi_overbought = self.parameters.get('mfi_overbought', 80)
        
        if current_mfi < mfi_oversold:
            confidence = (mfi_oversold - current_mfi) / mfi_oversold
            return 'BUY', confidence
        elif current_mfi > mfi_overbought:
            confidence = (current_mfi - mfi_overbought) / (100 - mfi_overbought)
            return 'SELL', confidence
        
        return 'HOLD', 0.0
    
    def _atr_breakout_signal(self, prices: pd.Series, symbol: str) -> Tuple[str, float]:
        """ATR-based volatility breakout strategy."""
        # Skip if we don't have real OHLCV data
        if symbol not in self.ohlcv_history:
            return 'HOLD', 0.0
            
        ohlcv = self.ohlcv_history[symbol]
        if len(ohlcv) < self.parameters.get('atr_period', 14):
            return 'HOLD', 0.0
        
        high = ohlcv['high']
        low = ohlcv['low']
        close = ohlcv['close']
        
        atr_values = atr(high, low, prices, self.parameters.get('atr_period', 14))
        current_atr = atr_values.iloc[-1]
        
        # Check for volatility expansion
        atr_sma = atr_values.rolling(20).mean()
        
        if pd.notna(atr_sma.iloc[-1]) and current_atr > atr_sma.iloc[-1] * 1.5:
            # High volatility - look for breakout
            price_change = prices.pct_change().iloc[-1]
            
            if price_change > 0:
                confidence = min(abs(price_change) * 10, 1.0)
                return 'BUY', confidence
            elif price_change < 0:
                confidence = min(abs(price_change) * 10, 1.0)
                return 'SELL', confidence
        
        return 'HOLD', 0.0
    
    def _fibonacci_signal(self, prices: pd.Series) -> Tuple[str, float]:
        """Fibonacci retracement strategy."""
        # Find recent high and low
        window = min(50, len(prices))
        recent_high = prices.iloc[-window:].max()
        recent_low = prices.iloc[-window:].min()
        
        fib_levels = fibonacci_retracement(recent_high, recent_low)
        current_price = prices.iloc[-1]
        
        # Check if price is near a Fibonacci level
        for level_name, level_price in fib_levels.items():
            if level_name in ['61.8%', '50.0%', '38.2%']:  # Key retracement levels
                if abs(current_price - level_price) / level_price < 0.002:  # Within 0.2%
                    # Price at support level
                    if prices.pct_change().iloc[-1] < 0:  # Coming from above
                        return 'BUY', 0.7
                    else:  # Bouncing off
                        return 'SELL', 0.7
        
        return 'HOLD', 0.0


def create_strategy_variants(asset_type: str, num_variants: int = 10) -> List[Dict[str, any]]:
    """
    Create multiple strategy variants with different parameters for testing.
    
    This is what the exploration manager would use to generate candidates.
    """
    variants = []
    strategy_types = [
        'sma_crossover', 'rsi_extremes', 'ema_momentum', 
        'bollinger_bands', 'mfi_volume', 'atr_breakout', 
        'fibonacci_retracement', 'combined'
    ]
    
    for i in range(num_variants):
        # Randomly select variant type
        variant_type = random.choice(strategy_types)
        
        # Generate random parameters within reasonable ranges
        params = {
            'name': f'{asset_type}_{variant_type}_{i}',
            'variant': variant_type,
            'fast_period': random.randint(5, 20),
            'slow_period': random.randint(20, 50),
            'rsi_period': random.randint(10, 20),
            'rsi_oversold': random.randint(20, 35),
            'rsi_overbought': random.randint(65, 80),
            'stop_loss': random.uniform(0.005, 0.02),  # 0.5% to 2%
            'take_profit': random.uniform(0.01, 0.04),  # 1% to 4%
        }
        
        # Ensure fast < slow
        if params['fast_period'] >= params['slow_period']:
            params['fast_period'], params['slow_period'] = params['slow_period'], params['fast_period']
        
        variants.append(params)
    
    return variants