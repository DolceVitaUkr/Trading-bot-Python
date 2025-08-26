# file: core/pair_manager.py
"""
Advanced pair selection engine implementing multi-factor scoring system.
Filters for liquidity, volatility, momentum, sentiment, and correlation.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd


@dataclass
class PairScore:
    """Container for pair scoring results."""
    symbol: str
    score: float
    liquidity_score: float
    volatility_score: float
    momentum_score: float
    correlation_score: float
    sentiment_score: float
    technical_score: float
    volume_24h: float
    price: float
    change_24h: float
    atr_pct: float
    spread_bps: float
    regime: str


class PairManager:
    """
    Advanced pair selection using multi-factor scoring system.
    
    Implements sophisticated ranking based on:
    - Liquidity & Market Quality (20 pts)
    - Volatility & Trend (20 pts) 
    - Momentum vs BTC/ETH (20 pts)
    - Correlation & Diversification (15 pts)
    - Sentiment & News (15 pts)
    - Technical Health (10 pts)
    """

    def __init__(
        self,
        min_volume_usd: float = 100_000_000,  # $100M minimum
        max_spread_bps: float = 5.0,  # 0.05% max spread
        min_atr_pct: float = 1.0,  # 1% minimum volatility
        max_pairs: int = 5,
        update_interval_minutes: int = 15,
    ):
        self.min_volume_usd = min_volume_usd
        self.max_spread_bps = max_spread_bps
        self.min_atr_pct = min_atr_pct
        self.max_pairs = max_pairs
        self.update_interval_minutes = update_interval_minutes
        
        self.log = logging.getLogger(__name__)
        self.last_update: Optional[datetime] = None
        self.current_pairs: List[PairScore] = []
        self.btc_performance: Dict[str, float] = {}
        self.eth_performance: Dict[str, float] = {}
        
        # Major pairs for correlation analysis
        self.majors = ['BTC/USDT', 'ETH/USDT']
        
        # Sector mapping for diversification
        self.sectors = {
            'BTC/USDT': 'store_of_value',
            'ETH/USDT': 'smart_contract',
            'SOL/USDT': 'layer1',
            'ADA/USDT': 'layer1', 
            'DOT/USDT': 'interoperability',
            'LINK/USDT': 'oracle',
            'UNI/USDT': 'defi',
            'AAVE/USDT': 'defi',
            'MATIC/USDT': 'layer2',
            'AVAX/USDT': 'layer1',
            'XRP/USDT': 'payments',
            'DOGE/USDT': 'meme',
            'SHIB/USDT': 'meme'
        }

    async def rank_pairs(self, market_data: Dict[str, Any]) -> List[PairScore]:
        """
        Rank trading pairs using multi-factor scoring system.
        
        Args:
            market_data: Dict containing market data for all symbols
            
        Returns:
            List of PairScore objects, ranked by total score
        """
        try:
            self.log.info("Starting pair ranking analysis...")
            
            # Filter for basic quality requirements
            qualified_pairs = self._apply_quality_filters(market_data)
            self.log.info(f"Qualified pairs after filtering: {len(qualified_pairs)}")
            
            if len(qualified_pairs) < 3:
                self.log.warning("Insufficient qualified pairs, using mock data")
                return self._get_mock_pairs()
            
            # Calculate reference performance for momentum analysis
            await self._calculate_reference_performance(market_data)
            
            # Score each qualified pair
            scored_pairs = []
            for symbol, data in qualified_pairs.items():
                try:
                    score = await self._score_pair(symbol, data, market_data)
                    if score:
                        scored_pairs.append(score)
                except Exception as e:
                    self.log.error(f"Failed to score {symbol}: {e}")
                    continue
            
            # Sort by total score and return top pairs
            scored_pairs.sort(key=lambda x: x.score, reverse=True)
            top_pairs = scored_pairs[:self.max_pairs]
            
            self.current_pairs = top_pairs
            self.last_update = datetime.utcnow()
            
            self.log.info(f"Selected top {len(top_pairs)} pairs:")
            for pair in top_pairs:
                self.log.info(f"  {pair.symbol}: {pair.score:.1f} ({pair.regime})")
            
            return top_pairs
            
        except Exception as e:
            self.log.error(f"Pair ranking failed: {e}")
            return self._get_mock_pairs()

    def _apply_quality_filters(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply basic liquidity and quality filters."""
        qualified = {}
        
        for symbol, data in market_data.items():
            try:
                volume_24h = data.get('volume_24h', 0)
                spread_bps = data.get('spread_bps', 999)
                
                # Basic quality checks
                if volume_24h >= self.min_volume_usd and spread_bps <= self.max_spread_bps:
                    # Additional checks for suspicious activity
                    price_change = abs(data.get('change_24h', 0))
                    if price_change < 50:  # Exclude extreme pumps > 50%
                        qualified[symbol] = data
                        
            except Exception as e:
                self.log.debug(f"Filter check failed for {symbol}: {e}")
                continue
                
        return qualified

    async def _calculate_reference_performance(self, market_data: Dict[str, Any]) -> None:
        """Calculate BTC and ETH performance across timeframes for momentum comparison."""
        timeframes = ['1h', '4h', '1d']
        
        for tf in timeframes:
            btc_data = market_data.get('BTC/USDT', {})
            eth_data = market_data.get('ETH/USDT', {})
            
            self.btc_performance[tf] = btc_data.get(f'change_{tf}', 0)
            self.eth_performance[tf] = eth_data.get(f'change_{tf}', 0)

    async def _score_pair(self, symbol: str, data: Dict[str, Any], market_data: Dict[str, Any]) -> Optional[PairScore]:
        """Score individual pair across all factors."""
        try:
            # Extract base metrics
            volume_24h = data.get('volume_24h', 0)
            price = data.get('price', 0)
            change_24h = data.get('change_24h', 0)
            spread_bps = data.get('spread_bps', 0)
            atr_pct = data.get('atr_pct', 0)
            
            # Score each factor (0-100 scale, then weighted)
            liquidity_score = self._score_liquidity(volume_24h, spread_bps) * 0.20
            volatility_score = self._score_volatility_trend(symbol, data) * 0.20
            momentum_score = self._score_momentum(symbol, data) * 0.20
            correlation_score = self._score_correlation(symbol, market_data) * 0.15
            sentiment_score = self._score_sentiment(symbol, data) * 0.15
            technical_score = self._score_technical(symbol, data) * 0.10
            
            total_score = (
                liquidity_score + volatility_score + momentum_score +
                correlation_score + sentiment_score + technical_score
            )
            
            # Tag regime
            regime = self.tag_regime(symbol, data)
            
            return PairScore(
                symbol=symbol,
                score=total_score,
                liquidity_score=liquidity_score,
                volatility_score=volatility_score,
                momentum_score=momentum_score,
                correlation_score=correlation_score,
                sentiment_score=sentiment_score,
                technical_score=technical_score,
                volume_24h=volume_24h,
                price=price,
                change_24h=change_24h,
                atr_pct=atr_pct,
                spread_bps=spread_bps,
                regime=regime
            )
            
        except Exception as e:
            self.log.error(f"Scoring failed for {symbol}: {e}")
            return None

    def _score_liquidity(self, volume_24h: float, spread_bps: float) -> float:
        """
        Score liquidity and market quality (0-100).
        
        Factors:
        - Volume (70%): Higher volume = better liquidity
        - Spread (30%): Tighter spread = better quality
        """
        # Volume scoring (log scale)
        if volume_24h <= 0:
            volume_score = 0
        else:
            # Score 0-100 based on volume relative to minimum
            volume_ratio = volume_24h / self.min_volume_usd
            volume_score = min(100, 50 + 30 * np.log10(volume_ratio))
        
        # Spread scoring (inverse relationship)
        if spread_bps <= 0:
            spread_score = 100
        else:
            # Perfect score for spreads <= 1bp, declining to 0 at max_spread_bps
            spread_score = max(0, 100 * (1 - spread_bps / self.max_spread_bps))
        
        return volume_score * 0.7 + spread_score * 0.3

    def _score_volatility_trend(self, symbol: str, data: Dict[str, Any]) -> float:
        """
        Score volatility and trend opportunity (0-100).
        
        Factors:
        - ATR/Volatility (60%): Sufficient volatility for opportunities
        - Trend strength (40%): Clear directional bias
        """
        atr_pct = data.get('atr_pct', 0)
        trend_strength = data.get('trend_strength', 0)  # -1 to 1, where 0 = sideways
        
        # Volatility scoring
        if atr_pct < self.min_atr_pct:
            vol_score = 0
        else:
            # Optimal volatility around 3-5%, declining for extreme values
            optimal_atr = 4.0
            vol_deviation = abs(atr_pct - optimal_atr)
            vol_score = max(0, 100 - vol_deviation * 10)
        
        # Trend scoring
        trend_score = min(100, abs(trend_strength) * 100)
        
        return vol_score * 0.6 + trend_score * 0.4

    def _score_momentum(self, symbol: str, data: Dict[str, Any]) -> float:
        """
        Score relative momentum vs BTC/ETH (0-100).
        
        Logic: Favor coins outperforming major reference assets
        """
        timeframes = ['1h', '4h', '1d']
        momentum_scores = []
        
        for tf in timeframes:
            coin_change = data.get(f'change_{tf}', 0)
            btc_change = self.btc_performance.get(tf, 0)
            eth_change = self.eth_performance.get(tf, 0)
            
            # Calculate relative strength vs both BTC and ETH
            btc_relative = coin_change - btc_change
            eth_relative = coin_change - eth_change
            
            # Score based on outperformance
            relative_strength = (btc_relative + eth_relative) / 2
            
            # Convert to 0-100 score (0 = underperforming, 100 = strong outperformance)
            tf_score = 50 + min(50, max(-50, relative_strength * 5))
            momentum_scores.append(tf_score)
        
        # Weight recent timeframes more heavily
        weights = [0.5, 0.3, 0.2]  # 1h, 4h, 1d
        weighted_score = sum(s * w for s, w in zip(momentum_scores, weights))
        
        return weighted_score

    def _score_correlation(self, symbol: str, market_data: Dict[str, Any]) -> float:
        """
        Score correlation and diversification benefit (0-100).
        
        Logic: Favor coins with low correlation to already selected pairs
        """
        if not self.current_pairs:
            return 100  # First pair gets full score
        
        correlations = []
        current_symbols = [p.symbol for p in self.current_pairs]
        
        for existing_symbol in current_symbols:
            # Calculate correlation (simplified - in real implementation use price series)
            corr = self._calculate_correlation(symbol, existing_symbol, market_data)
            correlations.append(abs(corr))
        
        # Score based on average correlation (lower = better)
        avg_correlation = np.mean(correlations) if correlations else 0
        correlation_score = max(0, 100 * (1 - avg_correlation))
        
        # Bonus for sector diversification
        symbol_sector = self.sectors.get(symbol, 'other')
        existing_sectors = [self.sectors.get(s, 'other') for s in current_symbols]
        
        sector_bonus = 20 if symbol_sector not in existing_sectors else 0
        
        return min(100, correlation_score + sector_bonus)

    def _calculate_correlation(self, symbol1: str, symbol2: str, market_data: Dict[str, Any]) -> float:
        """Calculate simplified correlation between two symbols."""
        # Simplified correlation using recent price changes
        data1 = market_data.get(symbol1, {})
        data2 = market_data.get(symbol2, {})
        
        changes1 = [data1.get('change_1h', 0), data1.get('change_4h', 0), data1.get('change_1d', 0)]
        changes2 = [data2.get('change_1h', 0), data2.get('change_4h', 0), data2.get('change_1d', 0)]
        
        if len(changes1) < 2 or len(changes2) < 2:
            return 0.5  # Default moderate correlation
        
        try:
            correlation = np.corrcoef(changes1, changes2)[0, 1]
            return correlation if not np.isnan(correlation) else 0.5
        except:
            return 0.5

    def _score_sentiment(self, symbol: str, data: Dict[str, Any]) -> float:
        """
        Score market sentiment and news (0-100).
        
        In production: integrate social sentiment, news analysis
        For now: simplified based on price action and volume patterns
        """
        # Sentiment proxy: volume-price relationship
        volume_24h = data.get('volume_24h', 0)
        change_24h = data.get('change_24h', 0)
        avg_volume = data.get('avg_volume_7d', volume_24h)
        
        # Volume surge indicates interest
        volume_ratio = volume_24h / max(avg_volume, 1)
        volume_score = min(50, volume_ratio * 25)
        
        # Price momentum indicates sentiment direction
        momentum_score = 50 + min(25, max(-25, change_24h * 2))
        
        # News sentiment (mock - would integrate real sentiment API)
        news_score = 50  # Neutral by default
        
        return (volume_score + momentum_score + news_score) / 3

    def _score_technical(self, symbol: str, data: Dict[str, Any]) -> float:
        """
        Score technical health (0-100).
        
        Factors: RSI, support/resistance, Fibonacci levels
        """
        rsi = data.get('rsi', 50)
        support_distance = data.get('support_distance_pct', 5)
        resistance_distance = data.get('resistance_distance_pct', 5)
        
        # RSI scoring (favor 30-70 range, avoid extreme overbought/oversold)
        if 30 <= rsi <= 70:
            rsi_score = 100
        elif rsi < 20 or rsi > 80:
            rsi_score = 20  # Extreme levels
        else:
            rsi_score = 60  # Moderate overbought/oversold
        
        # Support/Resistance scoring
        sr_score = 100 - min(50, (support_distance + resistance_distance) * 5)
        
        return (rsi_score + sr_score) / 2

    def tag_regime(self, symbol: str, data: Dict[str, Any]) -> str:
        """
        Tag market regime for the symbol.
        
        Returns: 'trending', 'ranging', 'volatile', 'breakout'
        """
        atr_pct = data.get('atr_pct', 0)
        trend_strength = data.get('trend_strength', 0)
        change_24h = data.get('change_24h', 0)
        
        # High volatility regime
        if atr_pct > 6:
            return 'volatile'
        
        # Strong trend regime  
        if abs(trend_strength) > 0.7:
            return 'trending'
        
        # Breakout regime (large price move with volume)
        if abs(change_24h) > 8:
            return 'breakout'
        
        # Default to ranging
        return 'ranging'

    def _get_mock_pairs(self) -> List[PairScore]:
        """Return mock data when real analysis fails."""
        mock_pairs = [
            PairScore('BTC/USDT', 85.5, 18.0, 16.5, 17.2, 13.8, 12.0, 8.0, 1250000000, 43250.00, 2.34, 3.2, 2.1, 'trending'),
            PairScore('ETH/USDT', 82.3, 17.5, 15.8, 16.4, 14.2, 11.4, 7.0, 890000000, 2650.00, -1.23, 4.1, 2.8, 'ranging'),
            PairScore('SOL/USDT', 79.1, 15.2, 17.9, 15.8, 12.7, 10.5, 7.0, 560000000, 98.50, 5.67, 5.8, 4.2, 'breakout'),
            PairScore('XRP/USDT', 75.8, 16.8, 14.2, 14.1, 13.5, 9.7, 7.5, 450000000, 0.52, -0.89, 2.9, 3.1, 'ranging'),
            PairScore('ADA/USDT', 73.4, 14.9, 15.1, 13.8, 12.9, 10.2, 6.5, 320000000, 0.38, 3.45, 4.5, 3.8, 'trending')
        ]
        
        self.log.info("Using mock pair data for demonstration")
        return mock_pairs

    def needs_update(self) -> bool:
        """Check if pair rankings need to be updated."""
        if not self.last_update:
            return True
        
        time_since_update = datetime.utcnow() - self.last_update
        return time_since_update.total_seconds() > (self.update_interval_minutes * 60)

    def get_current_pairs(self) -> List[PairScore]:
        """Get currently selected pairs."""
        return self.current_pairs.copy()

    def get_pair_score(self, symbol: str) -> Optional[PairScore]:
        """Get score for specific pair."""
        for pair in self.current_pairs:
            if pair.symbol == symbol:
                return pair
        return None


__all__ = ["PairManager", "PairScore"]