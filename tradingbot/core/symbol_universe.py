# file: tradingbot/core/symbol_universe.py
# module_version: v1.00

"""
Symbol Universe Management Module

Owns: crypto universe selection & refresh rules (rank by 24h turnover, filters).
Public API:
- get_symbols(asset)
- refresh_crypto_universe()
- get_universe_stats()

Calls: brokers/bybit_client for tickers/instruments.
Never does: trading, risk.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Tuple
import json
from pathlib import Path
from dataclasses import dataclass

from .interfaces import Asset

logger = logging.getLogger(__name__)

@dataclass
class SymbolMetrics:
    """Metrics for a trading symbol"""
    symbol: str
    volume_24h_usd: float
    price_change_24h_pct: float
    last_price: float
    market_cap_rank: Optional[int] = None
    is_stable: bool = False
    spike_score: float = 0.0
    last_updated: datetime = None

@dataclass
class UniverseConfig:
    """Configuration for universe selection"""
    target_crypto_count: int = 30
    min_crypto_count: int = 20
    max_crypto_count: int = 40
    min_volume_24h_usd: float = 10_000_000  # $10M minimum
    exclude_stablecoins: bool = True
    refresh_interval_hours: int = 6
    spike_threshold_pct: float = 15.0  # 15% price change triggers spike
    
class SymbolUniverse:
    """Manages trading symbol universe selection and refresh"""
    
    def __init__(self, 
                 bybit_client=None,
                 state_dir: str = "tradingbot/state",
                 config: UniverseConfig = None):
        self.bybit_client = bybit_client
        self.state_dir = Path(state_dir)
        self.universe_file = self.state_dir / "symbol_universe.json"
        
        # Ensure state directory exists
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.config = config or UniverseConfig()
        
        # Current universe state
        self._universes: Dict[Asset, List[str]] = {}
        self._symbol_metrics: Dict[str, SymbolMetrics] = {}
        self._last_refresh: Dict[Asset, datetime] = {}
        
        # Stable core symbols that should always be included
        self._core_crypto_symbols = {
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT',
            'XRPUSDT', 'DOTUSDT', 'LINKUSDT', 'LTCUSDT', 'BCHUSDT'
        }
        
        # Known stablecoins to exclude
        self._stablecoins = {
            'USDTUSDT', 'USDCUSDT', 'BUSDUSDT', 'TUSDUSDT', 'DAIUSDT',
            'USTUSDT', 'FRAXUSDT', 'LUSDUSDT'
        }
        
        # Load existing state
        self._load_universe_state()
        
    def get_symbols(self, asset: Asset) -> List[str]:
        """
        Get current symbol list for an asset
        
        Args:
            asset: Asset type to get symbols for
            
        Returns:
            List of symbol strings
        """
        # Check if refresh is needed
        if self._needs_refresh(asset):
            self.refresh_crypto_universe()
            
        return self._universes.get(asset, [])
        
    def refresh_crypto_universe(self) -> bool:
        """
        Refresh crypto universe based on 24h volume ranking
        
        Returns:
            bool: True if successful refresh
        """
        logger.info("Refreshing crypto symbol universe")
        
        try:
            # Fetch current market data
            tickers = self._fetch_crypto_tickers()
            if not tickers:
                logger.error("Failed to fetch crypto tickers")
                return False
                
            # Process and rank symbols
            symbol_metrics = self._process_tickers(tickers)
            
            # Select universe
            selected_symbols = self._select_crypto_universe(symbol_metrics)
            
            # Update state
            self._universes[Asset.CRYPTO] = selected_symbols
            self._symbol_metrics.update(symbol_metrics)
            self._last_refresh[Asset.CRYPTO] = datetime.now()
            
            # Detect and log spikes
            spikes = self._detect_spikes(symbol_metrics)
            if spikes:
                logger.info(f"Detected price spikes in: {', '.join(spikes)}")
                
            # Save state
            self._save_universe_state()
            
            logger.info(f"Refreshed crypto universe: {len(selected_symbols)} symbols")
            return True
            
        except Exception as e:
            logger.error(f"Failed to refresh crypto universe: {e}")
            return False
            
    def get_universe_stats(self, asset: Asset = None) -> Dict:
        """
        Get statistics about current universe
        
        Args:
            asset: Optional asset filter
            
        Returns:
            Dict with universe statistics
        """
        stats = {}
        
        assets_to_check = [asset] if asset else list(Asset)
        
        for asset_type in assets_to_check:
            if asset_type not in self._universes:
                continue
                
            symbols = self._universes[asset_type]
            last_refresh = self._last_refresh.get(asset_type)
            
            asset_stats = {
                "symbol_count": len(symbols),
                "last_refresh": last_refresh.isoformat() if last_refresh else None,
                "needs_refresh": self._needs_refresh(asset_type),
                "symbols": symbols[:10]  # First 10 symbols
            }
            
            # Add volume stats for crypto
            if asset_type == Asset.CRYPTO and symbols:
                volumes = [
                    self._symbol_metrics[sym].volume_24h_usd 
                    for sym in symbols 
                    if sym in self._symbol_metrics
                ]
                if volumes:
                    asset_stats.update({
                        "total_volume_24h_usd": sum(volumes),
                        "avg_volume_24h_usd": sum(volumes) / len(volumes),
                        "min_volume_24h_usd": min(volumes),
                        "max_volume_24h_usd": max(volumes)
                    })
                    
            stats[asset_type.value] = asset_stats
            
        return stats
        
    def get_symbol_metrics(self, symbol: str) -> Optional[SymbolMetrics]:
        """Get metrics for a specific symbol"""
        return self._symbol_metrics.get(symbol)
        
    def is_spike_candidate(self, symbol: str) -> bool:
        """Check if symbol is currently showing a price spike"""
        metrics = self._symbol_metrics.get(symbol)
        if not metrics:
            return False
            
        return abs(metrics.price_change_24h_pct) >= self.config.spike_threshold_pct
        
    def _needs_refresh(self, asset: Asset) -> bool:
        """Check if universe needs refresh for an asset"""
        if asset not in self._last_refresh:
            return True
            
        last_refresh = self._last_refresh[asset]
        refresh_interval = timedelta(hours=self.config.refresh_interval_hours)
        
        return datetime.now() - last_refresh > refresh_interval
        
    def _fetch_crypto_tickers(self) -> List[Dict]:
        """Fetch crypto ticker data from exchange"""
        if not self.bybit_client:
            # Return mock data for testing
            logger.warning("No Bybit client available, using mock data")
            return self._get_mock_tickers()
            
        try:
            # Fetch 24h ticker data from Bybit
            response = self.bybit_client.get_tickers(category="spot")
            
            if response and "result" in response and "list" in response["result"]:
                return response["result"]["list"]
            else:
                logger.error("Unexpected ticker response format")
                return []
                
        except Exception as e:
            logger.error(f"Failed to fetch tickers from Bybit: {e}")
            return []
            
    def _process_tickers(self, tickers: List[Dict]) -> Dict[str, SymbolMetrics]:
        """Process raw ticker data into SymbolMetrics"""
        processed = {}
        
        for ticker in tickers:
            try:
                symbol = ticker.get("symbol", "")
                if not symbol.endswith("USDT"):
                    continue
                    
                volume_24h = float(ticker.get("turnover24h", 0))
                price_change_pct = float(ticker.get("price24hPcnt", 0)) * 100
                last_price = float(ticker.get("lastPrice", 0))
                
                # Skip if volume too low
                if volume_24h < self.config.min_volume_24h_usd:
                    continue
                    
                # Skip stablecoins if configured
                if self.config.exclude_stablecoins and symbol in self._stablecoins:
                    continue
                    
                metrics = SymbolMetrics(
                    symbol=symbol,
                    volume_24h_usd=volume_24h,
                    price_change_24h_pct=price_change_pct,
                    last_price=last_price,
                    spike_score=abs(price_change_pct),
                    last_updated=datetime.now()
                )
                
                processed[symbol] = metrics
                
            except (ValueError, KeyError) as e:
                logger.debug(f"Skipping invalid ticker {ticker}: {e}")
                continue
                
        return processed
        
    def _select_crypto_universe(self, symbol_metrics: Dict[str, SymbolMetrics]) -> List[str]:
        """Select crypto universe based on volume ranking"""
        
        # Sort by 24h volume descending
        sorted_symbols = sorted(
            symbol_metrics.items(),
            key=lambda x: x[1].volume_24h_usd,
            reverse=True
        )
        
        selected = []
        
        # Always include core symbols if available
        for symbol in self._core_crypto_symbols:
            if symbol in symbol_metrics and symbol not in selected:
                selected.append(symbol)
                
        # Add highest volume symbols up to target
        for symbol, metrics in sorted_symbols:
            if len(selected) >= self.config.target_crypto_count:
                break
                
            if symbol not in selected:
                selected.append(symbol)
                
        # Ensure minimum count
        if len(selected) < self.config.min_crypto_count:
            # Add more symbols even if below preferred volume
            remaining_symbols = [
                symbol for symbol, _ in sorted_symbols 
                if symbol not in selected
            ]
            
            needed = self.config.min_crypto_count - len(selected)
            selected.extend(remaining_symbols[:needed])
            
        # Enforce maximum count
        if len(selected) > self.config.max_crypto_count:
            selected = selected[:self.config.max_crypto_count]
            
        return selected
        
    def _detect_spikes(self, symbol_metrics: Dict[str, SymbolMetrics]) -> List[str]:
        """Detect symbols with significant price spikes"""
        spikes = []
        
        for symbol, metrics in symbol_metrics.items():
            if abs(metrics.price_change_24h_pct) >= self.config.spike_threshold_pct:
                spikes.append(f"{symbol}({metrics.price_change_24h_pct:+.1f}%)")
                
        return spikes
        
    def _get_mock_tickers(self) -> List[Dict]:
        """Get mock ticker data for testing"""
        mock_tickers = []
        
        base_symbols = [
            'BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'XRP', 'DOT', 'LINK', 
            'LTC', 'BCH', 'MATIC', 'AVAX', 'UNI', 'ATOM', 'FTT', 
            'NEAR', 'ALGO', 'VET', 'ICP', 'FIL', 'TRX', 'ETC', 
            'XLM', 'HBAR', 'SAND', 'MANA', 'AXS', 'THETA', 'EGLD', 'XTZ'
        ]
        
        import random
        
        for i, symbol in enumerate(base_symbols):
            mock_ticker = {
                "symbol": f"{symbol}USDT",
                "turnover24h": str(random.uniform(10_000_000, 1_000_000_000)),
                "price24hPcnt": str(random.uniform(-0.15, 0.15)),  # -15% to +15%
                "lastPrice": str(random.uniform(0.1, 50000))
            }
            mock_tickers.append(mock_ticker)
            
        return mock_tickers
        
    def _save_universe_state(self):
        """Save universe state to persistent storage"""
        try:
            state_data = {
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "target_crypto_count": self.config.target_crypto_count,
                    "min_volume_24h_usd": self.config.min_volume_24h_usd,
                    "refresh_interval_hours": self.config.refresh_interval_hours
                },
                "universes": {
                    asset.value: symbols 
                    for asset, symbols in self._universes.items()
                },
                "last_refresh": {
                    asset.value: timestamp.isoformat()
                    for asset, timestamp in self._last_refresh.items()
                },
                "symbol_metrics": {
                    symbol: {
                        "volume_24h_usd": metrics.volume_24h_usd,
                        "price_change_24h_pct": metrics.price_change_24h_pct,
                        "last_price": metrics.last_price,
                        "spike_score": metrics.spike_score,
                        "last_updated": metrics.last_updated.isoformat()
                    }
                    for symbol, metrics in self._symbol_metrics.items()
                }
            }
            
            with open(self.universe_file, 'w') as f:
                json.dump(state_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save universe state: {e}")
            
    def _load_universe_state(self):
        """Load universe state from persistent storage"""
        if not self.universe_file.exists():
            # Initialize with default crypto universe
            self._universes[Asset.CRYPTO] = list(self._core_crypto_symbols)
            return
            
        try:
            with open(self.universe_file, 'r') as f:
                data = json.load(f)
                
            # Load universes
            universes_data = data.get("universes", {})
            for asset_str, symbols in universes_data.items():
                try:
                    asset = Asset(asset_str)
                    self._universes[asset] = symbols
                except ValueError:
                    logger.warning(f"Unknown asset in universe state: {asset_str}")
                    
            # Load last refresh times
            refresh_data = data.get("last_refresh", {})
            for asset_str, timestamp_str in refresh_data.items():
                try:
                    asset = Asset(asset_str)
                    timestamp = datetime.fromisoformat(timestamp_str)
                    self._last_refresh[asset] = timestamp
                except (ValueError, TypeError):
                    logger.warning(f"Invalid refresh timestamp for {asset_str}")
                    
            # Load symbol metrics
            metrics_data = data.get("symbol_metrics", {})
            for symbol, metrics_dict in metrics_data.items():
                try:
                    metrics = SymbolMetrics(
                        symbol=symbol,
                        volume_24h_usd=metrics_dict["volume_24h_usd"],
                        price_change_24h_pct=metrics_dict["price_change_24h_pct"],
                        last_price=metrics_dict["last_price"],
                        spike_score=metrics_dict["spike_score"],
                        last_updated=datetime.fromisoformat(metrics_dict["last_updated"])
                    )
                    self._symbol_metrics[symbol] = metrics
                except (KeyError, ValueError, TypeError):
                    logger.debug(f"Skipping invalid metrics for {symbol}")
                    
            logger.info(f"Loaded universe state: {len(self._universes)} asset universes")
            
        except Exception as e:
            logger.error(f"Failed to load universe state: {e}")
            # Fallback to default crypto universe
            self._universes[Asset.CRYPTO] = list(self._core_crypto_symbols)