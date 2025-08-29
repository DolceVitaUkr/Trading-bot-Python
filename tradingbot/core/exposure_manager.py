# file: tradingbot/core/exposure_manager.py
# module_version: v1.00

"""
Exposure Manager - Portfolio and correlation exposure control.
This is the ONLY module that manages aggregate exposure and correlations.
"""

import asyncio
from typing import Dict, Any, Optional, List, Set
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from .configmanager import config_manager
from .loggerconfig import get_logger


class SymbolCluster(Enum):
    """Symbol correlation clusters"""
    BTC_CLUSTER = "btc_cluster"  # BTC and highly correlated
    ETH_L1_CLUSTER = "eth_l1_cluster"  # ETH and L1s
    DEFI_CLUSTER = "defi_cluster"  # DeFi tokens
    MEME_CLUSTER = "meme_cluster"  # Meme coins
    USD_MAJORS = "usd_majors"  # USD forex pairs
    COMMODITIES = "commodities"  # Gold, oil, etc
    UNCLUSTERED = "unclustered"  # Low correlation


@dataclass
class ExposurePosition:
    """Active position for exposure tracking"""
    symbol: str
    side: str  # long/short
    size: float
    notional_usd: float
    cluster: SymbolCluster
    asset_type: str
    entry_time: datetime
    correlation_factor: float = 1.0


class ExposureManager:
    """
    Manages portfolio-wide exposure and correlation limits.
    Prevents concentration risk and correlated exposure buildup.
    """
    
    def __init__(self):
        self.log = get_logger("exposure_manager")
        self.config = config_manager
        
        # Active positions by symbol
        self.positions: Dict[str, ExposurePosition] = {}
        
        # Cluster definitions
        self._define_clusters()
        
        # Exposure limits
        self._load_exposure_limits()
        
        # Dependencies (will be injected)
        self.pnl_reconciler = None
        
        self.log.info("Exposure Manager initialized - Portfolio exposure controlled")
    
    def _define_clusters(self):
        """Define symbol correlation clusters"""
        self.clusters = {
            SymbolCluster.BTC_CLUSTER: {
                'symbols': ['BTCUSDT', 'BTCUSD', 'WBTCUSDT', 'BTCETH'],
                'correlation': 0.95
            },
            SymbolCluster.ETH_L1_CLUSTER: {
                'symbols': ['ETHUSDT', 'ETHUSD', 'SOLUSDT', 'AVAXUSDT', 'ADAUSDT', 'DOTUSDT'],
                'correlation': 0.75
            },
            SymbolCluster.DEFI_CLUSTER: {
                'symbols': ['UNIUSDT', 'AAVEUSDT', 'SUSHIUSDT', 'COMPUSDT', 'MKRUSDT', 'SNXUSDT'],
                'correlation': 0.70
            },
            SymbolCluster.MEME_CLUSTER: {
                'symbols': ['DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT', 'FLOKIUSDT'],
                'correlation': 0.65
            },
            SymbolCluster.USD_MAJORS: {
                'symbols': ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY'],
                'correlation': 0.60
            }
        }
        
        # Build reverse lookup
        self.symbol_to_cluster: Dict[str, SymbolCluster] = {}
        for cluster, info in self.clusters.items():
            for symbol in info['symbols']:
                self.symbol_to_cluster[symbol] = cluster
    
    def _load_exposure_limits(self):
        """Load exposure limits from config"""
        exposure_config = self.config.config.get('exposure', {})
        
        self.limits = {
            'max_single_position_pct': exposure_config.get('max_single_position_pct', 0.1),  # 10% max per position
            'max_cluster_exposure_pct': exposure_config.get('max_cluster_exposure_pct', 0.3),  # 30% max per cluster
            'max_directional_exposure_pct': exposure_config.get('max_directional_exposure_pct', 0.6),  # 60% max long or short
            'max_concurrent_per_cluster': exposure_config.get('max_concurrent_per_cluster', 2),  # Max 2 positions per cluster
            'max_total_positions': {
                'spot': exposure_config.get('max_spot_positions', 6),
                'futures': exposure_config.get('max_futures_positions', 3),
                'forex': exposure_config.get('max_forex_positions', 4),
                'options': exposure_config.get('max_options_positions', 3)
            }
        }
    
    def set_dependencies(self, pnl_reconciler=None):
        """Inject dependencies"""
        if pnl_reconciler:
            self.pnl_reconciler = pnl_reconciler
    
    async def can_open(self, symbol: str, side: str, size: float, 
                      asset_type: str = 'spot', price: float = None) -> Dict[str, Any]:
        """
        Check if a new position can be opened within exposure limits.
        
        Args:
            symbol: Trading symbol
            side: 'long' or 'short'
            size: Position size
            asset_type: spot/futures/forex/options
            price: Current price (for notional calculation)
        
        Returns:
            {
                'pass': bool,
                'reason': str,
                'current_exposure': dict
            }
        """
        
        # Get cluster for symbol
        cluster = self.symbol_to_cluster.get(symbol, SymbolCluster.UNCLUSTERED)
        
        # Calculate notional value
        notional_usd = size * (price or 1)
        
        # Get current exposures
        current = self._calculate_current_exposure()
        
        violations = []
        
        # 1. Check max concurrent positions for asset type
        asset_position_count = len([p for p in self.positions.values() 
                                   if p.asset_type == asset_type])
        
        if asset_position_count >= self.limits['max_total_positions'].get(asset_type, 10):
            violations.append(f"Max {asset_type} positions reached ({asset_position_count})")
        
        # 2. Check single position size limit
        total_portfolio_value = current.get('total_notional', 10000)  # Default 10k
        position_pct = notional_usd / total_portfolio_value
        
        if position_pct > self.limits['max_single_position_pct']:
            violations.append(f"Position size {position_pct:.1%} exceeds max {self.limits['max_single_position_pct']:.1%}")
        
        # 3. Check cluster exposure limit
        cluster_exposure = current.get('cluster_exposure', {}).get(cluster, 0)
        new_cluster_exposure = (cluster_exposure + notional_usd) / total_portfolio_value
        
        if new_cluster_exposure > self.limits['max_cluster_exposure_pct']:
            violations.append(f"Cluster exposure {new_cluster_exposure:.1%} exceeds max {self.limits['max_cluster_exposure_pct']:.1%}")
        
        # 4. Check concurrent positions in cluster
        cluster_positions = [p for p in self.positions.values() if p.cluster == cluster]
        
        if len(cluster_positions) >= self.limits['max_concurrent_per_cluster']:
            violations.append(f"Max concurrent in {cluster.value} reached ({len(cluster_positions)})")
        
        # 5. Check directional exposure
        if side == 'long':
            new_long_exposure = (current.get('long_exposure', 0) + notional_usd) / total_portfolio_value
            if new_long_exposure > self.limits['max_directional_exposure_pct']:
                violations.append(f"Long exposure {new_long_exposure:.1%} exceeds max {self.limits['max_directional_exposure_pct']:.1%}")
        else:
            new_short_exposure = (current.get('short_exposure', 0) + notional_usd) / total_portfolio_value
            if new_short_exposure > self.limits['max_directional_exposure_pct']:
                violations.append(f"Short exposure {new_short_exposure:.1%} exceeds max {self.limits['max_directional_exposure_pct']:.1%}")
        
        # Check for existing position in same symbol
        if symbol in self.positions:
            existing = self.positions[symbol]
            if existing.side != side:
                violations.append(f"Cannot open opposite side for {symbol} (existing {existing.side})")
        
        if violations:
            self.log.warning(f"Exposure check failed for {symbol}: {violations}")
            return {
                'pass': False,
                'reason': '; '.join(violations),
                'current_exposure': current
            }
        
        self.log.info(f"Exposure check passed for {symbol} {side} ${notional_usd:.2f}")
        
        return {
            'pass': True,
            'reason': 'Within exposure limits',
            'current_exposure': current
        }
    
    def add_position(self, symbol: str, side: str, size: float, 
                    price: float, asset_type: str):
        """Add a new position to exposure tracking"""
        
        cluster = self.symbol_to_cluster.get(symbol, SymbolCluster.UNCLUSTERED)
        correlation_factor = self.clusters.get(cluster, {}).get('correlation', 1.0)
        
        position = ExposurePosition(
            symbol=symbol,
            side=side,
            size=size,
            notional_usd=size * price,
            cluster=cluster,
            asset_type=asset_type,
            entry_time=datetime.now(),
            correlation_factor=correlation_factor
        )
        
        self.positions[symbol] = position
        
        self.log.info(f"Added position: {symbol} {side} ${position.notional_usd:.2f} "
                     f"in {cluster.value}")
    
    def update_position(self, symbol: str, new_size: float = None, 
                       new_price: float = None):
        """Update an existing position"""
        
        if symbol not in self.positions:
            self.log.warning(f"Position {symbol} not found for update")
            return
        
        position = self.positions[symbol]
        
        if new_size is not None:
            position.size = new_size
        
        if new_price is not None:
            position.notional_usd = position.size * new_price
        
        self.log.info(f"Updated position: {symbol} size={position.size} "
                     f"notional=${position.notional_usd:.2f}")
    
    def remove_position(self, symbol: str):
        """Remove a closed position"""
        
        if symbol in self.positions:
            position = self.positions.pop(symbol)
            self.log.info(f"Removed position: {symbol} was ${position.notional_usd:.2f}")
    
    def _calculate_current_exposure(self) -> Dict[str, Any]:
        """Calculate current portfolio exposure metrics"""
        
        total_notional = 0
        long_exposure = 0
        short_exposure = 0
        cluster_exposure: Dict[SymbolCluster, float] = {}
        asset_exposure: Dict[str, float] = {}
        
        for position in self.positions.values():
            total_notional += position.notional_usd
            
            if position.side == 'long':
                long_exposure += position.notional_usd
            else:
                short_exposure += position.notional_usd
            
            # Aggregate by cluster
            if position.cluster not in cluster_exposure:
                cluster_exposure[position.cluster] = 0
            cluster_exposure[position.cluster] += position.notional_usd
            
            # Aggregate by asset type
            if position.asset_type not in asset_exposure:
                asset_exposure[position.asset_type] = 0
            asset_exposure[position.asset_type] += position.notional_usd
        
        # Calculate percentages
        exposure_summary = {
            'total_notional': total_notional,
            'long_exposure': long_exposure,
            'short_exposure': short_exposure,
            'net_exposure': long_exposure - short_exposure,
            'position_count': len(self.positions),
            'cluster_exposure': {k.value: v for k, v in cluster_exposure.items()},
            'asset_exposure': asset_exposure,
            'long_pct': (long_exposure / total_notional * 100) if total_notional > 0 else 0,
            'short_pct': (short_exposure / total_notional * 100) if total_notional > 0 else 0
        }
        
        return exposure_summary
    
    def current_exposure(self) -> Dict[str, Any]:
        """Get current exposure summary"""
        return self._calculate_current_exposure()
    
    def get_cluster_symbols(self, cluster: SymbolCluster) -> List[str]:
        """Get all symbols in a cluster"""
        return self.clusters.get(cluster, {}).get('symbols', [])
    
    def get_correlated_positions(self, symbol: str) -> List[ExposurePosition]:
        """Get positions correlated with a given symbol"""
        
        cluster = self.symbol_to_cluster.get(symbol, SymbolCluster.UNCLUSTERED)
        
        if cluster == SymbolCluster.UNCLUSTERED:
            return []
        
        correlated = []
        for pos in self.positions.values():
            if pos.cluster == cluster and pos.symbol != symbol:
                correlated.append(pos)
        
        return correlated
    
    async def check_correlation_risk(self) -> Dict[str, Any]:
        """Check for excessive correlation risk in portfolio"""
        
        risks = []
        current = self._calculate_current_exposure()
        
        # Check each cluster
        for cluster, exposure in current.get('cluster_exposure', {}).items():
            exposure_pct = exposure / current.get('total_notional', 1) if current.get('total_notional', 0) > 0 else 0
            
            if exposure_pct > self.limits['max_cluster_exposure_pct'] * 0.8:  # 80% of limit = warning
                risks.append({
                    'cluster': cluster,
                    'exposure_pct': exposure_pct,
                    'limit_pct': self.limits['max_cluster_exposure_pct'],
                    'severity': 'high' if exposure_pct > self.limits['max_cluster_exposure_pct'] else 'medium'
                })
        
        # Check directional bias
        long_pct = current.get('long_pct', 0) / 100
        short_pct = current.get('short_pct', 0) / 100
        
        if long_pct > self.limits['max_directional_exposure_pct'] * 0.8:
            risks.append({
                'type': 'directional_long',
                'exposure_pct': long_pct,
                'limit_pct': self.limits['max_directional_exposure_pct'],
                'severity': 'high' if long_pct > self.limits['max_directional_exposure_pct'] else 'medium'
            })
        
        if short_pct > self.limits['max_directional_exposure_pct'] * 0.8:
            risks.append({
                'type': 'directional_short',
                'exposure_pct': short_pct,
                'limit_pct': self.limits['max_directional_exposure_pct'],
                'severity': 'high' if short_pct > self.limits['max_directional_exposure_pct'] else 'medium'
            })
        
        return {
            'has_risks': len(risks) > 0,
            'risks': risks,
            'current_exposure': current
        }


# Module initialization
exposure_manager = ExposureManager()