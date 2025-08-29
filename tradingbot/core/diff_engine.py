# file: tradingbot/core/diff_engine.py
# module_version: v1.00

"""
Dry-Run Diff Module

Owns: compute "what live would have done vs paper" for audits.
Public API:
- compare(asset, since_ts) -> diff report
- simulate_live_orders(asset, paper_trades)

Calls: data_manager, order_router (simulation mode), pnl_reconciler.
Never does: live order placement.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .interfaces import Asset, OrderSide, OrderType

logger = logging.getLogger(__name__)

class DiffType(Enum):
    """Types of differences between paper and live"""
    EXECUTION_PRICE = "execution_price"
    SLIPPAGE = "slippage" 
    LATENCY = "latency"
    REJECTION = "rejection"
    PARTIAL_FILL = "partial_fill"
    SL_TP_EXECUTION = "sl_tp_execution"
    MARKET_IMPACT = "market_impact"

@dataclass
class TradeDiff:
    """Difference between paper and simulated live execution"""
    paper_trade_id: str
    timestamp: datetime
    symbol: str
    side: OrderSide
    diff_type: DiffType
    paper_value: Any
    live_value: Any
    impact_usd: float
    description: str
    confidence: float = 0.0  # 0-1 confidence in simulation accuracy

@dataclass
class DiffReport:
    """Comprehensive diff report"""
    asset: Asset
    start_time: datetime
    end_time: datetime
    paper_trades_count: int
    simulated_trades_count: int
    total_diffs: int
    diffs_by_type: Dict[DiffType, int]
    cumulative_impact_usd: float
    individual_diffs: List[TradeDiff]
    summary_stats: Dict[str, Any]
    
class DiffEngine:
    """Computes differences between paper and hypothetical live execution"""
    
    def __init__(self, 
                 data_manager,
                 order_router,
                 pnl_reconciler):
        self.data_manager = data_manager
        self.order_router = order_router
        self.pnl_reconciler = pnl_reconciler
        
        # Simulation parameters
        self.slippage_bps = {
            Asset.CRYPTO: 2.0,    # 0.02% average slippage
            Asset.FOREX: 0.5,     # 0.005% tighter spreads
            Asset.FUTURES: 1.0,   # 0.01% futures slippage
            Asset.OPTIONS: 5.0    # 0.05% wider spreads
        }
        
        self.latency_ms = {
            Asset.CRYPTO: 50,     # 50ms average execution latency
            Asset.FOREX: 30,      # 30ms institutional FX
            Asset.FUTURES: 40,    # 40ms futures execution  
            Asset.OPTIONS: 100    # 100ms options complexity
        }
        
    def compare(self, asset: Asset, since_ts: datetime) -> DiffReport:
        """
        Compare paper trades vs simulated live execution
        
        Args:
            asset: Asset to analyze
            since_ts: Start time for comparison
            
        Returns:
            DiffReport: Comprehensive comparison report
        """
        logger.info(f"Computing paper vs live diff for {asset.value} since {since_ts}")
        
        # Get paper trades from reconciler
        paper_trades = self.pnl_reconciler.get_trades(
            asset=asset,
            start_time=since_ts,
            end_time=datetime.now()
        )
        
        if not paper_trades:
            return self._empty_diff_report(asset, since_ts)
            
        # Simulate live execution for each trade
        diffs = []
        simulated_count = 0
        
        for trade in paper_trades:
            try:
                trade_diffs = self._simulate_trade_execution(trade)
                diffs.extend(trade_diffs)
                simulated_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to simulate trade {trade.get('id', 'unknown')}: {e}")
                
        # Aggregate results
        diff_report = self._build_diff_report(
            asset=asset,
            start_time=since_ts,
            end_time=datetime.now(),
            paper_trades_count=len(paper_trades),
            simulated_trades_count=simulated_count,
            diffs=diffs
        )
        
        logger.info(f"Generated diff report: {diff_report.total_diffs} differences, "
                   f"${diff_report.cumulative_impact_usd:.2f} total impact")
                   
        return diff_report
        
    def simulate_live_orders(self, asset: Asset, paper_trades: List[Dict]) -> List[Dict]:
        """
        Simulate what live orders would have looked like
        
        Args:
            asset: Asset type
            paper_trades: List of paper trades to simulate
            
        Returns:
            List of simulated live order results
        """
        simulated_orders = []
        
        for trade in paper_trades:
            try:
                # Enable simulation mode on order router
                self.order_router.set_simulation_mode(True)
                
                # Simulate the order with live market conditions
                sim_result = self._simulate_order_with_market_data(trade, asset)
                simulated_orders.append(sim_result)
                
            except Exception as e:
                logger.error(f"Failed to simulate order: {e}")
                
            finally:
                # Always disable simulation mode
                self.order_router.set_simulation_mode(False)
                
        return simulated_orders
        
    def _simulate_trade_execution(self, paper_trade: Dict) -> List[TradeDiff]:
        """Simulate live execution of a single paper trade"""
        diffs = []
        
        symbol = paper_trade.get("symbol")
        side = OrderSide(paper_trade.get("side"))
        paper_price = paper_trade.get("execution_price", 0.0)
        quantity = paper_trade.get("quantity", 0.0)
        timestamp = datetime.fromisoformat(paper_trade.get("timestamp"))
        trade_id = paper_trade.get("id", "unknown")
        
        # Get market data at execution time
        market_data = self._get_historical_market_data(symbol, timestamp)
        if not market_data:
            return diffs
            
        # Simulate execution price with slippage
        live_price = self._simulate_execution_price(
            paper_price, side, quantity, market_data
        )
        
        if abs(live_price - paper_price) > 0.001:  # Significant price difference
            price_diff = TradeDiff(
                paper_trade_id=trade_id,
                timestamp=timestamp,
                symbol=symbol,
                side=side,
                diff_type=DiffType.EXECUTION_PRICE,
                paper_value=paper_price,
                live_value=live_price,
                impact_usd=(live_price - paper_price) * quantity * (1 if side == OrderSide.BUY else -1),
                description=f"Execution price diff: paper ${paper_price:.6f} vs live ${live_price:.6f}",
                confidence=0.8
            )
            diffs.append(price_diff)
            
        # Simulate latency impact
        latency_impact = self._simulate_latency_impact(
            symbol, timestamp, side, quantity, market_data
        )
        
        if latency_impact["impact_usd"] != 0:
            latency_diff = TradeDiff(
                paper_trade_id=trade_id,
                timestamp=timestamp,
                symbol=symbol,
                side=side,
                diff_type=DiffType.LATENCY,
                paper_value=0,  # Paper has no latency
                live_value=latency_impact["latency_ms"],
                impact_usd=latency_impact["impact_usd"],
                description=f"Latency impact: {latency_impact['latency_ms']}ms delay",
                confidence=0.6
            )
            diffs.append(latency_diff)
            
        # Simulate potential rejection scenarios
        rejection_prob = self._calculate_rejection_probability(
            symbol, side, quantity, market_data
        )
        
        if rejection_prob > 0.1:  # 10% or higher rejection chance
            rejection_diff = TradeDiff(
                paper_trade_id=trade_id,
                timestamp=timestamp,
                symbol=symbol,
                side=side,
                diff_type=DiffType.REJECTION,
                paper_value=False,  # Paper never rejects
                live_value=rejection_prob,
                impact_usd=paper_trade.get("pnl", 0.0) * rejection_prob,  # Lost opportunity
                description=f"Rejection risk: {rejection_prob:.1%} probability",
                confidence=0.4
            )
            diffs.append(rejection_diff)
            
        return diffs
        
    def _simulate_execution_price(self, 
                                 paper_price: float,
                                 side: OrderSide, 
                                 quantity: float,
                                 market_data: Dict) -> float:
        """Simulate live execution price with slippage and market impact"""
        
        bid = market_data.get("bid", paper_price)
        ask = market_data.get("ask", paper_price)
        spread = ask - bid
        
        # Base slippage
        asset = Asset.CRYPTO  # Default, should be passed as parameter in real implementation
        slippage_factor = self.slippage_bps[asset] / 10000  # Convert bps to decimal
        
        # Market impact based on order size vs volume
        volume_24h = market_data.get("volume_24h", 1000000)
        order_value = paper_price * quantity
        market_impact_factor = min(order_value / volume_24h * 100, 0.01)  # Cap at 1%
        
        # Total adjustment
        total_adjustment = slippage_factor + market_impact_factor
        
        if side == OrderSide.BUY:
            # Buying: price goes up due to slippage and impact
            live_price = ask + (ask * total_adjustment)
        else:
            # Selling: price goes down due to slippage and impact  
            live_price = bid - (bid * total_adjustment)
            
        return live_price
        
    def _simulate_latency_impact(self, 
                               symbol: str,
                               timestamp: datetime,
                               side: OrderSide,
                               quantity: float,
                               market_data: Dict) -> Dict:
        """Simulate impact of execution latency"""
        
        asset = Asset.CRYPTO  # Should be determined from symbol
        latency = self.latency_ms[asset]
        
        # Get price movement during latency window
        future_timestamp = timestamp + timedelta(milliseconds=latency)
        future_market_data = self._get_historical_market_data(symbol, future_timestamp)
        
        if not future_market_data:
            return {"latency_ms": latency, "impact_usd": 0.0}
            
        current_price = market_data.get("mid_price", 0)
        future_price = future_market_data.get("mid_price", current_price)
        
        # Calculate impact based on price movement direction vs order side
        price_movement = future_price - current_price
        
        if side == OrderSide.BUY and price_movement > 0:
            # Price went up while trying to buy - negative impact
            impact_usd = -price_movement * quantity
        elif side == OrderSide.SELL and price_movement < 0:
            # Price went down while trying to sell - negative impact
            impact_usd = price_movement * quantity  # price_movement is negative
        else:
            # Favorable movement or no movement
            impact_usd = 0.0
            
        return {
            "latency_ms": latency,
            "impact_usd": impact_usd
        }
        
    def _calculate_rejection_probability(self,
                                       symbol: str,
                                       side: OrderSide,
                                       quantity: float,
                                       market_data: Dict) -> float:
        """Calculate probability of order rejection"""
        
        # Base rejection rate (very low for most cases)
        base_rejection = 0.01  # 1%
        
        # Increase based on market conditions
        volatility = market_data.get("volatility_1h", 0.02)  # 2% default
        volume_24h = market_data.get("volume_24h", 1000000)
        order_value = market_data.get("mid_price", 1) * quantity
        
        # Higher volatility increases rejection risk
        volatility_factor = min(volatility * 10, 0.1)  # Cap at 10%
        
        # Large orders vs volume increase rejection risk
        size_factor = min(order_value / volume_24h * 100, 0.05)  # Cap at 5%
        
        total_rejection_prob = base_rejection + volatility_factor + size_factor
        
        return min(total_rejection_prob, 0.2)  # Cap at 20%
        
    def _get_historical_market_data(self, symbol: str, timestamp: datetime) -> Optional[Dict]:
        """Get historical market data for a symbol at a specific time"""
        try:
            # This would typically query the data_manager for historical data
            # For now, return mock data based on current market state
            
            current_quote = self.data_manager.get_last_quote(symbol)
            if not current_quote:
                return None
                
            # Mock some market data (in real implementation, this would be historical)
            return {
                "bid": current_quote.get("bid", 0),
                "ask": current_quote.get("ask", 0),
                "mid_price": (current_quote.get("bid", 0) + current_quote.get("ask", 0)) / 2,
                "volume_24h": 1000000,  # Mock volume
                "volatility_1h": 0.02   # Mock volatility
            }
            
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol} at {timestamp}: {e}")
            return None
            
    def _build_diff_report(self, 
                          asset: Asset,
                          start_time: datetime,
                          end_time: datetime,
                          paper_trades_count: int,
                          simulated_trades_count: int,
                          diffs: List[TradeDiff]) -> DiffReport:
        """Build comprehensive diff report"""
        
        # Count diffs by type
        diffs_by_type = {}
        for diff_type in DiffType:
            diffs_by_type[diff_type] = sum(1 for d in diffs if d.diff_type == diff_type)
            
        # Calculate cumulative impact
        cumulative_impact = sum(d.impact_usd for d in diffs)
        
        # Summary statistics
        summary_stats = {
            "avg_diff_impact_usd": cumulative_impact / len(diffs) if diffs else 0.0,
            "max_single_impact_usd": max((d.impact_usd for d in diffs), default=0.0),
            "min_single_impact_usd": min((d.impact_usd for d in diffs), default=0.0),
            "positive_impacts": sum(1 for d in diffs if d.impact_usd > 0),
            "negative_impacts": sum(1 for d in diffs if d.impact_usd < 0),
            "avg_confidence": sum(d.confidence for d in diffs) / len(diffs) if diffs else 0.0
        }
        
        return DiffReport(
            asset=asset,
            start_time=start_time,
            end_time=end_time,
            paper_trades_count=paper_trades_count,
            simulated_trades_count=simulated_trades_count,
            total_diffs=len(diffs),
            diffs_by_type=diffs_by_type,
            cumulative_impact_usd=cumulative_impact,
            individual_diffs=diffs,
            summary_stats=summary_stats
        )
        
    def _empty_diff_report(self, asset: Asset, since_ts: datetime) -> DiffReport:
        """Create empty diff report when no trades found"""
        return DiffReport(
            asset=asset,
            start_time=since_ts,
            end_time=datetime.now(),
            paper_trades_count=0,
            simulated_trades_count=0,
            total_diffs=0,
            diffs_by_type={diff_type: 0 for diff_type in DiffType},
            cumulative_impact_usd=0.0,
            individual_diffs=[],
            summary_stats={}
        )