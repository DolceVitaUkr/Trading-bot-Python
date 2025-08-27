"""
Strategy Development Manager
Handles the complete workflow from paper trading to live deployment.
"""
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from .validation_manager import ValidationManager
from .loggerconfig import get_logger


class StrategyStatus(Enum):
    """Strategy development status."""
    DEVELOPING = "developing"      # Currently in paper trading development
    PENDING_VALIDATION = "pending_validation"  # Ready for validation
    IN_VALIDATION = "in_validation"  # Currently being validated
    VALIDATED = "validated"       # Passed validation, approved for live
    REJECTED = "rejected"         # Failed validation
    LIVE_TESTING = "live_testing" # Currently in live testing phase
    APPROVED = "approved"         # Approved for production trading
    DEPRECATED = "deprecated"     # No longer in use


@dataclass
class StrategyMetrics:
    """Strategy performance metrics."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    development_days: int
    created_at: str
    last_updated: str


@dataclass
class StrategyRecord:
    """Complete strategy development record."""
    strategy_id: str
    asset_type: str
    status: StrategyStatus
    metrics: StrategyMetrics
    validation_results: Optional[Dict[str, Any]] = None
    validation_reasons: Optional[List[str]] = None
    paper_trading_start: Optional[str] = None
    validation_date: Optional[str] = None
    live_deployment_date: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['status'] = self.status.value
        return data


class StrategyDevelopmentManager:
    """Manages the complete strategy development lifecycle."""
    
    def __init__(self):
        self.log = get_logger("strategy_development")
        self.validator = ValidationManager()
        
        # Storage paths
        self.strategies_dir = Path("tradingbot/state/strategies")
        self.strategies_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing strategies
        self.strategies: Dict[str, StrategyRecord] = {}
        self._load_strategies()
        
        # Development thresholds
        self.min_development_days = 7  # Minimum days in paper trading
        self.min_trades_for_validation = 50  # Minimum trades before validation
        
    def _load_strategies(self):
        """Load strategies from persistent storage."""
        try:
            strategies_file = self.strategies_dir / "strategies.json"
            if strategies_file.exists():
                with open(strategies_file, 'r') as f:
                    data = json.load(f)
                    for strategy_id, strategy_data in data.items():
                        # Convert status string back to enum
                        strategy_data['status'] = StrategyStatus(strategy_data['status'])
                        # Reconstruct dataclasses
                        metrics_data = strategy_data['metrics']
                        metrics = StrategyMetrics(**metrics_data)
                        strategy_data['metrics'] = metrics
                        # Create strategy record
                        self.strategies[strategy_id] = StrategyRecord(**strategy_data)
                        
                self.log.info(f"Loaded {len(self.strategies)} strategies from storage")
        except Exception as e:
            self.log.warning(f"Could not load strategies: {e}")
            self.strategies = {}
    
    def _save_strategies(self):
        """Save strategies to persistent storage."""
        try:
            strategies_file = self.strategies_dir / "strategies.json"
            data = {sid: strategy.to_dict() for sid, strategy in self.strategies.items()}
            with open(strategies_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            self.log.error(f"Could not save strategies: {e}")
    
    def register_paper_strategy(
        self, 
        asset_type: str, 
        paper_trader_data: Dict[str, Any]
    ) -> str:
        """Register a new strategy from paper trading data."""
        # Generate strategy ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        strategy_id = f"{asset_type}_strategy_{timestamp}"
        
        # Extract metrics from paper trader
        stats = paper_trader_data.get('trading_stats', {})
        
        metrics = StrategyMetrics(
            total_trades=stats.get('total_trades', 0),
            winning_trades=stats.get('winning_trades', 0),
            losing_trades=stats.get('losing_trades', 0),
            win_rate=stats.get('win_rate', 0.0),
            total_pnl=stats.get('total_pnl', 0.0),
            max_drawdown=stats.get('max_drawdown', 0.0),
            sharpe_ratio=stats.get('sharpe_ratio', 0.0),
            profit_factor=stats.get('profit_factor', 0.0),
            avg_win=stats.get('avg_win', 0.0),
            avg_loss=stats.get('avg_loss', 0.0),
            development_days=self._calculate_development_days(paper_trader_data),
            created_at=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat()
        )
        
        strategy = StrategyRecord(
            strategy_id=strategy_id,
            asset_type=asset_type,
            status=StrategyStatus.DEVELOPING,
            metrics=metrics,
            paper_trading_start=datetime.now().isoformat()
        )
        
        self.strategies[strategy_id] = strategy
        self._save_strategies()
        
        self.log.info(f"Registered new strategy: {strategy_id}")
        return strategy_id
    
    def _calculate_development_days(self, paper_trader_data: Dict[str, Any]) -> int:
        """Calculate how many days strategy has been in development."""
        try:
            pnl_history = paper_trader_data.get('pnl_history', [])
            if len(pnl_history) < 2:
                return 0
            
            first_trade = datetime.fromisoformat(pnl_history[0]['timestamp'])
            last_trade = datetime.fromisoformat(pnl_history[-1]['timestamp'])
            return (last_trade - first_trade).days
        except:
            return 0
    
    def update_strategy_metrics(
        self, 
        strategy_id: str, 
        paper_trader_data: Dict[str, Any]
    ):
        """Update strategy metrics from current paper trading data."""
        if strategy_id not in self.strategies:
            return
        
        strategy = self.strategies[strategy_id]
        stats = paper_trader_data.get('trading_stats', {})
        
        # Update metrics
        strategy.metrics.total_trades = stats.get('total_trades', 0)
        strategy.metrics.winning_trades = stats.get('winning_trades', 0)
        strategy.metrics.losing_trades = stats.get('losing_trades', 0)
        strategy.metrics.win_rate = stats.get('win_rate', 0.0)
        strategy.metrics.total_pnl = stats.get('total_pnl', 0.0)
        strategy.metrics.max_drawdown = stats.get('max_drawdown', 0.0)
        strategy.metrics.profit_factor = stats.get('profit_factor', 0.0)
        strategy.metrics.avg_win = stats.get('avg_win', 0.0)
        strategy.metrics.avg_loss = stats.get('avg_loss', 0.0)
        strategy.metrics.development_days = self._calculate_development_days(paper_trader_data)
        strategy.metrics.last_updated = datetime.now().isoformat()
        
        # Check if ready for validation
        if (strategy.status == StrategyStatus.DEVELOPING and 
            self._is_ready_for_validation(strategy)):
            strategy.status = StrategyStatus.PENDING_VALIDATION
            self.log.info(f"Strategy {strategy_id} is ready for validation")
        
        self._save_strategies()
    
    def _is_ready_for_validation(self, strategy: StrategyRecord) -> bool:
        """Check if strategy meets criteria for validation."""
        metrics = strategy.metrics
        
        # Must have minimum trades
        if metrics.total_trades < self.min_trades_for_validation:
            return False
        
        # Must have been in development for minimum days
        if metrics.development_days < self.min_development_days:
            return False
        
        # Must have some winning trades
        if metrics.winning_trades == 0:
            return False
        
        # Must be profitable overall
        if metrics.total_pnl <= 0:
            return False
        
        return True
    
    async def validate_strategy(self, strategy_id: str) -> Tuple[bool, List[str]]:
        """Run validation on a strategy using the ValidationManager."""
        if strategy_id not in self.strategies:
            return False, ["Strategy not found"]
        
        strategy = self.strategies[strategy_id]
        strategy.status = StrategyStatus.IN_VALIDATION
        self._save_strategies()
        
        try:
            # Use actual strategy metrics from paper trading instead of recomputing
            metrics = {
                "sharpe": strategy.metrics.sharpe_ratio,
                "max_dd": abs(strategy.metrics.max_drawdown),
                "profit_factor": strategy.metrics.profit_factor,
                "win_rate": strategy.metrics.win_rate / 100.0 if strategy.metrics.win_rate > 1 else strategy.metrics.win_rate,
                "total_pnl": strategy.metrics.total_pnl
            }
            
            # Apply gating rules using real metrics
            passed, reasons = self.validator.gating(
                metrics, 
                n_trades=strategy.metrics.total_trades
            )
            
            # Update strategy with validation results
            strategy.validation_results = metrics
            strategy.validation_reasons = reasons
            strategy.validation_date = datetime.now().isoformat()
            
            if passed:
                strategy.status = StrategyStatus.VALIDATED
                self.log.info(f"Strategy {strategy_id} passed validation")
            else:
                strategy.status = StrategyStatus.REJECTED
                self.log.warning(f"Strategy {strategy_id} failed validation: {reasons}")
            
            self._save_strategies()
            return passed, reasons
            
        except Exception as e:
            strategy.status = StrategyStatus.DEVELOPING
            self._save_strategies()
            self.log.error(f"Validation failed for {strategy_id}: {e}")
            return False, [f"Validation error: {e}"]
    
    def approve_for_live_testing(self, strategy_id: str) -> bool:
        """Approve a validated strategy for live testing."""
        if strategy_id not in self.strategies:
            return False
        
        strategy = self.strategies[strategy_id]
        if strategy.status != StrategyStatus.VALIDATED:
            return False
        
        strategy.status = StrategyStatus.LIVE_TESTING
        strategy.live_deployment_date = datetime.now().isoformat()
        self._save_strategies()
        
        self.log.info(f"Strategy {strategy_id} approved for live testing")
        return True
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get summary of all strategies by status."""
        summary = {
            "total_strategies": len(self.strategies),
            "by_status": {},
            "by_asset": {},
            "validation_stats": {
                "pending_validation": 0,
                "approved_for_live": 0,
                "rejected": 0
            }
        }
        
        # Count by status
        for status in StrategyStatus:
            summary["by_status"][status.value] = 0
        
        # Count by asset
        asset_types = ["crypto", "futures", "forex", "forex_options"]
        for asset in asset_types:
            summary["by_asset"][asset] = {
                "developing": 0,
                "validated": 0,
                "live": 0,
                "total": 0
            }
        
        # Process all strategies
        for strategy in self.strategies.values():
            # Count by status
            summary["by_status"][strategy.status.value] += 1
            
            # Count by asset
            if strategy.asset_type in summary["by_asset"]:
                asset_summary = summary["by_asset"][strategy.asset_type]
                asset_summary["total"] += 1
                
                if strategy.status in [StrategyStatus.DEVELOPING, StrategyStatus.PENDING_VALIDATION]:
                    asset_summary["developing"] += 1
                elif strategy.status == StrategyStatus.VALIDATED:
                    asset_summary["validated"] += 1
                elif strategy.status in [StrategyStatus.LIVE_TESTING, StrategyStatus.APPROVED]:
                    asset_summary["live"] += 1
            
            # Validation stats
            if strategy.status == StrategyStatus.PENDING_VALIDATION:
                summary["validation_stats"]["pending_validation"] += 1
            elif strategy.status in [StrategyStatus.VALIDATED, StrategyStatus.LIVE_TESTING, StrategyStatus.APPROVED]:
                summary["validation_stats"]["approved_for_live"] += 1
            elif strategy.status == StrategyStatus.REJECTED:
                summary["validation_stats"]["rejected"] += 1
        
        return summary
    
    def get_strategies_by_asset(self, asset_type: str) -> Dict[str, Any]:
        """Get detailed strategy information for a specific asset."""
        asset_strategies = {
            strategy_id: strategy 
            for strategy_id, strategy in self.strategies.items() 
            if strategy.asset_type == asset_type
        }
        
        return {
            "asset_type": asset_type,
            "strategies": {sid: s.to_dict() for sid, s in asset_strategies.items()},
            "summary": {
                "total": len(asset_strategies),
                "developing": sum(1 for s in asset_strategies.values() if s.status == StrategyStatus.DEVELOPING),
                "pending_validation": sum(1 for s in asset_strategies.values() if s.status == StrategyStatus.PENDING_VALIDATION),
                "validated": sum(1 for s in asset_strategies.values() if s.status == StrategyStatus.VALIDATED),
                "live": sum(1 for s in asset_strategies.values() if s.status in [StrategyStatus.LIVE_TESTING, StrategyStatus.APPROVED]),
                "rejected": sum(1 for s in asset_strategies.values() if s.status == StrategyStatus.REJECTED)
            }
        }
    
    def register_strategy(self, strategy_data: Dict[str, Any]) -> str:
        """Register a strategy with the strategy development manager."""
        strategy_id = strategy_data.get("id", f"strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        asset_type = strategy_data.get("asset_type", "crypto")
        
        # Create metrics from strategy data
        performance = strategy_data.get("performance", {})
        metrics = StrategyMetrics(
            total_trades=performance.get("total_trades", 0),
            winning_trades=performance.get("winning_trades", 0),
            losing_trades=performance.get("losing_trades", 0),
            win_rate=performance.get("win_rate", 0.0),
            total_pnl=performance.get("total_pnl", 0.0),
            max_drawdown=performance.get("max_drawdown", 0.0),
            sharpe_ratio=performance.get("sharpe_ratio", 0.0),
            profit_factor=performance.get("profit_factor", 0.0),
            avg_win=performance.get("avg_win", 0.0),
            avg_loss=performance.get("avg_loss", 0.0),
            development_days=0,
            created_at=datetime.now().isoformat(),
            last_updated=strategy_data.get("last_updated", datetime.now().isoformat())
        )
        
        # Create strategy record
        strategy = StrategyRecord(
            strategy_id=strategy_id,
            asset_type=asset_type,
            status=StrategyStatus(strategy_data.get("status", "developing")),
            metrics=metrics,
            paper_trading_start=datetime.now().isoformat()
        )
        
        self.strategies[strategy_id] = strategy
        self._save_strategies()
        
        self.log.info(f"Registered strategy: {strategy_id}")
        return strategy_id

    async def run_validation_check(self):
        """Check for strategies ready for validation and process them."""
        pending_strategies = [
            (sid, strategy) for sid, strategy in self.strategies.items()
            if strategy.status == StrategyStatus.PENDING_VALIDATION
        ]
        
        for strategy_id, strategy in pending_strategies:
            self.log.info(f"Running validation for strategy {strategy_id}")
            await self.validate_strategy(strategy_id)