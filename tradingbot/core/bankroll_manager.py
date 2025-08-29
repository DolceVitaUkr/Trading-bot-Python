# file: tradingbot/core/bankroll_manager.py
# module_version: v1.00

"""
Auto Top-Ups Module

Owns: paper equity top-ups (+$1000 when ≤$10), epoch logging.
Public API:
- ensure_min_equity()
- record_epoch()
- initialize_asset_bankroll()
- restore_asset_bankroll()

Calls: session_manager.
Never does: trading.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
import json
from pathlib import Path
from dataclasses import dataclass

from .interfaces import Asset

logger = logging.getLogger(__name__)

@dataclass
class EpochRecord:
    """Record of a bankroll epoch (top-up event)"""
    epoch_id: str
    timestamp: datetime
    asset: Asset
    trigger_equity: float
    topup_amount: float
    new_equity: float
    reason: str
    session_id: Optional[str] = None

class BankrollManager:
    """Manages paper trading bankroll and auto top-ups"""
    
    def __init__(self, state_dir: str = "tradingbot/state"):
        self.state_dir = Path(state_dir)
        self.epochs_file = self.state_dir / "bankroll_epochs.jsonl"
        self.bankroll_file = self.state_dir / "bankroll_state.json"
        
        # Ensure state directory exists
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.MIN_EQUITY_THRESHOLD = 10.0  # Top-up when equity <= $10
        self.TOPUP_AMOUNT = 1000.0       # Add $1,000 per top-up
        self.INITIAL_EQUITY = 1000.0     # Initial equity per asset
        
        # Current bankroll state
        self._bankroll_state: Dict[Asset, float] = {}
        self._epoch_counter = 0
        
        # Load existing state
        self._load_bankroll_state()
        
    def initialize_asset_bankroll(self, asset: Asset, initial_equity: float = None) -> float:
        """
        Initialize bankroll for an asset
        
        Args:
            asset: Asset to initialize
            initial_equity: Starting equity (default: $1,000)
            
        Returns:
            float: Initial equity amount
        """
        if initial_equity is None:
            initial_equity = self.INITIAL_EQUITY
            
        self._bankroll_state[asset] = initial_equity
        self._save_bankroll_state()
        
        logger.info(f"Initialized {asset.value} bankroll with ${initial_equity}")
        return initial_equity
        
    def restore_asset_bankroll(self, asset: Asset, equity: float):
        """
        Restore bankroll state for an asset (used during session resume)
        
        Args:
            asset: Asset to restore
            equity: Current equity amount
        """
        self._bankroll_state[asset] = equity
        self._save_bankroll_state()
        
        logger.info(f"Restored {asset.value} bankroll to ${equity}")
        
    def ensure_min_equity(self, asset: Asset, current_equity: float, session_id: str = None) -> float:
        """
        Ensure minimum equity for an asset, applying top-up if needed
        
        Args:
            asset: Asset to check
            current_equity: Current equity amount
            session_id: Optional session identifier
            
        Returns:
            float: Equity amount after any top-up
        """
        if current_equity > self.MIN_EQUITY_THRESHOLD:
            # Update internal state but no top-up needed
            self._bankroll_state[asset] = current_equity
            return current_equity
            
        # Need to top-up
        new_equity = current_equity + self.TOPUP_AMOUNT
        
        # Record epoch
        epoch_record = EpochRecord(
            epoch_id=f"epoch_{asset.value}_{self._epoch_counter:04d}",
            timestamp=datetime.now(),
            asset=asset,
            trigger_equity=current_equity,
            topup_amount=self.TOPUP_AMOUNT,
            new_equity=new_equity,
            reason=f"Auto top-up triggered (equity ${current_equity:.2f} <= ${self.MIN_EQUITY_THRESHOLD})",
            session_id=session_id
        )
        
        self._record_epoch(epoch_record)
        self._epoch_counter += 1
        
        # Update bankroll state
        self._bankroll_state[asset] = new_equity
        self._save_bankroll_state()
        
        logger.info(f"Applied ${self.TOPUP_AMOUNT} top-up to {asset.value}: "
                   f"${current_equity:.2f} -> ${new_equity:.2f}")
        
        return new_equity
        
    def get_current_equity(self, asset: Asset) -> float:
        """Get current equity for an asset"""
        return self._bankroll_state.get(asset, 0.0)
        
    def get_all_equity(self) -> Dict[Asset, float]:
        """Get current equity for all assets"""
        return dict(self._bankroll_state)
        
    def record_epoch(self, asset: Asset, reason: str, equity_change: float = 0.0, session_id: str = None):
        """
        Record a manual epoch event
        
        Args:
            asset: Asset affected
            reason: Reason for epoch
            equity_change: Change in equity
            session_id: Optional session identifier
        """
        current_equity = self._bankroll_state.get(asset, 0.0)
        new_equity = current_equity + equity_change
        
        epoch_record = EpochRecord(
            epoch_id=f"epoch_{asset.value}_{self._epoch_counter:04d}",
            timestamp=datetime.now(),
            asset=asset,
            trigger_equity=current_equity,
            topup_amount=equity_change,
            new_equity=new_equity,
            reason=reason,
            session_id=session_id
        )
        
        self._record_epoch(epoch_record)
        self._epoch_counter += 1
        
        # Update bankroll state
        self._bankroll_state[asset] = new_equity
        self._save_bankroll_state()
        
        logger.info(f"Recorded epoch for {asset.value}: {reason} "
                   f"(${current_equity:.2f} -> ${new_equity:.2f})")
        
    def get_epoch_history(self, asset: Asset = None, limit: int = 100) -> List[EpochRecord]:
        """
        Get epoch history
        
        Args:
            asset: Optional asset filter
            limit: Maximum number of records to return
            
        Returns:
            List of EpochRecord objects
        """
        if not self.epochs_file.exists():
            return []
            
        epochs = []
        try:
            with open(self.epochs_file, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        epoch_asset = Asset(data["asset"])
                        
                        if asset is None or epoch_asset == asset:
                            epoch = EpochRecord(
                                epoch_id=data["epoch_id"],
                                timestamp=datetime.fromisoformat(data["timestamp"]),
                                asset=epoch_asset,
                                trigger_equity=data["trigger_equity"],
                                topup_amount=data["topup_amount"],
                                new_equity=data["new_equity"],
                                reason=data["reason"],
                                session_id=data.get("session_id")
                            )
                            epochs.append(epoch)
                            
                            if len(epochs) >= limit:
                                break
                                
        except Exception as e:
            logger.error(f"Failed to load epoch history: {e}")
            
        return epochs
        
    def get_topup_stats(self, asset: Asset = None) -> Dict[str, Any]:
        """
        Get top-up statistics
        
        Args:
            asset: Optional asset filter
            
        Returns:
            Dict with statistics
        """
        epochs = self.get_epoch_history(asset)
        auto_topups = [e for e in epochs if "Auto top-up" in e.reason]
        
        stats = {
            "total_epochs": len(epochs),
            "auto_topups": len(auto_topups),
            "total_topup_amount": sum(e.topup_amount for e in auto_topups if e.topup_amount > 0),
            "avg_topup_amount": 0.0,
            "last_topup": None
        }
        
        if auto_topups:
            stats["avg_topup_amount"] = stats["total_topup_amount"] / len(auto_topups)
            stats["last_topup"] = auto_topups[0].timestamp.isoformat()
            
        return stats
        
    def _record_epoch(self, epoch: EpochRecord):
        """Record an epoch event to persistent storage"""
        try:
            epoch_data = {
                "epoch_id": epoch.epoch_id,
                "timestamp": epoch.timestamp.isoformat(),
                "asset": epoch.asset.value,
                "trigger_equity": epoch.trigger_equity,
                "topup_amount": epoch.topup_amount,
                "new_equity": epoch.new_equity,
                "reason": epoch.reason,
                "session_id": epoch.session_id
            }
            
            with open(self.epochs_file, 'a') as f:
                f.write(json.dumps(epoch_data) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to record epoch: {e}")
            
    def _save_bankroll_state(self):
        """Save current bankroll state to persistent storage"""
        try:
            state_data = {
                "timestamp": datetime.now().isoformat(),
                "epoch_counter": self._epoch_counter,
                "bankroll": {asset.value: equity for asset, equity in self._bankroll_state.items()}
            }
            
            with open(self.bankroll_file, 'w') as f:
                json.dump(state_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save bankroll state: {e}")
            
    def _load_bankroll_state(self):
        """Load bankroll state from persistent storage"""
        if not self.bankroll_file.exists():
            return
            
        try:
            with open(self.bankroll_file, 'r') as f:
                data = json.load(f)
                
            self._epoch_counter = data.get("epoch_counter", 0)
            bankroll_data = data.get("bankroll", {})
            
            for asset_str, equity in bankroll_data.items():
                try:
                    asset = Asset(asset_str)
                    self._bankroll_state[asset] = equity
                except ValueError:
                    logger.warning(f"Unknown asset in bankroll state: {asset_str}")
                    
            logger.info(f"Loaded bankroll state: {len(self._bankroll_state)} assets")
            
        except Exception as e:
            logger.error(f"Failed to load bankroll state: {e}")