# file: tradingbot/core/session_manager.py
# module_version: v1.00

"""
Paper Session Control Module

Owns: paper start ($1,000), reward reset (0), resume open paper trades after restarts.
Public API:
- start_session()
- resume_session()
- get_session_state()

Calls: bankroll_manager, pnl_reconciler.
Never does: scheduling, order placement.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional
import json
import os
from pathlib import Path

from .interfaces import SessionState, Asset
from .bankroll_manager import BankrollManager
from .pnl_reconciler import PnlReconciler

logger = logging.getLogger(__name__)

class SessionManager:
    """Manages paper trading session lifecycle and state"""
    
    def __init__(self, 
                 bankroll_manager: BankrollManager,
                 pnl_reconciler: PnlReconciler,
                 state_dir: str = "tradingbot/state"):
        self.bankroll_manager = bankroll_manager
        self.pnl_reconciler = pnl_reconciler
        self.state_dir = Path(state_dir)
        self.session_file = self.state_dir / "session_state.json"
        
        # Ensure state directory exists
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        self._session_state: Optional[SessionState] = None
        
    def start_session(self, assets: list[Asset] = None) -> SessionState:
        """
        Start a new paper trading session
        
        Args:
            assets: List of assets to initialize sessions for
            
        Returns:
            SessionState: New session state
        """
        logger.info("Starting new paper trading session")
        
        if assets is None:
            assets = [Asset.CRYPTO, Asset.FOREX, Asset.FUTURES, Asset.OPTIONS]
            
        session_state = SessionState(
            session_id=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            start_time=datetime.now(),
            assets=assets,
            initial_equity_per_asset=1000.0,  # $1,000 per asset
            current_equity={asset: 1000.0 for asset in assets},
            reward={asset: 0.0 for asset in assets},
            open_positions={asset: [] for asset in assets},
            session_metadata={
                "version": "v1.00",
                "created_by": "session_manager",
                "paper_mode": True
            }
        )
        
        # Initialize bankroll for each asset
        for asset in assets:
            self.bankroll_manager.initialize_asset_bankroll(asset, 1000.0)
            
        # Reset reconciler state
        self.pnl_reconciler.reset_session()
        
        # Save session state
        self._session_state = session_state
        self._save_session_state()
        
        logger.info(f"Started session {session_state.session_id} for assets: {assets}")
        return session_state
        
    def resume_session(self) -> Optional[SessionState]:
        """
        Resume existing paper trading session after restart
        
        Returns:
            SessionState: Resumed session state or None if no session to resume
        """
        logger.info("Attempting to resume paper trading session")
        
        if not self.session_file.exists():
            logger.warning("No session file found to resume")
            return None
            
        try:
            session_state = self._load_session_state()
            if session_state is None:
                logger.error("Failed to load session state")
                return None
                
            # Restore bankroll state
            for asset in session_state.assets:
                current_equity = session_state.current_equity.get(asset, 1000.0)
                self.bankroll_manager.restore_asset_bankroll(asset, current_equity)
                
            # Resume reconciler state 
            self.pnl_reconciler.resume_session(session_state.session_id)
            
            # Restore open positions (this would typically involve
            # syncing with broker state for live positions)
            for asset, positions in session_state.open_positions.items():
                if positions:
                    logger.info(f"Resuming {len(positions)} positions for {asset}")
                    
            self._session_state = session_state
            logger.info(f"Resumed session {session_state.session_id}")
            return session_state
            
        except Exception as e:
            logger.error(f"Failed to resume session: {e}")
            return None
            
    def get_session_state(self) -> Optional[SessionState]:
        """Get current session state"""
        return self._session_state
        
    def update_session_equity(self, asset: Asset, new_equity: float, reward: float = None):
        """
        Update session equity for an asset
        
        Args:
            asset: Asset to update
            new_equity: New equity value
            reward: Optional reward value to set
        """
        if self._session_state is None:
            logger.warning("No active session to update")
            return
            
        self._session_state.current_equity[asset] = new_equity
        
        if reward is not None:
            self._session_state.reward[asset] = reward
            
        self._save_session_state()
        
    def end_session(self) -> Dict[str, Any]:
        """
        End current session and return summary
        
        Returns:
            Dict with session summary
        """
        if self._session_state is None:
            logger.warning("No active session to end")
            return {}
            
        session_id = self._session_state.session_id
        duration = datetime.now() - self._session_state.start_time
        
        summary = {
            "session_id": session_id,
            "duration_hours": duration.total_seconds() / 3600,
            "final_equity": dict(self._session_state.current_equity),
            "final_reward": dict(self._session_state.reward),
            "total_pnl": {
                asset: equity - self._session_state.initial_equity_per_asset
                for asset, equity in self._session_state.current_equity.items()
            }
        }
        
        # Archive session state
        archive_path = self.state_dir / f"session_archive_{session_id}.json"
        self._save_session_state(archive_path)
        
        # Clear current session
        self._session_state = None
        if self.session_file.exists():
            self.session_file.unlink()
            
        logger.info(f"Ended session {session_id}, archived to {archive_path}")
        return summary
        
    def _save_session_state(self, path: Path = None):
        """Save current session state to file"""
        if self._session_state is None:
            return
            
        save_path = path or self.session_file
        
        try:
            session_data = {
                "session_id": self._session_state.session_id,
                "start_time": self._session_state.start_time.isoformat(),
                "assets": [asset.value for asset in self._session_state.assets],
                "initial_equity_per_asset": self._session_state.initial_equity_per_asset,
                "current_equity": {asset.value: equity for asset, equity in self._session_state.current_equity.items()},
                "reward": {asset.value: reward for asset, reward in self._session_state.reward.items()},
                "open_positions": {asset.value: positions for asset, positions in self._session_state.open_positions.items()},
                "session_metadata": self._session_state.session_metadata
            }
            
            with open(save_path, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save session state: {e}")
            
    def _load_session_state(self) -> Optional[SessionState]:
        """Load session state from file"""
        try:
            with open(self.session_file, 'r') as f:
                data = json.load(f)
                
            return SessionState(
                session_id=data["session_id"],
                start_time=datetime.fromisoformat(data["start_time"]),
                assets=[Asset(asset) for asset in data["assets"]],
                initial_equity_per_asset=data["initial_equity_per_asset"],
                current_equity={Asset(asset): equity for asset, equity in data["current_equity"].items()},
                reward={Asset(asset): reward for asset, reward in data["reward"].items()},
                open_positions={Asset(asset): positions for asset, positions in data["open_positions"].items()},
                session_metadata=data.get("session_metadata", {})
            )
            
        except Exception as e:
            logger.error(f"Failed to load session state: {e}")
            return None