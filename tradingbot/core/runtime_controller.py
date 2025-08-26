"""Runtime controller managing live toggles and kill switch state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Any

from .configmanager import ConfigManager, config_manager
from .notifier import Notifier


class RuntimeController:
    """Persist and manipulate runtime trading state."""

    def __init__(
        self,
        state_path: Path | None = None,
        validator: Callable[[str], bool] | None = None,
        notifier: Notifier | None = None,
    ) -> None:
        self.validator = validator or (lambda asset: True)
        self.notifier = notifier or Notifier()
        self.config: ConfigManager = config_manager
        self.state_path = state_path or (
            Path(__file__).resolve().parent.parent / "state" / "runtime.json"
        )
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        if self.state_path.exists():
            with open(self.state_path, "r", encoding="utf-8") as fh:
                self.state: Dict[str, Any] = json.load(fh)
        else:
            self.state = {"assets": {}, "global": {"kill_switch": False}, "trading": {}}
            self._save()

    # ------------------------------------------------------------------
    def _save(self) -> None:
        with open(self.state_path, "w", encoding="utf-8") as fh:
            json.dump(self.state, fh, indent=2)

    # ------------------------------------------------------------------
    def enable_live(self, asset: str) -> None:
        if not self.validator(asset):
            raise ValueError("Validation gate failed")
        asset_state = self.state["assets"].setdefault(
            asset, {"live": False, "close_only": False, "consecutive_losses": 0}
        )
        asset_state["live"] = True
        asset_state["close_only"] = False
        self._save()
        self.notifier.send(f"Live trading enabled for {asset}")

    def disable_live(self, asset: str, close_only: bool = False) -> None:
        asset_state = self.state["assets"].setdefault(
            asset, {"live": False, "close_only": False, "consecutive_losses": 0}
        )
        asset_state["live"] = False
        asset_state["close_only"] = bool(close_only)
        self._save()
        self.notifier.send(f"Live trading disabled for {asset}")

    def set_global_kill(self, active: bool) -> None:
        self.state["global"]["kill_switch"] = bool(active)
        if active:
            for state in self.state["assets"].values():
                state["live"] = False
                state["close_only"] = True
        self._save()
        self.notifier.send(f"Global kill switch {'ON' if active else 'OFF'}")

    def record_trade_result(self, asset: str, is_live: bool, pnl_after_fees: float) -> None:
        if not is_live:
            return
        asset_state = self.state["assets"].setdefault(
            asset, {"live": False, "close_only": False, "consecutive_losses": 0}
        )
        if pnl_after_fees < 0:
            asset_state["consecutive_losses"] = asset_state.get("consecutive_losses", 0) + 1
            limit = self.config.get("safety.CONSECUTIVE_LOSS_KILL", 0)
            if asset_state["consecutive_losses"] >= limit > 0:
                self.set_global_kill(True)
        else:
            asset_state["consecutive_losses"] = 0
        self._save()

    def start_asset_trading(self, asset: str, mode: str) -> None:
        """Start trading for specific asset and mode."""
        trading_state = self.state.setdefault("trading", {})
        trading_state[asset] = {"status": "running", "mode": mode}
        self._save()
        self.notifier.send(f"Started {asset} {mode} trading")
        
    def stop_asset_trading(self, asset: str, mode: str) -> None:
        """Stop trading for specific asset and mode."""
        trading_state = self.state.setdefault("trading", {})
        if asset in trading_state:
            trading_state[asset]["status"] = "stopped"
        self._save()
        self.notifier.send(f"Stopped {asset} {mode} trading")
        
    def kill_asset_trading(self, asset: str) -> None:
        """Kill switch for specific asset."""
        trading_state = self.state.setdefault("trading", {})
        trading_state[asset] = {"status": "killed"}
        self._save()
        self.notifier.send(f"Kill switch activated for {asset}")
        
    def emergency_stop_all(self) -> None:
        """Emergency stop all trading across all assets."""
        self.set_global_kill(True)
        trading_state = self.state.setdefault("trading", {})
        for asset in trading_state:
            trading_state[asset] = {"status": "killed"}
        self._save()
        self.notifier.send("EMERGENCY STOP: All trading halted")
        
    def get_portfolio_stats(self) -> Dict[str, Any]:
        """Get portfolio statistics (mock implementation)."""
        # In real implementation, this would query actual portfolio data
        return {
            "total_pnl": 0.0,
            "active_trades": 0, 
            "win_rate": 0.0,
            "balance": self.config.get("safety", {}).get("PAPER_EQUITY_START", 10000.0)
        }
    
    def get_state(self) -> Dict[str, Any]:
        return self.state.copy()


__all__ = ["RuntimeController"]
