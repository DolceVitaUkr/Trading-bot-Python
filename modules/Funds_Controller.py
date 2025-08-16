import logging
from datetime import datetime, timezone
from typing import Dict, Any

logger = logging.getLogger(__name__)

class FundsController:
    """
    Manages runtime permissions and capital allocation caps for the trading bot.
    This module acts as a safety layer to enable/disable trading activities
    globally or on a per-asset-class basis.
    """

    def __init__(self, initial_state: Dict[str, Any] = None):
        """
        Initializes the FundsController.
        State can be loaded from a saved configuration or defaults.

        Args:
            initial_state (Dict[str, Any], optional): A dictionary to restore state from.
                                                     Defaults to None, which uses default values.
        """
        if initial_state is None:
            initial_state = {}

        # --- State ---
        self.allow_bot_funds: bool = initial_state.get('allow_bot_funds', False)
        self.asset_enabled: Dict[str, bool] = initial_state.get('asset_enabled', {
            "SPOT": True,
            "PERP": False,
            "OPTIONS": False,
            "FOREX": False,
        })
        self.max_pair_allocation_pct: float = initial_state.get('max_pair_allocation_pct', 0.10)

        logger.info(f"FundsController initialized. Bot funds allowed: {self.allow_bot_funds}. "
                    f"Asset classes enabled: {self.asset_enabled}. "
                    f"Max pair allocation: {self.max_pair_allocation_pct:.2%}")

    def _log_state_change(self, parameter: str, old_value: Any, new_value: Any, source: str = "UI"):
        """Logs any change in the controller's state."""
        timestamp = datetime.now(timezone.utc).isoformat()
        logger.info(f"STATE_CHANGE: Parameter '{parameter}' changed from '{old_value}' to '{new_value}'. "
                    f"Source: {source}, Timestamp: {timestamp}")

    # --- Public API ---

    def is_allowed(self, asset_class: str, symbol: str) -> bool:
        """
        Checks if trading is permitted for a given asset class and symbol.
        This is the primary guard function for the entire trading pipeline.

        Args:
            asset_class (str): The asset class (e.g., 'SPOT', 'PERP').
            symbol (str): The trading symbol (e.g., 'BTC/USDT').

        Returns:
            bool: True if trading is allowed, False otherwise.
        """
        if not self.allow_bot_funds:
            return False

        if not self.asset_enabled.get(asset_class, False):
            return False

        # Future placeholder for symbol-specific blacklists/whitelists
        # For now, if the asset class is enabled, all its symbols are.

        return True

    def pair_cap_pct(self) -> float:
        """
        Returns the maximum percentage of a sub-ledger's equity that can be
        allocated to a single trading pair.

        Returns:
            float: The maximum allocation percentage (e.g., 0.10 for 10%).
        """
        return self.max_pair_allocation_pct

    def snapshot(self) -> Dict[str, Any]:
        """
        Returns a dictionary representing the current state of the controller.
        Useful for UI updates or periodic state saving.

        Returns:
            Dict[str, Any]: The current state.
        """
        return {
            "allow_bot_funds": self.allow_bot_funds,
            "asset_enabled": self.asset_enabled.copy(),
            "max_pair_allocation_pct": self.max_pair_allocation_pct,
            "timestamp_utc": datetime.now(timezone.utc).isoformat()
        }

    # --- State Modification Methods (UI Hooks) ---

    def set_allow_bot_funds(self, allow: bool, source: str = "UI"):
        """
        Globally enables or disables the bot's ability to use funds.
        This is the master switch.
        """
        if self.allow_bot_funds != allow:
            old_value = self.allow_bot_funds
            self.allow_bot_funds = allow
            self._log_state_change("allow_bot_funds", old_value, allow, source)

    def set_asset_enabled(self, asset_class: str, is_enabled: bool, source: str = "UI"):
        """
        Enables or disables a specific asset class for trading.
        """
        old_value = self.asset_enabled.get(asset_class)
        if old_value != is_enabled:
            self.asset_enabled[asset_class] = is_enabled
            self._log_state_change(f"asset_enabled.{asset_class}", old_value, is_enabled, source)

    def set_max_pair_allocation_pct(self, new_cap: float, source: str = "UI"):
        """
        Sets the maximum percentage of a sub-ledger that can be allocated
        to one pair.
        """
        if not (0.01 <= new_cap <= 0.5):
            logger.warning(f"Invalid max_pair_allocation_pct '{new_cap}'. Must be between 0.01 and 0.5.")
            return

        if self.max_pair_allocation_pct != new_cap:
            old_value = self.max_pair_allocation_pct
            self.max_pair_allocation_pct = new_cap
            self._log_state_change("max_pair_allocation_pct", old_value, new_cap, source)
