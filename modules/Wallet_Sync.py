import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class WalletSync:
    """
    Pulls live wallet balances from active exchange adapters and provides
    a consolidated view of equity per asset class.
    """

    def __init__(self, exchange_adapters: Dict[str, Any]):
        """
        Initializes the WalletSync.

        Args:
            exchange_adapters (Dict[str, object]): A dictionary mapping asset classes
                                                   (e.g., 'SPOT', 'PERP') to their
                                                   corresponding exchange adapter instances.
                                                   Each adapter must have a `get_balance()` method.
        """
        self.exchange_adapters = exchange_adapters
        self._last_known_balances: Dict[str, float] = {asset: 0.0 for asset in self.exchange_adapters.keys()}

        # Determine if we are in live mode by checking the simulation status of the adapters.
        # If any adapter is not in simulation, the whole system is considered live.
        self.is_live = any(not getattr(adapter, 'is_simulation', True) for adapter in self.exchange_adapters.values())

        # If not live, initialize with simulation balances from config
        if not self.is_live:
            # This part is a bit of a guess, assuming there's a config structure for this.
            # In a real scenario, this would be more robust.
            # For now, let's assume PortfolioManager will handle initial sim balances.
            pass

    def sync(self) -> Dict[str, float]:
        """
        Synchronizes balances from all configured exchange adapters.
        If an API call fails for an adapter, it returns the last known balance
        for that asset class and logs a warning.

        Returns:
            Dict[str, float]: A dictionary of asset classes and their corresponding balances.
        """
        # In simulation mode, balances are managed by PortfolioManager, so we don't sync.
        if not self.is_live:
            logger.debug("WalletSync is in simulation mode. Sync is managed by PortfolioManager.")
            return self._last_known_balances.copy()

        current_balances = {}
        for asset, adapter in self.exchange_adapters.items():
            try:
                # We assume the adapter's get_balance() returns the total equity for that asset class.
                balance = adapter.get_balance()
                current_balances[asset] = balance
                self._last_known_balances[asset] = balance
            except Exception as e:
                logger.warning(f"Failed to sync wallet for asset '{asset}': {e}. "
                               f"Using last known balance of {self._last_known_balances.get(asset, 0.0):.2f}.")
                current_balances[asset] = self._last_known_balances.get(asset, 0.0)

        logger.info(f"Live wallet sync complete. Balances: {current_balances}")
        return current_balances

    def get_equity(self, asset: str) -> float:
        """
        Returns the last known equity for a specific asset class.

        Args:
            asset (str): The asset class (e.g., 'SPOT').

        Returns:
            float: The last known balance for the asset.
        """
        return self._last_known_balances.get(asset, 0.0)

    def set_simulation_balance(self, asset: str, balance: float):
        """
        Allows setting the balance for an asset class in simulation mode.
        This would be called by the PortfolioManager.
        """
        if not self.is_live:
            self._last_known_balances[asset] = balance
