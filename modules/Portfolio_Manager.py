import logging
import uuid
from typing import Dict, Optional, Union, Any

logger = logging.getLogger(__name__)

class PortfolioManager:
    """
    Manages capital allocation across different asset classes (sub-ledgers),
    handles trade reservations, and books P&L.
    """

    def __init__(self,
                 allocations: Dict[str, Union[float, int]],
                 wallet_sync: Optional[Any] = None, # Use Any to avoid circular dependency issues
                 is_percentage: bool = False,
                 total_capital: Optional[float] = None):
        """
        Initializes the PortfolioManager.

        Args:
            allocations (Dict[str, Union[float, int]]):
                Allocation per asset class.
                If is_percentage is True, values are percentages (e.g., 0.6 for 60%).
                If is_percentage is False, values are fixed USD amounts.
            wallet_sync (Optional[WalletSync], optional):
                The wallet sync instance for live equity capping. Defaults to None.
            is_percentage (bool, optional):
                Whether the allocations are percentages or fixed amounts. Defaults to False.
            total_capital (Optional[float], optional):
                The total capital to use for percentage-based allocations.
                Required if is_percentage is True.
        """
        self.wallet_sync = wallet_sync
        self.ledgers: Dict[str, Dict[str, float]] = {}
        self.reservations: Dict[str, Dict[str, Any]] = {} # {reservation_id: {"asset": str, "amount": float}}

        if is_percentage:
            if total_capital is None or total_capital <= 0:
                raise ValueError("Total capital must be provided for percentage-based allocations.")
            for asset, pct in allocations.items():
                amount = total_capital * float(pct)
                self.ledgers[asset] = {
                    "total": amount,
                    "available": amount,
                    "realized_pnl": 0.0,
                    "fees": 0.0,
                }
        else:
            for asset, amount in allocations.items():
                self.ledgers[asset] = {
                    "total": float(amount),
                    "available": float(amount),
                    "realized_pnl": 0.0,
                    "fees": 0.0,
                }

        # In simulation mode, inform WalletSync about initial balances.
        if self.wallet_sync and not getattr(self.wallet_sync, 'is_live', True):
            for asset, ledger in self.ledgers.items():
                self.wallet_sync.set_simulation_balance(asset, ledger["total"])

    def available_budget(self, asset: str) -> float:
        """
        Returns the available budget for a given asset class.
        In live mode, this is capped by the actual wallet balance from WalletSync.

        Args:
            asset (str): The asset class (e.g., 'SPOT').

        Returns:
            float: The available trading budget in USD.
        """
        ledger = self.ledgers.get(asset)
        if not ledger:
            logger.warning(f"No ledger found for asset class: {asset}")
            return 0.0

        available_in_ledger = ledger["available"]

        if self.wallet_sync and getattr(self.wallet_sync, 'is_live', False):
            live_equity = self.wallet_sync.get_equity(asset)
            if available_in_ledger > live_equity:
                logger.warning(f"Portfolio budget for {asset} ({available_in_ledger:.2f}) exceeds live wallet "
                               f"equity ({live_equity:.2f}). Capping available budget.")
                return live_equity

        return available_in_ledger

    def reserve(self, asset: str, amount_usd: float) -> Optional[str]:
        """
        Reserves a specific amount of capital for a trade.

        Args:
            asset (str): The asset class to reserve from.
            amount_usd (float): The amount to reserve in USD.

        Returns:
            Optional[str]: A unique reservation ID if successful, None otherwise.
        """
        if amount_usd <= 0:
            logger.error("Reservation amount must be positive.")
            return None

        if asset not in self.ledgers:
            logger.error(f"Cannot reserve for unknown asset class: {asset}")
            return None

        if self.available_budget(asset) < amount_usd:
            logger.warning(f"Cannot reserve {amount_usd:.2f} for {asset}. "
                           f"Available budget is {self.available_budget(asset):.2f}.")
            return None

        ledger = self.ledgers[asset]
        ledger["available"] -= amount_usd

        reservation_id = str(uuid.uuid4())
        self.reservations[reservation_id] = {"asset": asset, "amount": amount_usd}

        logger.info(f"Reserved {amount_usd:.2f} for {asset}. Reservation ID: {reservation_id}. "
                    f"New available budget: {ledger['available']:.2f}")
        return reservation_id

    def release(self, reservation_id: str):
        """
        Releases a previously made reservation. This is used after a trade is
        completed (booked) or cancelled. The reserved amount is returned to the
        available budget.

        Args:
            reservation_id (str): The ID of the reservation to release.
        """
        reservation = self.reservations.pop(reservation_id, None)
        if not reservation:
            logger.error(f"Reservation ID '{reservation_id}' not found for release.")
            return

        asset = reservation["asset"]
        amount = reservation["amount"]

        ledger = self.ledgers.get(asset)
        if ledger:
            # The capital is now free, but it was the principal of the trade.
            # The PnL from the trade is already accounted for in book_trade.
            # The principal amount is returned to the available pool.
            ledger["available"] += amount
            logger.info(f"Released reservation {reservation_id} ({amount:.2f} for {asset}). "
                        f"New available budget: {ledger['available']:.2f}")
        else:
            logger.error(f"Ledger for asset {asset} not found during release of reservation {reservation_id}.")

    def book_trade(self, asset: str, pnl_net: float, fees: float):
        """
        Books the financial result of a trade to the corresponding ledger.
        This adjusts the total and available capital based on the trade's outcome.

        Args:
            asset (str): The asset class where the trade occurred.
            pnl_net (float): The net profit or loss from the trade.
            fees (float): The fees paid for the trade.
        """
        ledger = self.ledgers.get(asset)
        if not ledger:
            logger.error(f"Cannot book trade for unknown asset class: {asset}")
            return

        ledger["total"] += pnl_net
        ledger["available"] += pnl_net

        ledger["realized_pnl"] += pnl_net
        ledger["fees"] += fees

        logger.info(f"Booked trade for {asset}: PnL={pnl_net:.2f}, Fees={fees:.2f}. "
                    f"New total equity for {asset}: {ledger['total']:.2f}. "
                    f"New available budget: {ledger['available']:.2f}")

        # In simulation, update WalletSync's balance to reflect new total equity
        if self.wallet_sync and not getattr(self.wallet_sync, 'is_live', True):
            self.wallet_sync.set_simulation_balance(asset, ledger["total"])

    def get_open_positions(self) -> list:
        """
        Returns a list of symbols for all open positions.
        NOTE: This is a placeholder. A real implementation would track open positions.
        """
        # This should be implemented by tracking active positions.
        # For now, returning an empty list as a placeholder.
        return []

    def get_total_equity(self, asset: str) -> float:
        """
        Returns the total equity for a given asset class.

        Args:
            asset (str): The asset class (e.g., 'SPOT').

        Returns:
            float: The total equity in USD for the asset class.
        """
        ledger = self.ledgers.get(asset)
        if not ledger:
            logger.warning(f"No ledger found for asset class: {asset}")
            return 0.0

        # In live mode, the true source of equity is the wallet sync
        if self.wallet_sync and getattr(self.wallet_sync, 'is_live', False):
            return self.wallet_sync.get_equity(asset)

        return ledger["total"]
