import logging
from typing import List, Dict, Any, Optional

from ib_insync import IB, Position, AccountValue

from TradingBot.brokers.ConnectIBKRAPI import IBKRConnectionManager

log = logging.getLogger(__name__)


class IBKRAccountFetcher:
    """
    Handles fetching of read-only account data from Interactive Brokers,
    such as balances, buying power, and portfolio positions.
    """

    def __init__(self, conn_manager: IBKRConnectionManager):
        self.conn_manager = conn_manager
        self.ib: Optional[IB] = None

    async def _ensure_connected(self):
        """Ensures the IB client is connected before making a request."""
        if not self.ib or not self.ib.isConnected():
            self.ib = await self.conn_manager.get_tws_client()

    async def get_account_summary(self) -> Dict[str, Any]:
        """
        Fetches a consolidated summary of the primary account.

        Returns:
            A dictionary containing key account values.
        """
        await self._ensure_connected()
        assert self.ib is not None
        log.info("Fetching account summary...")

        # Request all summary tags
        account_values: List[AccountValue] = self.ib.accountSummary()

        summary = {
            "account_id": self.ib.managedAccounts()[0],
            "summary": {val.tag: val.value for val in account_values if val.currency == 'BASE'}
        }
        log.info(f"Account summary fetched successfully: {summary['summary']}")
        return summary

    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        Fetches all current positions in the portfolio.

        Returns:
            A list of dictionaries, where each dictionary represents a position.
        """
        await self._ensure_connected()
        assert self.ib is not None
        log.info("Fetching portfolio positions...")

        # Using portfolio() to get a complete snapshot of portfolio items
        portfolio_items = self.ib.portfolio()

        position_list = [
            {
                "conId": item.contract.conId,
                "symbol": item.contract.localSymbol,
                "secType": item.contract.secType,
                "currency": item.contract.currency,
                "position": item.position,
                "market_price": item.marketPrice,
                "market_value": item.marketValue,
                "average_cost": item.averageCost,
                "unrealized_pnl": item.unrealizedPNL,
                "realized_pnl": item.realizedPNL,
            }
            for item in portfolio_items
        ]
        log.info(f"Found {len(position_list)} positions.")
        return position_list
