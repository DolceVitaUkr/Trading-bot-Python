import logging
from typing import List, Dict, Any

from ib_insync import IB, Position, AccountValue

from modules.brokers.ibkr.Connect_IBKR_API import IBKRConnectionManager

log = logging.getLogger(__name__)


class IBKRAccountFetcher:
    """
    Handles fetching of read-only account data from Interactive Brokers,
    such as balances, buying power, and portfolio positions.
    """

    def __init__(self, conn_manager: IBKRConnectionManager):
        self.conn_manager = conn_manager
        self.ib: IB = None

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
        log.info("Fetching account summary...")

        # Request all summary tags
        summary_tags = "NetLiquidation,TotalCashValue,BuyingPower,GrossPositionValue,MaintMarginReq,InitMarginReq"
        account_values: List[AccountValue] = await self.ib.reqAccountSummaryAsync(tags=summary_tags)

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
        log.info("Fetching portfolio positions...")

        # Using reqPositionsAsync to avoid blocking and potential timeouts on large accounts
        positions: List[Position] = await self.ib.reqPositionsAsync()

        position_list = [
            {
                "conId": pos.contract.conId,
                "symbol": pos.contract.localSymbol,
                "secType": pos.contract.secType,
                "currency": pos.contract.currency,
                "position": pos.position,
                "market_price": pos.marketPrice,
                "market_value": pos.marketValue,
                "average_cost": pos.avgCost,
                "unrealized_pnl": pos.unrealizedPNL,
                "realized_pnl": pos.realizedPNL,
            }
            for pos in positions
        ]
        log.info(f"Found {len(position_list)} positions.")
        return position_list
