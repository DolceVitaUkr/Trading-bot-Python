from decimal import Decimal
from typing import Literal, Optional


class OptionsExchange:
    """
    Options Exchange adapter for options instruments (crypto or FX).
    Designed to mirror the Exchange API structure.
    """

    def __init__(self, venue: str, testnet: bool = False):
        """
        Args:
            venue: Exchange identifier (e.g., 'deribit', 'bybit')
            testnet: Connect to test/sandbox environment if available
        """
        self.venue = venue
        self.testnet = testnet
        # TODO: implement connection/authentication for specific venue

    async def get_chain_async(self, symbol: str, expiry: str) -> list[dict]:
        """
        Fetch the available option chain for a given underlying and expiry.

        Args:
            symbol: Underlying asset (e.g., 'BTCUSDT')
            expiry: Expiry date in YYYY-MM-DD format

        Returns:
            List of options contracts
        """
        raise NotImplementedError(
            "Option chain retrieval not yet implemented.")

    async def create_order_async(
        self,
        *,
        instrument_id: str,
        side: Literal["buy", "sell"],
        qty: Decimal,
        order_type: Literal["market", "limit"] = "market",
        price: Optional[Decimal] = None,
        reduce_only: bool = False,
        client_id: Optional[str] = None
    ) -> dict:
        """
        Place an order on an options contract.

        Args:
            instrument_id: Unique ID for the options instrument
            side: 'buy' or 'sell'
            qty: Quantity to trade
            order_type: 'market' or 'limit'
            price: Limit price (required if order_type='limit')
            reduce_only: Close-only flag
            client_id: Optional client order ID

        Returns:
            Order confirmation dict from the exchange
        """
        raise NotImplementedError(
            "Options order placement not yet implemented.")

    async def reconcile_async(self) -> None:
        """
        Reconcile open options positions and orders after restart.
        """
        raise NotImplementedError("Options reconcile not yet implemented.")
