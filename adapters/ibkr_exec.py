"""
IBKR Execution Adapter using ib_insync.
"""
import asyncio
import logging
from ib_insync import IB, Forex, MarketOrder, Order, Trade
from typing import List, Dict, Any, Tuple

from core.interfaces import Execution
from adapters.ibkr_market import IbkrMarketData

logger = logging.getLogger(__name__)

# Minimum notional values for major IBKR FX pairs.
# These are approximate and should be verified and updated.
# Values are in the second currency of the pair (e.g., for EUR/USD, it's USD 25,000).
FX_MIN_NOTIONAL = {
    "EURUSD": 25000, "GBPUSD": 25000, "AUDUSD": 25000, "USDCAD": 25000,
    "USDCHF": 25000, "USDJPY": 2500000, # JPY notional is larger
    "EURGBP": 20000, "EURCHF": 25000, "EURJPY": 1250000,
}

class IbkrExecution(Execution):
    """
    Implementation of the Execution interface for Interactive Brokers.
    """

    def __init__(self, ib_client: IB, market_data_provider: IbkrMarketData):
        self.ib = ib_client
        self.market_data = market_data_provider

    def _symbol_to_contract(self, symbol: str):
        """Converts a symbol like 'EUR/USD' to an IBKR Forex contract."""
        if "/" in symbol:
            return Forex(symbol.replace('/', ''))
        raise ValueError(f"Symbol format not supported for IBKR: {symbol}")

    async def forex_allowed(self, symbol: str, qty: float) -> Tuple[bool, str]:
        """
        Checks if a forex trade is allowed based on several criteria.
        Returns (is_allowed, reason_code).
        """
        # 1. Check for market data subscription (indirectly)
        # We try to fetch a ticker. If it fails after retries, we assume no data subscription.
        try:
            price_data = await self.market_data.ticker(symbol)
            if price_data.get("price", 0.0) == 0.0:
                return False, "NO_MARKET_DATA_SUBSCRIPTION"
        except Exception:
            return False, "MARKET_DATA_FETCH_FAILED"

        # 2. Check minimum notional size
        pair = symbol.replace('/', '')
        if pair in FX_MIN_NOTIONAL:
            price = price_data["price"]
            notional_value = qty * price
            min_notional = FX_MIN_NOTIONAL[pair]

            if notional_value < min_notional:
                reason = f"NOTIONAL_TOO_SMALL:{notional_value:.2f}<{min_notional}"
                return False, reason
        else:
            # For pairs not in our map, we'll allow the trade but log a warning.
            # In a production system, this might be a hard failure.
            logger.warning(f"No minimum notional defined for {symbol}. Allowing trade.")

        # 3. Check pacing/concurrency (placeholder for a real budgeter)
        # A real implementation would check a token bucket or similar mechanism here.

        return True, "ALLOWED"


    async def place_order(self, symbol: str, side: str, qty: float, **params: Any) -> Dict[str, Any]:
        """
        Places an order on IBKR, with support for simulated bracket orders.
        `side` should be 'BUY' or 'SELL'.
        `params` can include 'stop_loss_price' and 'take_profit_price'.
        """
        await asyncio.sleep(0.05) # Pacing to avoid rate limits
        is_allowed, reason = await self.forex_allowed(symbol, qty)
        if not is_allowed:
            logger.warning(f"Order for {symbol} blocked: {reason}")
            return {"status": "REJECTED", "reason": reason}

        contract = self._symbol_to_contract(symbol)

        # --- Bracket Order Simulation ---
        stop_loss_price = params.get("stop_loss_price")
        take_profit_price = params.get("take_profit_price")

        if stop_loss_price and take_profit_price:
            logger.info(f"Simulating bracket order for {symbol} with SL={stop_loss_price}, TP={take_profit_price}")
            bracket_orders = self.ib.bracketOrder(
                action=side.upper(),
                quantity=qty,
                limitPrice=take_profit_price,
                stopPrice=stop_loss_price
            )
            # Place all three orders in one go
            trades = []
            for order in bracket_orders:
                trade = self.ib.placeOrder(contract, order)
                trades.append(trade)

            # For simulation, we'll just report the status of the parent trade
            trade = trades[0]

        else: # Simple Market Order
            order = MarketOrder(side.upper(), qty)
            trade = self.ib.placeOrder(contract, order)

        try:
            await self.ib.sleep(1) # Allow time for the order status to update
            if trade.orderStatus.status in ['Submitted', 'Filled']:
                return {
                    "status": trade.orderStatus.status,
                    "id": trade.order.orderId,
                    "avgFillPrice": trade.orderStatus.avgFillPrice,
                    "filled": trade.orderStatus.filled,
                }
            else:
                return {
                    "status": trade.orderStatus.status,
                    "id": trade.order.orderId,
                    "reason": trade.orderStatus.log[-1].message if trade.orderStatus.log else "Unknown reason"
                }
        except Exception as e:
            logger.error(f"Error placing order for {symbol}: {e}")
            return {"status": "ERROR", "reason": str(e)}


    async def positions(self) -> List[Dict[str, Any]]:
        """Fetches current open positions."""
        await asyncio.sleep(0.05) # Pacing to avoid rate limits
        await self.ib.reqPositionsAsync()
        positions = self.ib.positions()

        return [
            {
                "symbol": pos.contract.symbol + pos.contract.currency,
                "quantity": pos.position,
                "average_cost": pos.avgCost,
            }
            for pos in positions
        ]
