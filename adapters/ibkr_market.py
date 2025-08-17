"""
IBKR Market Data Adapter using ib_insync.
"""
import os
import asyncio
from ib_insync import IB, Forex, util
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
import pandas as pd
from typing import Dict

from core.interfaces import MarketData, OHLCV

class IbkrMarketData(MarketData):
    """
    Implementation of the MarketData interface for Interactive Brokers.
    """

    def __init__(self, ib_client: IB):
        self.ib = ib_client

    def _symbol_to_contract(self, symbol: str):
        """Converts a symbol like 'EUR/USD' to an IBKR Forex contract."""
        if "/" in symbol:
            # Assuming format 'EUR/USD' for Forex
            return Forex(symbol.replace('/', ''))
        # This can be extended for other asset types
        raise ValueError(f"Symbol format not supported for IBKR: {symbol}")

    def _calculate_duration(self, limit: int, timeframe: str) -> str:
        """
        Calculates the duration string needed for a historical data request.
        This is a heuristic and may need refinement.
        """
        # Based on https://interactivebrokers.github.io/tws-api/historical_limitations.html
        timeframe_parts = timeframe.split()
        value = int(timeframe_parts[0])
        unit = timeframe_parts[1]

        if 'min' in unit:
            total_minutes = limit * value
            if total_minutes <= 60: return f"{total_minutes * 60} S"
            if total_minutes <= 1440: return "1 D"
            return f"{min(total_minutes // 1440, 5)} D" # Max 5 days for 1-min bars
        elif 'hour' in unit:
            total_hours = limit * value
            if total_hours <= 8: return f"{total_hours * 3600} S"
            if total_hours <= 24: return "1 D"
            return f"{min(total_hours // 24, 20)} D" # Max ~20 days for 1-hour bars
        elif 'day' in unit:
            return f"{limit} D" # Can go up to 1Y for daily bars

        # Fallback for other timeframes
        return "1 D"

    @retry(wait=wait_exponential(multiplier=1, min=2, max=60), stop=stop_after_attempt(5))
    async def candles(self, symbol: str, timeframe: str, limit: int) -> OHLCV:
        """
        Fetches historical OHLCV data from IBKR.

        IBKR timeframe format examples: '1 min', '5 mins', '1 hour', '1 day'
        IBKR duration format examples: '60 S', '3600 S', '1 D', '1 M', '1 Y'
        """
        await asyncio.sleep(0.05) # Pacing to avoid rate limits
        if not self.ib.isConnected():
            raise ConnectionError("IBKR client is not connected.")
        contract = self._symbol_to_contract(symbol)

        # A simple mapping for duration, this needs to be robust
        # This is a very rough estimation to fetch 'limit' candles
        # e.g., for 1 min timeframe, limit 100 -> '100 M' is not valid.
        # A better approach is needed. For now, we'll fetch for a day.
        duration = self._calculate_duration(limit, timeframe)

        try:
            bars = await self.ib.reqHistoricalDataAsync(
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=timeframe,
                whatToShow='MIDPOINT',
                useRTH=True
            )
            if not bars:
                print(f"No historical data returned for {symbol}.")
                # This could be due to no market data subscription.
                # The error message from IBKR would be in the logs.
                return pd.DataFrame()

            df = util.df(bars)
            df.rename(columns={
                'date': 'timestamp',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }, inplace=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            return df.tail(limit)
        except Exception as e:
            print(f"Error fetching historical data for {symbol}: {e}")
            # This could be a pacing violation or other API error.
            raise RetryError(f"Failed after retries for {symbol}") from e

    @retry(wait=wait_exponential(multiplier=1, min=2, max=60), stop=stop_after_attempt(3))
    async def ticker(self, symbol: str) -> Dict[str, float]:
        """Fetches the latest ticker price."""
        await asyncio.sleep(0.05) # Pacing to avoid rate limits
        if not self.ib.isConnected():
            raise ConnectionError("IBKR client is not connected.")
        contract = self._symbol_to_contract(symbol)

        try:
            # Request streaming data, wait for a tick, then cancel.
            self.ib.reqMktData(contract, '', False, False)
            await self.ib.sleep(2) # Wait for the ticker to arrive
            ticker = self.ib.ticker(contract)
            self.ib.cancelMktData(contract)

            if ticker and (ticker.last or ticker.close):
                return {"price": ticker.last if not pd.isna(ticker.last) else ticker.close}
            else:
                # Fallback to historical data for last price if streaming fails
                bars = await self.candles(symbol, '1 min', 1)
                if not bars.empty:
                    return {"price": bars['close'].iloc[-1]}
                return {"price": 0.0}
        except Exception as e:
            print(f"Error fetching ticker for {symbol}: {e}")
            return {"price": 0.0}

    async def volume_24h(self, symbol: str) -> float:
        """
        Fetches 24h volume. Not directly available for Forex in the same way as CEXs.
        We can approximate by summing up daily volume bars.
        """
        # This is a simplification. A real implementation would need more logic.
        daily_bars = await self.candles(symbol, '1 day', 1)
        if not daily_bars.empty:
            return daily_bars['volume'].iloc[-1]
        return 0.0
