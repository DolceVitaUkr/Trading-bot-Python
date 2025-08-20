"""
Core interfaces for the trading bot, defined as Protocols.
"""
from typing import Protocol, List, Dict, Any, Tuple, TypeAlias
import pandas as pd

# Placeholder for OHLCV data, assuming pandas DataFrame
OHLCV: TypeAlias = pd.DataFrame

class MarketData(Protocol):
    """Interface for market data providers."""

    def candles(self, symbol: str, timeframe: str, limit: int) -> OHLCV:
        """
        Fetches historical OHLCV data.
        """
        ...

    def ticker(self, symbol: str) -> Dict[str, float]:
        """
        Fetches the latest ticker price.
        """
        ...

    def volume_24h(self, symbol: str) -> float:
        """
        Fetches the 24-hour trading volume.
        """
        ...

class Execution(Protocol):
    """Interface for execution venues."""

    def place_order(self, symbol: str, side: str, qty: float, **params: Any) -> Dict[str, Any]:
        """
        Places an order.
        """
        ...

    def positions(self) -> List[Dict[str, Any]]:
        """
        Fetches current open positions.
        """
        ...

class WalletSync(Protocol):
    """Interface for syncing wallet/sub-ledger balances."""

    def subledger_equity(self) -> Dict[str, float]:
        """
        Returns equity for each sub-ledger (e.g., SPOT, FUTURES, FX).
        """
        ...

class NewsFeed(Protocol):
    """Interface for news and sentiment analysis."""

    def sentiment(self, symbols: List[str]) -> Dict[str, float]:
        """
        Gets sentiment score for a list of symbols.
        """
        ...

    def macro_blockers(self, symbols: List[str]) -> Dict[str, bool]:
        """
        Checks for any macro-economic events that should block trading.
        """
        ...

class ValidationRunner(Protocol):
    """Interface for running strategy validation."""

    async def approved(self, strategy_id: str, market: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Checks if a strategy is approved for live trading based on its performance.
        Returns a tuple of (is_approved, metadata).
        """
        ...
