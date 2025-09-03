"""
Core interfaces for the trading bot, defined as Protocols.
"""
from typing import Protocol, List, Dict, Any, Tuple, TypeAlias, Optional
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
import pandas as pd

# Placeholder for OHLCV data, assuming pandas DataFrame
OHLCV: TypeAlias = pd.DataFrame

# Core enums
class Asset(Enum):
    """Supported asset classes"""
    CRYPTO = "crypto"
    FUTURES = "futures"
    FOREX = "forex"
    OPTIONS = "options"

class OrderSide(Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

# Core data structures
@dataclass
class SessionState:
    """Paper trading session state"""
    session_id: str
    start_time: datetime
    assets: List[Asset]
    initial_equity_per_asset: float
    current_equity: Dict[Asset, float]
    reward: Dict[Asset, float] 
    open_positions: Dict[Asset, List[Any]]
    session_metadata: Dict[str, Any]

class MarketData(Protocol):
    """Adapter-facing market data interface. Timestamps UTC."""
    async def candles(self, symbol: str, timeframe: str, limit: int):
        ...
    async def ticker(self, symbol: str):
        ...
    async def volume_24h(self, symbol: str):
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

class MacroBlockers(Protocol):
    async def macro_blockers(self, symbols):
        ...

class ValidationRunner(Protocol):
    """Interface for running strategy validation."""

    async def approved(self, strategy_id, market):
        ...
