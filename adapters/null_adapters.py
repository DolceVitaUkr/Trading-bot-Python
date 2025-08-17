"""
Null adapter implementations for the core interfaces.
These are used when a feature is disabled to prevent the bot from crashing.
"""
from typing import List, Dict, Any, Tuple
import pandas as pd
from core.interfaces import MarketData, Execution, WalletSync, NewsFeed, ValidationRunner

class NullMarketData(MarketData):
    """Null implementation of the MarketData interface."""

    def candles(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        print("NullMarketData: candles() called, returning empty DataFrame.")
        return pd.DataFrame()

    def ticker(self, symbol: str) -> Dict[str, float]:
        print("NullMarketData: ticker() called, returning empty dict.")
        return {"price": 0.0}

    def volume_24h(self, symbol: str) -> float:
        print("NullMarketData: volume_24h() called, returning 0.")
        return 0.0

class NullExecution(Execution):
    """Null implementation of the Execution interface."""

    def place_order(self, symbol: str, side: str, qty: float, **params: Any) -> Dict[str, Any]:
        print(f"NullExecution: place_order({symbol}, {side}, {qty}) called, doing nothing.")
        return {"status": "simulated", "id": "null-order-id"}

    def positions(self) -> List[Dict[str, Any]]:
        print("NullExecution: positions() called, returning empty list.")
        return []

class NullWalletSync(WalletSync):
    """Null implementation of the WalletSync interface."""

    def subledger_equity(self) -> Dict[str, float]:
        print("NullWalletSync: subledger_equity() called, returning zero balances.")
        return {"SPOT": 0.0, "FUTURES": 0.0, "FX": 0.0}

class NullNewsFeed(NewsFeed):
    """Null implementation of the NewsFeed interface."""

    def sentiment(self, symbols: List[str]) -> Dict[str, float]:
        print("NullNewsFeed: sentiment() called, returning neutral sentiment.")
        return {symbol: 0.5 for symbol in symbols}

    def macro_blockers(self, symbols: List[str]) -> Dict[str, bool]:
        print("NullNewsFeed: macro_blockers() called, returning no blockers.")
        return {symbol: False for symbol in symbols}

class NullValidationRunner(ValidationRunner):
    """Null implementation of the ValidationRunner interface."""

    def approved(self, strategy_id: str, market: str) -> Tuple[bool, Dict[str, Any]]:
        print(f"NullValidationRunner: approved({strategy_id}, {market}) called, returning approved.")
        # Default to approved for components that don't need validation to run.
        return True, {"reason": "Null runner, default to approved"}
