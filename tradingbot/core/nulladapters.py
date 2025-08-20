from tradingbot.core.interfaces import MarketData, Execution, WalletSync
from typing import List, Dict, Any
import pandas as pd

class NullMarketData(MarketData):
    def candles(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        return pd.DataFrame()

    def ticker(self, symbol: str) -> Dict[str, float]:
        return {}

    def volume_24h(self, symbol: str) -> float:
        return 0.0

class NullExecution(Execution):
    def place_order(self, symbol: str, side: str, qty: float, **params: Any) -> Dict[str, Any]:
        return {}

    def positions(self) -> List[Dict[str, Any]]:
        return []

class NullWalletSync(WalletSync):
    def subledger_equity(self) -> Dict[str, float]:
        return {}
