"""
Strategy Manager to handle strategy registration and metadata.
"""
import os
from dataclasses import dataclass
from typing import Any, Dict

from trading_bot.core.schemas import StrategyMeta

try:
    import orjson as _json
except ImportError:
    import ujson as _json


@dataclass
class Decision:
    """Data class to hold a trading decision."""

    signal: str  # 'buy' or 'sell'
    sl: float  # stop loss
    tp: float  # take profit
    meta: Dict[str, Any]


STRATEGY_FILE = "state/strategies.jsonl"


class StrategyManager:
    """
    Manages the lifecycle and metadata of trading strategies.
    """

    def __init__(self, strategy_file: str = STRATEGY_FILE):
        self.strategy_file = strategy_file
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Ensures the strategy JSONL file exists."""
        os.makedirs(os.path.dirname(self.strategy_file), exist_ok=True)
        with open(self.strategy_file, "a"):
            pass

    def register_strategy(self, meta: StrategyMeta):
        """
        Registers a new strategy by saving its metadata to the JSONL file.
        """
        print(f"Registering strategy: {meta.strategy_id}")
        with open(self.strategy_file, "ab") as f:
            f.write(_json.dumps(meta.dict(by_alias=True)))
            f.write(b"\n")
