"""
Strategy Manager to handle strategy registration and metadata.
"""
import os
import orjson
from typing import Dict, Any

from core.schemas import StrategyMeta

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
            f.write(orjson.dumps(meta.dict(by_alias=True)))
            f.write(b"\n")
