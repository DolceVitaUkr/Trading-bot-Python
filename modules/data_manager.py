# modules/data_manager.py

import os
import logging
import pandas as pd
import time
from datetime import datetime
from typing import List, Dict, Optional

import config
from modules.exchange import Exchange        # updated import
from utils.utilities import ensure_directory, retry

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)


class DataManager:
    def __init__(self, test_mode: bool = False):
        """
        Manages historical OHLCV data storage and retrieval.
        If test_mode is True, skips real API calls and only uses mock data.
        """
        self.data_folder = config.HISTORICAL_DATA_PATH
        self.test_mode = test_mode

        # Initialize exchange only in production mode
        self.exchange: Optional[Exchange] = None
        if not self.test_mode:
            self.exchange = Exchange()

        self._ensure_data_folder()
        self.cache: Dict[str, pd.DataFrame] = {}

    def _ensure_data_folder(self):
        """Create data folder if missing."""
        ensure_directory(self.data_folder)

    def _get_filename(self, symbol: str, timeframe: str) -> str:
        """Compute parquet filename; prepend 'test_' in test mode."""
        base = f"{symbol.replace('/', '').lower()}_{timeframe}.parquet"
        if self.test_mode:
            base = f"test_{base}"
        return os.path.join(self.data_folder, base)

    def _generate_mock_klines(self, periods: int, timeframe: str) -> List[list]:
        """Create synthetic OHLCV data for testing."""
        interval_ms = timeframe_to_minutes(timeframe) * 60 * 1000
        start = int(time.time() * 1000) - periods * interval_ms
        return [
            [
                start + i * interval_ms,
                50000 + i,
                50000 + i + 50,
                50000 + i - 50,
                50000 + i + 25,
                1000 + i,
            ]
            for i in range(periods)
        ]

    def update_klines(
        self, symbol: str, timeframe: str, klines: Optional[List[list]] = None
    ) -> bool:
        """
        Append new klines to the local parquet file.
        - In test_mode, if klines is None, uses generated mock data.
        - Returns True on success, False otherwise.
        """
        fname = self._get_filename(symbol, timeframe)
        os.makedirs(os.path.dirname(fname), exist_ok=True)

        # If no data provided in test_mode, generate some
        if self.test_mode and klines is None:
            klines = self._generate_mock_klines(periods=100, timeframe=timeframe)

        if not klines:
            logger.warning("No new klines to update.")
            return False

        # Load existing data if present
        existing = pd.DataFrame()
        if os.path.exists(fname):
            try:
                existing = self._load_data(fname)
            except Exception:
                logger.exception("Failed to load existing data, will overwrite.")

        # Process and merge
        new_df = self._process_data(klines)
        combined = pd.concat([existing, new_df])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)

        # Save to parquet
        try:
            self._save_data(combined, fname)
            return True
        except Exception:
            logger.exception("Failed to save updated data.")
            return False

    def load_historical_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Load price data from parquet. Raises FileNotFoundError if missing.
        Caches loaded DataFrames in memory.
        """
        key = f"{symbol}_{timeframe}_{'test' if self.test_mode else 'prod'}"
        if key in self.cache:
            return self.cache[key]

        fname = self._get_filename(symbol, timeframe)
        if not os.path.exists(fname):
            raise FileNotFoundError(f"No historical data at {fname}")

        df = self._load_data(fname)
        self.cache[key] = df
        return df

    def _process_data(self, raw: List[list]) -> pd.DataFrame:
        """Convert raw OHLCV lists into a cleaned DataFrame with a DatetimeIndex."""
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        df = pd.DataFrame(raw, columns=cols)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        df = df[~df.index.duplicated(keep="last")]
        df.sort_index(inplace=True)
        return df.astype(float)

    def _save_data(self, df: pd.DataFrame, path: str):
        """Write DataFrame to parquet (tz-naive)."""
        df = df.copy()
        df.index = df.index.tz_localize(None)
        df.to_parquet(path, engine="pyarrow", compression="snappy")

    def _load_data(self, path: str) -> pd.DataFrame:
        """Read parquet into DataFrame with UTC DatetimeIndex."""
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index, utc=True)
        return df.sort_index()


def timeframe_to_minutes(tf: str) -> int:
    """Convert a timeframe string like '15m' or '1h' to total minutes."""
    unit_map = {"m": 1, "h": 60, "d": 1440, "w": 10080}
    unit = tf[-1]
    num = int(tf[:-1])
    return num * unit_map.get(unit, 1)
