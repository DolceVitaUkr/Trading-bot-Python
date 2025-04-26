# modules/data_manager.py

import os
import time
import logging
import pandas as pd
from typing import List, Optional, Dict

import config
from modules.exchange import ExchangeAPI
from utils.utilities import ensure_directory

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class DataManager:
    """
    Manages OHLCV data for both real and test modes.
    In test_mode, all data operations stay local and can use mock data.
    """

    def __init__(self, test_mode: bool = False):
        self.data_folder = config.HISTORICAL_DATA_PATH
        self.test_mode = test_mode

        # Only initialize real API client when not in test mode
        self.exchange = ExchangeAPI() if not self.test_mode else None

        # Ensure data directory exists
        ensure_directory(self.data_folder)

        # In-memory cache for loaded DataFrames
        self.cache: Dict[str, pd.DataFrame] = {}

    def _get_filename(self, symbol: str, timeframe: str) -> str:
        """
        Compute the parquet filename, prefixing with 'test_' when in test_mode.
        """
        base = f"{symbol.replace('/', '').lower()}_{timeframe}.parquet"
        if self.test_mode:
            base = f"test_{base}"
        return os.path.join(self.data_folder, base)

    def _generate_mock_klines(self, periods: int, timeframe: str) -> List[list]:
        """
        Create synthetic OHLCV data for testing.
        """
        interval_ms = timeframe_to_minutes(timeframe) * 60 * 1000
        start = int(time.time() * 1000) - periods * interval_ms
        return [
            [
                start + i * interval_ms,    # timestamp
                50000 + i,                  # open
                50000 + i + 50,             # high
                50000 + i - 50,             # low
                50000 + i + 25,             # close
                1000 + i                    # volume
            ]
            for i in range(periods)
        ]

    def _generate_continuous_klines(self, periods: int, start_time: int, timeframe: str) -> List[list]:
        """
        Generate perfectly spaced OHLCV data starting at start_time.
        """
        interval_ms = timeframe_to_minutes(timeframe) * 60 * 1000
        return [
            [
                start_time + i * interval_ms,
                50000 + i,
                50000 + i + 50,
                50000 + i - 50,
                50000 + i + 25,
                1000 + i
            ]
            for i in range(periods)
        ]

    def update_klines(
        self,
        symbol: str,
        timeframe: str,
        klines: Optional[List[list]] = None
    ) -> bool:
        """
        Append new klines to the local parquet file.
        - In test_mode: if klines is None, generates mock data.
        - Returns True on success, False on failure or if no data.
        """
        fname = self._get_filename(symbol, timeframe)
        os.makedirs(os.path.dirname(fname), exist_ok=True)

        # In test mode, generate if not provided
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

        # Process incoming raw data
        new_df = self._process_data(klines)

        # Concatenate & dedupe (keep last = newest)
        combined = pd.concat([existing, new_df])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)

        # Save back to parquet
        try:
            self._save_data(combined, fname)
            return True
        except Exception:
            logger.exception("Failed to save updated data.")
            return False

    def load_historical_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Load OHLCV data from parquet. Raises FileNotFoundError if missing.
        Caches loaded DataFrames in memory.
        """
        key = f"{symbol}_{timeframe}_{'test' if self.test_mode else 'prod'}"
        if key in self.cache:
            return self.cache[key]

        fname = self._get_filename(symbol, timeframe)
        if not os.path.exists(fname):
            raise FileNotFoundError(f"No historical data found for {symbol}/{timeframe}")

        df = self._load_data(fname)
        self.cache[key] = df
        return df

    def _process_data(self, raw: List[list]) -> pd.DataFrame:
        """
        Convert raw OHLCV lists into a cleaned DataFrame with UTC DatetimeIndex.
        """
        columns = ["timestamp", "open", "high", "low", "close", "volume"]
        df = pd.DataFrame(raw, columns=columns)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        # drop any exact-duplicate timestamps
        df = df[~df.index.duplicated(keep="last")]
        df.sort_index(inplace=True)
        return df.astype(float)

    def _save_data(self, df: pd.DataFrame, path: str):
        """
        Write DataFrame to parquet (tz-naive index) with Snappy compression.
        """
        df_copy = df.copy()
        df_copy.index = df_copy.index.tz_localize(None)
        df_copy.to_parquet(path, engine="pyarrow", compression="snappy")

    def _load_data(self, path: str) -> pd.DataFrame:
        """
        Read parquet into DataFrame, restoring UTC DatetimeIndex.
        """
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index, utc=True)
        return df.sort_index()


def timeframe_to_minutes(tf: str) -> int:
    """
    Convert a string like '15m','1h','1d','1w' into total minutes.
    """
    unit_map = {"m": 1, "h": 60, "d": 1440, "w": 10080}
    unit = tf[-1]
    num = int(tf[:-1])
    return num * unit_map.get(unit, 1)
