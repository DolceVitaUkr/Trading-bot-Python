# modules/data_manager.py

import os
import logging
import time
from datetime import datetime
from typing import List, Dict, Optional, Any

import pandas as pd
import requests

import config
from utils.utilities import ensure_directory, retry

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)


def timeframe_to_minutes(tf: str) -> int:
    """
    Convert a timeframe string like '15m', '1h', '1d', or '1w' to minutes.
    """
    unit_map = {"m": 1, "h": 60, "d": 1440, "w": 10080}
    unit = tf[-1]
    num = int(tf[:-1])
    return num * unit_map.get(unit, 1)


class DataManager:
    """
    Manages historical OHLCV data:
    - In test_mode: generates mock candles.
    - In live mode: fetches new candles from Bybit public API.
    Data is stored in Parquet under config.HISTORICAL_DATA_PATH.
    """

    def __init__(self, test_mode: bool = False):
        self.data_folder = config.HISTORICAL_DATA_PATH
        self.test_mode = test_mode
        # Bybit REST base URL (can override via config.BYBIT_BASE_URL)
        self.base_url = getattr(config, "BYBIT_BASE_URL", "https://api.bybit.com")
        self._ensure_data_folder()
        self.cache: Dict[str, pd.DataFrame] = {}

    def _ensure_data_folder(self):
        """Create the data folder if it doesn't exist."""
        ensure_directory(self.data_folder)

    def _get_filename(self, symbol: str, timeframe: str) -> str:
        """
        Build the Parquet filename for a given symbol/timeframe.
        Prefix with 'test_' when in test_mode.
        """
        base = f"{symbol.replace('/', '').lower()}_{timeframe}.parquet"
        if self.test_mode:
            base = f"test_{base}"
        return os.path.join(self.data_folder, base)

    def _generate_mock_klines(self, periods: int, timeframe: str) -> List[list]:
        """
        Generate synthetic OHLCV data for testing:
        intervals of perfect spacing, prices incrementing by 1.
        """
        interval_ms = timeframe_to_minutes(timeframe) * 60 * 1000
        start = int(time.time() * 1000) - periods * interval_ms
        return [
            [
                start + i * interval_ms,
                50000 + i,          # open
                50000 + i + 50,     # high
                50000 + i - 50,     # low
                50000 + i + 25,     # close
                1000 + i            # volume
            ]
            for i in range(periods)
        ]

    @retry(times=3, backoff=2)
    def fetch_recent_klines(
        self,
        symbol: str,
        timeframe: str,
        since: Optional[int] = None,
        limit: int = 200
    ) -> List[list]:
        """
        Fetch OHLCV data from Bybit public API.
        - symbol: e.g. 'BTC/USDT'
        - timeframe: '1m', '5m', '1h', etc.
        - since: timestamp in ms to fetch data after
        - limit: number of candle points
        Returns list of [timestamp_ms, open, high, low, close, volume].
        """
        sym = symbol.replace("/", "")
        interval = timeframe_to_minutes(timeframe)
        params: Dict[str, Any] = {
            "symbol": sym,
            "interval": str(interval),
            "limit": limit,
        }
        if since:
            # Bybit expects 'from' in seconds
            params["from"] = int(since / 1000)

        url = f"{self.base_url}/public/linear/kline"
        resp = requests.get(url, params=params, timeout=(5, 15))
        resp.raise_for_status()
        data = resp.json()
        if data.get("ret_code") != 0 or "result" not in data:
            raise RuntimeError(f"Invalid kline response: {data}")

        klines = []
        for entry in data["result"]:
            ts = entry.get("open_time")
            if ts is None:
                continue
            klines.append([
                int(ts) * 1000,
                float(entry["open"]),
                float(entry["high"]),
                float(entry["low"]),
                float(entry["close"]),
                float(entry["volume"]),
            ])
        return klines

    def update_klines(
        self,
        symbol: str,
        timeframe: str,
        klines: Optional[List[list]] = None
    ) -> bool:
        """
        Append or create OHLCV data for a given symbol/timeframe:
        - In test_mode with no klines: generate mock data.
        - In live mode with no klines: fetch from API since last timestamp.
        Saves merged data to Parquet.
        Returns True on success.
        """
        fname = self._get_filename(symbol, timeframe)
        os.makedirs(os.path.dirname(fname), exist_ok=True)

        # Test mode: mock data if none provided
        if self.test_mode and klines is None:
            klines = self._generate_mock_klines(periods=100, timeframe=timeframe)

        # Live mode: fetch if none provided
        if not self.test_mode and klines is None:
            try:
                if os.path.exists(fname):
                    existing = self._load_data(fname)
                    since_ts = int(existing.index[-1].timestamp() * 1000)
                else:
                    existing = pd.DataFrame()
                    since_ts = None
                klines = self.fetch_recent_klines(symbol, timeframe, since=since_ts)
            except Exception as e:
                logger.error(f"Failed to fetch recent klines: {e}", exc_info=True)
                return False

        if not klines:
            logger.warning("No new klines to update.")
            return False

        # Load existing if present
        existing = pd.DataFrame()
        if os.path.exists(fname):
            try:
                existing = self._load_data(fname)
            except Exception:
                logger.exception("Failed to load existing data, overwriting.")

        # Process and merge
        new_df = self._process_data(klines)
        combined = pd.concat([existing, new_df])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)

        # Save
        try:
            self._save_data(combined, fname)
            return True
        except Exception:
            logger.exception("Failed to save updated data.")
            return False

    def load_historical_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Load OHLCV data from Parquet, with caching.
        Raises FileNotFoundError if missing.
        """
        key = f"{symbol}_{timeframe}_{'test' if self.test_mode else 'prod'}"
        if key in self.cache:
            return self.cache[key]

        fname = self._get_filename(symbol, timeframe)
        if not os.path.exists(fname):
            raise FileNotFoundError(f"No historical data at {fname}")

        try:
            df = self._load_data(fname)
        except Exception as e:
            logger.exception(f"Error loading historical data: {e}")
            raise

        self.cache[key] = df
        return df

    def _process_data(self, raw: List[list]) -> pd.DataFrame:
        """
        Convert raw OHLCV lists to a UTC DatetimeIndex DataFrame.
        """
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        df = pd.DataFrame(raw, columns=cols)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        df = df[~df.index.duplicated(keep="last")]
        df.sort_index(inplace=True)
        return df.astype(float)

    def _save_data(self, df: pd.DataFrame, path: str):
        """
        Write to Parquet (tz-naive), raising on failure.
        """
        try:
            tmp = df.copy()
            tmp.index = tmp.index.tz_localize(None)
            tmp.to_parquet(path, engine="pyarrow", compression="snappy")
        except Exception as e:
            logger.exception(f"Error saving data to {path}: {e}")
            raise

    def _load_data(self, path: str) -> pd.DataFrame:
        """
        Read Parquet into DataFrame with UTC index, raising on failure.
        """
        try:
            df = pd.read_parquet(path)
            df.index = pd.to_datetime(df.index, utc=True)
            return df.sort_index()
        except Exception as e:
            logger.exception(f"Error reading data from {path}: {e}")
            raise
