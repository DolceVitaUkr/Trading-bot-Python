# modules/data_manager.py
import os
import logging
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import config
from modules.exchange import ExchangeAPI
from utils.utilities import ensure_directory, retry
from config import HISTORICAL_DATA_PATH

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

class DataManager:
    def __init__(self):
        self.data_folder = config.HISTORICAL_DATA_PATH
        self.exchange = ExchangeAPI()
        self._ensure_data_folder()
        self.cache: Dict[str, pd.DataFrame] = {}
        self.test_mode = False  # Add this line
        
    def _ensure_data_folder(self):
        ensure_directory(self.data_folder)

    def _get_filename(self, symbol: str, timeframe: str, test_mode: bool = False) -> str:
        """Generate filename with test mode support"""
        base_name = f"{symbol.replace('/', '').lower()}_{timeframe}.parquet"
        if test_mode:
            return os.path.join(HISTORICAL_DATA_PATH, f"test_{base_name}")
        return os.path.join(HISTORICAL_DATA_PATH, base_name)

    def _generate_mock_klines(self, periods: int, timeframe: str):
        """Generate test data with accurate intervals"""
        interval_ms = timeframe_to_minutes(timeframe) * 60 * 1000
        base_time = int(time.time() * 1000) - (periods * interval_ms)
        return [
            [
                base_time + (i * interval_ms),
                50000 + i,
                50000 + i + 50,
                50000 + i - 50,
                50000 + i + 25, 
                1000 + i
            ] for i in range(periods)
        ]

    def _generate_continuous_klines(self, periods: int, start_time: int, timeframe: str):
        """Generate continuous klines with perfect intervals"""
        interval = timeframe_to_minutes(timeframe) * 60 * 1000
        return [
            [
                start_time + (i * interval),
                50000 + i,
                50000 + i + 50,
                50000 + i - 50,
                50000 + i + 25,
                1000 + i
            ] for i in range(periods)
        ]

    
    def update_klines(self, symbol: str, timeframe: str, klines: List[list], test_mode=False) -> bool:
        """Updated version with test mode support"""
        filename = self._get_filename(symbol, timeframe, test_mode)
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Existing data handling
            existing_data = pd.DataFrame()
            if os.path.exists(filename):
                existing_data = self._load_data(filename)
                
            # Process new klines
            if klines:
                new_df = self._process_data(klines)
                combined = pd.concat([existing_data, new_df])
                combined = combined[~combined.index.duplicated(keep='last')]
                self._save_data(combined, filename)
                return True
                
            return False  # Add proper error handling
        except Exception as e:
            logger.error(f"Update failed: {str(e)}")
            return False
    

    def _create_initial_dataset(self, symbol: str, timeframe: str) -> bool:
        initial_data = self.exchange.fetch_market_data(symbol, timeframe, limit=1000)
        if not initial_data:
            return False
        processed_df = self._process_data(initial_data)
        self._save_data(processed_df, self._get_filename(symbol, timeframe))
        return True

    def _fetch_new_data(self, symbol: str, timeframe: str, existing_df: pd.DataFrame) -> pd.DataFrame:
        last_timestamp = existing_df.index[-1].timestamp() * 1000
        new_data = self.exchange.fetch_market_data(symbol, timeframe, since=int(last_timestamp))
        return self._process_data(new_data) if new_data else pd.DataFrame()

    def _process_data(self, raw_data: List[list]) -> pd.DataFrame:
        columns = ["timestamp", "open", "high", "low", "close", "volume"]
        df = pd.DataFrame(raw_data, columns=columns)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df[df['timestamp'].notna()]
        df.set_index('timestamp', inplace=True)
        df = df[~df.index.duplicated(keep='last')]
        df.sort_index(inplace=True)
        return df.astype(float)

    def _merge_datasets(self, existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
        if new.empty:
            return existing
            
        # Convert to UTC and remove timezone info for comparison
        existing.index = existing.index.tz_convert(None)
        new.index = new.index.tz_convert(None)
        
        # Find first new timestamp after existing data
        cutoff = existing.index[-1]
        new = new[new.index > cutoff]
        
        return pd.concat([existing, new]).sort_index().drop_duplicates()

    def _validate_klines(self, klines: List[list], timeframe: str) -> bool:
        """Validate klines timestamps with 2% tolerance"""
        interval = timeframe_to_minutes(timeframe) * 60 * 1000
        for i in range(1, len(klines)):
            diff = klines[i][0] - klines[i-1][0]
            if not (0.98 * interval <= diff <= 1.02 * interval):
                logger.warning(f"Invalid interval at index {i}: {diff}ms (expected ~{interval}ms)")
                return False
        return True

    def _save_data(self, df: pd.DataFrame, filename: str):
        df.index = df.index.tz_localize(None)  # Remove timezone for parquet storage
        df.to_parquet(filename, engine='pyarrow', compression='snappy')

    def _load_data(self, filename: str) -> pd.DataFrame:
        df = pd.read_parquet(filename)
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Invalid data format: missing DatetimeIndex")
        df.index = pd.to_datetime(df.index, utc=True)  # Read as UTC
        return df.sort_index()

    def load_historical_data(self, symbol: str, timeframe: str = '15m', 
                            test_mode: bool = False) -> pd.DataFrame:
        """Load historical data with test/prod mode support"""
        cache_key = f"{symbol}_{timeframe}_{'test' if test_mode else 'prod'}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        filename = self._get_filename(symbol, timeframe, test_mode)
        
        if os.path.exists(filename):
            try:
                df = self._load_data(filename)
                self.cache[cache_key] = df
                return df
            except Exception as e:
                logger.error(f"Failed to load data: {str(e)}", exc_info=True)
                raise
        
        # Fallback check for test files if in test mode
        if test_mode and not os.path.exists(filename):
            raise FileNotFoundError(f"No historical data found for {symbol}/{timeframe} (test mode)")
        
        raise FileNotFoundError(f"No historical data found for {symbol}/{timeframe}")

def timeframe_to_minutes(timeframe: str) -> int:
    units = {'m': 1, 'h': 60, 'd': 1440, 'w': 10080}
    return int(timeframe[:-1]) * units[timeframe[-1]]