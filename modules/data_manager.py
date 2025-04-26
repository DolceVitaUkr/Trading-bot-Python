# modules/data_manager.py
import os, time, logging
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
import config
from modules.exchange import Exchange
from utils.utilities import ensure_directory

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, test_mode: bool = False):
        self.data_folder = config.HISTORICAL_DATA_PATH
        self.test_mode = test_mode
        self.exchange: Optional[Exchange] = None
        if not self.test_mode:
            self.exchange = Exchange()
        ensure_directory(self.data_folder)
        self.cache: Dict[str, pd.DataFrame] = {}

    def _get_filename(self, symbol: str, timeframe: str) -> str:
        base = f"{symbol.replace('/','').lower()}_{timeframe}.parquet"
        if self.test_mode:
            base = f"test_{base}"
        return os.path.join(self.data_folder, base)

    def _generate_mock_klines(self, periods: int, timeframe: str) -> List[list]:
        interval = timeframe_to_minutes(timeframe) * 60_000
        start = int(time.time()*1000) - periods * interval
        return [
            [start + i*interval,
             50000 + i,
             50000 + i + 50,
             50000 + i - 50,
             50000 + i + 25,
             1000 + i]
            for i in range(periods)
        ]

    def update_klines(self,
                      symbol: str,
                      timeframe: str,
                      klines: Optional[List[list]] = None) -> bool:
        fname = self._get_filename(symbol, timeframe)
        os.makedirs(os.path.dirname(fname), exist_ok=True)

        if self.test_mode and klines is None:
            klines = self._generate_mock_klines(100, timeframe)
        if not klines:
            return False

        existing = pd.DataFrame()
        if os.path.exists(fname):
            try:
                existing = self._load_data(fname)
            except:
                logger.exception("Could not load existing; overwriting.")

        new_df = self._process_data(klines)
        combined = pd.concat([existing, new_df])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)

        try:
            self._save_data(combined, fname)
            return True
        except:
            logger.exception("Save failed")
            return False

    def load_historical_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
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
        cols = ["timestamp","open","high","low","close","volume"]
        df = pd.DataFrame(raw, columns=cols)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        df = df[~df.index.duplicated(keep="last")]
        df.sort_index(inplace=True)
        return df.astype(float)

    def _save_data(self, df: pd.DataFrame, path: str):
        df = df.copy()
        df.index = df.index.tz_localize(None)
        df.to_parquet(path, engine="pyarrow", compression="snappy")

    def _load_data(self, path: str) -> pd.DataFrame:
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index, utc=True)
        return df.sort_index()

def timeframe_to_minutes(tf: str) -> int:
    unit_map = {"m":1,"h":60,"d":1440,"w":10080}
    num, unit = int(tf[:-1]), tf[-1]
    return num * unit_map.get(unit,1)
