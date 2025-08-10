# modules/data_manager.py

import os
import time
import math
import logging
from typing import Optional, List, Tuple
from datetime import datetime, timezone

import ccxt  # REST only (stable cross-platform)
import pandas as pd

import config
from utils.utilities import ensure_directory, retry

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(getattr(logging, str(getattr(config, "LOG_LEVEL", "INFO")), logging.INFO)
                if isinstance(getattr(config, "LOG_LEVEL", "INFO"), str) else getattr(config, "LOG_LEVEL", logging.INFO))


# ------------------------------ Helpers ------------------------------ #

def _sanitize_symbol_for_file(symbol: str) -> str:
    # "BTC/USDT" -> "BTCUSDT"
    return symbol.replace("/", "").replace("-", "").upper()


def _now_ms() -> int:
    return int(time.time() * 1000)


# ------------------------------ DataManager ------------------------------ #

class DataManager:
    """
    Data access layer for OHLCV and latest price.

    Design goals:
      - Keep local CSV per (symbol,timeframe), append incrementally
      - Never exceed `max_bars` rows (default 900)
      - Use Bybit via CCXT REST (stable on Windows)
      - Avoid gigabyte downloads (only fetch missing tail)
      - Filenames use slashless symbols: BTCUSDT_5m.csv

    File schema:
      index: pandas datetime (UTC)
      columns: ["open","high","low","close","volume"]
    """

    def __init__(
        self,
        base_path: str = None,
        exchange: str = "bybit",
        use_websocket: bool = False,  # reserved (off in this implementation)
        spot_market: bool = True,
    ):
        self.base_path = base_path or getattr(config, "HISTORICAL_DATA_PATH", "historical_data")
        ensure_directory(self.base_path)

        # CCXT exchange (Bybit)
        self.exchange_id = exchange.lower()
        self.use_websocket = use_websocket and False  # no WS client here; REST-only by design
        self.spot_market = spot_market

        self.ccxt = self._build_ccxt_client()
        self._timeframe_limits = {
            "1m": 1000, "3m": 1000, "5m": 1000, "15m": 1000, "30m": 1000,
            "1h": 1000, "4h": 1000, "1d": 1000
        }

    def _build_ccxt_client(self):
        cls = getattr(ccxt, self.exchange_id)
        # Use testnet keys if simulation, else live keys
        if config.USE_SIMULATION:
            api_key = getattr(config, "SIMULATION_BYBIT_API_KEY", None)
            secret = getattr(config, "SIMULATION_BYBIT_API_SECRET", None)
            testnet = True
        else:
            api_key = getattr(config, "BYBIT_API_KEY", None)
            secret = getattr(config, "BYBIT_API_SECRET", None)
            testnet = False

        params = {"enableRateLimit": True}
        inst = cls({
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": "spot" if self.spot_market else "swap",
            },
        })
        # Bybit testnet host (ccxt auto-handles in most versions; keep param for clarity)
        if hasattr(inst, "urls") and "api" in inst.urls and testnet:
            try:
                inst.urls["api"]["public"] = inst.urls["test"]["public"]
                inst.urls["api"]["private"] = inst.urls["test"]["private"]
            except Exception:
                pass
        return inst

    # ------------------------------ Public API ------------------------------ #

    @retry(max_attempts=3, delay=1.0, backoff=2.0)
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Lightweight last price."""
        sym = self.ccxt.market(symbol)["symbol"]
        t = self.ccxt.fetch_ticker(sym)
        p = t.get("last") or t.get("close")
        return float(p) if p is not None else None

    def _csv_path(self, symbol: str, timeframe: str) -> str:
        name = f"{_sanitize_symbol_for_file(symbol)}_{timeframe}.csv"
        return os.path.join(self.base_path, name)

    def _read_local_csv(self, path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            # enforce columns and dtypes
            for c in ["open", "high", "low", "close", "volume"]:
                if c not in df.columns:
                    df[c] = float("nan")
            df = df[["open", "high", "low", "close", "volume"]]
            df.index = pd.to_datetime(df.index, utc=True)
            return df
        except Exception as e:
            logger.warning(f"Failed reading {path}: {e} (recreating)")
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    def _write_local_csv(self, path: str, df: pd.DataFrame) -> None:
        ensure_directory(os.path.dirname(path))
        df.to_csv(path, index=True)

    @retry(max_attempts=3, delay=1.0, backoff=2.0)
    def _fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since_ms: Optional[int],
        limit: int,
    ) -> List[List[float]]:
        """Wrapper around ccxt.fetch_ohlcv."""
        sym = self.ccxt.market(symbol)["symbol"]
        # NOTE: Bybit supports since + limit for many timeframes; limit <= 1000
        bars = self.ccxt.fetch_ohlcv(sym, timeframe=timeframe, since=since_ms, limit=limit)
        return bars or []

    def _bars_to_df(self, bars: List[List[float]]) -> pd.DataFrame:
        if not bars:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        df = df.astype({
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
        })
        return df

    def _timeframe_ms(self, timeframe: str) -> int:
        # minimal set we use (5m, 15m)
        mapping = {
            "1m": 60_000,
            "3m": 180_000,
            "5m": 300_000,
            "15m": 900_000,
            "30m": 1_800_000,
            "1h": 3_600_000,
            "4h": 14_400_000,
            "1d": 86_400_000,
        }
        if timeframe not in mapping:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        return mapping[timeframe]

    def _trim_tail(self, df: pd.DataFrame, max_bars: int) -> pd.DataFrame:
        if len(df) <= max_bars:
            return df
        return df.iloc[-max_bars:].copy()

    def _calc_next_since(self, df: pd.DataFrame, timeframe: str) -> int:
        """Return 'since' (ms) for the next REST call so we only get new bars."""
        if df.empty:
            # roughly ~900 bars back to fill the file initially
            return _now_ms() - (self._timeframe_ms(timeframe) * 900)
        last_ts_ms = int(df.index[-1].timestamp() * 1000)
        # ask from the next bar
        return last_ts_ms + self._timeframe_ms(timeframe)

    # ------------------------------ Main entry ------------------------------ #

    def load_historical_data(
        self,
        symbol: str,
        timeframe: str = "5m",
        *,
        max_bars: int = 900,
        append_increment: int = 5,
    ) -> pd.DataFrame:
        """
        Return a DataFrame of up to `max_bars` bars, saving latest to disk.
        Only fetch up to `append_increment` NEW bars from REST (default 5).
        """
        path = self._csv_path(symbol, timeframe)
        local = self._read_local_csv(path)

        # Compute the REST request window
        since = self._calc_next_since(local, timeframe)
        # Bybit allows up to 1000; keep it small to avoid overfetch
        limit_cap = self._timeframe_limits.get(timeframe, 1000)
        limit = max(1, min(append_increment, limit_cap))

        # Fetch new tail
        try:
            new_bars = self._fetch_ohlcv(symbol, timeframe, since_ms=since, limit=limit)
        except Exception as e:
            logger.warning(f"fetch_ohlcv failed for {symbol} {timeframe}: {e}")
            new_bars = []

        if new_bars:
            df_new = self._bars_to_df(new_bars)
            # Sometimes the newest bar can overlap (re-run); concat+drop dup
            combined = pd.concat([local, df_new])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)
            combined = self._trim_tail(combined, max_bars=max_bars)
            self._write_local_csv(path, combined)
            return combined

        # No new bars => return local (trimmed) as-is
        local = self._trim_tail(local, max_bars=max_bars)
        if not local.empty:
            # write back trimmed copy to keep file from growing
            self._write_local_csv(path, local)
        return local

    # Convenience bulk method for 5m + 15m
    def load_dual_timeframe(
        self,
        symbol: str,
        tf_entry: str = "5m",
        tf_setup: str = "15m",
        *,
        max_bars_entry: int = 900,
        max_bars_setup: int = 900,
        append_increment: int = 5,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_entry = self.load_historical_data(symbol, tf_entry, max_bars=max_bars_entry, append_increment=append_increment)
        df_setup = self.load_historical_data(symbol, tf_setup, max_bars=max_bars_setup, append_increment=append_increment)
        return df_entry, df_setup
