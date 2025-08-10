# modules/data_manager.py

import os
import gzip
import io
import logging
from typing import Optional, Literal, Dict
from datetime import datetime, timezone

import pandas as pd

import config
from modules.exchange import ExchangeAPI
from utils.utilities import ensure_directory, retry

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(getattr(logging, str(getattr(config, "LOG_LEVEL", "INFO")), logging.INFO)
                if isinstance(getattr(config, "LOG_LEVEL", "INFO"), str) else getattr(config, "LOG_LEVEL", logging.INFO))


VALID_TIMEFRAMES = ("5m", "15m")
TF_TO_MS: Dict[str, int] = {"5m": 5 * 60 * 1000, "15m": 15 * 60 * 1000}


class DataManager:
    """
    Handles historical bar storage & incremental updates.

    - Supports ONLY 5m and 15m.
    - Stores CSV.gz per symbol/timeframe under HISTORICAL_DATA_PATH/<timeframe>/<SYMBOL_NO_SLASH>.csv.gz
    - Always appends *only new* bars (uses since=last_ts+1).
    - Never requests more than `max_request_limit` (default 900).
    - Keeps file row-count bounded via `max_rows_keep` (optional ring-buffer behavior).
    """

    def __init__(
        self,
        exchange: Optional[ExchangeAPI] = None,
        data_root: Optional[str] = None,
        *,
        max_request_limit: int = 900,
        max_rows_keep: Optional[int] = None,  # e.g., 50_000 to cap file size; None = unlimited
    ):
        self.exchange = exchange or ExchangeAPI()
        self.data_root = data_root or getattr(config, "HISTORICAL_DATA_PATH", "historical_data")
        self.max_request_limit = max(1, min(int(max_request_limit), 900))
        self.max_rows_keep = max_rows_keep

        ensure_directory(self.data_root)

    # ─────────────────────────────
    # Public API
    # ─────────────────────────────

    def storage_path(self, symbol: str, timeframe: Literal["5m", "15m"]) -> str:
        self._assert_tf(timeframe)
        sym_noslash = self._noslash(symbol)
        folder = os.path.join(self.data_root, timeframe)
        ensure_directory(folder)
        return os.path.join(folder, f"{sym_noslash}.csv.gz")

    def load_historical_data(
        self,
        symbol: str,
        timeframe: Literal["5m", "15m"],
        *,
        auto_update: bool = True
    ) -> pd.DataFrame:
        """
        Load bars from local storage; optionally fetch & append fresh bars first.
        Returns columns: [timestamp, open, high, low, close, volume], index = pandas.DatetimeIndex (UTC)
        """
        self._assert_tf(timeframe)
        if auto_update:
            try:
                self.update_bars(symbol, timeframe)
            except Exception as e:
                logger.warning(f"update_bars failed for {symbol} {timeframe}: {e}")

        path = self.storage_path(symbol, timeframe)
        if not os.path.exists(path):
            # ensure empty frame with expected columns
            return self._empty_df()

        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                df = pd.read_csv(f)
            return self._normalize_df(df)
        except Exception as e:
            logger.error(f"Failed reading {path}: {e}")
            return self._empty_df()

    @retry(max_attempts=3, delay=1.0, backoff=2.0)
    def update_bars(
        self,
        symbol: str,
        timeframe: Literal["5m", "15m"],
        *,
        bootstrap_candles: int = 600,  # first-time pull size (kept < 900)
    ) -> int:
        """
        Append only new bars for the symbol/timeframe.
        Returns number of bars appended.
        """
        self._assert_tf(timeframe)
        path = self.storage_path(symbol, timeframe)
        tf_ms = TF_TO_MS[timeframe]

        # Load current
        if os.path.exists(path):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                existing = pd.read_csv(f)
            existing = self._normalize_df(existing)
            last_ts = int(existing["timestamp"].iloc[-1]) if len(existing) else None
        else:
            existing = self._empty_df()
            last_ts = None

        # Determine since
        if last_ts is None:
            # Bootstrap a chunk; keep well under 900
            limit = min(self.max_request_limit, max(200, bootstrap_candles))
            since = None
        else:
            # Ask for the NEXT candle after the last saved (plus 1 ms to be safe)
            since = last_ts + 1
            limit = min(self.max_request_limit, 300)

        new_rows = self._fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if new_rows is None or len(new_rows) == 0:
            return 0

        new_df = pd.DataFrame(
            new_rows,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        # drop any duplicates or overlaps
        if len(existing):
            new_df = new_df[new_df["timestamp"] > existing["timestamp"].iloc[-1]]

        if len(new_df) == 0:
            return 0

        # Ensure candle alignment to timeframe boundary
        new_df = new_df[new_df["timestamp"] % tf_ms == 0]

        # Append & cap
        out = pd.concat([existing, new_df], ignore_index=True)

        if self.max_rows_keep and len(out) > self.max_rows_keep:
            out = out.iloc[-self.max_rows_keep :].reset_index(drop=True)

        self._write_gz_csv(path, out)
        return len(new_df)

    # ─────────────────────────────
    # Internals
    # ─────────────────────────────

    def _fetch_ohlcv(
        self,
        symbol: str,
        timeframe: Literal["5m", "15m"],
        *,
        since: Optional[int],
        limit: int,
    ):
        """
        Fetch bars from exchange (ccxt under the hood via ExchangeAPI).
        The ExchangeAPI resolves symbol mapping for Bybit automatically.
        """
        try:
            # ExchangeAPI already handles spot/perp symbol normalization.
            # We cap limit further to be safe.
            limit = min(int(limit), self.max_request_limit)
            if limit <= 0:
                return []

            rows = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
            # Expect ccxt format: [timestamp, open, high, low, close, volume]
            return rows or []
        except Exception as e:
            logger.warning(f"_fetch_ohlcv error {symbol} {timeframe}: {e}")
            return []

    def _write_gz_csv(self, path: str, df: pd.DataFrame) -> None:
        # Always write UTF-8 csv.gz atomically
        tmp = f"{path}.tmp"
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        with gzip.open(tmp, "wb") as f:
            f.write(csv_bytes)
        os.replace(tmp, path)

    def _normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return self._empty_df()
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        df = df.loc[:, cols].copy()
        # enforce int timestamp
        df["timestamp"] = df["timestamp"].astype("int64")
        # ensure sorted unique
        df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
        # attach UTC index for plotting/convenience (optional)
        try:
            df.index = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        except Exception:
            pass
        return df

    def _empty_df(self) -> pd.DataFrame:
        df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        try:
            df.index = pd.to_datetime([], utc=True)
        except Exception:
            pass
        return df

    def _assert_tf(self, timeframe: str) -> None:
        if timeframe not in VALID_TIMEFRAMES:
            raise ValueError(f"timeframe must be one of {VALID_TIMEFRAMES}, got '{timeframe}'")

    def _noslash(self, symbol: str) -> str:
        return symbol.replace("/", "").replace(":", "")
