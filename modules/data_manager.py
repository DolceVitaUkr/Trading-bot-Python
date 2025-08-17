import os
import time
import logging
from typing import List, Optional, Tuple, Dict

import pandas as pd

import config
from utils.utilities import ensure_directory, write_json, retry, format_timestamp
from modules.exchange import ExchangeAPI
from Data_Registry import Data_Registry

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
log_level_str = str(getattr(config, "LOG_LEVEL", "INFO"))
log_level = getattr(logging, log_level_str, logging.INFO)
logger.setLevel(log_level)


# ccxt timeframe to milliseconds
TF_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
    "6h": 21_600_000, "12h": 43_200_000, "1d": 86_400_000,
    "1w": 604_800_000
}


def _symbol_to_filename(symbol: str) -> str:
    # Remove "/" and ":" to keep it filesystem-safe
    return symbol.replace("/", "").replace(":", "").upper()


def _build_paths(symbol: str,
                 timeframe: str,
                 exchange_name: str = "bybit") -> Tuple[str, str]:
    """
    Return (csv_path, meta_path) for a symbol/timeframe.
    """
    csv_path = Data_Registry.get_historical_data_path(exchange_name, symbol, timeframe)
    meta_path = Data_Registry.get_historical_meta_path(exchange_name, symbol, timeframe)
    return str(csv_path), str(meta_path)


class DataManager:
    """
    Persisted OHLCV store with incremental backfill using ccxt.

    Key design:
      - CSV per symbol/timeframe at:
        historical_data/bybit/<timeframe>/<SYMBOL>.csv
      - Metadata JSON alongside CSV tracks last sync timestamp
      - Incremental fetch respects Bybit/ccxt limits
        (use <= 900 bars per call)
      - Returns pandas.DataFrame with UTC DatetimeIndex,
        columns: open, high, low, close, volume

    Notes:
      - We default to Bybit spot unless config.EXCHANGE_PROFILE indicates perp.
      - No API keys required for public data; keys (if present) won’t hurt.
    """

    def __init__(
        self,
        exchange: Optional[ExchangeAPI] = None,
        *,
        max_request_bars: int = 900,
    ):
        """
        Initializes the DataManager.

        Args:
            exchange: An optional ExchangeAPI instance.
            max_request_bars: The maximum number of bars to request in a
                              single API call.
        """
        self.max_request_bars = max(10, min(900, int(max_request_bars)))
        if exchange:
            self.exchange = exchange
        else:
            self.exchange = ExchangeAPI()
            self.exchange.load_markets()


    # ──────────────────────────────────────────────────────────────────────
    # Public API - Mock Data Methods for Filters
    # ──────────────────────────────────────────────────────────────────────
    def get_daily_volume(self, symbol: str) -> float:
        """(Mock) Returns the 24h trading volume for a symbol."""
        mock_volumes = {
            "BTC/USDT": 2_500_000_000.0,
            "ETH/USDT": 1_800_000_000.0,
            "LOW_LIQ_COIN/USDT": 100_000.0, # For testing liquidity filter
        }
        volume = mock_volumes.get(symbol, 10_000_000.0) # Default to liquid
        logger.debug(f"Mock daily volume for {symbol}: ${volume:,.0f}")
        return volume

    def get_funding_rate(self, symbol: str) -> float:
        """(Mock) Returns the current funding rate for a perpetual contract."""
        mock_rates = {
            "BTC/USDT": 0.0001,  # Standard positive funding
            "ETH/USDT": -0.0003, # High negative funding (costly to long)
            "DOGE/USDT": 0.0008, # High positive funding (costly to short)
        }
        rate = mock_rates.get(symbol, 0.0001) # Default to standard
        logger.debug(f"Mock funding rate for {symbol}: {rate:.4%}")
        return rate

    # ──────────────────────────────────────────────────────────────────────
    # Public API - Historical Data
    # ──────────────────────────────────────────────────────────────────────
    def load_historical_data(
        self,
        symbol: str,
        timeframe: str = "5m",
        *,
        backfill_bars: int = 900,
        incremental: bool = True
    ) -> pd.DataFrame:
        """
        Ensure CSV exists; backfill if missing; optionally incremental sync;
        return dataframe.
        """
        csv_path, meta_path = _build_paths(
            symbol, timeframe, self.exchange.client.id)
        df = self._read_csv(csv_path)

        if df is None or df.empty:
            # Backfill last N bars
            logger.info(
                f"[Data] Backfilling {symbol} {timeframe} for "
                f"~{backfill_bars} bars…")
            self._backfill_from_scratch(
                symbol, timeframe, lookback_bars=backfill_bars)
            df = self._read_csv(csv_path)

        if incremental:
            try:
                self.sync_incremental(symbol, timeframe)
                df = self._read_csv(csv_path)
            except Exception as e:
                logger.warning(
                    f"[Data] Incremental sync failed for {symbol} "
                    f"{timeframe}: {e}")

        # Normalize index and types
        if df is None:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"])
        df = self._normalize_df(df)
        return df

    def sync_incremental(self, symbol: str, timeframe: str) -> int:
        """
        Fetch only *new* bars since last saved candle.
        Returns number of rows appended.
        """
        csv_path, meta_path = _build_paths(
            symbol, timeframe, self.exchange.client.id)
        df = self._read_csv(csv_path)
        last_ts = None
        if df is not None and not df.empty:
            last_ts = int(df["timestamp"].iloc[-1])

        since = last_ts + 1 if last_ts else None
        appended = self._fetch_append(symbol, timeframe, since=since)
        if appended > 0:
            write_json(meta_path,
                       {"last_sync": format_timestamp(int(time.time()))})
        return appended

    def ensure_symbols(self,
                       symbols: List[str],
                       timeframes: List[str]) -> Dict[str, Dict[str, int]]:
        """
        Ensure storage exists and is up-to-date for each (symbol, timeframe).
        Returns counts of appended rows per pair/timeframe.
        """
        result: Dict[str, Dict[str, int]] = {}
        for sym in symbols:
            result[sym] = {}
            for tf in timeframes:
                try:
                    appended = self.sync_incremental(sym, tf)
                    result[sym][tf] = appended
                except Exception as e:
                    logger.warning(
                        f"[Data] ensure_symbols failed for {sym} {tf}: {e}")
                    result[sym][tf] = 0
        return result

    # ──────────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────────
    def _read_csv(self, path: str) -> Optional[pd.DataFrame]:
        try:
            if not os.path.exists(path):
                return None
            df = pd.read_csv(path)
            # Expect columns: timestamp, open, high, low, close, volume
            return df
        except Exception as e:
            logger.warning(f"[Data] failed to read {path}: {e}")
            return None

    def _normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        df = df.loc[:, [c for c in cols if c in df.columns]].copy()
        df["timestamp"] = df["timestamp"].astype("int64")
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.dropna(subset=["timestamp", "close"], inplace=True)
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        # Set DatetimeIndex (UTC)
        df.index = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df

    def _write_csv(self, path: str, df: pd.DataFrame) -> None:
        ensure_directory(os.path.dirname(path) or ".")
        df.to_csv(path, index=False)

    def _append_to_csv(self, path: str, rows: List[List[float]]) -> int:
        """
        Append OHLCV rows if they are strictly newer than last timestamp in file.
        rows are ccxt OHLCV lists: [ts, open, high, low, close, volume]
        """
        if not rows:
            return 0
        if not os.path.exists(path):
            df = pd.DataFrame(
                rows,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ],
            )
            self._write_csv(path, df)
            return len(rows)

        # Read last timestamp to avoid duplication
        try:
            last_line = None
            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                block = 1024
                data = b""
                while size > 0 and b"\n" not in data:
                    step = min(block, size)
                    size -= step
                    f.seek(size)
                    data = f.read(step) + data
                lines = data.splitlines()
                if lines:
                    last_line = lines[-1].decode("utf-8")
            last_ts = None
            if last_line:
                parts = last_line.split(",")
                if parts and parts[0].isdigit():
                    last_ts = int(parts[0])

            new_rows = [
                r for r in rows if (last_ts is None or int(r[0]) > last_ts)
            ]
            if not new_rows:
                return 0

            with open(path, "a", encoding="utf-8") as f:
                for r in new_rows:
                    f.write(
                        ",".join(
                            [
                                str(int(r[0])),
                                f"{float(r[1])}",
                                f"{float(r[2])}",
                                f"{float(r[3])}",
                                f"{float(r[4])}",
                                f"{float(r[5])}",
                            ]
                        )
                        + "\n"
                    )
            return len(new_rows)
        except Exception as e:
            logger.warning(f"[Data] append failed for {path}: {e}")
            # Fallback: rewrite full file (rare)
            df_old = self._read_csv(path)
            df_new = pd.DataFrame(
                rows,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ],
            )
            if df_old is not None:
                df_all = pd.concat([df_old, df_new], ignore_index=True)
            else:
                df_all = df_new
            df_all = self._normalize_df(df_all)
            self._write_csv(
                path,
                df_all.reset_index(drop=False)[
                    [
                        "timestamp",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                    ]
                ],
            )
            return len(df_new)

    def _backfill_from_scratch(self,
                                symbol: str,
                                timeframe: str,
                                *,
                                lookback_bars: int = 900) -> None:
        """
        Fetch recent `lookback_bars` in as few requests as possible
        (<= max_request_bars per call).
        """
        csv_path, meta_path = _build_paths(
            symbol, timeframe, self.exchange.client.id)
        tf_ms = TF_MS.get(timeframe)
        if not tf_ms:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        now_ms = self._now_ms()
        since = now_ms - lookback_bars * tf_ms
        all_rows: List[List[float]] = []
        remaining = lookback_bars

        while remaining > 0:
            limit = min(self.max_request_bars, remaining)
            batch = self._fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not batch:
                break
            all_rows.extend(batch)
            since = int(batch[-1][0]) + tf_ms
            got = len(batch)
            remaining -= got
            if got < limit:
                break
            # tiny pause to stay friendly
            time.sleep(0.2)

        if all_rows:
            df = pd.DataFrame(
                all_rows,
                columns=["timestamp", "open", "high", "low", "close", "volume"])
            self._write_csv(csv_path, df)
            write_json(meta_path,
                       {"last_sync": format_timestamp(int(time.time()))})

    def _fetch_append(self,
                      symbol: str,
                      timeframe: str,
                      *,
                      since: Optional[int]) -> int:
        """
        Loop fetching in chunks <= max_request_bars until caught up to 'now'.
        """
        csv_path, meta_path = _build_paths(
            symbol, timeframe, self.exchange.client.id)
        tf_ms = TF_MS.get(timeframe)
        if not tf_ms:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        # If no since provided, fetch a small recent window
        # (e.g., last 2*max bars)
        if since is None:
            since = self._now_ms() - (self.max_request_bars * 2 * tf_ms)

        appended_total = 0
        while True:
            batch = self._fetch_ohlcv(
                symbol, timeframe, since=since, limit=self.max_request_bars)
            if not batch:
                break
            appended = self._append_to_csv(csv_path, batch)
            appended_total += appended

            last_ts = int(batch[-1][0])
            since = last_ts + tf_ms

            # Stop if we’re close enough to the latest bar
            if self._now_ms() - last_ts <= tf_ms:
                break

            # Be polite to API
            time.sleep(0.2)

        if appended_total > 0:
            write_json(meta_path,
                       {"last_sync": format_timestamp(int(time.time()))})

        return appended_total

    @retry(max_attempts=3, delay=1.0, backoff=2.0, logger=logger)
    def _fetch_ohlcv(self,
                     symbol: str,
                     timeframe: str,
                     *,
                     since: Optional[int],
                     limit: int) -> List[List[float]]:
        """
        Single ccxt fetchOHLCV call with retry.
        Returns list of [ts, o, h, l, c, v].
        """
        # Ensure symbol exists in exchange markets mapping
        sym = symbol
        markets = self.exchange.client.markets
        if symbol not in markets and symbol.replace("/", ":") in markets:
            sym = symbol.replace("/", ":")

        data = self.exchange.client.fetch_ohlcv(
            sym, timeframe=timeframe, since=since, limit=limit)
        # Bybit sometimes returns duplicates or gaps;
        # we’ll clean when appending
        return data or []

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)
