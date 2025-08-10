# modules/data_manager.py

import os
import time
import math
import threading
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple

import pandas as pd

import config
from utils.utilities import ensure_directory, retry, format_timestamp

# Optional deps (graceful fallback if missing)
try:
    import ccxt  # REST (stable)
except Exception:  # pragma: no cover
    ccxt = None

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(getattr(logging, str(getattr(config, "LOG_LEVEL", "INFO")), logging.INFO)
                if isinstance(getattr(config, "LOG_LEVEL", "INFO"), str) else getattr(config, "LOG_LEVEL", logging.INFO))


SUPPORTED_TIMEFRAMES = {"5m": 5, "15m": 15}  # minutes ⇒ integer
MAX_BACKFILL_BARS = 900  # stay under Bybit/ccxt conservative limit


def normalize_symbol_for_storage(symbol: str) -> str:
    """BTC/USDT -> BTCUSDT (safe for filenames and Bybit symbols)."""
    return symbol.replace("/", "").upper()


def human_symbol(symbol: str) -> str:
    """BTCUSDT -> BTC/USDT (for UI/logs)."""
    if "/" in symbol:
        return symbol
    if symbol.endswith("USDT"):
        return f"{symbol[:-4]}/USDT"
    return symbol


def tf_to_ms(tf: str) -> int:
    if tf not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"Only {list(SUPPORTED_TIMEFRAMES)} timeframes are supported")
    return SUPPORTED_TIMEFRAMES[tf] * 60_000


class DataManager:
    """
    Handles:
      - Backfill (<=900 bars) via ccxt REST
      - Lightweight incremental updates (polling) for latest bars
      - Scheduled refresh of "top pairs" (every 60 minutes)
      - CSV storage per symbol/timeframe, deduped by timestamp
    Disk layout:
      {HISTORICAL_DATA_PATH}/{SYMBOL_NO_SLASH}/{timeframe}.csv
    """

    def __init__(self, data_root: Optional[str] = None, exchange_profile: Optional[str] = None):
        self.data_root = data_root or config.HISTORICAL_DATA_PATH
        ensure_directory(self.data_root)

        self.exchange_profile = exchange_profile or config.EXCHANGE_PROFILE
        self.use_testnet = getattr(config, "USE_TESTNET", True)

        self.exchange = self._init_ccxt()

        self._top_pairs_cache: Tuple[float, List[str]] = (0.0, [])
        self._poll_threads: Dict[Tuple[str, str], threading.Thread] = {}
        self._poll_flags: Dict[Tuple[str, str], threading.Event] = {}

    def _init_ccxt(self):
        if ccxt is None:
            logger.warning("ccxt not installed; data fetching is disabled.")
            return None
        ex = ccxt.bybit({
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},  # we’re doing spot klines for learning
        })
        # Testnet only affects trading endpoints; for market data Bybit serves public endpoints.
        # We leave it as-is so it works on either network.
        return ex

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def data_path(self, symbol: str, timeframe: str) -> str:
        sym = normalize_symbol_for_storage(symbol)
        d = os.path.join(self.data_root, sym)
        ensure_directory(d)
        return os.path.join(d, f"{timeframe}.csv")

    @retry(max_attempts=3, delay=1.0, backoff=2.0)
    def load_historical_data(self, symbol: str, timeframe: str = "5m") -> pd.DataFrame:
        """
        Return local OHLCV DataFrame (UTC index).
        Backfills (<=900) on first call; later calls read existing file.
        """
        path = self.data_path(symbol, timeframe)
        if not os.path.exists(path):
            logger.info(f"No local data for {symbol} {timeframe}; backfilling up to {MAX_BACKFILL_BARS} bars…")
            self.backfill(symbol, timeframe, limit=MAX_BACKFILL_BARS)

        try:
            df = pd.read_csv(path)
            if "timestamp" not in df.columns:
                raise ValueError("CSV missing 'timestamp' column")
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)
            return df[["open", "high", "low", "close", "volume"]]
        except Exception as e:
            logger.warning(f"Failed reading {path}: {e}. Rebuilding from scratch.")
            self.backfill(symbol, timeframe, limit=MAX_BACKFILL_BARS)
            return self.load_historical_data(symbol, timeframe)

    def start_incremental_updates(self, symbols: List[str], timeframes: List[str] = ["5m", "15m"], interval_sec: float = 5.0) -> None:
        """
        Starts lightweight polling threads to append only new bars.
        One thread per (symbol, timeframe). Safe to call multiple times (idempotent).
        """
        for sym in symbols:
            for tf in timeframes:
                key = (sym, tf)
                if key in self._poll_threads and self._poll_threads[key].is_alive():
                    continue
                stop_event = threading.Event()
                self._poll_flags[key] = stop_event
                t = threading.Thread(target=self._poll_loop, args=(sym, tf, interval_sec, stop_event), daemon=True)
                self._poll_threads[key] = t
                t.start()
                logger.info(f"Started incremental update thread for {human_symbol(sym)} {tf}")

    def stop_incremental_updates(self):
        for key, ev in self._poll_flags.items():
            ev.set()
        for key, th in list(self._poll_threads.items()):
            th.join(timeout=2.0)
            self._poll_threads.pop(key, None)
        self._poll_flags.clear()

    @retry(max_attempts=3, delay=1.0, backoff=2.0)
    def backfill(self, symbol: str, timeframe: str = "5m", limit: int = MAX_BACKFILL_BARS) -> None:
        """
        Fetch <= limit recent bars via REST and write CSV (overwrite).
        """
        if self.exchange is None:
            raise RuntimeError("ccxt not available for backfill")

        if timeframe not in SUPPORTED_TIMEFRAMES:
            raise ValueError(f"Timeframe {timeframe} not supported")

        # ccxt expects symbols like 'BTC/USDT'
        human = human_symbol(symbol)
        tf = timeframe

        # Fetch most recent window directly (Bybit/ccxt supports 'limit')
        ohlcv = self.exchange.fetch_ohlcv(human, timeframe=tf, limit=limit)
        df = self._ohlcv_to_df(ohlcv)
        path = self.data_path(symbol, timeframe)
        df.to_csv(path, index=True)
        logger.info(f"Backfilled {len(df)} bars into {path}")

    def append_latest(self, symbol: str, timeframe: str = "5m") -> int:
        """
        Append only new bars to CSV using REST. Returns number of new rows appended.
        """
        path = self.data_path(symbol, timeframe)
        last_ts = None
        try:
            if os.path.exists(path):
                tmp = pd.read_csv(path, usecols=["timestamp"], nrows=1)  # fast existence check
                df_all = pd.read_csv(path)
                if not df_all.empty:
                    last_ts = pd.to_datetime(df_all["timestamp"].iloc[-1], utc=True)
        except Exception:
            last_ts = None

        # Compute 'since' a bit before the last bar to be safe (one bar overlap)
        if last_ts is not None:
            since_ms = int(last_ts.timestamp() * 1000 - tf_to_ms(timeframe))
            since_ms = max(0, since_ms)
            limit = 200  # small page (fast)
        else:
            since_ms = None
            limit = MAX_BACKFILL_BARS

        if self.exchange is None:
            logger.debug("ccxt unavailable; cannot append")
            return 0

        human = human_symbol(symbol)
        ohlcv = self.exchange.fetch_ohlcv(human, timeframe=timeframe, since=since_ms, limit=limit)
        df_new = self._ohlcv_to_df(ohlcv)
        if df_new.empty:
            return 0

        # Merge-dedup by timestamp
        if os.path.exists(path):
            try:
                df_old = pd.read_csv(path)
                df_old["timestamp"] = pd.to_datetime(df_old["timestamp"], utc=True)
                merged = pd.concat([df_old, df_new]).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
            except Exception:
                merged = df_new
        else:
            merged = df_new

        # Write back (atomic-ish)
        tmp_path = f"{path}.tmp"
        merged.to_csv(tmp_path, index=True)
        os.replace(tmp_path, path)
        added = max(0, len(merged) - (len(df_old) if os.path.exists(path) else 0))
        return added

    def get_top_pairs(self, quote: str = "USDT", max_pairs: Optional[int] = None) -> List[str]:
        """
        Returns top symbols (by 24h quote volume, then volatility proxy) with / in name.
        Cached for 60 minutes to avoid spamming.
        """
        max_pairs = max_pairs or config.MAX_SIMULATION_PAIRS
        now = time.time()
        cache_ts, cache_syms = self._top_pairs_cache
        if now - cache_ts < 60 * 60 and cache_syms:
            return cache_syms[:max_pairs]

        if self.exchange is None:
            return [config.DEFAULT_SYMBOL]

        try:
            tickers = self.exchange.fetch_tickers()
        except Exception as e:
            logger.warning(f"fetch_tickers failed: {e}")
            return [config.DEFAULT_SYMBOL]

        candidates: List[Tuple[str, float, float]] = []  # (symbol, volUSDT, volProxy)
        for sym, t in tickers.items():
            # Bybit spot symbols use format "BTC/USDT"
            if not sym.endswith(f"/{quote}"):
                continue
            vol_quote = float(t.get("quoteVolume", 0) or 0.0)
            # crude vol proxy using high/low if present
            high = float(t.get("high", 0) or 0.0)
            low = float(t.get("low", 0) or 0.0)
            vol_proxy = (high - low) / high if (high and low and high > 0 and high > low) else 0.0
            if vol_quote <= 0:
                continue
            candidates.append((sym, vol_quote, vol_proxy))

        # Sort by volume first, then volatility proxy
        candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
        out = [c[0] for c in candidates[:max_pairs]]

        # Cache
        self._top_pairs_cache = (now, out)
        return out

    # ─────────────────────────────────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────────────────────────────────

    def _poll_loop(self, symbol: str, timeframe: str, interval_sec: float, stop_event: threading.Event):
        """
        Polls REST for the newest bar and appends it. Tiny and cheap.
        """
        path = self.data_path(symbol, timeframe)
        logger.info(f"[poll] {human_symbol(symbol)} {timeframe} → {path}")
        while not stop_event.is_set():
            try:
                added = self.append_latest(symbol, timeframe)
                if added:
                    logger.debug(f"[poll] {human_symbol(symbol)} {timeframe}: +{added} bars")
            except Exception as e:
                logger.debug(f"[poll] append failed for {symbol} {timeframe}: {e}")
            stop_event.wait(interval_sec)

    @staticmethod
    def _ohlcv_to_df(ohlcv: List[List[float]]) -> pd.DataFrame:
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        if not ohlcv:
            return pd.DataFrame(columns=cols).set_index("timestamp")
        df = pd.DataFrame(ohlcv, columns=cols)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        return df[["open", "high", "low", "close", "volume"]]
