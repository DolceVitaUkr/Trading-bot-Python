# modules/data_manager.py

import asyncio
import csv
import gzip
import io
import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import websockets

import config
from modules.exchange import ExchangeAPI, normalize_symbol
from utils.utilities import ensure_directory, retry

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(getattr(logging, str(getattr(config, "LOG_LEVEL", "INFO")), logging.INFO)
                if isinstance(getattr(config, "LOG_LEVEL", "INFO"), str) else getattr(config, "LOG_LEVEL", logging.INFO))


# ────────────────────────────────────────────────────────────────────────────────
# DataManager: persistent OHLCV store + Bybit WS backfill/append
# ────────────────────────────────────────────────────────────────────────────────

TIMEFRAME_MINUTES = {
    "5m": 5,
    "15m": 15,
}

KLINE_BACKFILL_HOURS = int(os.getenv("KLINE_BACKFILL_HOURS", "72")) if "KLINE_BACKFILL_HOURS" in os.environ else 72


def _csv_path(symbol: str, timeframe: str) -> str:
    base = getattr(config, "HISTORICAL_DATA_PATH", "historical_data")
    sym = normalize_symbol(symbol)
    return os.path.join(base, sym, f"{timeframe}.csv")


class DataManager:
    """
    Responsibilities:
      - Keep per-symbol, per-timeframe CSV of OHLCV with headers: ts,open,high,low,close,volume
      - On load: append-only style; backfill only missing bars (limit<=900 per request)
      - WebSocket: subscribe to kline 5m/15m; append closed bars (confirm tick 'confirm' flag)
      - Top pairs refresh every N minutes
    """

    def __init__(self, exchange: Optional[ExchangeAPI] = None):
        self.exchange = exchange or ExchangeAPI()
        self._ws_task: Optional[asyncio.Task] = None
        self._ws_symbols: List[str] = []
        self._ws_intervals: List[str] = []
        self._stop_ws = asyncio.Event()

        # in-memory last-known closed candle timestamp by (symbol,timeframe)
        self._last_ts: Dict[Tuple[str, str], int] = {}

        # debounce map to avoid frequent backfill while WS is alive
        self._last_backfill_wallclock: Dict[Tuple[str, str], float] = defaultdict(lambda: 0.0)

    # ───────────────────────────────
    # Public API
    # ───────────────────────────────

    @retry(max_attempts=3, delay=1.5, backoff=2.0)
    def load_historical_data(self, symbol: str, timeframe: str = "5m") -> pd.DataFrame:
        """
        Return DataFrame indexed by datetime (UTC), columns: [open,high,low,close,volume]
        """
        path = _csv_path(symbol, timeframe)
        if not os.path.exists(path):
            # ensure directory and create empty file with header
            ensure_directory(os.path.dirname(path))
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["ts", "open", "high", "low", "close", "volume"])

        df = pd.read_csv(path)
        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = df.drop_duplicates(subset=["ts"]).sort_values("ts")
        df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        return df

    def backfill_missing(
        self,
        symbol: str,
        timeframe: str = "5m",
        hours: int = KLINE_BACKFILL_HOURS,
        max_chunk: int = 900,
    ) -> int:
        """
        Append only missing bars. Uses ccxt.fetch_ohlcv (limit<=900).
        Returns number of appended bars.
        """
        path = _csv_path(symbol, timeframe)
        ensure_directory(os.path.dirname(path))
        tf_minutes = TIMEFRAME_MINUTES.get(timeframe)
        if not tf_minutes:
            raise ValueError("Unsupported timeframe; use '5m' or '15m'.")

        # Determine last timestamp in file
        last_ts = None
        if os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                # Read last few lines quickly
                with open(path, "rb") as f:
                    f.seek(max(f.seek(0, 2) - 2048, 0))
                    tail = f.read().decode("utf-8", errors="ignore")
                *_, last_line = [ln for ln in tail.splitlines() if ln.strip()]
                if last_line and last_line.replace(",", "").strip() != "tsopehighlowclosevolume":
                    parts = last_line.split(",")
                    if parts and parts[0].isdigit():
                        last_ts = int(parts[0])
            except Exception:
                last_ts = None

        now_ms = int(time.time() * 1000)
        since_ms = now_ms - int(hours * 3600 * 1000)
        if last_ts:
            # Shift since to last_ts + one bar
            since_ms = max(since_ms, last_ts + tf_minutes * 60 * 1000)

        appended = 0
        if since_ms >= now_ms - tf_minutes * 60 * 1000:
            # Nothing to fetch
            return 0

        # Fetch in one chunk (bounded by limit)
        data = self.exchange.fetch_klines(symbol, timeframe=timeframe, since_ms=since_ms, limit=max_chunk)
        if not data:
            return 0

        # Append
        new_rows = []
        for ts, o, h, l, c, v in data:
            if last_ts and ts <= last_ts:
                continue
            new_rows.append([ts, o, h, l, c, v])

        if not new_rows:
            return 0

        file_exists = os.path.exists(path) and os.path.getsize(path) > 0
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not file_exists:
                w.writerow(["ts", "open", "high", "low", "close", "volume"])
            w.writerows(new_rows)

        self._last_ts[(normalize_symbol(symbol), timeframe)] = new_rows[-1][0]
        appended = len(new_rows)
        logger.info(f"[Data] Backfilled {appended} bars for {normalize_symbol(symbol)} {timeframe}")
        return appended

    async def start_ws(self, symbols: List[str], intervals: List[str]) -> None:
        """
        Launch a WS reader for Bybit spot klines and append CLOSED candles to CSV.
        Re-runnable; will restart on disconnect.
        """
        self._ws_symbols = [normalize_symbol(s) for s in symbols]
        self._ws_intervals = intervals[:]
        self._stop_ws.clear()

        if self._ws_task and not self._ws_task.done():
            # already running
            return

        self._ws_task = asyncio.create_task(self._ws_loop())

    async def stop_ws(self) -> None:
        self._stop_ws.set()
        if self._ws_task:
            try:
                await asyncio.wait_for(self._ws_task, timeout=5)
            except Exception:
                pass

    # ───────────────────────────────
    # Top pairs refresh
    # ───────────────────────────────

    def fetch_top_pairs(self, max_pairs: int = None) -> List[str]:
        max_pairs = max_pairs or int(getattr(config, "MAX_SIMULATION_PAIRS", 5))
        return self.exchange.fetch_top_pairs(max_pairs=max_pairs)

    # ───────────────────────────────
    # Internals
    # ───────────────────────────────

    async def _ws_loop(self) -> None:
        url = ExchangeAPI.ws_public_endpoint_spot()
        subs = []
        for tf in self._ws_intervals:
            minutes = TIMEFRAME_MINUTES.get(tf)
            if not minutes:
                continue
            for s in self._ws_symbols:
                subs.append({"topic": f"kline.{minutes}.{s}"})

        if not subs:
            logger.warning("WS loop started with no subscriptions.")
            return

        while not self._stop_ws.is_set():
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                    # subscribe
                    sub_msg = {
                        "op": "subscribe",
                        "args": [x["topic"] for x in subs],
                    }
                    await ws.send(json.dumps(sub_msg))
                    logger.info(f"[WS] Subscribed to {len(subs)} topics on {url}")

                    # read loop
                    while not self._stop_ws.is_set():
                        raw = await asyncio.wait_for(ws.recv(), timeout=60)
                        if isinstance(raw, (bytes, bytearray)):
                            raw = raw.decode("utf-8", errors="ignore")
                        msg = json.loads(raw)

                        # heartbeat / info messages
                        if "type" in msg and msg["type"] in ("welcome", "pong"):
                            continue
                        if "success" in msg and msg.get("op") == "subscribe":
                            continue

                        topic = msg.get("topic") or ""
                        data = msg.get("data")
                        if not topic or not data:
                            continue

                        if not topic.startswith("kline."):
                            continue

                        # topic format: kline.<minutes>.<symbol>
                        try:
                            _, minutes, symbol = topic.split(".")
                            minutes = int(minutes)
                        except Exception:
                            continue
                        tf = "5m" if minutes == 5 else "15m" if minutes == 15 else None
                        if not tf:
                            continue

                        # Bybit WS sends a list; closed candles have "confirm": True
                        if isinstance(data, dict):
                            data = [data]
                        for k in data:
                            # field names: start, end, interval, open, high, low, close, volume, turnover, confirm
                            if not k.get("confirm", False):
                                # only append when candle closes
                                continue
                            ts_ms = int(k.get("start", 0))
                            o = float(k.get("open", 0))
                            h = float(k.get("high", 0))
                            l = float(k.get("low", 0))
                            c = float(k.get("close", 0))
                            v = float(k.get("volume", 0))
                            self._append_bar(symbol, tf, [ts_ms, o, h, l, c, v])

            except asyncio.TimeoutError:
                logger.info("[WS] timeout; reconnecting…")
            except Exception as e:
                logger.warning(f"[WS] error: {e} (reconnecting in 3s)")
                await asyncio.sleep(3)

    def _append_bar(self, symbol: str, timeframe: str, row: List[Any]) -> None:
        sym = normalize_symbol(symbol)
        path = _csv_path(sym, timeframe)
        ensure_directory(os.path.dirname(path))

        # Avoid duplicates by ts
        ts = int(row[0])
        last_key = (sym, timeframe)
        last_ts = self._last_ts.get(last_key)

        if last_ts and ts <= last_ts:
            return

        # small disk write
        file_exists = os.path.exists(path) and os.path.getsize(path) > 0
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not file_exists:
                w.writerow(["ts", "open", "high", "low", "close", "volume"])
            w.writerow(row)

        self._last_ts[last_key] = ts

    # Convenience: refresh for a list of symbols (backfill + optional ws)
    async def ensure_ready(
        self,
        symbols: List[str],
        intervals: List[str] = ("5m", "15m"),
        backfill_hours: int = KLINE_BACKFILL_HOURS,
        start_ws: bool = True,
    ) -> None:
        # backfill sequentially to respect rate limits
        for s in symbols:
            for tf in intervals:
                # throttle backfill per (s,tf) no more than every 2 minutes
                key = (normalize_symbol(s), tf)
                now = time.time()
                if now - self._last_backfill_wallclock[key] < 120:
                    continue
                self.backfill_missing(s, timeframe=tf, hours=backfill_hours, max_chunk=900)
                self._last_backfill_wallclock[key] = now

        if start_ws:
            await self.start_ws(symbols, list(intervals))
