# modules/data_manager.py

import os
import time
import json
import math
import gzip
import queue
import atexit
import signal
import threading
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple, Any

import pandas as pd

import config
from utils.utilities import ensure_directory, retry, utc_now, format_timestamp

try:
    import requests
except Exception:
    requests = None

try:
    import websockets
    import asyncio
except Exception:
    websockets = None
    asyncio = None


BYBIT_API = "https://api.bybit.com"
WS_PUBLIC = "wss://stream.bybit.com/v5/public/spot"  # change to linear if using futures

# Map internal timeframe -> Bybit interval string
INTERVAL_MAP = {
    "5m": "5",
    "15m": "15",
}

DEFAULT_CATEGORY = getattr(config, "BYBIT_CATEGORY", "spot")  # "spot" or "linear"
DATA_ROOT = getattr(config, "DATA_ROOT", "./data/bybit")
BACKFILL_BARS_5M = int(getattr(config, "BACKFILL_BARS_5M", 10000))   # ~34 days
BACKFILL_BARS_15M = int(getattr(config, "BACKFILL_BARS_15M", 8000))  # ~83 days
REST_PAGE_LIMIT = 900  # stay under Bybit max 1000
TOP_PAIRS_COUNT = int(getattr(config, "TOP_PAIRS_COUNT", 10))
TOP_PAIRS_REFRESH_MIN = int(getattr(config, "TOP_PAIRS_REFRESH_MIN", 60))
REQUEST_SLEEP = float(getattr(config, "REQUEST_SLEEP", 0.15))  # polite pacing


def _now_ms() -> int:
    return int(time.time() * 1000)


def _to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def normalize_symbol_for_rest(sym: str) -> str:
    """BTC/USDT -> BTCUSDT"""
    return sym.replace("/", "").upper()


def pretty_symbol(sym: str) -> str:
    """BTCUSDT -> BTC/USDT"""
    if "/" in sym:
        return sym.upper()
    if sym.endswith("USDT"):
        return f"{sym[:-4]}/USDT"
    return sym.upper()


def _tf_path(timeframe: str) -> str:
    return os.path.join(DATA_ROOT, timeframe)


def _file_path(symbol: str, timeframe: str) -> str:
    ensure_directory(_tf_path(timeframe))
    return os.path.join(_tf_path(timeframe), f"{normalize_symbol_for_rest(symbol)}.parquet")


def _sleep_polite():
    time.sleep(REQUEST_SLEEP)


class DataManager:
    """
    Bybit market data manager:
      - Parquet storage per symbol/timeframe
      - REST backfill (<=900/page) + incremental sync
      - WebSocket append on CLOSED candle
      - Top pairs hourly refresh with 24h volume rank and 2h/24h change (spike watch)
    """

    def __init__(self, category: str = DEFAULT_CATEGORY, timeframes: Optional[List[str]] = None):
        self.category = category
        self.timeframes = timeframes or ["15m", "5m"]

        self._stop = threading.Event()
        self._ws_thread: Optional[threading.Thread] = None
        self._ws_symbols: List[str] = []  # normalized REST style (BTCUSDT)
        self._ws_connected = False

        self._top_pairs_cache: List[str] = []
        self._top_pairs_ts: float = 0.0
        self._spike_state: Dict[str, Dict[str, float]] = {}  # symbol -> last pct changes snapshot

        # in-process queue for closed candles from WS
        self._ws_queue: "queue.Queue[Tuple[str, str, dict]]" = queue.Queue()

        atexit.register(self.shutdown)
        try:
            signal.signal(signal.SIGINT, lambda *_: self.shutdown())
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def start_ws(self, symbols: List[str]) -> None:
        """
        Start WebSocket (if available) to receive kline CLOSE events and append to Parquet.
        `symbols` should be pretty style (BTC/USDT). We'll normalize internally.
        """
        if websockets is None or asyncio is None:
            # websockets lib not available; fallback to REST only
            return
        rest_syms = [normalize_symbol_for_rest(s) for s in symbols]
        self._ws_symbols = sorted(set(rest_syms))
        if self._ws_thread and self._ws_thread.is_alive():
            # already running; update subs by restarting
            self.stop_ws()
        self._ws_thread = threading.Thread(target=self._ws_worker, daemon=True)
        self._ws_thread.start()

    def stop_ws(self) -> None:
        self._stop.set()
        if self._ws_thread and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=5)
        self._stop.clear()

    def shutdown(self):
        self.stop_ws()

    def load_historical_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Return a DataFrame with columns: open, high, low, close, volume (index = datetime UTC).
        Ensures local Parquet exists & is fresh up to near-now (by REST incremental if WS not running).
        """
        assert timeframe in INTERVAL_MAP, f"Unsupported timeframe: {timeframe}"
        path = _file_path(symbol, timeframe)

        if os.path.exists(path):
            df = pd.read_parquet(path)
        else:
            df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            df.index.name = "timestamp"

        # Incremental sync (REST) to fill gaps
        latest_ts = int(df.index[-1].timestamp() * 1000) if not df.empty else None
        needed_bars = BACKFILL_BARS_5M if timeframe == "5m" else BACKFILL_BARS_15M
        self._sync_rest(symbol, timeframe, latest_ts=latest_ts, min_total=needed_bars)

        # Reload
        if os.path.exists(path):
            df = pd.read_parquet(path)
        return df

    def ensure_symbols_ready(self, symbols: List[str]) -> None:
        """
        Make sure local data exists and is up-to-date for each symbol/timeframe.
        Keeps initial backfill manageable & then incremental.
        """
        for sym in symbols:
            for tf in self.timeframes:
                try:
                    self.load_historical_data(sym, tf)
                except Exception:
                    # don't crash whole loop
                    pass

    def get_top_pairs(self) -> List[str]:
        """
        Return cached Top Pairs list (pretty style symbols), refreshing hourly by 24h volume & pct change.
        """
        now = time.time()
        if not self._top_pairs_cache or (now - self._top_pairs_ts) >= TOP_PAIRS_REFRESH_MIN * 60:
            pairs = self._refresh_top_pairs()
            self._top_pairs_cache = pairs
            self._top_pairs_ts = now
        return self._top_pairs_cache[:]

    # ─────────────────────────────────────────────────────────────────────
    # Internals: REST Backfill & Incremental
    # ─────────────────────────────────────────────────────────────────────

    def _sync_rest(self, symbol: str, timeframe: str, latest_ts: Optional[int], min_total: int) -> None:
        """
        Ensure we have at least `min_total` bars and up to latest-1 closed bar, pulling only missing pages.
        """
        path = _file_path(symbol, timeframe)
        rest_sym = normalize_symbol_for_rest(symbol)
        interval = INTERVAL_MAP[timeframe]

        existing = pd.DataFrame()
        if os.path.exists(path):
            existing = pd.read_parquet(path)

        have = len(existing)
        now_ms = _now_ms()
        tf_ms = int(INTERVAL_MAP[timeframe]) * 60 * 1000
        last_wanted_end = now_ms - tf_ms  # last fully closed bar end

        # Determine starting point
        if latest_ts is None or have == 0:
            # fresh backfill: request min_total in pages backward from last_wanted_end
            required = min_total
            end = last_wanted_end
            frames = []
            while required > 0:
                limit = min(REST_PAGE_LIMIT, required)
                start = end - limit * tf_ms
                chunk = self._fetch_klines(rest_sym, interval, start, end)
                if chunk.empty:
                    break
                frames.append(chunk)
                required -= len(chunk)
                end = int(chunk.index[0].timestamp() * 1000)  # page backward
                _sleep_polite()
            if frames:
                newdf = pd.concat(frames, axis=0).sort_index()
            else:
                newdf = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
                newdf.index.name = "timestamp"
            # write
            if not newdf.empty:
                newdf = newdf[~newdf.index.duplicated(keep="last")]
                newdf.to_parquet(path)
            return

        # incremental forward
        start = latest_ts + tf_ms
        if start > last_wanted_end:
            return  # already up-to-date

        frames = []
        while start <= last_wanted_end:
            # fetch a page forward
            limit = min(REST_PAGE_LIMIT, math.ceil((last_wanted_end - start) / tf_ms))
            end = start + limit * tf_ms
            chunk = self._fetch_klines(rest_sym, interval, start, end)
            if chunk.empty:
                break
            frames.append(chunk)
            start = int(chunk.index[-1].timestamp() * 1000) + tf_ms
            _sleep_polite()

        if frames:
            add = pd.concat(frames, axis=0).sort_index()
            add = add[~add.index.duplicated(keep="last")]
            combined = pd.concat([existing, add], axis=0).sort_index()
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.to_parquet(path)

    @retry(max_attempts=5, delay=0.3, backoff=1.8)
    def _fetch_klines(self, symbol_rest: str, interval: str, start: int, end: int) -> pd.DataFrame:
        """
        Fetch klines from Bybit /v5/market/kline within [start, end), respecting limit 900/page.
        """
        if requests is None:
            raise RuntimeError("requests library not available")
        url = f"{BYBIT_API}/v5/market/kline"
        params = {
            "category": self.category,
            "symbol": symbol_rest,
            "interval": interval,
            "start": start,
            "end": end,
            "limit": REST_PAGE_LIMIT,
        }
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            raise RuntimeError(f"Bybit kline error {resp.status_code}: {resp.text}")
        data = resp.json()
        if str(data.get("retCode")) != "0":
            raise RuntimeError(f"Bybit kline retCode={data.get('retCode')} msg={data.get('retMsg')}")
        rows = data.get("result", {}).get("list", []) or []
        # rows are typically newest first; normalize to oldest->newest
        rows.sort(key=lambda x: int(x[0]))
        # Bybit v5 list cols: startTime, open, high, low, close, volume, turnover
        records = []
        for r in rows:
            ts = int(r[0])
            records.append({
                "timestamp": datetime.fromtimestamp(ts / 1000, tz=timezone.utc),
                "open": float(r[1]),
                "high": float(r[2]),
                "low": float(r[3]),
                "close": float(r[4]),
                "volume": float(r[5]),
            })
        df = pd.DataFrame.from_records(records)
        if df.empty:
            df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        df.set_index("timestamp", inplace=True)
        return df

    # ─────────────────────────────────────────────────────────────────────
    # WebSocket handling
    # ─────────────────────────────────────────────────────────────────────

    def _ws_worker(self):
        """
        Run WebSocket listener in a thread, push CLOSED candles into queue, and writer loop appends to parquet.
        """
        self._ws_connected = False

        def writer_loop():
            while not self._stop.is_set():
                try:
                    sym_rest, tf, payload = self._ws_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                try:
                    ts_ms = int(payload["start"])
                    # process only on confirm close
                    if not payload.get("confirm"):
                        continue
                    df = pd.DataFrame([{
                        "timestamp": datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc),
                        "open": float(payload["open"]),
                        "high": float(payload["high"]),
                        "low": float(payload["low"]),
                        "close": float(payload["close"]),
                        "volume": float(payload["volume"]),
                    }]).set_index("timestamp")
                    path = _file_path(pretty_symbol(sym_rest), tf)
                    # append
                    if os.path.exists(path):
                        cur = pd.read_parquet(path)
                        merged = pd.concat([cur, df], axis=0).sort_index()
                        merged = merged[~merged.index.duplicated(keep="last")]
                        merged.to_parquet(path)
                    else:
                        df.to_parquet(path)
                except Exception:
                    # swallow in writer to keep loop alive
                    pass

        writer = threading.Thread(target=writer_loop, daemon=True)
        writer.start()

        if websockets is None or asyncio is None:
            return

        async def ws_main():
            self._ws_connected = True
            try:
                async with websockets.connect(WS_PUBLIC, ping_interval=20, ping_timeout=20) as ws:
                    # subscribe to klines for requested symbols on 5 and 15
                    topics = []
                    for sym in self._ws_symbols:
                        for tf in self.timeframes:
                            interval = INTERVAL_MAP[tf]
                            topics.append(f"kline.{interval}.{sym}")
                    sub = {"op": "subscribe", "args": topics}
                    await ws.send(json.dumps(sub))

                    while not self._stop.is_set():
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=5)
                        except asyncio.TimeoutError:
                            # keep-alive: Bybit will ping/pong automatically, continue
                            continue
                        data = json.loads(msg)
                        # kline format: {"topic":"kline.5.BTCUSDT","data":[{...}],"type":"snapshot/update"}
                        topic = data.get("topic", "")
                        if not topic.startswith("kline."):
                            continue
                        parts = topic.split(".")
                        if len(parts) != 3:
                            continue
                        interval, sym_rest = parts[1], parts[2]
                        tf = "5m" if interval == "5" else "15m" if interval == "15" else None
                        if tf not in self.timeframes:
                            continue
                        for item in data.get("data", []):
                            # Bybit sends both live & confirm on close; we only enqueue confirm
                            self._ws_queue.put((sym_rest, tf, item))
            finally:
                self._ws_connected = False

        # run the async loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(ws_main())
        finally:
            try:
                loop.stop()
            except Exception:
                pass
            try:
                loop.close()
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────────────
    # Top pairs + spike detection
    # ─────────────────────────────────────────────────────────────────────

    @retry(max_attempts=4, delay=0.4, backoff=2.0)
    def _refresh_top_pairs(self) -> List[str]:
        """
        Rank USDT-quoted pairs by 24h volume; track 2h/24h change to flag spikes.
        Returns pretty symbols list.
        """
        if requests is None:
            return ["BTC/USDT", "ETH/USDT"]

        url = f"{BYBIT_API}/v5/market/tickers"
        params = {"category": self.category}
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            raise RuntimeError(f"Bybit tickers error {resp.status_code}: {resp.text}")
        data = resp.json()
        if str(data.get("retCode")) != "0":
            raise RuntimeError(f"Bybit tickers retCode={data.get('retCode')} msg={data.get('retMsg')}")

        items = data.get("result", {}).get("list", []) or []
        # filter USDT quote
        usdt = [it for it in items if str(it.get("symbol", "")).upper().endswith("USDT")]

        # parse fields
        parsed: List[Tuple[str, float, float, float]] = []
        for it in usdt:
            sym = pretty_symbol(it["symbol"])
            vol24 = float(it.get("turnover24h", it.get("volume24h", 0.0)) or 0.0)
            chg24 = float(it.get("price24hPcnt", 0.0) or 0.0) * 100.0  # often in fraction
            chg2h = float(it.get("usdIndexPrice2hPcnt", it.get("lastPrice2hPcnt", 0.0)) or 0.0) * 100.0
            parsed.append((sym, vol24, chg2h, chg24))

        # sort by 24h volume desc
        parsed.sort(key=lambda x: x[1], reverse=True)
        top = parsed[:TOP_PAIRS_COUNT]

        # spike detection vs previous snapshot
        for sym, vol, chg2, chg24 in top:
            last = self._spike_state.get(sym, {})
            last2 = last.get("chg2h", 0.0)
            last24 = last.get("chg24h", 0.0)
            spike_2h = chg2 - last2
            spike_24h = chg24 - last24
            self._spike_state[sym] = {"chg2h": chg2, "chg24h": chg24, "last_spike2h": spike_2h, "last_spike24h": spike_24h}
        return [sym for sym, *_ in top]
