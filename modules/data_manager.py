# modules/data_manager.py
from __future__ import annotations

import os
import json
import time
import logging
import pathlib
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Any, Tuple

import pandas as pd
import requests

import config
from utils.utilities import ensure_directory, retry

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    logger.addHandler(handler)

# Bybit v5 market kline endpoint (public)
BYBIT_V5_KLINE = (getattr(config, "BYBIT_BASE_URL", "https://api.bybit.com").rstrip("/") + "/v5/market/kline")


def timeframe_to_minutes(tf: str) -> int:
    unit_map = {"m": 1, "h": 60, "d": 1440, "w": 10080}
    unit = tf[-1]
    num = int(tf[:-1])
    return num * unit_map.get(unit, 1)


def _utc_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _ms_to_utc(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def _floor_to_tf(ts: int, tf_ms: int) -> int:
    return ts - (ts % tf_ms)


class DataManager:
    """
    Historical OHLCV manager for Bybit v5 (HTTP).
    Partitioned parquet by day:
      {HISTORICAL_DATA_PATH}/{SYMBOL}/{TF}/YYYY-MM-DD.parquet
    """

    TF_MAP = {
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
    }

    INTERVAL_MAP = {
        "5m": "5",
        "15m": "15",
    }

    DEFAULT_CATEGORY = "spot"  # "spot" or "linear"/"inverse"

    _CONNECT_TIMEOUT = 3.05
    _READ_TIMEOUT = 10.0
    _REQS_PER_MIN_BUDGET = 40
    _PAGE_LIMIT = 1000  # v5 max rows/page

    def __init__(self, test_mode: bool = False):
        self.data_folder = config.HISTORICAL_DATA_PATH
        ensure_directory(self.data_folder)
        self.data_root = pathlib.Path(self.data_folder)

        self.cache: Dict[str, pd.DataFrame] = {}
        self._rate_bucket: List[float] = []
        self._meta_cache: Dict[str, Dict[str, Any]] = {}

    # ---------------- Public API ---------------- #

    def update_klines(
        self,
        symbol: str,
        timeframe: str,
        klines: Optional[List[list]] = None,
        category: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> bool:
        try:
            self._validate_tf(timeframe)
            category = (category or self.DEFAULT_CATEGORY).lower()

            if klines:
                new_df = self._process_data(klines)
                self._write_partition_df(symbol, timeframe, new_df)
                if not new_df.empty:
                    last_ts = int(new_df.index[-1].value // 10**6)
                    meta = self._load_meta(symbol, timeframe)
                    if meta.get("last_ts") is None or last_ts > int(meta["last_ts"]):
                        meta["last_ts"] = last_ts
                        self._save_meta(symbol, timeframe, meta)
                return True

            now_utc = datetime.now(timezone.utc)
            if until is None:
                until = now_utc

            last_ts = self.last_timestamp(symbol, timeframe)
            tf_ms = self.TF_MAP[timeframe]
            if since is None:
                if last_ts is None:
                    days = 120 if timeframe == "15m" else 30
                    since = now_utc - timedelta(days=days)
                else:
                    since = _ms_to_utc(last_ts - (3 * tf_ms))

            rows, parts = self._fetch_and_persist(
                symbol=symbol,
                timeframe=timeframe,
                category=category,
                start=since,
                end=until,
            )
            logger.info(f"[{symbol} {timeframe}] added rows={rows}, partitions={parts}")
            return True
        except Exception as e:
            logger.exception(f"update_klines failed: {e}")
            return False

    def load_historical_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        key = f"{symbol}_{timeframe}_all"
        if key in self.cache:
            return self.cache[key]
        df = self._read_all(symbol, timeframe)
        self.cache[key] = df
        return df

    def load_window(
        self,
        symbol: str,
        timeframe: str,
        lookback: int = 500,
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        self._validate_tf(timeframe)
        end_time = end_time or datetime.now(timezone.utc)
        tf_ms = self.TF_MAP[timeframe]
        start_time = end_time - timedelta(milliseconds=lookback * tf_ms)
        df = self._read_range(symbol, timeframe, start_time, end_time)
        return self._finalize_df(df)

    def last_timestamp(self, symbol: str, timeframe: str) -> Optional[int]:
        meta = self._load_meta(symbol, timeframe)
        ts = meta.get("last_ts")
        if ts is not None:
            return int(ts)
        tail = self._read_tail(symbol, timeframe, 1)
        if tail is None or tail.empty:
            return None
        return int(tail.index[-1].value // 10**6)

    def has_gap(self, symbol: str, timeframe: str) -> bool:
        self._validate_tf(timeframe)
        tf_ms = self.TF_MAP[timeframe]
        df = self.load_window(symbol, timeframe, lookback=220)
        if df.empty:
            return False
        diffs = df.index.to_series().diff().dropna().view("i8") // 10**6
        return (diffs != tf_ms).any()

    def upsert_bar(self, symbol: str, timeframe: str, bar: Dict[str, Any], category: Optional[str] = None) -> None:
        self._validate_tf(timeframe)
        ts = int(bar["timestamp"])
        tf_ms = self.TF_MAP[timeframe]
        if ts % tf_ms != 0:
            raise ValueError(f"Bar ts {ts} not aligned to {timeframe}")

        df = (
            pd.DataFrame([bar])
            .set_index(pd.to_datetime([ts], unit="ms", utc=True))
            .astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
        )
        df.index.name = "timestamp"
        self._write_partition_df(symbol, timeframe, df)

        meta = self._load_meta(symbol, timeframe)
        if meta.get("last_ts") is None or ts > int(meta["last_ts"]):
            meta["last_ts"] = ts
            self._save_meta(symbol, timeframe, meta)

    # ---------------- Fetch & Persist ---------------- #

    @retry(times=3, backoff=2)
    def _fetch_page(
        self,
        symbol: str,
        timeframe: str,
        category: str,
        start_ms: int,
        end_ms: int,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {
            "category": category,
            "symbol": symbol.replace("/", ""),
            "interval": self.INTERVAL_MAP[timeframe],
            "start": start_ms,
            "end": end_ms,
            "limit": str(self._PAGE_LIMIT),
        }
        resp = requests.get(
            BYBIT_V5_KLINE,
            params=params,
            timeout=(self._CONNECT_TIMEOUT, self._READ_TIMEOUT),
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("retCode") != 0:
            raise RuntimeError(f"Bybit retCode={data.get('retCode')} msg={data.get('retMsg')}")

        rows = data.get("result", {}).get("list", []) or []
        if not rows:
            return []

        recs = []
        for k in rows:
            ts = int(k[0])
            recs.append({
                "timestamp": ts,
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            })
        recs.sort(key=lambda x: x["timestamp"])
        return recs

    def _fetch_and_persist(
        self,
        symbol: str,
        timeframe: str,
        category: str,
        start: datetime,
        end: datetime,
    ) -> Tuple[int, int]:
        start_ms = _utc_ms(start)
        end_ms = _utc_ms(end)
        tf_ms = self.TF_MAP[timeframe]
        if end_ms <= start_ms:
            return 0, 0

        cursor = _floor_to_tf(start_ms, tf_ms)
        end_ms = _floor_to_tf(end_ms, tf_ms)

        rows_added = 0
        partitions_touched = set()

        while cursor <= end_ms:
            self._throttle_bucket()
            page_end = min(cursor + (self._PAGE_LIMIT - 1) * tf_ms, end_ms)
            recs = self._fetch_page(symbol, timeframe, category, cursor, page_end)
            if not recs:
                cursor = page_end + tf_ms
                continue

            df = (
                pd.DataFrame(recs)
                .drop_duplicates(subset=["timestamp"])
                .set_index(pd.to_datetime([r["timestamp"] for r in recs], unit="ms", utc=True))
                [["open", "high", "low", "close", "volume"]]
            )
            df.index.name = "timestamp"

            aligned = ((df.index.view('i8') // 10**6) % tf_ms == 0).all()
            if not aligned:
                raise RuntimeError("Fetched bars not aligned to timeframe")

            touched = self._write_partition_df(symbol, timeframe, df)
            partitions_touched |= touched
            rows_added += len(df)

            last_ts = int(df.index[-1].value // 10**6)
            meta = self._load_meta(symbol, timeframe)
            if meta.get("last_ts") is None or last_ts > int(meta["last_ts"]):
                meta["last_ts"] = last_ts
                self._save_meta(symbol, timeframe, meta)

            cursor = last_ts + tf_ms

        return rows_added, len(partitions_touched)

    # ---------------- IO Layout: partitioned parquet ---------------- #

    def _symbol_dir(self, symbol: str, timeframe: str) -> pathlib.Path:
        return self.data_root / self._sanitize_symbol(symbol) / timeframe

    def _meta_path(self, symbol: str, timeframe: str) -> pathlib.Path:
        return self._symbol_dir(symbol, timeframe) / "_meta.json"

    def _day_path(self, symbol: str, timeframe: str, day: datetime) -> pathlib.Path:
        return self._symbol_dir(symbol, timeframe) / f"{day.strftime('%Y-%m-%d')}.parquet"

    def _write_partition_df(self, symbol: str, timeframe: str, df: pd.DataFrame) -> set:
        if df is None or df.empty:
            return set()

        symdir = self._symbol_dir(symbol, timeframe)
        symdir.mkdir(parents=True, exist_ok=True)

        touched = set()
        for day, g in df.groupby(df.index.date):
            day_dt = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc)
            path = self._day_path(symbol, timeframe, day_dt)

            if path.exists():
                old = pd.read_parquet(path)
                if old.index.tz is None:
                    old.index = old.index.tz_localize(timezone.utc)
                merged = pd.concat([old, g], axis=0)
            else:
                merged = g

            merged = self._finalize_df(merged)
            tmp = merged.copy()
            tmp.index = tmp.index.tz_localize(None)
            tmp.to_parquet(path, engine="pyarrow", compression="snappy")
            touched.add(path.name)

        return touched

    # ---------------- Readers ---------------- #

    def _read_all(self, symbol: str, timeframe: str) -> pd.DataFrame:
        symdir = self._symbol_dir(symbol, timeframe)
        if not symdir.exists():
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        dfs = []
        for fn in sorted(os.listdir(symdir)):
            if not fn.endswith(".parquet"):
                continue
            try:
                d = pd.read_parquet(symdir / fn)
                if d.index.tz is None:
                    d.index = d.index.tz_localize(timezone.utc)
                dfs.append(d)
            except Exception as e:
                logger.error(f"Failed reading {symdir / fn}: {e}")

        if not dfs:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.concat(dfs, axis=0)
        return self._finalize_df(df)

    def _read_range(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
        symdir = self._symbol_dir(symbol, timeframe)
        if not symdir.exists():
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        day = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
        end_day = datetime(end.year, end.month, end.day, tzinfo=timezone.utc)

        dfs = []
        while day <= end_day:
            path = self._day_path(symbol, timeframe, day)
            if path.exists():
                try:
                    d = pd.read_parquet(path)
                    if d.index.tz is None:
                        d.index = d.index.tz_localize(timezone.utc)
                    dfs.append(d)
                except Exception as e:
                    logger.error(f"Failed reading {path}: {e}")
            day += timedelta(days=1)

        if not dfs:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.concat(dfs, axis=0)
        df = df[(df.index >= start) & (df.index <= end)]
        return self._finalize_df(df)

    def _read_tail(self, symbol: str, timeframe: str, n: int) -> Optional[pd.DataFrame]:
        symdir = self._symbol_dir(symbol, timeframe)
        if not symdir.exists():
            return None
        parts = sorted([p for p in symdir.iterdir() if p.name.endswith(".parquet")])
        if not parts:
            return None
        dfs = []
        for p in parts[-2:]:
            try:
                d = pd.read_parquet(p)
                if d.index.tz is None:
                    d.index = d.index.tz_localize(timezone.utc)
                dfs.append(d)
            except Exception as e:
                logger.error(f"Failed reading {p}: {e}")
        if not dfs:
            return None
        df = pd.concat(dfs, axis=0)
        df = self._finalize_df(df)
        return df.tail(n)

    # ---------------- Meta ---------------- #

    def _load_meta(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        path = self._meta_path(symbol, timeframe)
        key = path.as_posix()
        if key in self._meta_cache:
            return self._meta_cache[key]
        if not path.exists():
            meta = {"last_ts": None}
            self._meta_cache[key] = meta
            return meta
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            meta = {"last_ts": None}
        self._meta_cache[key] = meta
        return meta

    def _save_meta(self, symbol: str, timeframe: str, meta: Dict[str, Any]) -> None:
        path = self._meta_path(symbol, timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
        self._meta_cache[path.as_posix()] = meta

    # ---------------- Internals ---------------- #

    def _sanitize_symbol(self, symbol: str) -> str:
        return symbol.replace("/", "_").upper()

    def _finalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = df[~df.index.duplicated(keep="last")].sort_index()
        cols = ["open", "high", "low", "close", "volume"]
        for c in cols:
            if c not in df.columns:
                df[c] = float("nan")
        return df[cols].astype(float)

    def _validate_tf(self, timeframe: str) -> None:
        if timeframe not in self.TF_MAP:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

    def _throttle_bucket(self) -> None:
        t = time.time()
        self._rate_bucket.append(t)
        self._rate_bucket = [x for x in self._rate_bucket if t - x <= 60.0]
        if len(self._rate_bucket) > self._REQS_PER_MIN_BUDGET:
            sleep_s = 60.0 - (t - self._rate_bucket[0])
            if sleep_s > 0:
                time.sleep(min(sleep_s, 1.5))

    # Small helper to convert raw klines -> DataFrame (used when klines passed in)
    def _process_data(self, klines: List[list]) -> pd.DataFrame:
        # Expect rows like [ts, open, high, low, close, volume]
        if not klines:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(klines, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        return self._finalize_df(df)
