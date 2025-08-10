# modules/top_pairs.py

import time
import logging
from typing import List, Dict, Optional, Tuple, Literal

import pandas as pd

from modules.exchange import ExchangeAPI
from modules.data_manager import DataManager, VALID_TIMEFRAMES
import config

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(getattr(logging, str(getattr(config, "LOG_LEVEL", "INFO")), logging.INFO)
                if isinstance(getattr(config, "LOG_LEVEL", "INFO"), str) else getattr(config, "LOG_LEVEL", logging.INFO))


class TopPairs:
    """
    Selects top USDT pairs by 24h volume (and/or change), with spike detection.

    - Refresh window: 60 minutes (configurable via constructor).
    - Spike check:
        * Short-term (2h): computed from recent 5m bars (~24 bars)
        * Daily (24h): taken from ticker percentage if present; otherwise from 15m bars
    - Returns symbols in both CCXT format ("BTC/USDT") and no-slash ("BTCUSDT").
    """

    def __init__(
        self,
        exchange: Optional[ExchangeAPI] = None,
        data_manager: Optional[DataManager] = None,
        refresh_minutes: int = 60,
        *,
        quote: str = "USDT",
        min_24h_volume_usd: float = 5_000_000.0,
        max_pairs: int = 10,
    ):
        self.exchange = exchange or ExchangeAPI()
        self.dm = data_manager or DataManager(exchange=self.exchange)
        self.refresh_seconds = max(10 * 60, int(refresh_minutes * 60))
        self.quote = quote.upper()
        self.min_24h_volume_usd = float(min_24h_volume_usd)
        self.max_pairs = max_pairs

        self._last_refresh_ts: float = 0.0
        self._cache: List[Dict] = []

    # ─────────────────────────────
    # Public API
    # ─────────────────────────────

    def needs_refresh(self) -> bool:
        return (time.time() - self._last_refresh_ts) >= self.refresh_seconds

    def current(self) -> List[Dict]:
        """
        Return cached list (refresh externally by calling refresh() if needs_refresh()).
        Each entry:
            {
              "symbol": "BTC/USDT",
              "symbol_id": "BTCUSDT",
              "vol24_usd": float,
              "pct24": float or None,
              "pct2h": float or None
            }
        """
        return list(self._cache)

    def refresh(self) -> List[Dict]:
        """
        Pull fresh tickers, filter by quote & volume, compute spike metrics.
        """
        try:
            pairs = self._select_by_volume()
            out = []
            for sym, vol_usd, pct24 in pairs[: self.max_pairs]:
                pct2h = self._compute_2h_change(sym)
                out.append({
                    "symbol": sym,
                    "symbol_id": self._noslash(sym),
                    "vol24_usd": vol_usd,
                    "pct24": pct24,
                    "pct2h": pct2h,
                })
            self._cache = out
            self._last_refresh_ts = time.time()
            return list(self._cache)
        except Exception as e:
            logger.warning(f"TopPairs.refresh failed: {e}")
            return list(self._cache)

    # ─────────────────────────────
    # Internals
    # ─────────────────────────────

    def _select_by_volume(self) -> List[Tuple[str, float, Optional[float]]]:
        """
        Use exchange tickers to rank symbols by 24h volume in quote currency (USDT).
        Returns list of (symbol_ccxt, vol24_usd, pct24)
        """
        tickers = self.exchange.fetch_tickers_safely()
        candidates: List[Tuple[str, float, Optional[float]]] = []

        for sym, t in tickers.items():
            # CCXT symbol format guard
            if "/" not in sym:
                continue
            base, quote = sym.split("/")
            if quote.upper() != self.quote:
                continue

            vol = t.get("quoteVolume") or t.get("info", {}).get("quoteVolume")
            if vol is None:
                # Approx with baseVolume * last price if present
                try:
                    vol = float(t.get("baseVolume", 0.0)) * float(t.get("last", t.get("close", 0.0) or 0.0))
                except Exception:
                    vol = 0.0
            try:
                vol = float(vol)
            except Exception:
                vol = 0.0

            if vol < self.min_24h_volume_usd:
                continue

            # 24h change if available
            pct = None
            ch = t.get("percentage")
            if ch is not None:
                try:
                    pct = float(ch)
                except Exception:
                    pct = None

            candidates.append((sym, vol, pct))

        # sort by 24h USD volume desc
        candidates.sort(key=lambda r: r[1], reverse=True)
        return candidates

    def _compute_2h_change(self, symbol: str) -> Optional[float]:
        """
        Compute percent change over ~2 hours from 5m bars.
        pct2h = (last_close / close_2h_ago - 1) * 100
        """
        try:
            df = self.dm.load_historical_data(symbol, "5m", auto_update=True)
            if len(df) < 24:  # 24 x 5m = 120m
                # try fetching a bit more explicitly
                self.dm.update_bars(symbol, "5m", bootstrap_candles=300)
                df = self.dm.load_historical_data(symbol, "5m", auto_update=False)
            if len(df) < 24:
                return None

            last_close = float(df["close"].iloc[-1])
            close_2h_ago = float(df["close"].iloc[-24])
            if close_2h_ago <= 0:
                return None
            return (last_close / close_2h_ago - 1.0) * 100.0
        except Exception as e:
            logger.debug(f"_compute_2h_change failed {symbol}: {e}")
            return None

    def _noslash(self, symbol: str) -> str:
        return symbol.replace("/", "").replace(":", "")
