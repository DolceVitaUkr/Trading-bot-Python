# modules/top_pairs.py

import os
import time
import json
import logging
from typing import List, Dict, Optional, Tuple

import ccxt

import config
from utils.utilities import ensure_directory, read_json, write_json, retry

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(getattr(logging, str(getattr(config, "LOG_LEVEL", "INFO")), logging.INFO)
                if isinstance(getattr(config, "LOG_LEVEL", "INFO"), str) else getattr(config, "LOG_LEVEL", logging.INFO))


CACHE_FILE = os.path.join(getattr(config, "HISTORICAL_DATA_PATH", "historical_data"), "_top_pairs_cache.json")


def _is_stable(symbol: str) -> bool:
    s = symbol.upper().replace("/", "")
    return any(x in s for x in ("USDC", "USDT", "BUSD", "FDUSD")) and (
        s.startswith("USDC") or s.startswith("USDT") or s.startswith("BUSD") or s.startswith("FDUSD")
    )


class TopPairs:
    """
    Finds liquid spot USDT pairs on Bybit (or chosen exchange) and caches for 60 minutes.
    """

    def __init__(self, exchange_id: str = "bybit", spot: bool = True, cache_minutes: int = 60):
        self.exchange_id = exchange_id
        self.cache_minutes = cache_minutes
        self.spot = spot
        self.ex = self._build_ccxt_client()
        ensure_directory(os.path.dirname(CACHE_FILE))

    def _build_ccxt_client(self):
        cls = getattr(ccxt, self.exchange_id)
        if config.USE_SIMULATION:
            api_key = getattr(config, "SIMULATION_BYBIT_API_KEY", None)
            secret = getattr(config, "SIMULATION_BYBIT_API_SECRET", None)
            testnet = True
        else:
            api_key = getattr(config, "BYBIT_API_KEY", None)
            secret = getattr(config, "BYBIT_API_SECRET", None)
            testnet = False

        inst = cls({
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": True,
            "options": {"defaultType": "spot" if self.spot else "swap"},
        })
        if hasattr(inst, "urls") and "api" in inst.urls and testnet:
            try:
                inst.urls["api"]["public"] = inst.urls["test"]["public"]
                inst.urls["api"]["private"] = inst.urls["test"]["private"]
            except Exception:
                pass
        return inst

    def _cache_read(self) -> Dict:
        return read_json(CACHE_FILE, default={"updated": 0, "pairs": []})

    def _cache_write(self, pairs: List[str]) -> None:
        write_json(CACHE_FILE, {"updated": int(time.time()), "pairs": pairs})

    def _cache_valid(self, updated_ts: int) -> bool:
        return (time.time() - updated_ts) < (self.cache_minutes * 60)

    @retry(max_attempts=3, delay=1.0, backoff=2.0)
    def _fetch_candidates(self) -> List[Dict]:
        """
        Fetch tickers, return a list of dicts with symbol, baseVolume and change stats.
        """
        tickers = self.ex.fetch_tickers()
        out: List[Dict] = []
        for sym, t in tickers.items():
            # spot USDT pairs only
            if "USDT" not in sym or "/" not in sym:
                continue
            # Exclude pure stable/stable
            if _is_stable(sym):
                continue
            vol = t.get("baseVolume") or t.get("quoteVolume")
            change = t.get("percentage")
            out.append({
                "symbol": sym,
                "volume": float(vol) if vol is not None else 0.0,
                "change": float(change) if change is not None else 0.0,
            })
        return out

    def refresh_top_pairs(self, *, max_pairs: int = 10, min_volume: float = 0.0) -> List[str]:
        """
        Query the exchange and cache the top USDT spot pairs (exclude stables) by volume + momentum.
        """
        try:
            cands = self._fetch_candidates()
        except Exception as e:
            logger.warning(f"TopPairs fetch failed, using cache if available: {e}")
            cache = self._cache_read()
            return cache.get("pairs", [])

        # Simple score: volume * (1 + change/100)
        for c in cands:
            c["score"] = max(0.0, c["volume"]) * (1.0 + (c["change"] / 100.0))

        # Filter + sort
        filt = [c for c in cands if c["volume"] >= min_volume]
        filt.sort(key=lambda x: x["score"], reverse=True)

        pairs = [c["symbol"] for c in filt[:max_pairs]]
        self._cache_write(pairs)
        return pairs

    def get_top_pairs(self, *, max_pairs: int = 10, min_volume: float = 0.0) -> List[str]:
        """
        Return cached pairs if fresh (< cache_minutes); otherwise refresh.
        """
        cache = self._cache_read()
        if self._cache_valid(int(cache.get("updated", 0))) and cache.get("pairs"):
            return cache["pairs"]
        return self.refresh_top_pairs(max_pairs=max_pairs, min_volume=min_volume)
