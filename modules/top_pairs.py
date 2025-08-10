# modules/top_pairs.py

import logging
from typing import List, Tuple, Optional, Dict

import ccxt

import config

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(getattr(logging, str(getattr(config, "LOG_LEVEL", "INFO")), logging.INFO)
                if isinstance(getattr(config, "LOG_LEVEL", "INFO"), str) else getattr(config, "LOG_LEVEL", logging.INFO))


class TopPairsProvider:
    """
    Rank Bybit symbols by 24h quote volume, filtered to USDT-quoted spot (default).
    Falls back to baseVolume*last if quoteVolume is missing.

    Usage:
        tp = TopPairsProvider()
        pairs = tp.get_top_pairs(quote="USDT", limit=20, min_24h_usdt=5e6)

    Return format:
        [
          {"symbol": "BTC/USDT", "quoteVolume": 123456789.0, "last": 68000.5},
          ...
        ]
    """

    def __init__(self, exchange: Optional[ccxt.Exchange] = None):
        self.exchange = exchange or self._make_exchange()
        self.exchange.load_markets()

    def _make_exchange(self) -> ccxt.Exchange:
        is_sim = getattr(config, "USE_SIMULATION", True)
        api_key = config.SIMULATION_BYBIT_API_KEY if is_sim else config.BYBIT_API_KEY
        secret = config.SIMULATION_BYBIT_API_SECRET if is_sim else config.BYBIT_API_SECRET

        default_type = "spot"
        if config.EXCHANGE_PROFILE in ("perp", "spot+perp"):
            default_type = "linear"

        ex = ccxt.bybit({
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": default_type,
                "adjustForTimeDifference": True
            }
        })
        if is_sim:
            ex.set_sandbox_mode(True)
        return ex

    def get_top_pairs(
        self,
        *,
        quote: str = "USDT",
        limit: int = 20,
        min_24h_usdt: float = 5_000_000.0,
        spot_only: bool = True
    ) -> List[Dict]:
        """
        Return top symbols sorted by 24h quote volume (USDT), with filters.
        """
        # Fetch tickers in one shot
        try:
            tickers = self.exchange.fetch_tickers()
        except Exception as e:
            logger.warning(f"[TopPairs] fetch_tickers failed: {e}")
            return []

        # Load markets mapping to filter spot / quote
        markets = self.exchange.markets

        ranked: List[Tuple[str, float, float]] = []  # (symbol, qvol, last)
        for sym, t in tickers.items():
            mkt = markets.get(sym)
            if not mkt:
                continue

            if spot_only and not mkt.get("spot", False):
                continue

            if str(mkt.get("quote", "")).upper() != quote.upper():
                continue

            last = float(t.get("last") or t.get("close") or 0.0)
            qvol = t.get("quoteVolume")
            if qvol is None:
                # fallback: baseVolume * last
                base = float(t.get("baseVolume") or 0.0)
                qvol = base * last
            qvol = float(qvol or 0.0)

            if qvol >= float(min_24h_usdt):
                ranked.append((sym, qvol, last))

        ranked.sort(key=lambda x: x[1], reverse=True)
        top = ranked[:limit]

        return [{"symbol": s, "quoteVolume": v, "last": p} for (s, v, p) in top]
