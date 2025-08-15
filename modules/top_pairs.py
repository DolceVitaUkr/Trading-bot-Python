# modules/top_pairs.py

import time
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional

import ccxt

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


@dataclass
class PairStats:
    """
    Represents statistics for a trading pair.
    """
    symbol: str
    base: str
    quote: str
    volume_usd_24h: float
    price_change_24h_pct: float
    ask: Optional[float]
    bid: Optional[float]


class TopPairs:
    """
    Fetch and rank top liquid Bybit spot pairs (default quote=USDT).
    Public-only; no API key required.

    - Uses ccxt's unified tickers where possible.
    - Filters by quote, status (active), and reasonable price/volume presence.
    - Caches results for `ttl_sec` to avoid hammering REST.
    """

    def __init__(
        self,
        exchange: Optional[ccxt.Exchange] = None,
        *,
        quote: str = "USDT",
        min_volume_usd_24h: float = 200_000,  # tune as you like
        ttl_sec: int = 60 * 60,               # refresh every 60 minutes
        max_pairs: int = 15
    ):
        """
        Initializes the TopPairs instance.
        """
        self.exchange = exchange or ccxt.bybit({"enableRateLimit": True})
        self.quote = quote.upper()
        self.min_volume_usd_24h = float(min_volume_usd_24h)
        self.ttl_sec = int(ttl_sec)
        self.max_pairs = int(max_pairs)

        self._cache_ts: float = 0.0
        self._cache: List[PairStats] = []

    def _spot_markets(self) -> Dict[str, Dict]:
        # Load markets once (ccxt caches internally too)
        self.exchange.load_markets()
        # Prefer spot markets; fall back to anything that looks like spot
        return {
            s: m for s, m in self.exchange.markets.items()
            if m.get("spot") and m.get("active", True)
        }

    def _as_pair_stats(
        self, symbol: str, t: Dict, mkt: Dict
    ) -> Optional[PairStats]:
        # t is a unified ticker dict from ccxt
        # We’ll compute a simple USD-ish volume:
        # ccxt often has `quoteVolume` ~ 24h volume in quote currency
        quote_vol = t.get("quoteVolume") or 0.0
        ask = t.get("ask")
        bid = t.get("bid")
        last = t.get("last") or (ask if ask else bid)
        # Some markets report only baseVolume; try to estimate quoteVol
        if (not quote_vol or quote_vol == 0.0) and last and t.get(
                "baseVolume"):
            quote_vol = float(t["baseVolume"]) * float(last)

        # Use 24h change if present, else compute from open/last
        chg_pct = 0.0
        if t.get("percentage") is not None:
            chg_pct = float(t["percentage"])
        else:
            open_ = t.get("open")
            if open_ and open_ > 0 and last:
                chg_pct = (float(last) - float(open_)) / float(open_) * 100.0

        base = mkt.get("base")
        quote = mkt.get("quote")

        vol_usd = float(
            quote_vol) if (quote and quote.upper() == "USDT") else float(
            quote_vol)

        return PairStats(
            symbol=symbol,
            base=base,
            quote=quote,
            volume_usd_24h=vol_usd,
            price_change_24h_pct=float(chg_pct),
            ask=float(ask) if ask else None,
            bid=float(bid) if bid else None,
        )

    def _refresh_cache(self) -> None:
        try:
            mkts = self._spot_markets()
            # Filter to preferred quote
            spot_symbols = [
                s for s, m in mkts.items()
                if m.get("quote", "").upper() == self.quote]
            if not spot_symbols:
                logger.warning(
                    f"No active spot symbols with quote={self.quote}")
                self._cache = []
                self._cache_ts = time.time()
                return

            tickers = self.exchange.fetch_tickers(spot_symbols)
            stats: List[PairStats] = []
            for sym in spot_symbols:
                t = tickers.get(sym)
                if not t:
                    continue
                ps = self._as_pair_stats(sym, t, mkts[sym])
                if not ps:
                    continue
                # Basic sanity gating
                if (ps.volume_usd_24h is None or
                        ps.volume_usd_24h < self.min_volume_usd_24h):
                    continue
                stats.append(ps)

            # Rank primarily by liquidity, then by positive momentum
            stats.sort(
                key=lambda x: (x.volume_usd_24h, x.price_change_24h_pct),
                reverse=True)
            self._cache = stats[: self.max_pairs]
            self._cache_ts = time.time()
            logger.info(
                f"[TopPairs] Cached {len(self._cache)} pairs for "
                f"quote={self.quote}")
        except Exception as e:
            logger.warning(f"[TopPairs] refresh failed: {e}")
            # Don’t nuke old cache on transient failures
            self._cache_ts = time.time()

    def get_top_pairs(self, *, force: bool = False) -> List[str]:
        """
        Return a list of top symbols like 'BTC/USDT'.
        """
        if force or (
                time.time() - self._cache_ts) > self.ttl_sec or not self._cache:
            self._refresh_cache()
        return [p.symbol for p in self._cache]

    def get_top_pairs_with_stats(
            self, *, force: bool = False) -> List[PairStats]:
        """
        Return a list of top pairs with statistics.
        """
        if force or (
                time.time() - self._cache_ts) > self.ttl_sec or not self._cache:
            self._refresh_cache()
        return list(self._cache)

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        """Keep ccxt/Bybit spot format: 'BTC/USDT'."""
        if "/" not in symbol and symbol.upper().endswith("USDT"):
            base = symbol[:-4]
            return f"{base}/USDT"
        return symbol
