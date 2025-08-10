# modules/exchange.py

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import ccxt
import ccxt.base.errors as ccxt_errors

import config

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(getattr(logging, str(getattr(config, "LOG_LEVEL", "INFO")), logging.INFO)
                if isinstance(getattr(config, "LOG_LEVEL", "INFO"), str) else getattr(config, "LOG_LEVEL", logging.INFO))


# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────

def normalize_symbol(sym: str) -> str:
    """Convert common formats to Bybit spot symbol (USDT quote) e.g. 'BTC/USDT' -> 'BTCUSDT'."""
    sym = sym.strip().upper()
    if "/" in sym:
        base, quote = sym.split("/", 1)
        return f"{base}{quote}"
    return sym


def denormalize_symbol(sym: str) -> str:
    """Inverse of normalize_symbol (best-effort) e.g. 'BTCUSDT' -> 'BTC/USDT'."""
    sym = sym.strip().upper()
    if sym.endswith("USDT"):
        return f"{sym[:-4]}/USDT"
    return sym


@dataclass
class MarketMeta:
    symbol: str
    base: str
    quote: str
    price_precision: int
    amount_precision: int
    min_cost: float


# ────────────────────────────────────────────────────────────────────────────────
# ExchangeAPI (Bybit via CCXT + light simulation helpers)
# ────────────────────────────────────────────────────────────────────────────────

class ExchangeAPI:
    """
    Thin wrapper around CCXT Bybit plus small simulation helpers that our TradeExecutor expects.
    - Works with testnet when config.USE_TESTNET=True.
    - Normalizes symbols to Bybit spot format (BTCUSDT).
    - Exposes basic helpers: get_price, fetch_klines, get_min_cost, precisions, fetch top pairs.
    - Simulation state for paper trading: _sim_cash_usd, _sim_positions, and _simulate_order().
    """

    def __init__(self) -> None:
        # CCXT Bybit client
        # (use the unified spot endpoints; ccxt handles testnet flag)
        self._ccxt = ccxt.bybit({
            "apiKey": (config.SIMULATION_BYBIT_API_KEY if config.USE_TESTNET else config.BYBIT_API_KEY),
            "secret": (config.SIMULATION_BYBIT_API_SECRET if config.USE_TESTNET else config.BYBIT_API_SECRET),
            "enableRateLimit": True,
            "options": {
                "defaultType": "spot",  # use spot for this project
            },
        })
        self._ccxt.set_sandbox_mode(config.USE_TESTNET)

        # Markets/metadata cache
        self._markets: Dict[str, MarketMeta] = {}

        # Light REST cache for prices (avoid hammering REST when WS is on)
        self._last_prices: Dict[str, Tuple[float, float]] = {}  # symbol -> (price, unix_ts)

        # Simulation state (paper)
        self._sim_cash_usd: float = float(getattr(config, "SIMULATION_START_BALANCE", 1000.0))
        self._sim_positions: Dict[str, Dict[str, Any]] = {}  # symbol -> {side, quantity, entry_price, ts}

        # Load markets once
        try:
            markets = self._ccxt.load_markets()
            for mkt in markets.values():
                if mkt.get("spot") and mkt.get("active"):
                    sym = mkt["id"]  # Bybit: 'BTCUSDT'
                    price_prec = mkt.get("precision", {}).get("price", 4)
                    amt_prec = mkt.get("precision", {}).get("amount", 6)
                    limits = mkt.get("limits", {}) or {}
                    min_cost = float(limits.get("cost", {}).get("min") or 0.0)
                    self._markets[sym] = MarketMeta(
                        symbol=sym,
                        base=mkt.get("base", ""),
                        quote=mkt.get("quote", ""),
                        price_precision=int(price_prec or 4),
                        amount_precision=int(amt_prec or 6),
                        min_cost=float(min_cost or 0.0),
                    )
        except Exception as e:
            logger.warning(f"load_markets failed (will lazily resolve later): {e}")

    # ───────────────────────────────
    # Basic Account / Market Helpers
    # ───────────────────────────────

    def _ensure_market(self, symbol: str) -> MarketMeta:
        sym = normalize_symbol(symbol)
        if sym in self._markets:
            return self._markets[sym]
        try:
            self._ccxt.load_markets(reload=True)
            m = self._ccxt.market(denormalize_symbol(sym))
            price_prec = m.get("precision", {}).get("price", 4)
            amt_prec = m.get("precision", {}).get("amount", 6)
            limits = m.get("limits", {}) or {}
            min_cost = float(limits.get("cost", {}).get("min") or 0.0)
            meta = MarketMeta(
                symbol=sym,
                base=m.get("base", ""),
                quote=m.get("quote", ""),
                price_precision=int(price_prec or 4),
                amount_precision=int(amt_prec or 6),
                min_cost=float(min_cost or 0.0),
            )
            self._markets[sym] = meta
            return meta
        except Exception as e:
            raise ccxt_errors.BadSymbol(f"Unknown symbol: {symbol} ({e})")

    def _resolve_symbol(self, symbol: str) -> str:
        return self._ensure_market(symbol).symbol

    def get_price(self, symbol: str) -> Optional[float]:
        """Return last price; uses 3s cache to avoid spamming REST."""
        sym = self._resolve_symbol(symbol)
        now = time.time()
        cached = self._last_prices.get(sym)
        if cached and now - cached[1] < 3.0:
            return cached[0]
        try:
            tkr = self._ccxt.fetch_ticker(denormalize_symbol(sym))
            price = float(tkr.get("last") or tkr.get("close") or tkr.get("bid") or tkr.get("ask") or 0.0)
            if price > 0:
                self._last_prices[sym] = (price, now)
            return price if price > 0 else None
        except Exception as e:
            logger.debug(f"get_price({sym}) failed: {e}")
            return cached[0] if cached else None

    def get_min_cost(self, symbol: str) -> float:
        return float(self._ensure_market(symbol).min_cost or getattr(config, "MIN_TRADE_AMOUNT_USD", 10.0))

    def get_price_precision(self, symbol: str) -> int:
        return int(self._ensure_market(symbol).price_precision or 4)

    def get_amount_precision(self, symbol: str) -> int:
        return int(self._ensure_market(symbol).amount_precision or 6)

    def get_balance(self) -> float:
        """
        - Simulation: simulated USD cash
        - Live: sum free+used in quote wallet (USDT)
        """
        if config.USE_SIMULATION:
            return float(self._sim_cash_usd)
        try:
            bal = self._ccxt.fetch_balance()
            total = bal.get("total", {})
            # Prefer USDT
            if "USDT" in total:
                return float(total["USDT"])
            # sum everything (rough)
            return float(sum(v for v in total.values() if isinstance(v, (int, float))))
        except Exception as e:
            logger.warning(f"fetch_balance failed: {e}")
            return 0.0

    # ───────────────────────────────
    # OHLCV & Top Pairs
    # ───────────────────────────────

    def fetch_klines(
        self,
        symbol: str,
        timeframe: str = "5m",
        since_ms: Optional[int] = None,
        limit: int = 300,
    ) -> List[List[Any]]:
        """
        Return ccxt-style OHLCV list [[ts, open, high, low, close, volume], ...]
        limit is hard-capped to 900 to respect your preference & Bybit constraints.
        """
        sym = denormalize_symbol(self._resolve_symbol(symbol))
        limit = int(max(1, min(limit, 900)))
        try:
            data = self._ccxt.fetch_ohlcv(sym, timeframe=timeframe, since=since_ms, limit=limit)
            return data or []
        except Exception as e:
            logger.warning(f"fetch_klines {sym} {timeframe} failed: {e}")
            return []

    def fetch_top_pairs(
        self,
        max_pairs: int = 5,
        quote: str = "USDT",
        min_quote_vol_usd: float = 1_000_000.0,
    ) -> List[str]:
        """
        Pull top Spot pairs by 24h volume with USDT quote.
        Returns list of Bybit-format symbols e.g. ['BTCUSDT', 'ETHUSDT', ...]
        """
        try:
            tickers = self._ccxt.fetch_tickers()
        except Exception as e:
            logger.warning(f"fetch_tickers failed: {e}")
            return [normalize_symbol(config.DEFAULT_SYMBOL)]

        ranked: List[Tuple[str, float]] = []
        for key, tkr in tickers.items():
            # ccxt unifies symbol like 'BTC/USDT'
            if not key.endswith(f"/{quote}"):
                continue
            # skip inactive markets if known
            try:
                m = self._ccxt.market(key)
                if not (m.get("spot") and m.get("active")):
                    continue
            except Exception:
                pass

            # Use quoteVolume or baseVolume → approximate USD volume
            qvol = tkr.get("quoteVolume")
            if qvol is None:
                base_vol = tkr.get("baseVolume") or 0
                last = tkr.get("last") or tkr.get("close") or 0
                qvol = (base_vol or 0) * (last or 0)
            try:
                qvol = float(qvol or 0.0)
            except Exception:
                qvol = 0.0
            if qvol <= 0:
                continue
            if qvol < min_quote_vol_usd:
                continue
            ranked.append((normalize_symbol(key), qvol))

        ranked.sort(key=lambda kv: kv[1], reverse=True)
        out = [s for s, _ in ranked[:max_pairs]]
        if not out:
            out = [normalize_symbol(config.DEFAULT_SYMBOL)]
        return out

    # ───────────────────────────────
    # Orders (we only simulate for now)
    # ───────────────────────────────

    def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        *,
        attach_sl: Optional[float] = None,
        attach_tp: Optional[float] = None,
        reduce_only: bool = False
    ) -> Dict[str, Any]:
        """
        For this project we keep real trading OFF; this raises if not simulation.
        (TradeExecutor handles simulation directly via _simulate_order.)
        """
        if config.USE_SIMULATION:
            raise RuntimeError("Live order route called in simulation mode")
        # If you ever flip to live, you can implement the CCXT createOrder call here.
        raise NotImplementedError("Live trading is disabled in this build")

    def fetch_positions(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Only used by live close flow; in simulation we keep positions locally.
        """
        if config.USE_SIMULATION:
            # TradeExecutor checks for None / empty
            return []
        try:
            # For spot there are no 'positions'; would need to emulate with balances/orders.
            return []
        except Exception:
            return []

    # ───────────────────────────────
    # Paper trading helpers (used by TradeExecutor)
    # ───────────────────────────────

    def _simulate_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float,
        *,
        attach_sl: Optional[float],
        attach_tp: Optional[float],
        reduce_only: bool = False
    ) -> Dict[str, Any]:
        """
        Very simple spot simulation:
        - Single net position per symbol (no hedging)
        - Buy adds long; Sell closes/reduces long (no shorting in spot)
        - Apply entry/exit at given price
        - Track PnL on closes; attach SL/TP kept as metadata
        """
        sym = self._resolve_symbol(symbol)
        side = side.lower()
        price = float(price)
        qty = float(quantity)

        # enforce min-notional
        min_cost = self.get_min_cost(sym)
        notional = price * qty
        if notional < max(min_cost, float(getattr(config, "MIN_TRADE_AMOUNT_USD", 10.0))):
            return {"status": "rejected_min_notional"}

        pos = self._sim_positions.get(sym)

        if side == "buy":
            # open or add long
            cost = price * qty
            if cost > self._sim_cash_usd + 1e-8:
                return {"status": "rejected_insufficient_cash"}
            self._sim_cash_usd -= cost
            if pos and pos["side"] == "long":
                new_qty = pos["quantity"] + qty
                avg = (pos["entry_price"] * pos["quantity"] + price * qty) / max(new_qty, 1e-12)
                pos.update({
                    "quantity": new_qty,
                    "entry_price": avg,
                    "sl": attach_sl or pos.get("sl"),
                    "tp": attach_tp or pos.get("tp"),
                    "ts": time.time(),
                })
                status = "open_increased"
            else:
                self._sim_positions[sym] = {
                    "side": "long",
                    "quantity": qty,
                    "entry_price": price,
                    "sl": attach_sl,
                    "tp": attach_tp,
                    "ts": time.time(),
                }
                status = "open"
            return {
                "status": status,
                "symbol": sym,
                "entry_price": price,
                "quantity": qty,
            }

        if side == "sell":
            # reduce/close long if exists
            if not pos or pos["side"] != "long" or pos["quantity"] <= 1e-12:
                return {"status": "no_position"}
            close_qty = min(qty, pos["quantity"])
            pnl = (price - pos["entry_price"]) * close_qty
            self._sim_cash_usd += price * close_qty
            pos["quantity"] -= close_qty
            if pos["quantity"] <= 1e-12:
                # full close
                del self._sim_positions[sym]
                status = "closed"
            else:
                status = "closed_partial"
            return {
                "status": status,
                "symbol": sym,
                "price": price,
                "quantity": close_qty,
                "pnl": float(pnl),
            }

        return {"status": "rejected_unknown_side"}

    # Expose internal store for UI/Executor convenience
    @property
    def positions(self) -> Dict[str, Dict[str, Any]]:
        return self._sim_positions

    # ───────────────────────────────
    # Simple WS (public) for kline push (optional, DataManager owns loop)
    # We keep endpoints here for reference.
    # ───────────────────────────────

    @staticmethod
    def ws_public_endpoint_spot() -> str:
        if config.USE_TESTNET:
            return "wss://stream-testnet.bybit.com/v5/public/spot"
        return "wss://stream.bybit.com/v5/public/spot"

    @staticmethod
    def ws_kline_topic(interval_min: int, symbol: str) -> str:
        # Bybit WS v5 spot kline topic format: "kline.<interval>.<symbol>", where interval is minutes ("5", "15")
        return f"kline.{interval_min}.{normalize_symbol(symbol)}"
