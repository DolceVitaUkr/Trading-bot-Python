# modules/exchange.py

import time
import math
import logging
from typing import Any, Dict, Optional

import ccxt
import config

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(getattr(logging, str(getattr(config, "LOG_LEVEL", "INFO")), logging.INFO)
                if isinstance(getattr(config, "LOG_LEVEL", "INFO"), str) else getattr(config, "LOG_LEVEL", logging.INFO))


class ExchangeAPI:
    """
    Thin wrapper around ccxt with an in-memory paper engine that mirrors
    the subset of calls used by TradeExecutor.

    Public methods used by the app:
      - get_balance()
      - get_price(symbol)
      - get_min_cost(symbol)
      - get_amount_precision(symbol)
      - get_price_precision(symbol)
      - fetch_positions(symbol)         # live only → list[dict]; paper returns internal
      - create_order(symbol, type, side, amount, price, attach_sl/attach_tp, reduce_only)
      - _simulate_order(...)            # paper engine for TradeExecutor

    Notes
    -----
    • Symbols use the ccxt standard with slash, e.g. "BTC/USDT".
    • In SIM mode, we keep:
        _sim_cash_usd  : float  (starting from SIMULATION_START_BALANCE)
        _sim_positions : dict[symbol] -> {side, quantity, entry_price, sl, tp}
        _last_price    : dict[symbol] -> float   (simple cache from last get_price)
    """

    def __init__(self):
        self.is_testnet = bool(getattr(config, "USE_TESTNET", True))
        self.profile = getattr(config, "EXCHANGE_PROFILE", "spot")
        self.fee_rate = float(getattr(config, "FEE_PERCENTAGE", 0.002))
        self.sim_delay = float(getattr(config, "SIMULATION_ORDER_DELAY", 0.5))

        # --- ccxt client (Bybit) ---
        # We prefer spot unless explicitly perp.
        bybit_kwargs = {
            "apiKey": config.SIMULATION_BYBIT_API_KEY if self.is_testnet else config.BYBIT_API_KEY,
            "secret": config.SIMULATION_BYBIT_API_SECRET if self.is_testnet else config.BYBIT_API_SECRET,
            "enableRateLimit": True,
            "options": {
                "defaultType": "spot" if self.profile == "spot" else "swap",
                "adjustForTimeDifference": True,
            },
        }
        self.client = ccxt.bybit(bybit_kwargs)
        # ccxt's Bybit testnet setting
        try:
            self.client.set_sandbox_mode(self.is_testnet)
        except Exception:
            # Older ccxt may not have set_sandbox_mode for Bybit; ignore.
            pass

        # Load markets once (best-effort)
        self._markets = {}
        try:
            self._markets = self.client.load_markets()
        except Exception as e:
            logger.warning(f"load_markets failed (continuing): {e}")

        # --- Paper engine state ---
        self._sim_cash_usd: float = float(getattr(config, "SIMULATION_START_BALANCE", 1000.0))
        self._sim_positions: Dict[str, Dict[str, Any]] = {}
        self._last_price: Dict[str, float] = {}

    # ──────────────────────────────────────────────────────────────────────
    # Helpers / Normalization
    # ──────────────────────────────────────────────────────────────────────
    def _resolve_symbol(self, symbol: str) -> str:
        # Normalize common forms: btcusdt → BTC/USDT
        s = symbol.replace(" ", "").upper()
        if "/" not in s:
            if s.endswith("USDT"):
                s = s[:-4] + "/USDT"
            elif s.endswith("USD"):
                s = s[:-3] + "/USD"
        return s

    # ──────────────────────────────────────────────────────────────────────
    # Live-ish helpers
    # ──────────────────────────────────────────────────────────────────────
    def get_price(self, symbol: str) -> Optional[float]:
        sym = self._resolve_symbol(symbol)
        # Try ticker first
        try:
            t = self.client.fetch_ticker(sym)
            last = float(t.get("last") or t.get("close") or 0.0)
            if last > 0:
                self._last_price[sym] = last
                return last
        except Exception as e:
            logger.debug(f"fetch_ticker {sym} failed: {e}")

        # Fallback to cache
        return self._last_price.get(sym)

    def get_balance(self) -> float:
        if self.is_testnet or getattr(config, "USE_SIMULATION", True):
            return float(self._sim_cash_usd)
        try:
            bal = self.client.fetch_balance()
            # Prefer USDT total (Bybit spot swaps use USDT commonly)
            total = bal.get("total", {})
            usdt = float(total.get("USDT") or 0.0)
            if usdt <= 0 and "free" in bal:
                usdt = float(bal["free"].get("USDT") or 0.0)
            return usdt
        except Exception as e:
            logger.warning(f"fetch_balance failed: {e}")
            return 0.0

    def get_min_cost(self, symbol: str) -> float:
        sym = self._resolve_symbol(symbol)
        m = self._markets.get(sym)
        if not m:
            # Generic fallback: $10
            return float(getattr(config, "MIN_TRADE_AMOUNT_USD", 10.0))
        # ccxt meta fields vary; use minCost/limits
        min_cost = None
        try:
            min_cost = m.get("limits", {}).get("cost", {}).get("min")
        except Exception:
            min_cost = None
        if min_cost is None:
            min_cost = m.get("minCost")
        if not min_cost:
            min_cost = float(getattr(config, "MIN_TRADE_AMOUNT_USD", 10.0))
        return float(min_cost)

    def get_amount_precision(self, symbol: str) -> int:
        sym = self._resolve_symbol(symbol)
        m = self._markets.get(sym)
        if not m:
            return 6
        # ccxt uses "precision": {"amount": n, "price": n}
        prec = m.get("precision", {}).get("amount")
        if prec is None:
            # derive from limits if available
            step = m.get("limits", {}).get("amount", {}).get("min")
            if step:
                return max(0, int(round(-math.log10(float(step)))))  # crude
            return 6
        return int(prec)

    def get_price_precision(self, symbol: str) -> int:
        sym = self._resolve_symbol(symbol)
        m = self._markets.get(sym)
        if not m:
            return 4
        prec = m.get("precision", {}).get("price")
        if prec is None:
            step = m.get("limits", {}).get("price", {}).get("min")
            if step:
                return max(0, int(round(-math.log10(float(step)))))
            return 4
        return int(prec)

    # ──────────────────────────────────────────────────────────────────────
    # Live order/positions
    # ──────────────────────────────────────────────────────────────────────
    def fetch_positions(self, symbol: str):
        """Live-only positions (paper uses _sim_positions)."""
        if self.is_testnet or getattr(config, "USE_SIMULATION", True):
            # map paper position to a ccxt-like list
            sym = self._resolve_symbol(symbol)
            p = self._sim_positions.get(sym)
            if not p:
                return []
            qty = float(p.get("quantity", 0.0))
            return [{
                "symbol": sym,
                "side": p.get("side"),
                "contracts": qty,
                "amount": qty,
                "entryPrice": float(p.get("entry_price", 0.0)),
            }]
        try:
            # Not all venues support fetch_positions for spot; handle gracefully
            positions = self.client.fetch_positions([self._resolve_symbol(symbol)])
            return positions or []
        except Exception as e:
            logger.debug(f"fetch_positions not supported/failed: {e}")
            return []

    def create_order(
        self,
        symbol: str,
        type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        *,
        attach_sl: Optional[float] = None,
        attach_tp: Optional[float] = None,
        reduce_only: bool = False
    ) -> Dict[str, Any]:
        """Live path only; paper is handled by TradeExecutor via _simulate_order()."""
        sym = self._resolve_symbol(symbol)
        try:
            params = {}
            if reduce_only:
                params["reduceOnly"] = True

            # Some ccxt/venue combos require price for limit only
            if type.lower() == "market":
                order = self.client.create_order(sym, "market", side, amount, None, params)
            else:
                order = self.client.create_order(sym, "limit", side, amount, price, params)

            # Inline SL/TP: best-effort, many venues require separate orders
            # so we skip here to keep compatibility.

            return order
        except Exception as e:
            # surface exception; TradeExecutor will wrap it
            raise

    # ──────────────────────────────────────────────────────────────────────
    # Paper engine
    # ──────────────────────────────────────────────────────────────────────
    def _simulate_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float,
        *,
        attach_sl: Optional[float] = None,
        attach_tp: Optional[float] = None,
        reduce_only: bool = False
    ) -> Dict[str, Any]:
        """
        Very lightweight spot-like simulator:
         - Long or Short "positions" (we allow short notionally for strategy testing)
         - Open: deduct entry notional from cash (fees applied in TradeExecutor)
         - Close: release notional and realize PnL to cash
         - Average price when adding to existing position of same side
         - If opening opposite side, we close existing (simple netting)

        Position schema:
          { "side": "long"|"short", "quantity": float, "entry_price": float,
            "sl": float|None, "tp": float|None }
        """
        sym = self._resolve_symbol(symbol)
        side = side.lower()
        order_type = order_type.lower()

        # Artificial delay to feel realistic
        try:
            time.sleep(max(0.0, self.sim_delay))
        except Exception:
            pass

        px = float(price)
        qty = float(quantity)
        if qty <= 0 or px <= 0:
            return {"status": "rejected", "reason": "invalid_qty_or_price"}

        pos = self._sim_positions.get(sym)

        # Reduce-only close path
        if reduce_only and pos:
            return self._close_position(sym, px, qty)

        # No position → open
        if not pos:
            self._open_position(sym, side, qty, px, attach_sl, attach_tp)
            return {
                "status": "open",
                "symbol": sym,
                "side": side,
                "quantity": qty,
                "entry_price": px,
            }

        # Existing position present
        if pos["side"] == side:
            # Increase / add; compute new average entry
            new_qty = pos["quantity"] + qty
            if new_qty <= 0:
                return {"status": "rejected", "reason": "nonpositive_qty"}

            avg_entry = (pos["entry_price"] * pos["quantity"] + px * qty) / new_qty
            pos["quantity"] = new_qty
            pos["entry_price"] = avg_entry
            if attach_sl is not None:
                pos["sl"] = float(attach_sl)
            if attach_tp is not None:
                pos["tp"] = float(attach_tp)

            return {
                "status": "open_increased",
                "symbol": sym,
                "side": side,
                "quantity": new_qty,
                "entry_price": float(avg_entry),
            }

        # Opposite side → close existing (netting). For simplicity, close fully.
        close_res = self._close_position(sym, px, pos["quantity"])
        # Optionally, open the new opposite side after close.
        self._open_position(sym, side, qty, px, attach_sl, attach_tp)
        return {
            "status": "open_added",
            "symbol": sym,
            "side": side,
            "quantity": qty,
            "entry_price": px,
            "closed_prev": close_res,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Paper internals
    # ──────────────────────────────────────────────────────────────────────
    def _open_position(
        self, symbol: str, side: str, qty: float, price: float,
        sl: Optional[float], tp: Optional[float]
    ):
        """
        Open a notional position:
          - We “reserve” notional by moving it out of cash, but actual fee
            handling is performed by TradeExecutor (so we don’t double-count).
        """
        notional = float(qty * price)
        # Reserve: move cash down by notional (naive spot-like accounting)
        try:
            self._sim_cash_usd -= notional
        except Exception:
            pass
        self._sim_positions[symbol] = {
            "side": "long" if side == "buy" or side == "long" else "short",
            "quantity": float(qty),
            "entry_price": float(price),
            "sl": float(sl) if sl else None,
            "tp": float(tp) if tp else None,
        }

    def _close_position(self, symbol: str, price: float, qty: float) -> Dict[str, Any]:
        pos = self._sim_positions.get(symbol)
        if not pos:
            return {"status": "noop", "symbol": symbol}

        close_qty = min(float(qty), float(pos["quantity"]))
        entry = float(pos["entry_price"])
        side = pos["side"]
        pnl = 0.0
        notional_release = 0.0

        if side == "long":
            pnl = (price - entry) * close_qty
            notional_release = price * close_qty
        else:
            # Short PnL: entry higher than exit → profit
            pnl = (entry - price) * close_qty
            # Notional release mirrors how we "reserved" at open:
            notional_release = entry * close_qty  # conservative

        # Return notional + PnL to cash (fees added by TradeExecutor separately)
        try:
            self._sim_cash_usd += notional_release + pnl
        except Exception:
            pass

        remaining = float(pos["quantity"]) - close_qty
        if remaining <= 0.0:
            self._sim_positions.pop(symbol, None)
            status = "closed"
        else:
            pos["quantity"] = remaining
            status = "closed_partial"

        return {
            "status": status,
            "symbol": symbol,
            "pnl": float(pnl),
            "quantity": float(close_qty),
            "price": float(price),
        }
