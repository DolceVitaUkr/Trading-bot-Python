# modules/exchange.py

import logging
from typing import Optional, Dict, Any

import config

try:
    import ccxt  # REST + live trading
except Exception:  # pragma: no cover
    ccxt = None

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(getattr(logging, str(getattr(config, "LOG_LEVEL", "INFO")), logging.INFO)
                if isinstance(getattr(config, "LOG_LEVEL", "INFO"), str) else getattr(config, "LOG_LEVEL", logging.INFO))


class ExchangeAPI:
    """
    Thin adapter around ccxt.bybit with a built-in, simple paper-trading engine
    used by TradeExecutor(simulation_mode=True).

    Public methods used elsewhere:
      - get_balance() -> float
      - get_price(symbol) -> float
      - get_min_cost(symbol) -> float
      - get_amount_precision(symbol) -> int
      - get_price_precision(symbol) -> int
      - _resolve_symbol(symbol) -> 'BTC/USDT'
      - _simulate_order(...)
      - fetch_positions(symbol) -> list[dict]        (live only)
      - create_order(symbol, type, side, amount, price, attach_sl=None, attach_tp=None, reduce_only=False)
    """

    def __init__(self):
        self.simulation_mode = bool(getattr(config, "USE_SIMULATION", True))
        self.use_testnet = bool(getattr(config, "USE_TESTNET", True))

        self._sim_cash_usd: float = float(getattr(config, "SIMULATION_START_BALANCE", 1000.0))
        # simple long-only position book keyed by normalized symbol (BTCUSDT)
        self._sim_positions: Dict[str, Dict[str, Any]] = {}
        self.positions = self._sim_positions  # public alias (used by TradeSimulator)

        self._markets_loaded = False
        self._markets = {}

        self.exchange = None
        if ccxt is not None:
            self.exchange = ccxt.bybit({
                "apiKey": (config.SIMULATION_BYBIT_API_KEY if self.simulation_mode else config.BYBIT_API_KEY),
                "secret": (config.SIMULATION_BYBIT_API_SECRET if self.simulation_mode else config.BYBIT_API_SECRET),
                "enableRateLimit": True,
                "options": {
                    # market data works the same; trades route to testnet when simulation_mode=False but USE_TESTNET=True
                    "defaultType": "spot",
                },
            })
            # ccxt testnet toggle (affects some exchanges; Bybit spot market-data is public anyway)
            try:
                if hasattr(self.exchange, "set_sandbox_mode"):
                    self.exchange.set_sandbox_mode(self.use_testnet)
            except Exception:
                pass

            # Preload markets lazily
            self._load_markets_safely()
        else:
            logger.warning("ccxt is not installed; live price/markets unavailable.")

    # ─────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────

    def _load_markets_safely(self):
        if self.exchange is None:
            return
        if self._markets_loaded:
            return
        try:
            self._markets = self.exchange.load_markets()
            self._markets_loaded = True
        except Exception as e:
            logger.debug(f"load_markets failed: {e}")

    @staticmethod
    def _normalize_for_storage(symbol: str) -> str:
        return symbol.replace("/", "").upper()

    @staticmethod
    def _human_symbol(symbol: str) -> str:
        if "/" in symbol:
            return symbol
        if symbol.endswith("USDT"):
            return f"{symbol[:-4]}/USDT"
        return symbol

    def _resolve_symbol(self, symbol: str) -> str:
        """Ensure 'BASE/QUOTE' format for ccxt."""
        return self._human_symbol(symbol)

    # ─────────────────────────────────────────────────────────────────────
    # Public: market data / precision helpers
    # ─────────────────────────────────────────────────────────────────────

    def get_price(self, symbol: str) -> Optional[float]:
        """Last price via fetch_ticker; returns None if unavailable."""
        sym = self._resolve_symbol(symbol)
        if self.exchange is None:
            return None
        try:
            t = self.exchange.fetch_ticker(sym)
            px = t.get("last") or t.get("close") or t.get("info", {}).get("lastPrice")
            return float(px) if px is not None else None
        except Exception as e:
            logger.debug(f"get_price({sym}) failed: {e}")
            return None

    def get_balance(self) -> float:
        """Wallet balance:
           - simulation: simulated USD cash
           - live: Bybit wallet USDT total (free + used)
        """
        if self.simulation_mode or self.exchange is None:
            return float(self._sim_cash_usd)
        try:
            bal = self.exchange.fetch_balance()
            # Prefer USDT total; fall back to 'total' balance sum
            if "USDT" in bal.get("total", {}):
                return float(bal["total"]["USDT"])
            # Sum all quote-equivalent if needed (very rough)
            total = 0.0
            for _, v in bal.get("total", {}).items():
                try:
                    total += float(v or 0.0)
                except Exception:
                    pass
            return float(total)
        except Exception as e:
            logger.debug(f"get_balance live failed: {e}")
            return 0.0

    def get_min_cost(self, symbol: str) -> float:
        """Minimum notional cost (USD) for a trade (best-effort)."""
        self._load_markets_safely()
        sym = self._resolve_symbol(symbol)
        market = self._markets.get(sym, {})
        limits = market.get("limits", {})
        cost = limits.get("cost", {})
        min_cost = cost.get("min")
        try:
            if min_cost is None:
                # Fallback to config minimum
                return float(getattr(config, "MIN_TRADE_AMOUNT_USD", 10.0))
            return float(min_cost)
        except Exception:
            return float(getattr(config, "MIN_TRADE_AMOUNT_USD", 10.0))

    def get_amount_precision(self, symbol: str) -> int:
        """Amount precision (number of decimals for quantity)."""
        self._load_markets_safely()
        sym = self._resolve_symbol(symbol)
        market = self._markets.get(sym, {})
        precision = market.get("precision", {})
        amt = precision.get("amount")
        try:
            return int(amt) if isinstance(amt, int) else 6
        except Exception:
            return 6

    def get_price_precision(self, symbol: str) -> int:
        """Price precision (number of decimals)."""
        self._load_markets_safely()
        sym = self._resolve_symbol(symbol)
        market = self._markets.get(sym, {})
        precision = market.get("precision", {})
        px = precision.get("price")
        try:
            return int(px) if isinstance(px, int) else 2
        except Exception:
            return 2

    # ─────────────────────────────────────────────────────────────────────
    # Simulation engine (spot, long-only, SL/TP shadow)
    # ─────────────────────────────────────────────────────────────────────

    def _simulate_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float,
        attach_sl: Optional[float],
        attach_tp: Optional[float],
        reduce_only: bool = False
    ) -> Dict[str, Any]:
        """
        Very small spot simulator:
          - BUY opens/increases a long position; subtract notional from cash
          - SELL closes a long (full or partial); adds notional back to cash
          - Tracks pnl on close and returns it
          - SL/TP stored as shadow levels (not auto-triggered here)
        """
        sym_h = self._resolve_symbol(symbol)
        sym_k = self._normalize_for_storage(symbol)
        side = side.lower()
        order_type = order_type.lower()

        qty = float(quantity)
        px = float(price)
        if qty <= 0 or px <= 0:
            raise ValueError("Quantity and price must be positive")

        pos = self._sim_positions.get(sym_k)

        # BUY
        if side == "buy" and not reduce_only:
            if pos:
                # increase / average price
                old_qty = float(pos["quantity"])
                old_entry = float(pos["entry_price"])
                new_qty = old_qty + qty
                new_entry = (old_entry * old_qty + px * qty) / max(new_qty, 1e-12)
                pos["quantity"] = new_qty
                pos["entry_price"] = new_entry
                if attach_sl is not None:
                    pos["sl"] = float(attach_sl)
                if attach_tp is not None:
                    pos["tp"] = float(attach_tp)
                self._sim_cash_usd -= (px * qty)
                return {
                    "status": "open_increased",
                    "symbol": sym_h,
                    "side": "buy",
                    "quantity": new_qty,
                    "entry_price": new_entry,
                    "price": px,
                }
            else:
                # open new
                self._sim_positions[sym_k] = {
                    "side": "long",
                    "quantity": qty,
                    "entry_price": px,
                    "sl": float(attach_sl) if attach_sl is not None else None,
                    "tp": float(attach_tp) if attach_tp is not None else None,
                }
                self._sim_cash_usd -= (px * qty)
                return {
                    "status": "open",
                    "symbol": sym_h,
                    "side": "buy",
                    "quantity": qty,
                    "entry_price": px,
                    "price": px,
                }

        # SELL (close long)
        if side == "sell":
            if not pos or pos.get("side") != "long":
                # nothing to close
                return {
                    "status": "noop",
                    "symbol": sym_h,
                    "side": "sell",
                    "quantity": 0.0,
                    "price": px,
                    "pnl": 0.0,
                }
            close_qty = min(qty, float(pos["quantity"]))
            entry_px = float(pos["entry_price"])
            pnl = (px - entry_px) * close_qty

            # reduce or close
            remaining = float(pos["quantity"]) - close_qty
            if remaining <= 1e-12:
                self._sim_positions.pop(sym_k, None)
                status = "closed"
                final_qty = 0.0
            else:
                pos["quantity"] = remaining
                status = "closed_partial"
                final_qty = remaining

            # add back cash for the notional sold
            self._sim_cash_usd += (px * close_qty)

            return {
                "status": status,
                "symbol": sym_h,
                "side": "sell",
                "quantity": close_qty if status == "closed" else final_qty,
                "price": px,
                "pnl": pnl,
            }

        # Unknown path
        return {
            "status": "rejected",
            "symbol": sym_h,
            "side": side,
            "quantity": qty,
            "price": px,
            "reason": "unsupported order path",
        }

    # ─────────────────────────────────────────────────────────────────────
    # Live trading helpers (used only when simulation_mode=False)
    # ─────────────────────────────────────────────────────────────────────

    def fetch_positions(self, symbol: str):
        """Live-only: return open position list for the symbol (reduce-only close).
        For spot, Bybit typically doesn’t expose positions (perps do). We fallback to empty.
        """
        if self.exchange is None or self.simulation_mode:
            return []
        try:
            # If you later use perps, you can switch defaultType=linear and call fetchPositions.
            return []
        except Exception:
            return []

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
        """Pass-through to ccxt.create_order with minimal normalization."""
        if self.exchange is None:
            raise RuntimeError("Exchange not available")
        sym = self._resolve_symbol(symbol)
        typ = order_type.lower()
        side = side.lower()

        params = {}
        # Bybit spot generally ignores reduce-only; included for compatibility
        if reduce_only:
            params["reduceOnly"] = True

        try:
            order = self.exchange.create_order(sym, typ, side, amount, price, params)
            order.setdefault("status", "open")
            return order
        except Exception as e:
            raise RuntimeError(f"create_order failed: {e}")
