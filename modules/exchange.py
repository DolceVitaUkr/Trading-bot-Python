# modules/exchange.py

import time
import logging
from typing import List, Optional, Union, Dict, Any

import ccxt
import config
from modules.error_handler import APIError, OrderExecutionError

logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))


class ThrottleGuard:
    """
    Very light global request budget with bursts.
    - max_per_minute: soft cap; if exceeded we sleep a bit.
    - burst: allow short bursts without immediate sleep.
    """
    def __init__(self, max_per_minute: int = 90, burst: int = 15):
        self.max_per_minute = max_per_minute
        self.burst = burst
        self.window: List[float] = []  # timestamps of calls

    def tick(self):
        now = time.time()
        one_min_ago = now - 60
        # prune old
        self.window = [t for t in self.window if t >= one_min_ago]
        self.window.append(now)

        n = len(self.window)
        if n <= self.burst:
            return  # allow burst

        if n > self.max_per_minute:
            # simple backoff
            sleep_for = min(1.0, (n - self.max_per_minute) * 0.05)
            logger.debug(f"[ThrottleGuard] Sleeping {sleep_for:.2f}s (n={n})")
            time.sleep(sleep_for)


def _retry(times: int = 3, backoff: float = 1.5):
    """Simple retry decorator with exponential backoff."""
    def deco(fun):
        def wrapper(self, *args, **kwargs):
            delay = 1.0
            last_exc = None
            for attempt in range(times):
                try:
                    return fun(self, *args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if attempt < times - 1:
                        logger.warning(f"{fun.__name__} failed (attempt {attempt+1}/{times}): {e}")
                        time.sleep(delay)
                        delay *= backoff
                    else:
                        raise
            raise last_exc  # not reached
        return wrapper
    return deco


class ExchangeAPI:
    """
    Unified BYBIT exchange wrapper (Spot + Perp profiles).
    - Paper mode: uses **live market data**; simulates fills in-memory.
    - Live mode: real orders via CCXT.
    - SL/TP attach on open (venue-specific via params), and reconciliation on restart.
    - Precision/min-size helpers from market metadata.
    """

    def __init__(self, profile: str = None):
        """
        profile: "spot" | "perp" | "spot+perp"
        """
        self.simulation: bool = bool(getattr(config, "USE_SIMULATION", False))
        # Default profile comes from config.EXCHANGE_PROFILE if present
        self.profile: str = profile or getattr(config, "EXCHANGE_PROFILE", "spot")

        # ccxt client for Bybit (testnet enabled when simulation or USE_TESTNET)
        self.use_testnet: bool = bool(getattr(config, "USE_TESTNET", False)) or self.simulation

        self.client: Optional[ccxt.bybit] = None
        self.markets: Dict[str, Any] = {}
        self._throttle = ThrottleGuard(max_per_minute=90, burst=20)

        # Simulation state (paper fills) — still uses live prices
        self._sim_positions: Dict[str, Dict[str, Any]] = {}   # symbol -> pos dict
        self._sim_open_orders: Dict[str, List[Dict[str, Any]]] = {}  # symbol -> list
        self._sim_cash_usd: float = float(getattr(config, "SIMULATION_START_BALANCE", 1000.0))

        if not self.simulation:
            self._init_ccxt()

    def _init_ccxt(self):
        try:
            self.client = ccxt.bybit({
                "apiKey": config.BYBIT_API_KEY,
                "secret": config.BYBIT_API_SECRET,
                "enableRateLimit": True,
            })
            if self.use_testnet:
                self.client.set_sandbox_mode(True)

            self._throttle.tick()
            self.markets = self.client.load_markets()
            logger.info(f"Loaded {len(self.markets)} markets (sandbox={self.use_testnet})")

        except Exception as e:
            logger.error("Failed to init/load Bybit markets", exc_info=True)
            raise APIError("Market load failure", context={"exception": str(e)})

    # ---------- Market helpers ----------

    def _resolve_symbol(self, symbol: str) -> str:
        """
        Ensure symbol format matches CCXT's market keys, e.g., 'BTC/USDT'.
        """
        if "/" in symbol:
            return symbol
        # naive normalize: e.g. BTCUSDT -> BTC/USDT
        if symbol.endswith("USDT"):
            return f"{symbol[:-4]}/USDT"
        return symbol

    def _get_market(self, symbol: str) -> Dict[str, Any]:
        s = self._resolve_symbol(symbol)
        m = self.markets.get(s) if self.markets else None
        if not m and not self.simulation and self.client:
            # try reloading if not found
            self._throttle.tick()
            self.markets = self.client.load_markets()
            m = self.markets.get(s)
        if not m:
            # In simulation (no ccxt), we can’t know exact limits; return defaults
            return {
                "symbol": s,
                "limits": {
                    "amount": {"min": 0.0001},
                    "cost": {"min": float(getattr(config, "MIN_TRADE_AMOUNT_USD", 10.0))}
                },
                "precision": {"price": 2, "amount": 6}
            }
        return m

    def get_min_cost(self, symbol: str) -> float:
        m = self._get_market(symbol)
        return float(m.get("limits", {}).get("cost", {}).get("min", 10.0))

    def get_min_amount(self, symbol: str) -> float:
        m = self._get_market(symbol)
        return float(m.get("limits", {}).get("amount", {}).get("min", 0.0001))

    def get_price_precision(self, symbol: str) -> int:
        m = self._get_market(symbol)
        return int(m.get("precision", {}).get("price", 2))

    def get_amount_precision(self, symbol: str) -> int:
        m = self._get_market(symbol)
        return int(m.get("precision", {}).get("amount", 6))

    # ---------- Public market data ----------

    @_retry()
    def get_price(self, symbol: str) -> float:
        """Latest traded price (live even in simulation)."""
        sym = self._resolve_symbol(symbol)
        if self.client:
            try:
                self._throttle.tick()
                ticker = self.client.fetch_ticker(sym)
                return float(ticker["last"] or ticker["close"])
            except Exception as e:
                logger.error(f"get_price failed for {sym}: {e}", exc_info=True)
                raise APIError(f"Failed to fetch price for {sym}", context={"symbol": sym})
        # Fallback in pure-sim with no client (should not happen if ccxt init ok)
        # Use last sim entry price or 0.0
        pos = self._sim_positions.get(sym)
        return float(pos["entry_price"]) if pos else 0.0

    @_retry()
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        since: Optional[int] = None,
        limit: int = 1000
    ) -> List[List[Union[int, float]]]:
        """
        Historical candles (OHLCV) from exchange (live even in simulation).
        Returns: [timestamp, open, high, low, close, volume]
        """
        sym = self._resolve_symbol(symbol)
        if not self.client:
            raise APIError("OHLCV requires ccxt client (should be available)", context={"symbol": sym})
        try:
            self._throttle.tick()
            return self.client.fetch_ohlcv(sym, timeframe=timeframe, since=since, limit=limit)
        except Exception as e:
            logger.error(f"fetch_ohlcv failed for {sym}/{timeframe}: {e}", exc_info=True)
            raise APIError("Failed to fetch OHLCV", context={"symbol": sym, "timeframe": timeframe})

    def fetch_market_data(self, symbol: str, timeframe: str, since: Optional[int] = None, limit: int = 1000):
        """Alias for DataManager compatibility."""
        return self.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)

    # ---------- Account state ----------

    @_retry()
    def get_balance(self) -> float:
        """
        Total USDT (or quote) balance.
        - In simulation: simulated cash + unrealized PnL not included (pure cash).
        - In live: sum free + used for USDT.
        """
        if self.simulation or not self.client:
            return float(self._sim_cash_usd)
        try:
            self._throttle.tick()
            bal = self.client.fetch_balance()
            usdt = bal.get("USDT") or bal.get("total", {}).get("USDT")
            if isinstance(usdt, dict):
                return float(usdt.get("free", 0.0) + usdt.get("used", 0.0))
            return float(usdt or 0.0)
        except Exception as e:
            logger.error(f"get_balance failed: {e}", exc_info=True)
            raise APIError("Failed to fetch balance")

    @_retry()
    def list_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        if self.simulation:
            if symbol:
                return list(self._sim_open_orders.get(self._resolve_symbol(symbol), []))
            # flatten
            all_orders = []
            for v in self._sim_open_orders.values():
                all_orders.extend(v)
            return all_orders

        try:
            self._throttle.tick()
            return self.client.fetch_open_orders(symbol=self._resolve_symbol(symbol) if symbol else None)
        except Exception as e:
            logger.error(f"list_open_orders failed: {e}", exc_info=True)
            raise APIError("Failed to fetch open orders")

    @_retry()
    def fetch_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        if self.simulation:
            if symbol:
                sym = self._resolve_symbol(symbol)
                pos = self._sim_positions.get(sym)
                return [pos] if pos else []
            return list(self._sim_positions.values())

        try:
            self._throttle.tick()
            positions = self.client.fetch_positions(symbols=[self._resolve_symbol(symbol)] if symbol else None)
            return positions or []
        except Exception as e:
            logger.error(f"fetch_positions failed: {e}", exc_info=True)
            raise APIError("Failed to fetch positions")

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Convenience: return single position or None."""
        ps = self.fetch_positions(symbol)
        return ps[0] if ps else None

    # ---------- Order placement & OCO/TP-SL attach ----------

    def _round_amount(self, symbol: str, amount: float) -> float:
        prec = self.get_amount_precision(symbol)
        return float(f"{amount:.{prec}f}")

    def _round_price(self, symbol: str, price: float) -> float:
        prec = self.get_price_precision(symbol)
        fmt = f"{{:.{prec}f}}"
        return float(fmt.format(price))

    @_retry()
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
        reduce_only: bool = False,
        params: Optional[dict] = None
    ) -> Dict[str, Any]:
        """
        Create an order. Always respects min-cost/precision; will raise if below.
        - In paper mode: simulate fill at market/limit price, maintain SL/TP as shadow OCO.
        - In live: pass venue-specific params to attach SL/TP (best-effort via CCXT).
        """
        sym = self._resolve_symbol(symbol)
        params = params or {}

        # Validate minimum cost/amount against live price
        mkt_price = self.get_price(sym)
        min_cost = self.get_min_cost(sym)
        notional = (price or mkt_price) * amount
        if notional < max(min_cost, float(getattr(config, "MIN_TRADE_AMOUNT_USD", 10.0))):
            raise OrderExecutionError(
                f"Order notional too small: {notional:.2f} < min {min_cost:.2f}",
                context={"symbol": sym, "amount": amount, "price": price or mkt_price}
            )

        # Round amount/price
        amount = self._round_amount(sym, amount)
        if price is not None:
            price = self._round_price(sym, price)

        if self.simulation:
            return self._simulate_order(
                sym, side.lower(), order_type.lower(), amount, price,
                attach_sl=attach_sl, attach_tp=attach_tp, reduce_only=reduce_only
            )

        # Live path
        try:
            # Bybit via CCXT supports takeProfit/stopLoss for swaps on some endpoints;
            # for spot we place base order, then separate protective orders if needed.
            order_params = dict(params)
            if attach_tp is not None:
                order_params.setdefault("takeProfit", float(attach_tp))
            if attach_sl is not None:
                order_params.setdefault("stopLoss", float(attach_sl))
            if reduce_only:
                order_params.setdefault("reduceOnly", True)

            self._throttle.tick()
            order = self.client.create_order(sym, order_type, side, amount, price, order_params)

            # Fallback: if venue rejected inline TP/SL, place separate stop/limit orders.
            if (attach_sl is not None or attach_tp is not None):
                try:
                    self._ensure_protective_orders_live(sym, side.lower(), amount, attach_sl, attach_tp)
                except Exception as sub_e:
                    logger.warning(f"Failed to attach protective orders via fallback: {sub_e}")

            return order
        except Exception as e:
            logger.error(f"create_order failed for {sym}: {e}", exc_info=True)
            raise OrderExecutionError("Order execution failed", context={"symbol": sym, "side": side})

    def _ensure_protective_orders_live(
        self, symbol: str, side: str, amount: float,
        sl: Optional[float], tp: Optional[float]
    ):
        """
        Fallback protective orders for live venue if inline attach is not supported.
        Places separate stop (SL) and limit (TP) orders with reduceOnly when applicable.
        """
        if sl is None and tp is None:
            return
        # For a LONG (buy) entry, SL is a sell-stop, TP is a sell-limit. Vice versa for SHORT.
        opp_side = "sell" if side in ("buy", "long") else "buy"
        if sl is not None:
            params = {"reduceOnly": True}
            self._throttle.tick()
            self.client.create_order(symbol, "stop", opp_side, amount, sl, params)
        if tp is not None:
            params = {"reduceOnly": True}
            self._throttle.tick()
            self.client.create_order(symbol, "limit", opp_side, amount, tp, params)

    # ---------- Simulation (paper) fills with live prices ----------

    def _sim_mark_to_market(self, symbol: str):
        pos = self._sim_positions.get(symbol)
        if not pos:
            return
        last = self.get_price(symbol)
        pos["unrealized_pnl"] = (
            (last - pos["entry_price"]) * pos["quantity"]
            if pos["side"] == "long"
            else (pos["entry_price"] - last) * pos["quantity"]
        )

    def _append_sim_order(self, symbol: str, order: Dict[str, Any]):
        self._sim_open_orders.setdefault(symbol, []).append(order)

    def _simulate_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: Optional[float],
        *,
        attach_sl: Optional[float],
        attach_tp: Optional[float],
        reduce_only: bool
    ) -> Dict[str, Any]:
        now = int(time.time() * 1000)
        last_price = self.get_price(symbol)
        fill_price = float(price or last_price)

        pos = self._sim_positions.get(symbol)
        opening_side = "long" if side in ("buy", "long") else "short"
        closing_side = "sell" if opening_side == "long" else "buy"

        # Closing logic if reduce_only or opposite direction
        if reduce_only or (pos and pos["side"] != opening_side):
            if not pos:
                return {"status": "no_position", "symbol": symbol, "time": now}

            qty_to_close = min(amount, pos["quantity"])
            pnl = (
                (fill_price - pos["entry_price"]) * qty_to_close
                if pos["side"] == "long"
                else (pos["entry_price"] - fill_price) * qty_to_close
            )
            self._sim_cash_usd += float(pnl)

            pos["quantity"] = round(pos["quantity"] - qty_to_close, 12)
            closed = {
                "symbol": symbol,
                "side": closing_side,
                "quantity": qty_to_close,
                "price": fill_price,
                "time": now,
                "pnl": pnl,
                "status": "closed_partial" if pos["quantity"] > 0 else "closed",
            }
            if pos["quantity"] <= 0:
                del self._sim_positions[symbol]
            return closed

        # Opening / adding
        cost = fill_price * amount
        if cost > self._sim_cash_usd:
            raise OrderExecutionError(
                "Insufficient simulated cash",
                context={"needed": cost, "cash": self._sim_cash_usd}
            )

        self._sim_cash_usd -= cost

        if pos and pos["side"] == opening_side:
            # Adjust avg price
            new_qty = pos["quantity"] + amount
            pos["entry_price"] = (pos["entry_price"] * pos["quantity"] + fill_price * amount) / new_qty
            pos["quantity"] = new_qty
            pos["time"] = now
        else:
            # New position
            self._sim_positions[symbol] = {
                "symbol": symbol,
                "side": opening_side,
                "quantity": amount,
                "entry_price": fill_price,
                "time": now,
                "unrealized_pnl": 0.0,
                "sl": attach_sl,
                "tp": attach_tp
            }

        # Shadow attach OCO as open orders for simulation
        if attach_sl is not None:
            self._append_sim_order(symbol, {
                "id": f"sim-sl-{now}",
                "type": "stop",
                "side": "sell" if opening_side == "long" else "buy",
                "price": float(attach_sl),
                "amount": float(amount),
                "reduceOnly": True,
                "status": "open"
            })
            self._sim_positions[symbol]["sl"] = float(attach_sl)
        if attach_tp is not None:
            self._append_sim_order(symbol, {
                "id": f"sim-tp-{now}",
                "type": "limit",
                "side": "sell" if opening_side == "long" else "buy",
                "price": float(attach_tp),
                "amount": float(amount),
                "reduceOnly": True,
                "status": "open"
            })
            self._sim_positions[symbol]["tp"] = float(attach_tp)

        return {
            "symbol": symbol,
            "side": opening_side,
            "quantity": self._sim_positions[symbol]["quantity"],
            "entry_price": self._sim_positions[symbol]["entry_price"],
            "status": "open",
            "time": now
        }

    # ---------- Reconciliation / restart safety ----------

    @_retry()
    def reconcile_open_state(self) -> Dict[str, Any]:
        """
        On startup: pull open positions and open orders, ensure SL/TP attached.
        - Paper: just validates shadow SL/TP objects.
        - Live: checks venue for missing protections; attempts to re-attach.
        Returns a summary dict.
        """
        summary: Dict[str, Any] = {"positions": [], "orders": [], "actions": []}

        if self.simulation:
            # Validate sim SL/TP presence (already stored in position)
            for sym, pos in list(self._sim_positions.items()):
                self._sim_mark_to_market(sym)
                sl, tp = pos.get("sl"), pos.get("tp")
                if sl is None or tp is None:
                    summary["actions"].append({"symbol": sym, "action": "missing_protection_sim"})
                summary["positions"].append(pos)
            for sym, orders in self._sim_open_orders.items():
                summary["orders"].extend(orders)
            return summary

        try:
            # Live: fetch per symbol (limited loop is fine at startup)
            symbols = [m for m in self.markets.keys() if m.endswith("/USDT")]
            for sym in symbols:
                self._throttle.tick()
                positions = self.fetch_positions(sym)
                if not positions:
                    continue
                summary["positions"].extend(positions)

                open_orders = self.list_open_orders(sym)
                summary["orders"].extend(open_orders)

                # Heuristic: if open position but no reduceOnly stop/limit on opposite side, attach
                has_sl = any(o for o in open_orders if o.get("type") in ("stop", "stop_loss") and o.get("reduceOnly"))
                has_tp = any(o for o in open_orders if o.get("type") in ("limit", "take_profit") and o.get("reduceOnly"))
                if positions and (not has_sl or not has_tp):
                    pos_qty = abs(float(
                        positions[0].get("contracts")
                        or positions[0].get("contractsSize")
                        or positions[0].get("amount")
                        or 0
                    ))
                    side = positions[0].get("side", "").lower()  # "long"|"short"
                    last = self.get_price(sym)
                    sl = last * (0.98 if side == "long" else 1.02)
                    tp = last * (1.02 if side == "long" else 0.98)
                    try:
                        self._ensure_protective_orders_live(sym, "buy" if side == "long" else "sell", pos_qty, sl, tp)
                        summary["actions"].append({"symbol": sym, "action": "reattach_sl_tp"})
                    except Exception as sub_e:
                        logger.warning(f"Failed to re-attach SL/TP for {sym}: {sub_e}")
                        summary["actions"].append({"symbol": sym, "action": "reattach_failed"})
            return summary
        except Exception as e:
            logger.error(f"reconcile_open_state failed: {e}", exc_info=True)
            raise APIError("Failed to reconcile state")

    # ---------- Cancel helpers ----------

    @_retry()
    def cancel_all(self, symbol: Optional[str] = None) -> int:
        """Cancel all open orders, returns count."""
        sym = self._resolve_symbol(symbol) if symbol else None
        count = 0
        if self.simulation:
            if sym:
                count = len(self._sim_open_orders.get(sym, []))
                self._sim_open_orders[sym] = []
            else:
                for k in list(self._sim_open_orders.keys()):
                    count += len(self._sim_open_orders[k])
                    self._sim_open_orders[k] = []
            return count

        try:
            orders = self.list_open_orders(sym)
            for o in orders:
                oid = o.get("id")
                if oid:
                    self._throttle.tick()
                    self.client.cancel_order(oid, sym)
                    count += 1
            return count
        except Exception as e:
            logger.error(f"cancel_all failed: {e}", exc_info=True)
            raise APIError("Failed to cancel orders", context={"symbol": sym})

    # ---------- Close / teardown ----------

    async def close(self):
        """For parity with ws adapters; ccxt has no persistent connection to close here."""
        return
