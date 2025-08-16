# modules/trade_executor.py

import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import config
from modules.exchange import ExchangeAPI
from modules.telegram_bot import TelegramNotifier
from modules.trade_calculator import calculate_trade_result
from modules.error_handler import OrderExecutionError, RiskViolationError, APIError

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
log_level_str = str(getattr(config, "LOG_LEVEL", "INFO"))
log_level = getattr(logging, log_level_str, logging.INFO)
logger.setLevel(log_level)


class TradeExecutor:
    """
    Executes trades in live or simulation mode.
    """

    def __init__(
        self,
        simulation_mode: Optional[bool] = None,
        notifier: Optional[TelegramNotifier] = None,
        risk_manager: Optional['RiskManager'] = None,
        notifications: Optional[Any] = None,
    ):
        """
        Initializes the TradeExecutor.
        """
        default_sim = config.USE_SIMULATION
        self.simulation_mode = default_sim if simulation_mode is None \
            else simulation_mode

        self.exchange = ExchangeAPI()
        self.risk_manager = risk_manager
        self.notifier = notifier or TelegramNotifier(
            disable_async=not getattr(config, "ASYNC_TELEGRAM", True))
        self.notifications = notifications
        self.fee_rate = float(getattr(config, "FEE_PERCENTAGE", 0.002))

        # Simulation state
        self.total_trades = 0
        self.winning_trades = 0
        self.realized_pnl = 0.0

    # ---------------------------- Public API ---------------------------- #

    def get_balance(self) -> float:
        """
        Return current account balance.
        """
        try:
            return float(self.exchange.get_balance())
        except Exception as e:
            logger.warning(f"get_balance failed: {e}")
            return 0.0

    def unrealized_pnl(self, symbol: str) -> float:
        """
        Unrealized PnL (paper only); returns 0 for live.
        """
        if not self.simulation_mode:
            return 0.0
        try:
            sym = self.exchange._resolve_symbol(symbol)
            pos = getattr(self.exchange, "_sim_positions", {}).get(sym)
            if not pos:
                return 0.0
            last = self.exchange.get_price(sym) or pos["entry_price"]
            qty = float(pos["quantity"])
            if pos["side"] == "long":
                return (last - pos["entry_price"]) * qty
            return (pos["entry_price"] - last) * qty
        except Exception as e:
            logger.debug(f"unrealized_pnl error: {e}")
            return 0.0

    def close_position(
        self,
        symbol: str,
        *,
        price: Optional[float] = None,
        quantity: Optional[float] = None,
        reduce_only: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Close an open position (market).
        """
        sym = self.exchange._resolve_symbol(symbol)

        if self.simulation_mode:
            return self._handle_simulation_close(sym, price, quantity, reduce_only)
        else:
            return self._handle_live_close(sym, price, reduce_only)

    def execute_order(
        self,
        symbol: str,
        side: str,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
        order_type: str = "market",
        *,
        attach_sl: Optional[float] = None,
        attach_tp: Optional[float] = None,
        reduce_only: bool = False
    ) -> Dict[str, Any]:
        """
        Execute a trade (open/add/close).
        """
        sym = self.exchange._resolve_symbol(symbol)
        side = side.lower()

        mkt_price = float(price) if price is not None else None
        if mkt_price is None:
            try:
                mkt_price = float(self.exchange.get_price(sym))
            except Exception as e:
                raise OrderExecutionError(f"Cannot fetch price for {sym}: {e}")
        if not mkt_price or mkt_price <= 0:
            raise OrderExecutionError(f"Invalid price for {sym}: {mkt_price}")

        min_cost = self.exchange.get_min_cost(sym)
        min_usd = max(
            min_cost, float(getattr(config, "MIN_TRADE_AMOUNT_USD", 10.0)))
        if quantity is None:
            quantity = max(
                min_usd / mkt_price,
                10 ** -self.exchange.get_amount_precision(sym))

        qty_rounded = float(
            f"{quantity:.{self.exchange.get_amount_precision(sym)}f}")
        px_rounded = float(
            f"{mkt_price:.{self.exchange.get_price_precision(sym)}f}")

        if self.simulation_mode:
            return self._handle_simulation_order(
                sym, side, order_type, qty_rounded, px_rounded,
                attach_sl, attach_tp, reduce_only)
        else:
            return self._handle_live_order(
                sym, side, order_type, qty_rounded, px_rounded,
                attach_sl, attach_tp, reduce_only)

    # ---------------------------- Internals ---------------------------- #

    def _handle_simulation_close(self, sym, price, quantity, reduce_only=True):
        now_iso = datetime.now(timezone.utc).isoformat()
        pos = getattr(self.exchange, "_sim_positions", {}).get(sym)
        if not pos:
            return {"status": "noop"}
        qty = float(quantity or pos["quantity"])
        side_for_close = "sell" if pos["side"] == "long" else "buy"
        px = float(price or self.exchange.get_price(sym) or pos["entry_price"])

        res = self.exchange._simulate_order(
            sym, side_for_close, "market", qty, px,
            attach_sl=None, attach_tp=None, reduce_only=reduce_only
        )
        try:
            exit_notional = px * qty
            exit_fee = exit_notional * self.fee_rate
            self.exchange._sim_cash_usd -= exit_fee
        except Exception:
            pass

        pnl = float(res.get("pnl", 0.0))
        self.realized_pnl += pnl
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1

        # Notify RiskManager of trade closure
        if self.risk_manager:
            self.risk_manager.unregister_position(sym)
            self.risk_manager.record_trade_closure(pnl, self.exchange._sim_cash_usd)

        event = {
            "symbol": sym, "side": side_for_close, "qty": qty, "price": px,
            "status": res.get("status"), "opened": None, "closed": now_iso,
            "pnl": pnl, "return_pct": None, "leverage": None,
            "meta": {"mode": "paper", "action": "close"},
        }
        self._emit_trade_event(event)
        res["timestamp"] = now_iso
        return res

    def _handle_live_close(self, sym, price, reduce_only=True):
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            pos_list = self.exchange.fetch_positions(sym)
            if not pos_list:
                return None
            p = pos_list[0]
            side = p.get("side", "").lower()
            qty_live = abs(float(
                p.get("contracts") or p.get("contractsSize") or
                p.get("amount") or 0.0
            ))
            if qty_live <= 0:
                return None
            close_side = "sell" if side == "long" else "buy"
            px_live = float(price or self.exchange.get_price(sym))
            order = self.exchange.create_order(
                sym, "market", close_side, qty_live, px_live, reduce_only=reduce_only
            )

            event = {
                "symbol": sym, "side": close_side, "qty": qty_live,
                "price": px_live, "status": order.get("status", "closed"),
                "opened": None, "closed": now_iso, "pnl": None,
                "return_pct": None, "leverage": None,
                "meta": {
                    "mode": "live", "action": "close",
                    "order_id": order.get("id")},
            }
            self._emit_trade_event(event)
            order["timestamp"] = now_iso
            return order
        except Exception as e:
            self._notify_alert(f"Failed to close live position {sym}: {e}")
            raise

    def _handle_simulation_order(
            self, sym, side, order_type, qty_rounded, px_rounded,
            attach_sl, attach_tp, reduce_only):
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            res = self.exchange._simulate_order(
                sym, side, order_type.lower(), qty_rounded, px_rounded,
                attach_sl=attach_sl, attach_tp=attach_tp,
                reduce_only=reduce_only
            )

            try:
                if res.get("status") in (
                        "open", "open_added", "open_increased"):
                    open_notional = px_rounded * qty_rounded
                    self.exchange._sim_cash_usd -= open_notional * self.fee_rate
                if res.get("status") in ("closed", "closed_partial"):
                    exit_notional = px_rounded * float(
                        res.get("quantity", qty_rounded))
                    self.exchange._sim_cash_usd -= exit_notional * self.fee_rate
            except Exception:
                pass

            if res.get("status") in ("closed", "closed_partial"):
                pnl = float(res.get("pnl", 0.0))
                self.realized_pnl += pnl
                self.total_trades += 1
                if pnl > 0:
                    self.winning_trades += 1

                # Notify RiskManager of trade closure
                if self.risk_manager:
                    self.risk_manager.unregister_position(sym)
                    self.risk_manager.record_trade_closure(pnl, self.exchange._sim_cash_usd)

                calc = calculate_trade_result(
                    entry_price=px_rounded,
                    exit_price=px_rounded,
                    quantity=float(res.get("quantity", qty_rounded)),
                    fee_percentage=self.fee_rate
                )
                event = {
                    "symbol": sym, "side": side,
                    "qty": float(res.get("quantity", qty_rounded)),
                    "price": px_rounded, "status": res.get("status"),
                    "opened": None, "closed": now_iso, "pnl": pnl,
                    "return_pct": float(calc["return_pct"]), "leverage": None,
                    "meta": {"mode": "paper"},
                }
                self._emit_trade_event(event)

                return {
                    "status": res.get("status", "simulated"), "symbol": sym,
                    "side": side,
                    "quantity": float(res.get("quantity", qty_rounded)),
                    "entry_price": float(res.get("price", px_rounded)),
                    "exit_price": float(res.get("price", px_rounded)),
                    "profit": float(pnl), "fees": float(calc["fees"]),
                    "return_pct": float(calc["return_pct"]),
                    "timestamp": now_iso,
                }

            event = {
                "symbol": sym, "side": side,
                "qty": float(res.get("quantity", qty_rounded)),
                "price": float(res.get("entry_price", px_rounded)),
                "status": res.get("status", "open"), "opened": now_iso,
                "closed": None, "pnl": None, "return_pct": None,
                "leverage": None, "meta": {"mode": "paper"},
            }
            self._emit_trade_event(event)

            return {
                "status": res.get("status", "open"), "symbol": sym,
                "side": side,
                "quantity": float(res.get("quantity", qty_rounded)),
                "entry_price": float(res.get("entry_price", px_rounded)),
                "exit_price": None, "profit": 0.0, "fees": 0.0,
                "return_pct": 0.0, "timestamp": now_iso,
            }

        except (RiskViolationError, OrderExecutionError, APIError):
            raise
        except Exception as e:
            self._notify_alert(f"Simulation error for {sym} {side}: {e}")
            raise

    def _handle_live_order(
            self, sym, side, order_type, qty_rounded, px_rounded,
            attach_sl, attach_tp, reduce_only):
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            order = self.exchange.create_order(
                sym, order_type, side, qty_rounded, px_rounded,
                attach_sl=attach_sl, attach_tp=attach_tp,
                reduce_only=reduce_only
            )

            event = {
                "symbol": sym, "side": side,
                "qty": float(order.get("amount", qty_rounded)),
                "price": float(order.get("price", px_rounded)),
                "status": order.get("status", "open"), "opened": now_iso,
                "closed": None, "pnl": None, "return_pct": None,
                "leverage": None,
                "meta": {"mode": "live", "order_id": order.get("id")},
            }
            self._emit_trade_event(event)

            return {
                "status": order.get("status", "open"),
                "symbol": order.get("symbol", sym),
                "side": order.get("side", side),
                "quantity": float(order.get("amount", qty_rounded)),
                "entry_price": float(order.get("price", px_rounded)),
                "exit_price": None, "profit": None, "fees": None,
                "return_pct": None, "order_id": order.get("id"),
                "timestamp": now_iso,
            }
        except Exception as e:
            self._notify_alert(f"Live order failed for {sym} {side}: {e}")
            raise OrderExecutionError(str(e))

    def _emit_trade_event(self, event: Dict[str, Any]):
        """
        Prefer NotificationManager-style dispatch.
        """
        if self.notifications and hasattr(self.notifications, "notify_trade"):
            try:
                self.notifications.notify_trade(event)
                return
            except Exception as e:
                logger.warning(f"notifications.notify_trade failed: {e}")

        side = str(event.get("side", "")).upper()
        sym = event.get("symbol")
        qty = event.get("qty")
        price = event.get("price")
        status = event.get("status", "")
        pnl = event.get("pnl")
        ret = event.get("return_pct")
        pieces = [
            f"Pair: {sym}", f"Amount: {qty}",
            f"Price: {price}", f"Status: {status}",
        ]
        if pnl is not None:
            pieces.append(f"PnL: {pnl:.4f}")
        if ret is not None:
            pieces.append(f"Return: {ret:.2f}%")
        text = f"📊 TRADE {side}\n" + "\n".join(pieces)
        self._notify_trade(text)

    def _notify_trade(self, text: str):
        try:
            self.notifier.send_message_sync(text, format="trade")
        except Exception as e:
            logger.warning(f"Telegram trade notify failed: {e}")

    def _notify_alert(self, text: str):
        if self.notifications and hasattr(self.notifications, "notify_error"):
            try:
                self.notifications.notify_error(text)
            except Exception:
                pass
        try:
            self.notifier.send_message_sync(
                {"type": "ALERT", "message": text}, format="alert")
        except Exception as e:
            logger.warning(f"Telegram alert notify failed: {e}")

    # ---------------------------- Simulation Helpers ---------------------------- #
    def reset_simulation(self):
        """
        Resets the simulation state for a new training run.
        """
        if not self.simulation_mode:
            return

        initial_balance = float(config.SIMULATION_START_BALANCE)
        self.exchange._sim_cash_usd = initial_balance
        self.exchange._sim_positions = {}
        self.total_trades = 0
        self.winning_trades = 0
        self.realized_pnl = 0.0
        logger.info(f"Simulation reset. Balance: {initial_balance}")

    def get_pnl(self) -> float:
        """Returns the total realized PnL for the simulation run."""
        if not self.simulation_mode:
            return 0.0
        return self.realized_pnl

    def get_total_trades(self) -> int:
        """Returns the total number of trades executed in the simulation."""
        if not self.simulation_mode:
            return 0
        return self.total_trades

    def get_win_rate(self) -> float:
        """Returns the win rate for the simulation run."""
        if not self.simulation_mode or self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades
