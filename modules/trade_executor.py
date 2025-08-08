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
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))


class TradeExecutor:
    """
    Executes trades in live or simulation mode.

    - Live mode: proxies to ExchangeAPI (CCXT).
    - Simulation mode: uses ExchangeAPI's in-memory fills but still applies our fee model.
    - Uses exchange helpers for min-notional and precision.
    - Optional SL/TP attachment (inline on live if supported; shadow in paper).
    - Telegram notifications (trade + alert).
    """

    def __init__(self, simulation_mode: Optional[bool] = None, notifier: Optional[TelegramNotifier] = None):
        self.simulation_mode = config.USE_SIMULATION if simulation_mode is None else simulation_mode
        self.exchange = ExchangeAPI()
        # Use sync notifier for deterministic logs; you can swap to async if preferred
        self.notifier = notifier or TelegramNotifier(disable_async=True)
        self.fee_rate = float(getattr(config, "FEE_PERCENTAGE", 0.002))

    # ---------------------------- Public API ---------------------------- #

    def get_balance(self) -> float:
        """
        Return current account balance:
        - paper: simulated USD cash from ExchangeAPI
        - live: venue wallet (USDT total)
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
        quantity: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Close an open position (market), optionally overriding price/quantity in paper mode.
        - live: we simulate a reduce-only market close by placing the opposite side.
        - paper: rely on _simulate_order with reduce_only=True; apply exit fees here.
        """
        sym = self.exchange._resolve_symbol(symbol)
        now_iso = datetime.now(timezone.utc).isoformat()

        if self.simulation_mode:
            # derive qty if not given
            pos = getattr(self.exchange, "_sim_positions", {}).get(sym)
            if not pos:
                return None
            qty = float(quantity or pos["quantity"])
            side_for_close = "sell" if pos["side"] == "long" else "buy"
            px = float(price or self.exchange.get_price(sym) or pos["entry_price"])

            # simulate reduce-only close
            res = self.exchange._simulate_order(
                sym, side_for_close, "market", qty, px, attach_sl=None, attach_tp=None, reduce_only=True
            )
            # apply exit fee in paper
            try:
                exit_notional = px * qty
                exit_fee = exit_notional * self.fee_rate
                # Deduct exit fee from sim cash
                self.exchange._sim_cash_usd -= exit_fee  # type: ignore[attr-defined]
            except Exception:
                pass

            self._notify_trade(
                f"SIM CLOSE {sym} x{qty:.8f} @ {px} | status={res.get('status')} | pnl={res.get('pnl', 0):.4f}"
            )
            res["timestamp"] = now_iso
            return res

        # live: fetch current position size then submit reduce-only market order
        try:
            pos_list = self.exchange.fetch_positions(sym)
            if not pos_list:
                return None
            p = pos_list[0]
            side = p.get("side", "").lower()  # "long"/"short"
            qty_live = abs(float(
                p.get("contracts") or p.get("contractsSize") or p.get("amount") or 0.0
            ))
            if qty_live <= 0:
                return None
            close_side = "sell" if side == "long" else "buy"
            px_live = float(price or self.exchange.get_price(sym))
            order = self.exchange.create_order(
                sym, "market", close_side, qty_live, px_live, reduce_only=True
            )
            self._notify_trade(f"LIVE CLOSE {sym} x{qty_live:.8f} | id={order.get('id')}")
            order["timestamp"] = now_iso
            return order
        except Exception as e:
            self._notify_alert(f"Failed to close live position {sym}: {e}")
            raise

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
        risk_close: bool = False
    ) -> Dict[str, Any]:
        """
        Execute a trade (open/add/close).
        - Uses exchange min-notional + precision helpers.
        - Paper: fees applied locally; SL/TP tracked as shadow orders in ExchangeAPI.
        - Live: attempts inline SL/TP attach; falls back to separate reduce-only orders.

        Returns a dict with normalized keys:
          status, symbol, side, quantity, entry_price, exit_price (paper close), profit/fees/return_pct (paper),
          order_id (live), timestamp
        """
        sym = self.exchange._resolve_symbol(symbol)
        side = side.lower()
        now_iso = datetime.now(timezone.utc).isoformat()

        # Resolve a usable price (for validation/precision)
        mkt_price = None
        if price is not None:
            mkt_price = float(price)
        else:
            try:
                mkt_price = float(self.exchange.get_price(sym))
            except Exception as e:
                raise OrderExecutionError(f"Cannot fetch price for {sym}: {e}")

        if not mkt_price or mkt_price <= 0:
            raise OrderExecutionError(f"Invalid price for {sym}: {mkt_price}")

        # Default quantity based on min-notional if not provided
        min_cost = self.exchange.get_min_cost(sym)
        min_usd = max(min_cost, float(getattr(config, "MIN_TRADE_AMOUNT_USD", 10.0)))
        if quantity is None:
            quantity = max(min_usd / mkt_price, 10 ** -self.exchange.get_amount_precision(sym))

        # Round price/amount to venue precision
        qty_rounded = float(f"{quantity:.{self.exchange.get_amount_precision(sym)}f}")
        px_rounded = float(f"{mkt_price:.{self.exchange.get_price_precision(sym)}f}")

        # --------- Simulation path --------- #
        if self.simulation_mode:
            try:
                # Sim fill; ExchangeAPI manages simulated cash for cost/PNL. We add fees ourselves.
                res = self.exchange._simulate_order(
                    sym, side, order_type.lower(), qty_rounded, px_rounded,
                    attach_sl=attach_sl, attach_tp=attach_tp, reduce_only=risk_close
                )

                # Calculate and deduct fees on the notional involved in this action
                try:
                    if res.get("status") in ("open", "closed_partial"):
                        # opening/add costs
                        open_notional = px_rounded * qty_rounded
                        open_fee = open_notional * self.fee_rate
                        self.exchange._sim_cash_usd -= open_fee  # type: ignore[attr-defined]
                    # For a closing event, we also charge exit fee
                    if res.get("status") in ("closed", "closed_partial"):
                        exit_notional = px_rounded * float(res.get("quantity", qty_rounded))
                        exit_fee = exit_notional * self.fee_rate
                        self.exchange._sim_cash_usd -= exit_fee  # type: ignore[attr-defined]
                except Exception:
                    pass

                # Build normalized response
                if res.get("status") in ("closed", "closed_partial"):
                    # We don't have entry/exit split in res; expose price and pnl
                    pnl = float(res.get("pnl", 0.0))
                    calc = calculate_trade_result(
                        entry_price=px_rounded,  # approximation for reporting; true entry is tracked inside exchange
                        exit_price=px_rounded,
                        quantity=float(res.get("quantity", qty_rounded)),
                        fee_percentage=self.fee_rate
                    )
                    msg = (
                        f"SIM {side.upper()} {sym} x{qty_rounded:.8f} @ {px_rounded} "
                        f"| status={res.get('status')} | pnl={pnl:.4f}"
                    )
                    self._notify_trade(msg)
                    return {
                        "status": res.get("status", "simulated"),
                        "symbol": sym,
                        "side": side,
                        "quantity": float(res.get("quantity", qty_rounded)),
                        "entry_price": float(res.get("price", px_rounded)),
                        "exit_price": float(res.get("price", px_rounded)),
                        "profit": float(pnl),
                        "fees": float(calc["fees"]),
                        "return_pct": float(calc["return_pct"]),
                        "timestamp": now_iso,
                    }

                # Open/added
                self._notify_trade(
                    f"SIM {side.upper()} {sym} x{qty_rounded:.8f} @ {px_rounded} | status={res.get('status')}"
                )
                return {
                    "status": res.get("status", "open"),
                    "symbol": sym,
                    "side": side,
                    "quantity": float(res.get("quantity", qty_rounded)),
                    "entry_price": float(res.get("entry_price", px_rounded)),
                    "exit_price": None,
                    "profit": 0.0,
                    "fees": 0.0,
                    "return_pct": 0.0,
                    "timestamp": now_iso,
                }

            except (RiskViolationError, OrderExecutionError, APIError):
                raise
            except Exception as e:
                self._notify_alert(f"Simulation error for {sym} {side}: {e}")
                raise

        # --------- Live path --------- #
        try:
            order = self.exchange.create_order(
                sym, order_type, side, qty_rounded, px_rounded,
                attach_sl=attach_sl, attach_tp=attach_tp, reduce_only=risk_close
            )
            self._notify_trade(f"LIVE {side.upper()} {sym} x{qty_rounded:.8f} @ {px_rounded} | id={order.get('id')}")
            return {
                "status": order.get("status", "open"),
                "symbol": order.get("symbol", sym),
                "side": order.get("side", side),
                "quantity": float(order.get("amount", qty_rounded)),
                "entry_price": float(order.get("price", px_rounded)),
                "exit_price": None,
                "profit": None,
                "fees": None,
                "return_pct": None,
                "order_id": order.get("id"),
                "timestamp": now_iso,
            }
        except Exception as e:
            self._notify_alert(f"Live order failed for {sym} {side}: {e}")
            raise OrderExecutionError(str(e))

    # ---------------------------- Internals ---------------------------- #

    def _notify_trade(self, text: str):
        try:
            self.notifier.send_message_sync(text, format="trade")
        except Exception as e:
            logger.warning(f"Telegram trade notify failed: {e}")

    def _notify_alert(self, text: str):
        try:
            self.notifier.send_message_sync({"type": "ALERT", "message": text}, format="alert")
        except Exception as e:
            logger.warning(f"Telegram alert notify failed: {e}")
