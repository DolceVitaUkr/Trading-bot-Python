# modules/trade_executor.py

import logging
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import config
from modules.exchange import ExchangeAPI
from modules.telegram_bot import TelegramNotifier
from modules.trade_calculator import calculate_trade_result
from modules.error_handler import OrderExecutionError, RiskViolationError

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


class TradeExecutor:
    """
    Executes trades in live or simulation mode.

    - Live mode: proxies to ExchangeAPI (CCXT).
    - Simulation mode: tracks cash balance and positions, uses ExchangeAPI's
      in-memory position logic and updates wallet P&L locally.
    - Sends Telegram notifications for each action.
    """

    def __init__(self, simulation_mode: Optional[bool] = None, notifier: Optional[TelegramNotifier] = None):
        self.simulation_mode = config.USE_SIMULATION if simulation_mode is None else simulation_mode
        self.exchange = ExchangeAPI()
        self.notifier = notifier or TelegramNotifier(disable_async=True)

        # Paper wallet balance for simulation
        self.fee_rate = getattr(config, "FEE_PERCENTAGE", 0.002)
        if self.simulation_mode:
            self._starting_balance = float(getattr(config, "SIMULATION_START_BALANCE", 1000.0))
            self._cash_balance = self._starting_balance
        else:
            self._starting_balance = None
            self._cash_balance = None  # unknown in live mode

    # ---------------------------- Public API ---------------------------- #

    def get_balance(self) -> float:
        """Return current cash balance (simulation) or 0.0 in live (placeholder)."""
        if self.simulation_mode:
            return float(self._cash_balance)
        # For live mode, wiring to exchange wallet/balance can be added later
        return 0.0

    def unrealized_pnl(self, symbol: str) -> float:
        """Unrealized PnL for the given symbol in simulation."""
        if not self.simulation_mode:
            return 0.0
        pos = self.exchange.positions.get(symbol)
        if not pos:
            return 0.0
        # Use provided get_price (may be 0.0 if unknown in pure-sim)
        current_price = self.exchange.get_price(symbol) or pos["entry_price"]
        qty = float(pos["quantity"])
        side = pos["side"]
        if side == "long":
            return (current_price - pos["entry_price"]) * qty
        else:
            return (pos["entry_price"] - current_price) * qty

    def close_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Close an open position at market (simulation or live)."""
        try:
            result = self.exchange.close_position(symbol)
            if self.simulation_mode and result:
                # credit or debit cash
                pnl_data = calculate_trade_result(
                    entry_price=result["entry_price"],
                    exit_price=result["exit_price"],
                    quantity=result["quantity"],
                    fee_percentage=self.fee_rate
                )
                self._cash_balance += float(result["exit_price"]) * float(result["quantity"]) - pnl_data["fees"]
                # pnl already captured in result["pnl"]; cash updated here for fees realism
            self._notify_trade(f"Close position {symbol}: {result}")
            return result
        except Exception as e:
            self._notify_alert(f"Failed to close position {symbol}: {e}")
            raise

    def execute_order(
        self,
        symbol: str,
        side: str,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
        order_type: str = "market",
    ) -> Dict[str, Any]:
        """
        Execute a trade.

        If `quantity` is None, we size to roughly MIN_TRADE_AMOUNT_USD worth:
            quantity = MIN_TRADE_AMOUNT_USD / price

        Returns a dict with keys:
          status, symbol, side, quantity, entry_price, exit_price (sim only), profit/fees/return_pct (sim only),
          order_id (live), timestamp
        """
        side = side.lower()
        timestamp = datetime.now(timezone.utc).isoformat()

        # Determine a usable price when not supplied
        if price is None:
            price = self.exchange.get_price(symbol)
        if not price or price <= 0:
            raise OrderExecutionError(f"Cannot execute {side} {symbol}: missing/invalid price")

        # Default quantity based on USD minimum notional if not provided
        if quantity is None:
            min_usd = float(getattr(config, "MIN_TRADE_AMOUNT_USD", 10.0))
            quantity = max(min_usd / float(price), 0.00000001)  # conservative floor

        # ------------- Simulation path ------------- #
        if self.simulation_mode:
            try:
                # Fees are charged on notional; simulate market fill
                notional = float(price) * float(quantity)
                fee = notional * float(self.fee_rate)

                if side in ("buy", "long"):
                    # Check cash
                    total_cost = notional + fee
                    if total_cost > self._cash_balance + 1e-9:
                        raise RiskViolationError(f"Insufficient cash for buy: need {total_cost:.2f}, have {self._cash_balance:.2f}")
                    # Submit to simulated exchange to open/increase position
                    res = self.exchange._simulate_order(symbol, "buy", float(quantity), float(price))
                    # Deduct cash
                    self._cash_balance -= total_cost
                    self._notify_trade(f"SIM BUY {symbol} x{quantity:.8f} @ {price} | fee={fee:.4f} | cash={self._cash_balance:.2f}")
                    return {
                        "status": "simulated",
                        "symbol": symbol,
                        "side": "buy",
                        "quantity": float(res["quantity"]),
                        "entry_price": float(res["entry_price"]),
                        "exit_price": float(res["entry_price"]),
                        "profit": 0.0,
                        "fees": float(fee),
                        "return_pct": 0.0,
                        "timestamp": timestamp,
                    }

                elif side in ("sell", "short"):
                    # Simulate closing or flipping via exchange
                    res = self.exchange._simulate_order(symbol, "sell", float(quantity), float(price))
                    if "pnl" in res:
                        # Closed a position
                        pnl_data = calculate_trade_result(
                            entry_price=res["entry_price"],
                            exit_price=res["exit_price"],
                            quantity=res["quantity"],
                            fee_percentage=self.fee_rate
                        )
                        proceeds = float(res["exit_price"]) * float(res["quantity"])
                        self._cash_balance += proceeds - float(pnl_data["fees"])
                        self._notify_trade(
                            f"SIM SELL {symbol} x{res['quantity']:.8f} @ {price} | "
                            f"pnl={pnl_data['profit']:.4f} | fee={pnl_data['fees']:.4f} | cash={self._cash_balance:.2f}"
                        )
                        return {
                            "status": "simulated",
                            "symbol": symbol,
                            "side": "sell",
                            "quantity": float(res["quantity"]),
                            "entry_price": float(res["entry_price"]),
                            "exit_price": float(res["exit_price"]),
                            "profit": float(pnl_data["profit"]),
                            "fees": float(pnl_data["fees"]),
                            "return_pct": float(pnl_data["return_pct"]),
                            "timestamp": timestamp,
                        }
                    else:
                        # Opened/added short (if you decide to support shorts later)
                        proceeds = notional - fee
                        self._cash_balance += proceeds
                        self._notify_trade(f"SIM SHORT {symbol} x{quantity:.8f} @ {price} | fee={fee:.4f} | cash={self._cash_balance:.2f}")
                        return {
                            "status": "simulated",
                            "symbol": symbol,
                            "side": "sell",
                            "quantity": float(res["quantity"]),
                            "entry_price": float(res["entry_price"]),
                            "exit_price": float(res["entry_price"]),
                            "profit": 0.0,
                            "fees": float(fee),
                            "return_pct": 0.0,
                            "timestamp": timestamp,
                        }

            except (RiskViolationError, OrderExecutionError):
                raise
            except Exception as e:
                self._notify_alert(f"Simulation error for {symbol} {side}: {e}")
                raise

        # ------------- Live path ------------- #
        try:
            order = self.exchange.create_order(symbol, order_type, side, float(quantity), float(price))
            self._notify_trade(f"LIVE {side.upper()} {symbol} x{quantity:.8f} @ {price} | id={order.get('id')}")
            return {
                "status": order.get("status", "open"),
                "symbol": order.get("symbol", symbol),
                "side": order.get("side", side),
                "quantity": float(order.get("amount", quantity)),
                "entry_price": float(order.get("price", price)),
                "exit_price": None,
                "profit": None,
                "fees": None,
                "return_pct": None,
                "order_id": order.get("id"),
                "timestamp": timestamp,
            }
        except Exception as e:
            self._notify_alert(f"Live order failed for {symbol} {side}: {e}")
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
