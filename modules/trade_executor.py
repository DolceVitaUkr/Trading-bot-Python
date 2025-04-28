# modules/trade_executor.py

import logging
import asyncio
import time
from datetime import datetime
from typing import Optional, Dict, Any

import config
from modules.exchange import ExchangeAPI
from modules.telegram_bot import TelegramNotifier
from modules.risk_management import RiskManager, RiskViolationError
from modules.trade_calculator import calculate_trade_result

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

def _send_telegram_sync(bot: TelegramNotifier, content: Any, fmt: str = 'text'):
    """Helper to send Telegram messages synchronously when needed."""
    try:
        # Force sync send for critical messages
        bot.send_message(content, format=fmt)
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}", exc_info=True)

class TradeExecutor:
    """
    Abstraction over real and simulated order execution.
    Uses ExchangeAPI under the hood, applies risk checks via RiskManager,
    and sends Telegram notifications for each executed order.
    """

    def __init__(self, simulation_mode: bool = True):
        self.simulation_mode = simulation_mode
        # Choose live/testnet based on config and simulation flag
        self.exchange = ExchangeAPI(
            use_testnet=self.simulation_mode and getattr(config, "USE_TESTNET", False)
        )
        # TelegramNotifier: disable async in contexts where sync is needed
        self.notifier = TelegramNotifier(disable_async=not config.ASYNC_TELEGRAM)
        # Risk manager for position sizing and risk checks
        self.risk_manager = RiskManager(
            account_balance=config.SIMULATION_START_BALANCE if self.simulation_mode else config.LIVE_ACCOUNT_BALANCE,
            max_portfolio_risk=config.MAX_PORTFOLIO_RISK
        )

    def execute_order(
        self,
        symbol: str,
        side: str,
        amount: Optional[float] = None,
        price: Optional[float] = None,
        order_type: str = "limit",
        risk_percent: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Place a new order (or simulate it).

        Returns a dict including:
          - entry_price, exit_price (same for market orders)
          - quantity
          - pnl (for simulation when position closes)
          - fees (simulated)
          - timestamp
          - status ("simulated" or exchange response status)
        """
        # --- Determine quantity via risk if not provided ---
        if amount is None:
            if risk_percent is None:
                risk_percent = config.DEFAULT_RISK_PERCENT
            # Need a stop price: use default stop-loss pct
            stop_price = self.risk_manager.calculate_stop_loss(
                entry_price=price,
                side=side,
                stop_loss_pct=config.DEFAULT_STOP_LOSS_PCT
            )
            amount = self.risk_manager.calculate_position_size(
                entry_price=price,
                stop_price=stop_price,
                risk_percent=risk_percent
            )

        # --- Enforce minimum order size ---
        min_order = max(config.MIN_ORDER_AMOUNT, self.exchange.get_min_order_size(symbol))
        if amount < min_order:
            raise ValueError(f"Order size {amount:.8f} < minimum allowed {min_order:.8f}")

        # --- Round price and amount to exchange precision ---
        price_precision = self.exchange.get_price_precision(symbol)
        amount_precision = self.exchange.get_amount_precision(symbol)
        if price is not None:
            price = round(price, price_precision)
        amount = round(amount, amount_precision)

        side = side.lower()
        timestamp = datetime.utcnow().isoformat()

        msg = {
            'symbol': symbol,
            'side': side,
            'amount': amount,
            'price': price,
            'time': timestamp
        }
        # --- Notify user ---
        log_msg = f"{'SIM' if self.simulation_mode else 'LIVE'} {side.upper()} {symbol} x{amount:.8f} @ {price}"
        logger.info(log_msg)
        _send_telegram_sync(self.notifier, log_msg, fmt='trade')

        # --- Execute or simulate ---
        try:
            if self.simulation_mode:
                time.sleep(config.SIMULATION_ORDER_DELAY)
                # In sim mode, we treat this as both entry and exit at the same price for P&L calc
                entry = price
                exit_ = price
                pnl_data = calculate_trade_result(entry, exit_, amount, fee_percentage=config.FEE_RATE)
                result = {
                    'status': 'simulated',
                    'symbol': symbol,
                    'side': side,
                    'quantity': amount,
                    'entry_price': entry,
                    'exit_price': exit_,
                    'profit': pnl_data['profit'],
                    'fees': pnl_data['fees'],
                    'return_pct': pnl_data['return_pct'],
                    'timestamp': timestamp
                }
            else:
                # Live mode: place order via exchange
                exec_res = self.exchange.create_order(symbol, order_type, side, amount, price)
                # Exchange may return order ID or position data; normalize keys
                entry = exec_res.get('filled_price', exec_res.get('price', price))
                # We don't know exit until later; set exit_price=None
                result = {
                    'status': exec_res.get('status', 'open'),
                    'symbol': exec_res.get('symbol', symbol),
                    'side': exec_res.get('side', side),
                    'quantity': exec_res.get('amount', amount),
                    'entry_price': entry,
                    'exit_price': None,
                    'profit': None,
                    'fees': None,
                    'return_pct': None,
                    'order_id': exec_res.get('id'),
                    'timestamp': timestamp
                }
            return result

        except RiskViolationError as e:
            # If risk check fails, notify and re-raise to trigger circuit breaker
            err_msg = f"Risk violation for {symbol} {side}: {e}"
            logger.critical(err_msg)
            _send_telegram_sync(self.notifier, err_msg, fmt='alert')
            raise

        except Exception as e:
            # Log and notify unexpected errors
            err_msg = f"Order execution failed for {symbol} {side}: {e}"
            logger.error(err_msg, exc_info=True)
            _send_telegram_sync(self.notifier, err_msg, fmt='alert')
            raise
