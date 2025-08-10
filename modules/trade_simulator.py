# modules/trade_simulator.py

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

import config
from modules.trade_executor import TradeExecutor
from modules.reward_system import calculate_points

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(getattr(logging, str(getattr(config, "LOG_LEVEL", "INFO")), logging.INFO)
                if isinstance(getattr(config, "LOG_LEVEL", "INFO"), str) else getattr(config, "LOG_LEVEL", logging.INFO))


class TradeSimulator:
    """
    Backtester that replays historical OHLCV data through the TradeExecutor in simulation mode.

    Each run starts with `starting_balance` USD and `starting_points` points.
    Tracks:
      - wallet_balance
      - points
      - portfolio_value
      - trade_history
    """

    def __init__(
        self,
        initial_wallet: float = None,
        initial_points: float = 100.0,
        trade_size_percent: float = None,
        slippage: float = 0.0005,
        trade_fee: float = None,
        symbol: str = "SIM"
    ):
        # Constants (pull from config if not supplied)
        self.starting_balance = float(initial_wallet if initial_wallet is not None
                                      else getattr(config, "SIMULATION_START_BALANCE", 1000.0))
        self.starting_points = float(initial_points)
        self.symbol = symbol

        # State
        self.wallet_balance = 0.0
        self.points = 0.0
        self.portfolio_value = 0.0
        self.trade_history: List[Dict[str, Any]] = []

        # Strategy params
        self.trade_size_percent = float(trade_size_percent if trade_size_percent is not None
                                        else getattr(config, "TRADE_SIZE_PERCENT", 0.05))
        self.slippage = float(slippage)
        self.trade_fee = float(trade_fee if trade_fee is not None
                               else getattr(config, "FEE_PERCENTAGE", 0.002))

        # Trade executor in simulation mode (always)
        self.executor = TradeExecutor(simulation_mode=True)

        # Internal
        self._entry_info: Dict[str, Dict[str, Any]] = {}
        self._summary_events: List[str] = []

    # -------------------- PUBLIC API -------------------- #

    def run(self, market_data: List[List[Any]], notify_summary: Optional[callable] = None) -> None:
        """
        Run backtest on market_data: list of [timestamp, open, high, low, close, volume].

        Optionally pass `notify_summary` to receive a compact run summary at the end.
        """
        if not market_data:
            logger.info("No market_data provided to TradeSimulator.run()")
            return

        # Reset
        self.wallet_balance = self.starting_balance
        self.points = self.starting_points
        self.trade_history.clear()
        self._entry_info.clear()
        self._summary_events.clear()
        self.portfolio_value = self.starting_balance

        for kline in market_data:
            try:
                ts, o, h, l, close_price, vol = kline
            except Exception:
                logger.debug(f"Skipping malformed kline: {kline}")
                continue

            signals = self._generate_trading_signals(kline)

            # Entry when flat
            if signals.get("buy") and self.symbol not in self._entry_info:
                self._execute_buy(ts, close_price)

            # Exit when in position
            elif signals.get("sell") and self.symbol in self._entry_info:
                self._execute_sell(ts, close_price)

            # Update portfolio value: wallet + position * current price
            qty = self.executor.exchange.positions.get(self.symbol, {}).get("quantity", 0)
            self.portfolio_value = self.wallet_balance + qty * close_price

        # Summary
        if notify_summary:
            summary_text = self._format_summary()
            notify_summary(summary_text)

    # -------------------- OVERRIDE IN STRATEGIES -------------------- #

    def _generate_trading_signals(self, kline: List[Any]) -> Dict[str, bool]:
        """
        Override in subclasses or tests to return {"buy": bool, "sell": bool}.
        Default: no-op.
        """
        return {"buy": False, "sell": False}

    # -------------------- INTERNAL EXECUTION -------------------- #

    def _execute_buy(self, timestamp: int, price: float):
        if price <= 0:
            return
        qty = (self.wallet_balance * self.trade_size_percent) / (
            price * (1 + self.slippage + self.trade_fee)
        )
        result = self.executor.execute_order(self.symbol, "buy", quantity=qty, price=price)
        # ExchangeAPI internal sim balance reflects fee debits; mirror to UI balance
        self.wallet_balance = self.executor.get_balance()
        entry_time = datetime.fromtimestamp(timestamp / 1000)
        self._entry_info[self.symbol] = {"price": price, "time": entry_time}

        self.trade_history.append({**result, "timestamp": timestamp})
        self._summary_events.append(
            f"Opened {self.symbol} LONG @ {price:.2f} ({qty:.6f} units)"
        )

    def _execute_sell(self, timestamp: int, price: float):
        if price <= 0:
            return
        result = self.executor.execute_order(self.symbol, "sell", price=price)
        self.wallet_balance = self.executor.get_balance()
        entry = self._entry_info.pop(self.symbol, None)
        pnl = float(result.get("profit", 0.0))

        points = 0.0
        if entry:
            points = calculate_points(
                profit=pnl,
                entry_time=entry["time"],
                exit_time=datetime.fromtimestamp(timestamp / 1000),
                stop_loss_triggered=False,
                risk_adjusted=True
            )
        self.points += points

        self.trade_history.append({**result, "timestamp": timestamp, "points": points})
        self._summary_events.append(
            f"Closed {self.symbol} LONG @ {price:.2f} | PnL ${pnl:.2f} | {points:.2f} pts"
        )

    def _format_summary(self) -> str:
        """
        Compact Telegram-friendly session summary.
        """
        return (
            f"Simulation Summary ({self.symbol}):\n"
            f"Start Balance: ${self.starting_balance:.2f}\n"
            f"End Balance:   ${self.wallet_balance:.2f}\n"
            f"Points:        {self.points:.2f}\n"
            f"Portfolio:     ${self.portfolio_value:.2f}\n"
            f"Trades:\n- " + "\n- ".join(self._summary_events)
        )
