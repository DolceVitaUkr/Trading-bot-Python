# modules/trade_simulator.py

import logging
from datetime import datetime
from typing import List, Dict, Any

from modules.trade_executor import TradeExecutor
from modules.reward_system import calculate_points

logger = logging.getLogger(__name__)


class TradeSimulator:
    """
    A simple backtester that trades a synthetic 'SIM' asset over historical k-line data.
    Each session starts with $1 000 and 100 points.  After one pass through the data,
    `self.wallet_balance`, `self.points`, and `self.portfolio_value` reflect the outcome.
    """

    def __init__(
        self,
        initial_wallet: float = 1000.0,
        initial_points: float = 100.0
    ):
        # === Session constants ===
        self.starting_balance = initial_wallet
        self.starting_points  = initial_points

        # === Runtime state ===
        self.wallet_balance = 0.0       # USD cash
        self.points         = 0.0       # performance points
        self.positions      = {}        # {"SIM": quantity}
        self.entry_info     = {}        # {"SIM": {"price": float, "time": datetime}}
        self.trade_history  = []        # list of executed trade dicts
        self.current_prices = {}        # {"SIM": last_price}
        self.portfolio_value = 0.0      # cash + position*price

        # === Strategy parameters ===
        self.trade_size_percent = 0.1   # use 10% of wallet per trade
        self.slippage          = 0.0005 # 0.05%
        self.trade_fee         = 0.0002 # 0.02%

        # always simulate here
        self.executor = TradeExecutor(simulation_mode=True)

    def run(self, market_data: List[List[Any]]) -> None:
        """
        Run a backtest on market_data, which is a list of [timestamp, open, high, low, close, volume].
        After this call returns, you can inspect:
          - self.wallet_balance
          - self.points
          - self.portfolio_value
          - self.trade_history
        """
        # reset at start
        self.wallet_balance = self.starting_balance
        self.points         = self.starting_points
        self.positions.clear()
        self.entry_info.clear()
        self.trade_history.clear()
        self.current_prices.clear()
        self.portfolio_value = self.starting_balance

        for kline in market_data:
            # unpack
            ts, o, h, l, close_price, vol = kline
            self.current_prices["SIM"] = close_price

            # generate buy/sell signals (override this method in tests or subclasses)
            signals = self._generate_trading_signals(kline)

            if signals.get("buy") and self.positions.get("SIM", 0) == 0:
                self._execute_buy(ts, close_price)
            elif signals.get("sell") and self.positions.get("SIM", 0) > 0:
                self._execute_sell(ts, close_price)

            # always update portfolio value: cash + current position * price
            qty = self.positions.get("SIM", 0)
            self.portfolio_value = self.wallet_balance + qty * close_price

        # done

    def _generate_trading_signals(self, kline: List[Any]) -> Dict[str, bool]:
        """
        Stub for generating 'buy' / 'sell' signals from a kline.
        Tests typically override this with:
            sim._generate_trading_signals = lambda k: {'buy': ..., 'sell': ...}
        """
        return {"buy": False, "sell": False}

    def _execute_buy(self, timestamp: int, price: float) -> None:
        # compute max capital to risk
        max_capital = self.wallet_balance * self.trade_size_percent
        buy_price   = price * (1 + self.slippage + self.trade_fee)
        qty         = max_capital / buy_price

        # deduct cost
        self.wallet_balance -= qty * buy_price

        # record position & entry info
        self.positions["SIM"] = qty
        entry_time = datetime.fromtimestamp(timestamp / 1000)
        self.entry_info["SIM"] = {"price": price, "time": entry_time}

        # log
        self.trade_history.append({
            "action": "buy",
            "price": price,
            "quantity": qty,
            "timestamp": timestamp
        })
        logger.debug(f"BUY  SIM @ {price:.4f} × {qty:.4f}")

    def _execute_sell(self, timestamp: int, price: float) -> None:
        qty = self.positions.get("SIM", 0.0)
        if qty <= 0:
            return  # nothing to sell

        # fee-adjusted exit price
        exit_price = price * (1 - self.slippage - self.trade_fee)

        # cash in
        self.wallet_balance += qty * exit_price

        # compute PnL and reward points
        entry = self.entry_info["SIM"]
        entry_price = entry["price"]
        entry_time  = entry["time"]
        pnl = qty * (exit_price - entry_price)

        pts = calculate_points(
            profit=pnl,
            entry_time=entry_time,
            exit_time=datetime.fromtimestamp(timestamp / 1000),
            stop_loss_triggered=False,
            risk_adjusted=True
        )
        self.points += pts

        # clear position
        del self.positions["SIM"]
        del self.entry_info["SIM"]

        # log
        self.trade_history.append({
            "action": "sell",
            "price": price,
            "quantity": qty,
            "timestamp": timestamp,
            "pnl": pnl,
            "points": pts
        })
        logger.debug(f"SELL SIM @ {price:.4f} × {qty:.4f} → PnL={pnl:.4f}, pts={pts:.2f}")
