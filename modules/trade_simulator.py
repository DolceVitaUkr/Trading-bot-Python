# modules/trade_simulator.py
import logging, time
from datetime import datetime
from typing import List, Dict
import modules.exchange as exchange_module
from modules.trade_executor import TradeExecutor
from modules.data_manager import DataManager

class TradeSimulator:
    def __init__(self, initial_wallet: float = 10000.0):
        self.wallet_balance = initial_wallet
        self.positions: Dict[str,float] = {}
        self.trading_pairs: List[str] = []
        self.trade_history = []
        self.current_prices: Dict[str,float] = {}
        self.trade_size_percent = 0.1
        self.slippage = 0.0005
        self.trade_fee = 0.0002

        self.executor = TradeExecutor(simulation_mode=True)
        self.data_manager = DataManager(test_mode=True)
        logging.basicConfig(level=logging.INFO)

    def run(self, market_data: List[list]):
        if not self.trading_pairs:
            raise ValueError("No trading pairs configured")
        for k in market_data:
            ts, o, h, l, c, v = k
            sym = self.trading_pairs[0]
            price = c
            self.current_prices[sym] = price
            sig = self._generate_trading_signals(k)
            if sig['buy']:
                self._execute_buy_order(sym, price)
            elif sig['sell']:
                self._execute_sell_order(sym, price)
            self._update_portfolio_value(sym, price)
            self._log_status(ts, sym, price)

    def _generate_trading_signals(self, kline):
        price = kline[4]
        return {'buy': price < 50015, 'sell': price > 50025}

    def _execute_buy_order(self, symbol, price):
        if self.positions.get(symbol,0)>0:
            return
        max_val = self.wallet_balance * self.trade_size_percent
        qty = max_val / (price*(1+self.slippage+self.trade_fee))
        try:
            order = self.executor.execute_order(symbol,'buy',qty,price,'market')
            self.positions[symbol] = qty
            self.wallet_balance -= qty * price * (1+self.slippage+self.trade_fee)
            self.trade_history.append(order)
        except Exception:
            pass

    def _execute_sell_order(self, symbol, price):
        qty = self.positions.get(symbol,0)
        if qty<=0:
            return
        try:
            order = self.executor.execute_order(symbol,'sell',qty,price,'market')
            self.wallet_balance += qty * price * (1-self.slippage-self.trade_fee)
            self.positions[symbol] = 0
            self.trade_history.append(order)
        except Exception:
            pass

    def _update_portfolio_value(self, symbol, price):
        pos_val = self.positions.get(symbol,0)*price
        self.portfolio_value = self.wallet_balance + pos_val

    def _log_status(self, ts, symbol, price):
        logging.info(f"{datetime.utcfromtimestamp(ts/1000)} | {symbol} @ {price:.2f} "
                     f"Bal: {self.wallet_balance:.2f} Pos: {self.positions.get(symbol,0):.4f} "
                     f"Total: {self.portfolio_value:.2f}")
