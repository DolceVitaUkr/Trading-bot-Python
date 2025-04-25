# modules/trade_executor.py
import logging
import config
from modules.exchange import ExchangeAPI
from modules.telegram_bot import TelegramNotifier
from utils.utilities import configure_logging
import asyncio
import time

def send_telegram_sync(bot, message):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    try:
        if loop.is_running():
            asyncio.create_task(bot.send_message(message))
        else:
            loop.run_until_complete(bot.send_message(message))
    except Exception as e:
        logging.error(f"Error sending telegram message: {e}", exc_info=True)

class TradeExecutor:
    def __init__(self, simulation_mode: bool = True):
        self.simulation_mode = simulation_mode
        self.exchange = ExchangeAPI()
        self.telegram_bot = TelegramNotifier(disable_async=True)

    def execute_order(self, symbol, side, amount, price, order_type="limit"):
        clean_symbol = symbol.replace('/', '')
        current_price = self.exchange.get_current_price(clean_symbol)
        # Add validation for minimum order size
        min_order = max(0.002, self.exchange.get_min_order_size(clean_symbol))
        if amount < min_order:
            raise ValueError(f"Order amount {amount} below minimum {min_order}")
                
        # Format price according to exchange rules
        price_precision = self.exchange.get_price_precision(clean_symbol)
        price = round(price, price_precision)

        side = side.lower()
        message = f"{'Simulating' if self.simulation_mode else 'Executing'} order: {side.upper()} {symbol} {amount:.2f} USDT at {price:.8f}"
        logging.info(message)
        send_telegram_sync(self.telegram_bot, message)
        
        if self.simulation_mode:
            time.sleep(config.SIMULATION_ORDER_DELAY)
            return {"status": "simulated", "symbol": symbol, "side": side, "amount": amount, "price": price}
        else:
            try:
                return self.exchange.create_order(clean_symbol, order_type, side, amount, price)
            except Exception as e:
                error_msg = f"Order failed: {e}"
                logging.error(error_msg, exc_info=True)
                send_telegram_sync(self.telegram_bot, error_msg)
                raise