# modules/trade_executor.py
import logging, asyncio, time
import modules.exchange as exchange_module
from modules.telegram_bot import TelegramNotifier
import config

def send_telegram_sync(bot, message):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    if loop.is_running():
        asyncio.create_task(bot.send_message(message))
    else:
        loop.run_until_complete(bot.send_message(message))

class TradeExecutor:
    def __init__(self, simulation_mode: bool = True):
        self.simulation_mode = simulation_mode
        self.exchange = exchange_module.ExchangeAPI()
        self.telegram_bot = TelegramNotifier(disable_async=True)

    def execute_order(self,
                      symbol: str,
                      side: str,
                      amount: float,
                      price: float,
                      order_type: str = "limit"):
        clean = symbol.replace('/','')
        current = self.exchange.get_current_price(clean)
        min_order = max(0.002, self.exchange.get_min_order_size(clean))
        if amount < min_order:
            raise ValueError(f"Order amount {amount} below minimum {min_order}")
        prec = self.exchange.get_price_precision(clean)
        price = round(price, prec)

        msg = f"{'Simulating' if self.simulation_mode else 'Executing'} " \
              f"{side.upper()} {symbol} {amount:.2f} USDT at {price:.8f}"
        logging.info(msg)
        send_telegram_sync(self.telegram_bot, msg)

        if self.simulation_mode:
            time.sleep(config.SIMULATION_ORDER_DELAY)
            return {"status":"simulated","symbol":symbol,"side":side,"amount":amount,"price":price}

        try:
            return self.exchange.create_order(clean, order_type, side, amount, price)
        except Exception as e:
            err = f"Order failed: {e}"
            logging.error(err, exc_info=True)
            send_telegram_sync(self.telegram_bot, err)
            raise
