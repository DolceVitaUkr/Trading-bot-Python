# main.py
import logging
import asyncio
import signal
from typing import Optional
from config import (
    ENVIRONMENT,
    SIMULATION_START_BALANCE,
    LOG_LEVEL,
    DEFAULT_TRADE_AMOUNT,
    FEE_PERCENTAGE,
    MAX_SIMULATION_PAIRS
)
from modules.exchange import ExchangeAPI
from modules.telegram_bot import TelegramBot
from modules.ui import TradingUI
from modules.data_manager import DataManager
from modules.error_handler import ErrorHandler
from modules.self_learning import SelfLearningBot
from modules.parameter_optimization import ParameterOptimizer
from modules.trade_simulator import TradeSimulator
from modules.trade_executor import TradeExecutor
from utils.utilities import configure_logging, ensure_directory

class TradingBot:
    def __init__(self):
        self._running = True
        self._setup_signals()
        
        # Initialize core components
        self.error_handler = ErrorHandler()
        self.data_manager = DataManager()
        self.exchange = ExchangeAPI()
        self.telegram_bot = TelegramBot()
        self.ui = TradingUI(log_callback=self._ui_log_handler)
        
        # Initialize trading components
        self.trade_executor = TradeExecutor(
            simulation_mode=(ENVIRONMENT == "simulation")
        )
        self.self_learning = SelfLearningBot(
            data_provider=self.data_manager,
            error_handler=self.error_handler
        )
        self.optimizer = ParameterOptimizer(
            exchange=self.exchange,
            data_manager=self.data_manager
        )
        
        # Configure trade simulator
        self.trade_simulator = TradeSimulator(
            initial_wallet=SIMULATION_START_BALANCE,
            trade_amount=DEFAULT_TRADE_AMOUNT,
            fee_percentage=FEE_PERCENTAGE
        )
        self.trade_simulator.log_callback = self._ui_log_handler

    def _setup_signals(self):
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _ui_log_handler(self, message: str):
        """Handle logging from both UI and other components"""
        logging.info(message)
        self.ui.update_log_display(message)
        self.telegram_bot.send_message(message)

    async def _run_simulation(self):
        """Run trading simulation with current configuration"""
        await self.trade_simulator.run()
        self.ui.update_simulation_results(
            self.trade_simulator.wallet_balance,
            self.trade_simulator.points
        )

    async def _run_live_trading(self):
        """Main live trading loop"""
        while self._running:
            try:
                # Implement live trading logic here
                await asyncio.sleep(1)
            except Exception as e:
                self.error_handler.handle(e)

    async def run(self):
        """Main async entry point"""
        logging.info(f"Starting in {ENVIRONMENT} mode")
        
        # Initialize UI components
        self.ui.add_action_handler(
            "start_simulation", 
            lambda: asyncio.create_task(self._run_simulation())
        )
        self.ui.add_action_handler(
            "start_optimization",
            lambda: asyncio.create_task(self.optimizer.run())
        )
        
        # Start main loop based on environment
        if ENVIRONMENT == "simulation":
            await self._run_simulation()
        else:
            await self._run_live_trading()

    def _shutdown(self, signum, frame):
        """Graceful shutdown procedure"""
        logging.info("Initiating shutdown sequence...")
        self._running = False
        self.exchange.close()
        self.data_manager.close()
        self.ui.shutdown()
        logging.info("Clean shutdown completed")

def configure_logging():
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("logs/bot.log"),
            logging.StreamHandler()
        ]
    )

async def main():
    configure_logging()
    bot = TradingBot()
    
    try:
        await bot.run()
    except Exception as e:
        logging.error(f"Critical error: {str(e)}")
        bot._shutdown(None, None)
        raise

if __name__ == "__main__":
    asyncio.run(main())