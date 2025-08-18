import asyncio
import logging

from trading_bot.core.Config_Manager import config_manager
from trading_bot.core.Logger_Config import setup_logging
from trading_bot.brokers.Exchange_Bybit import Exchange_Bybit
from trading_bot.core.Portfolio_Manager import Portfolio_Manager
from trading_bot.core.Strategy_Manager import Strategy_Manager, Decision
from trading_bot.core.Risk_Manager import Risk_Manager
from trading_bot.core.Trade_Executor import Trade_Executor
from trading_bot.core.schemas import Order

async def main():
    """
    Main entry point for the trading bot.
    Initializes all components and runs the main trading loop.
    """
    # 1. Setup Logging
    log_config = config_manager.get_config().get("logging", {})
    setup_logging(log_level=log_config.get("level", "INFO"))
    log = logging.getLogger(__name__)
    log.info("==================================================")
    log.info("           STARTING TRADING BOT                   ")
    log.info("==================================================")

    # 2. Initialize Components
    log.info("Initializing core components...")
    try:
        # Broker Adapters
        bybit_adapter = Exchange_Bybit(product_name="CRYPTO", mode="paper")

        # Core Managers
        portfolio_manager = Portfolio_Manager(bybit_adapter=bybit_adapter)
        trade_executor = Trade_Executor(bybit_adapter=bybit_adapter)
        # These managers have more complex dependencies, which have been simplified in the refactoring.
        Strategy_Manager()

        # Mock dependencies for Risk_Manager
        mock_kill_switch = type('KillSwitch', (), {'is_active': lambda self, asset: False})()
        mock_data_provider = type('DataProvider', (), {'get_funding_rate': lambda self, symbol: 0.0001})()
        Risk_Manager(account_balance=10000, sizing_policy={}, kill_switch=mock_kill_switch, data_provider=mock_data_provider)

        log.info("Core components initialized successfully.")
    except Exception as e:
        log.error(f"Failed to initialize components: {e}", exc_info=True)
        return

    # 3. Main Trading Loop (Simplified for demonstration)
    log.info("--- Starting Simplified Trading Cycle ---")
    try:
        # a. Get portfolio state
        log.info("Fetching portfolio state...")
        portfolio_state = await portfolio_manager.get_total_portfolio_state()
        if portfolio_state:
            log.info(f"Current Portfolio State: Total Balance: ${portfolio_state.total_balance_usd:.2f}, Available: ${portfolio_state.available_balance_usd:.2f}")
        else:
            log.warning("Could not fetch portfolio state.")

        # b. Get a trading decision (mocked for this example)
        log.info("Getting trading decision from Strategy_Manager...")
        # In a real scenario, the strategy manager would use market data to make a decision.
        # Here, we create a mock decision.
        mock_decision = Decision(signal="buy", sl=40000.0, tp=45000.0, meta={"symbol": "BTCUSDT", "strategy": "mock_trend"})
        log.info(f"Strategy decided: {mock_decision.signal} {mock_decision.meta['symbol']}")

        # c. Create a standardized order
        order_to_execute = Order(
            symbol=mock_decision.meta["symbol"],
            side=mock_decision.signal,
            quantity=0.001,
            order_type="market",
            stop_loss=mock_decision.sl,
            take_profit=mock_decision.tp
        )
        log.info(f"Created standardized order: {order_to_execute.dict()}")

        # d. Execute the order
        log.info("Sending order to Trade_Executor...")
        execution_result = await trade_executor.execute_order(order_to_execute)
        log.info(f"Execution result: {execution_result}")

    except Exception as e:
        log.error(f"An error occurred during the trading cycle: {e}", exc_info=True)

    log.info("--- Simplified Trading Cycle Complete ---")
    log.info("==================================================")
    log.info("           TRADING BOT SHUTDOWN                   ")
    log.info("==================================================")


if __name__ == "__main__":
    # Note: This script requires a running event loop.
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Bot shutdown requested by user.")
    except Exception as e:
        logging.getLogger(__name__).critical(f"A fatal error occurred: {e}", exc_info=True)
