import asyncio
import os
from typing import Dict, Any

# Setup logging first
from modules.Logger_Config import setup_logging, get_logger

# Import new IBKR modules and other components
from modules.brokers.ibkr.Connect_IBKR_API import IBKRConnectionManager
from modules.brokers.ibkr.Fetch_IBKR_Account import IBKRAccountFetcher
from modules.brokers.ibkr.Fetch_IBKR_MarketData import IBKRMarketDataFetcher
from modules.brokers.ibkr.Place_IBKR_Order import IBKROrderPlacer, TrainingOnlyError
from modules.training.Train_Forex_Spot import ForexSpotTrainer
from modules.training.Train_Forex_Options import ForexOptionsTrainer
from modules.storage.Save_AI_Update import save_model_artifact
from managers.validation_manager import ValidationManager

# Import existing adapters for other brokers like Bybit
from adapters.wallet_bybit import BybitWalletSync
from adapters.null_adapters import NullMarketData, NullExecution, NullWalletSync

# Import config variables
import config

# Initialize logger
setup_logging(log_level=config.LOG_LEVEL)
log = get_logger(__name__)


async def build_pipelines() -> Dict[str, Dict[str, Any]]:
    """
    Builds and initializes all the necessary components for each enabled product pipeline.
    """
    pipelines = {}
    ibkr_conn_manager = None

    # Initialize IBKR connection manager only if an IBKR product is enabled
    if any(config.ACCOUNT_SCOPE.get(p) == "IBKR" for p in config.PRODUCTS_ENABLED):
        log.info("IBKR product enabled, initializing connection manager.")
        ibkr_conn_manager = IBKRConnectionManager()
        try:
            await ibkr_conn_manager.connect_tws()
            if "FOREX_OPTIONS" in config.PRODUCTS_ENABLED:
                await ibkr_conn_manager.connect_web_api()
        except Exception as e:
            log.error("Failed to connect to IBKR, disabling IBKR-based products.", exc_info=True)
            ibkr_conn_manager = None # Disable if connection fails

    # A single validation manager for all pipelines
    validation_manager = ValidationManager()

    for product in config.PRODUCTS_ENABLED:
        broker = config.ACCOUNT_SCOPE.get(product)
        log.info(f"Building pipeline for product '{product}' on broker '{broker}'")

        pipeline = {
            "product_name": product,
            "broker": broker,
            "market_fetcher": NullMarketData(),
            "order_placer": NullExecution(),
            "account_fetcher": NullWalletSync(),
            "trainer": None,
            "validation_manager": validation_manager,
            "log": log.bind(product=product) # Bind product context to logger
        }

        if broker == "IBKR":
            if not ibkr_conn_manager:
                log.warning(f"Skipping IBKR product '{product}' due to connection failure.")
                continue

            market_fetcher = IBKRMarketDataFetcher(ibkr_conn_manager)
            pipeline["market_fetcher"] = market_fetcher
            pipeline["order_placer"] = IBKROrderPlacer(ibkr_conn_manager)
            pipeline["account_fetcher"] = IBKRAccountFetcher(ibkr_conn_manager)

            if product == "FOREX_SPOT":
                pipeline["trainer"] = ForexSpotTrainer(market_fetcher)
            elif product == "FOREX_OPTIONS":
                pipeline["trainer"] = ForexOptionsTrainer(market_fetcher)

        elif broker == "BYBIT":
            # Here you would initialize Bybit-specific components
            # For now, we use placeholders
            log.info("Bybit components are placeholders in this version.")
            # pipeline['account_fetcher'] = BybitWalletSync() # Example
            pass

        else:
            log.warning(f"Broker '{broker}' for product '{product}' is not supported. Using null components.")

        pipelines[product] = pipeline

    return pipelines, ibkr_conn_manager


async def run_product_pipeline(pipeline: Dict[str, Any]):
    """
    Runs a single iteration of a product's training and execution pipeline.
    """
    product_name = pipeline["product_name"]
    trainer = pipeline["trainer"]
    p_log = pipeline["log"] # Product-specific logger

    p_log.info("Starting pipeline iteration.")

    try:
        if not trainer:
            p_log.warning("No trainer configured for this product, skipping training.")
            return

        # This is where the core logic for a product's training/analysis runs
        # We call the appropriate method based on the product
        model_artifact, metrics = None, None
        if product_name == "FOREX_SPOT":
            model_artifact, metrics = await trainer.train_strategy("EURUSD", "15 mins", "30 D")
        elif product_name == "FOREX_OPTIONS":
            model_artifact, metrics = await trainer.analyze_atm_greeks("EURUSD")

        if model_artifact and metrics:
            p_log.info("Pipeline run successful.", metrics=metrics)

            # Persist the trained model/artifact
            strategy_id = f"{product_name.lower()}_default_strategy"
            save_model_artifact(product_name, strategy_id, model_artifact)

            # Update validation manager with trade count
            trade_count = metrics.get("num_trades", 0)
            if trade_count > 0:
                pipeline["validation_manager"].update_trade_count(product_name, trade_count)
        else:
            p_log.error("Pipeline run failed to produce an artifact.")

    except Exception as e:
        p_log.error("An error occurred during pipeline execution.", exc_info=True)


async def main():
    """
    Main entry point for the bot.
    Initializes and runs all enabled product pipelines concurrently.
    """
    log.info("--- Starting Trading Bot ---")
    pipelines, ibkr_manager = await build_pipelines()

    if not pipelines:
        log.warning("No product pipelines were successfully built. Shutting down.")
        return

    try:
        log.info(f"Running pipelines for enabled products: {list(pipelines.keys())}")
        while True:
            # Run all pipelines concurrently
            tasks = [run_product_pipeline(p) for p in pipelines.values()]
            await asyncio.gather(*tasks)

            log.info(f"All pipelines completed an iteration. Waiting for next run...")
            await asyncio.sleep(config.LIVE_LOOP_INTERVAL)

    except KeyboardInterrupt:
        log.info("Shutdown signal received.")
    except Exception as e:
        log.critical("An unhandled exception occurred in the main loop.", exc_info=True)
    finally:
        log.info("--- Shutting Down Trading Bot ---")
        if ibkr_manager:
            await ibkr_manager.disconnect_tws()
            await ibkr_manager.disconnect_web_api()


if __name__ == "__main__":
    if os.name == 'nt' and sys.version_info >= (3, 8):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    main_log = get_logger("main")
    try:
        asyncio.run(main())
    except Exception as e:
        main_log.critical("Fatal error during bot execution.", exc_info=True)
