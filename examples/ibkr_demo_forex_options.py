import asyncio
import logging

# Configure logging first
from modules.Logger_Config import setup_logging, get_logger
setup_logging(log_level="INFO")
log = get_logger(__name__)

from modules.brokers.ibkr.Connect_IBKR_API import IBKRConnectionManager
from modules.brokers.ibkr.Fetch_IBKR_MarketData import IBKRMarketDataFetcher
from modules.training.Train_Forex_Options import ForexOptionsTrainer
from modules.storage.Save_AI_Update import save_model_artifact

async def main():
    """
    This script demonstrates the full pipeline for analyzing Forex Options:
    1. Connect to IBKR TWS (for data) and the Web API (for contract lookups).
    2. Perform market data checks.
    3. Run the Forex Options analysis pipeline to get ATM greeks.
    4. Save the resulting analysis artifact.
    """
    log.info("--- Starting IBKR Forex Options Demo ---")

    manager = IBKRConnectionManager()

    try:
        # 1. Connect to TWS/Gateway and the Client Portal Web API
        log.info("Connecting to IBKR...")
        await manager.connect_tws()
        await manager.connect_web_api() # Needed for option chain lookups

        # 2. Initialize Fetchers and Trainers
        market_fetcher = IBKRMarketDataFetcher(manager)
        options_trainer = ForexOptionsTrainer(market_fetcher)

        # 3. Perform pre-flight checks
        log.info("Performing market data subscription check...")
        await market_fetcher.market_data_check()

        # 4. Run the analysis pipeline
        log.info("Running the Forex Options analysis pipeline for EUR/USD...")
        pair = "EURUSD"
        strategy_id = f"options_atm_greeks_{pair}"

        model_artifact, metrics = await options_trainer.analyze_atm_greeks(
            pair=pair
        )

        if model_artifact and metrics:
            log.info("Analysis successful.", model=model_artifact, metrics=metrics)

            # 5. Save the artifact
            save_model_artifact(
                product_name=metrics["product"],
                strategy_id=strategy_id,
                artifact=model_artifact
            )
        else:
            log.error("Analysis pipeline failed to produce an artifact.")

    except ConnectionError as e:
        log.error(f"Connection failed: {e}", exc_info=True)
    except PermissionError as e:
        log.error(f"A permission error occurred, likely missing market data subscriptions: {e}", exc_info=True)
    except Exception as e:
        log.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        log.info("Disconnecting from IBKR...")
        await manager.disconnect_tws()
        await manager.disconnect_web_api()
        log.info("--- IBKR Forex Options Demo Finished ---")

if __name__ == "__main__":
    # To run this script, ensure TWS or IB Gateway is running and you are logged in.
    # You must also run the Client Portal Gateway for contract searches.
    # Also, make sure your .env file is correctly configured.
    asyncio.run(main())
