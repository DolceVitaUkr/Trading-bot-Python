import asyncio
import logging

# Configure logging first
from modules.Logger_Config import setup_logging, get_logger
setup_logging(log_level="INFO")
log = get_logger(__name__)

from modules.brokers.ibkr.Connect_IBKR_API import IBKRConnectionManager
from modules.brokers.ibkr.Fetch_IBKR_MarketData import IBKRMarketDataFetcher
from modules.training.Train_Forex_Spot import ForexSpotTrainer
from modules.storage.Save_AI_Update import save_model_artifact

async def main():
    """
    This script demonstrates the full pipeline for training a Forex Spot model:
    1. Connect to IBKR.
    2. Perform market data checks.
    3. Run the Forex Spot training pipeline.
    4. Save the resulting model artifact.
    """
    log.info("--- Starting IBKR Forex Spot Demo ---")

    manager = IBKRConnectionManager()

    try:
        # 1. Connect to TWS/Gateway
        # Make sure TWS or Gateway is running and you are logged in.
        # Your .env file should be configured with the correct host, port, etc.
        log.info("Connecting to IBKR...")
        await manager.connect_tws()

        # 2. Initialize Fetchers and Trainers
        market_fetcher = IBKRMarketDataFetcher(manager)
        spot_trainer = ForexSpotTrainer(market_fetcher)

        # 3. Perform pre-flight checks
        log.info("Performing market data subscription check...")
        await market_fetcher.market_data_check()

        # 4. Run the training pipeline
        log.info("Running the Forex Spot training pipeline for EUR/USD...")
        pair = "EURUSD"
        strategy_id = f"spot_sma_crossover_{pair}"

        model_artifact, metrics = await spot_trainer.train_strategy(
            pair=pair,
            timeframe="15 mins",
            duration="30 D" # Fetch 30 days of 15-minute bars
        )

        if model_artifact and metrics:
            log.info("Training successful.", model=model_artifact, metrics=metrics)

            # 5. Save the artifact
            save_model_artifact(
                product_name=metrics["product"],
                strategy_id=strategy_id,
                artifact=model_artifact
            )
        else:
            log.error("Training pipeline failed to produce an artifact.")

    except ConnectionError as e:
        log.error(f"Connection failed: {e}", exc_info=True)
    except PermissionError as e:
        log.error(f"A permission error occurred, likely missing market data subscriptions: {e}", exc_info=True)
    except Exception as e:
        log.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        log.info("Disconnecting from IBKR...")
        await manager.disconnect_tws()
        log.info("--- IBKR Forex Spot Demo Finished ---")

if __name__ == "__main__":
    # To run this script, ensure TWS or IB Gateway is running and you are logged in.
    # Also, make sure your .env file is correctly configured.
    asyncio.run(main())
