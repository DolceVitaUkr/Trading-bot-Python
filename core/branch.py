import asyncio
import multiprocessing
from typing import Dict, Any, Optional

from adapters.null_adapters import NullMarketData, NullExecution, NullWalletSync
from core.schemas import BranchStatus
from modules.Logger_Config import setup_logging, get_logger
import config

# Import brokers
from modules.brokers.ibkr.Connect_IBKR_API import IBKRConnectionManager
from modules.brokers.ibkr.Fetch_IBKR_Account import IBKRAccountFetcher
from modules.brokers.ibkr.Fetch_IBKR_MarketData import IBKRMarketDataFetcher
from modules.brokers.ibkr.Place_IBKR_Order import IBKROrderPlacer
from modules.training.Train_Forex_Spot import ForexSpotTrainer
from modules.training.Train_Forex_Options import ForexOptionsTrainer
from adapters.bybit_v5 import BybitV5Adapter
from modules.training.Train_Crypto import CryptoTrainer


import json
import time

class Branch:
    def __init__(self, product_name: str, telemetry_queue: multiprocessing.Queue, mode: str = "paper", ibkr_conn_manager: Optional[IBKRConnectionManager] = None):
        self.product_name = product_name
        self.broker = config.ACCOUNT_SCOPE.get(product_name)
        self.ibkr_conn_manager = ibkr_conn_manager
        self.mode = mode
        self.telemetry_queue = telemetry_queue

        self.status = BranchStatus.STOPPED
        self.process: Optional[multiprocessing.Process] = None
        self.log = get_logger(f"branch.{product_name.lower()}")

        self.pipeline: Dict[str, Any] = {
            "product_name": self.product_name,
            "broker": self.broker,
            "market_fetcher": NullMarketData(),
            "order_placer": NullExecution(),
            "account_fetcher": NullWalletSync(),
            "trainer": None,
            "log": self.log,
        }

    def _build_pipeline(self):
        """Builds the pipeline of components for this specific branch."""
        self.log.info(f"Building pipeline for product '{self.product_name}' on broker '{self.broker}'")

        if self.broker == "IBKR":
            if not self.ibkr_conn_manager or not self.ibkr_conn_manager.is_connected():
                self.log.error("IBKR connection not available, cannot build IBKR pipeline.")
                raise ConnectionError("IBKR connection manager not available for IBKR branch.")

            market_fetcher = IBKRMarketDataFetcher(self.ibkr_conn_manager)
            self.pipeline["market_fetcher"] = market_fetcher
            self.pipeline["order_placer"] = IBKROrderPlacer(self.ibkr_conn_manager)
            self.pipeline["account_fetcher"] = IBKRAccountFetcher(self.ibkr_conn_manager)

            if self.product_name == "FOREX_SPOT":
                self.pipeline["trainer"] = ForexSpotTrainer(market_fetcher)
            elif self.product_name == "FOREX_OPTIONS":
                self.pipeline["trainer"] = ForexOptionsTrainer(market_fetcher)

        elif self.broker == "BYBIT":
            self.log.info(f"Initializing Bybit V5 adapter in '{self.mode}' mode.")
            bybit_adapter = BybitV5Adapter(product_name=self.product_name, mode=self.mode)
            self.pipeline["market_fetcher"] = bybit_adapter
            self.pipeline["order_placer"] = bybit_adapter
            self.pipeline["account_fetcher"] = bybit_adapter
            self.pipeline["trainer"] = CryptoTrainer(market_fetcher=bybit_adapter)

        else:
            self.log.warning(f"Broker '{self.broker}' is not supported. Using null components.")

        self.log.info("Pipeline build complete.")

    async def _run_pipeline_iteration(self):
        """Runs a single iteration of the product's training and execution pipeline."""
        trainer = self.pipeline.get("trainer")
        if not trainer:
            self.log.warning("No trainer configured, skipping iteration.")
            return

        try:
            self.log.info("Starting pipeline iteration.")
            # This logic will be expanded to be more generic
            if self.product_name == "FOREX_SPOT":
                await trainer.train_strategy("EURUSD", "15 mins", "30 D")
            elif self.product_name == "FOREX_OPTIONS":
                await trainer.analyze_atm_greeks("EURUSD")
            elif self.product_name in ["CRYPTO_SPOT", "CRYPTO_FUTURES"]:
                # The symbol can be made dynamic later
                await trainer.run_crypto_strategy("BTCUSDT")
            self.log.info("Pipeline iteration finished.")
        except Exception as e:
            self.log.error(f"An error occurred during pipeline execution: {e}", exc_info=True)

    async def _main_loop(self):
        """The main execution loop for the branch."""
        self.status = BranchStatus.RUNNING
        self.log.info("Branch main loop started.")
        self._build_pipeline()

        while self.status == BranchStatus.RUNNING:
            await self._run_pipeline_iteration()

            # Send telemetry update
            telemetry_data = {
                "branch": self.product_name,
                "status": self.status.value,
                "mode": self.mode,
                "timestamp": time.time()
            }
            try:
                self.telemetry_queue.put_nowait(json.dumps(telemetry_data))
            except Exception as e:
                self.log.error(f"Failed to put telemetry message in queue: {e}")

            await asyncio.sleep(config.LIVE_LOOP_INTERVAL)

        self.log.info("Branch main loop stopped.")

    def start(self):
        """Starts the branch in a new process."""
        if self.process and self.process.is_alive():
            self.log.warning("Branch is already running.")
            return

        self.log.info("Starting branch process...")
        # Note: The target function for multiprocessing must be a regular def, not async.
        # We create a simple wrapper to run the async main loop.
        self.process = multiprocessing.Process(target=self._run_process_target, name=f"Branch-{self.product_name}")
        self.process.start()
        self.status = BranchStatus.RUNNING

    def stop(self):
        """Stops the branch process."""
        if not self.process or not self.process.is_alive():
            self.log.warning("Branch is not running or already stopped.")
            return

        self.log.info("Stopping branch process...")
        self.status = BranchStatus.STOPPED
        self.process.terminate()
        self.process.join(timeout=5)
        if self.process.is_alive():
            self.log.warning("Process did not terminate gracefully, killing.")
            self.process.kill()
        self.process = None
        self.log.info("Branch process stopped.")

    def set_mode(self, mode: str):
        """
        Sets the trading mode for the branch.
        Note: A real implementation would require restarting the branch for the change to take full effect.
        """
        if mode not in ["paper", "live"]:
            self.log.error(f"Invalid mode '{mode}' specified.")
            return
        self.log.info(f"Setting mode to '{mode}'.")
        self.mode = mode
        # In a real scenario, we would likely need to restart the process
        # for the adapter to be re-initialized with the correct settings.

    def trigger_kill_switch(self):
        """
        Triggers the kill switch for the branch.
        This is a placeholder for the actual kill switch logic integration.
        """
        self.log.warning("Kill switch triggered via API.")
        # Here you would interact with the actual KillSwitchManager
        # For now, we just log the event.

    def _run_process_target(self):
        """A wrapper to run the async main loop in a synchronous process."""
        setup_logging(log_level=config.LOG_LEVEL) # Re-initialize logging for the new process
        try:
            asyncio.run(self._main_loop())
        except KeyboardInterrupt:
            self.log.info("Process interrupted by user.")
        except Exception as e:
            self.log.critical(f"A fatal error occurred in the branch process: {e}", exc_info=True)
            self.status = BranchStatus.ERROR
