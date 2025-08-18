import asyncio
from typing import Dict, Optional
import multiprocessing

from trading_bot.core.branch import Branch
from trading_bot.core.Logger_Config import get_logger
from trading_bot.brokers.Connect_IBKR_API import IBKRConnectionManager
from trading_bot.core.Config_Manager import config_manager


class Branch_Manager:
    def __init__(self):
        self.log = get_logger("branch_manager")
        self.branches: Dict[str, Branch] = {}
        self.ibkr_conn_manager: Optional[IBKRConnectionManager] = None
        self.telemetry_queue: multiprocessing.Queue = multiprocessing.Queue()

    async def initialize(self):
        """
        Initializes the connection managers and creates all branch instances.
        This must be called before starting any branches.
        """
        self.log.info("Initializing Branch Manager...")

        bot_settings = config_manager.get_config().get("bot_settings", {})
        products_enabled = bot_settings.get("products_enabled", [])
        account_scope = bot_settings.get("account_scope", {})

        # Initialize IBKR connection manager only if an IBKR product is enabled
        if any(account_scope.get(p) == "IBKR" for p in products_enabled):
            self.log.info("IBKR product enabled, initializing connection manager.")
            self.ibkr_conn_manager = IBKRConnectionManager()
            try:
                await self.ibkr_conn_manager.connect_tws()
                if "FOREX_OPTIONS" in products_enabled:
                    await self.ibkr_conn_manager.connect_web_api()
            except Exception as e:
                self.log.error(f"Failed to connect to IBKR: {e}", exc_info=True)
                # If connection fails, we can't proceed with IBKR branches
                self.ibkr_conn_manager = None

        # Create branch instances
        for product_name in products_enabled:
            broker = account_scope.get(product_name)

            if broker == "IBKR" and not self.ibkr_conn_manager:
                self.log.warning(f"Skipping IBKR branch '{product_name}' due to connection failure.")
                continue

            self.branches[product_name] = Branch(
                product_name=product_name,
                telemetry_queue=self.telemetry_queue,
                ibkr_conn_manager=self.ibkr_conn_manager if broker == "IBKR" else None
            )
            self.log.info(f"Created branch for '{product_name}'.")

        self.log.info("Branch Manager initialized.")

    def start_all(self):
        """Starts all created branch processes."""
        if not self.branches:
            self.log.warning("No branches to start.")
            return

        self.log.info(f"Starting all branches: {list(self.branches.keys())}")
        for branch in self.branches.values():
            branch.start()

    def stop_all(self):
        """Stops all running branch processes."""
        self.log.info("Stopping all branches...")
        for branch in self.branches.values():
            branch.stop()
        self.log.info("All branches stopped.")

    async def shutdown(self):
        """Stops all branches and disconnects connection managers."""
        self.stop_all()
        if self.ibkr_conn_manager:
            self.log.info("Disconnecting IBKR connection manager...")
            await self.ibkr_conn_manager.disconnect_tws()
            await self.ibkr_conn_manager.disconnect_web_api()
            self.log.info("IBKR connection manager disconnected.")

    def get_branch(self, product_name: str) -> Optional[Branch]:
        """Returns a specific branch instance by name."""
        return self.branches.get(product_name)

    def get_all_statuses(self) -> Dict[str, str]:
        """Returns the status of all branches."""
        return {name: branch.status.value for name, branch in self.branches.items()}
