import asyncio
import logging
from typing import Optional

from ib_insync import IB
import aiohttp

from trading_bot.core.configmanager import config_manager

ibkr_config = config_manager.get_config().get("api_keys", {}).get("ibkr", {})
IBKR_TWS_HOST = ibkr_config.get("host", "127.0.0.1")
IBKR_TWS_PORT = ibkr_config.get("port", 7497)
IBKR_TWS_CLIENT_ID = ibkr_config.get("client_id", 1)
IBKR_API_MODE = (
    config_manager.get_config().get("bot_settings", {}).get("ibkr_api_mode", "paper")
)
IBKR_CPAPI_GATEWAY_URL = (
    config_manager.get_config()
    .get("bot_settings", {})
    .get("ibkr_cpapi_gateway_url", "https://localhost:5000")
)
TRAINING_MODE = (
    config_manager.get_config().get("bot_settings", {}).get("training_mode", True)
)

log = logging.getLogger(__name__)


class IBKRConnectionManager:
    """
    Manages the connection to Interactive Brokers, for both the TWS/Gateway
    socket API and the Client Portal Web API.
    """

    def __init__(self):
        self.ib: IB = IB()
        self.web_session: Optional[aiohttp.ClientSession] = None
        self._is_paper_account: Optional[bool] = None

        self.ib.connectedEvent += self.on_tws_connect
        self.ib.disconnectedEvent += self.on_tws_disconnect

    def on_tws_connect(self):
        """Event handler for successful TWS connection."""
        log.info("TWS/Gateway connection successful.")
        log.info(f"Server version: {self.ib.serverVersion()}")
        log.info(f"TWS connection time: {self.ib.twsConnectionTime()}")

    def on_tws_disconnect(self):
        """Event handler for TWS disconnection."""
        log.warning("TWS/Gateway disconnected.")

    async def connect_tws(self):
        """
        Connects to the TWS/Gateway socket API.
        Handles retries and enforces paper trading mode if configured.
        """
        if self.ib.isConnected():
            log.info("TWS client is already connected.")
            return

        log.info(f"Connecting to TWS/Gateway at {IBKR_TWS_HOST}:{IBKR_TWS_PORT} with ClientID {IBKR_TWS_CLIENT_ID}...")
        try:
            await self.ib.connectAsync(
                host=IBKR_TWS_HOST,
                port=IBKR_TWS_PORT,
                clientId=IBKR_TWS_CLIENT_ID,
                timeout=15,
                readonly=TRAINING_MODE  # Set readonly based on TRAINING_MODE
            )
            await self._validate_account_mode()
        except asyncio.TimeoutError:
            log.error("Connection to TWS/Gateway timed out. Is TWS/Gateway running and API enabled?")
            raise ConnectionError("TWS/Gateway connection timed out.")
        except Exception as e:
            log.error(f"Failed to connect to TWS/Gateway: {e}")
            raise ConnectionError(f"TWS/Gateway connection failed: {e}")

    async def _validate_account_mode(self):
        """
        Validates that the connected account matches the configured API mode (paper/live).
        """
        accounts = self.ib.managedAccounts()
        if not accounts:
            log.error("No managed accounts found. Cannot validate account mode.")
            await self.disconnect_tws()
            raise ConnectionError("No managed accounts found.")

        account_id = accounts[0]
        log.info(f"Connected to account: {account_id}")

        is_paper = account_id.startswith('D')
        self._is_paper_account = is_paper

        if IBKR_API_MODE == "paper" and not is_paper:
            log.error(f"Configuration requires a paper account, but connected to a live account ({account_id}).")
            await self.disconnect_tws()
            raise ConnectionError("Connected to live account in paper mode.")
        elif IBKR_API_MODE == "live" and is_paper:
            log.warning(f"Configuration is for a live account, but connected to a paper account ({account_id}).")

        if TRAINING_MODE:
            log.info("TRAINING_MODE is ON. Order placement will be blocked.")
            if not self.ib.client.isReadOnly():
                 log.warning("TRAINING_MODE is ON, but TWS API is not set to Read-Only. Orders can still be placed.")

    async def disconnect_tws(self):
        """Disconnects from the TWS/Gateway."""
        if self.ib.isConnected():
            log.info("Disconnecting from TWS/Gateway...")
            self.ib.disconnect()

    async def connect_web_api(self):
        """
        Initializes the aiohttp session for the Client Portal Web API and checks its health.
        """
        if self.web_session and not self.web_session.closed:
            log.info("Web API session already initialized.")
            return

        log.info(f"Initializing Web API session for {IBKR_CPAPI_GATEWAY_URL}...")
        self.web_session = aiohttp.ClientSession()

        # Perform a health check
        try:
            # The /tickle endpoint re-authenticates, /iserver/auth/status is better for a simple check.
            async with self.web_session.get(f"{IBKR_CPAPI_GATEWAY_URL}/v1/api/iserver/auth/status", ssl=False) as response:
                response.raise_for_status()
                data = await response.json()
                if data.get("connected"):
                    log.info("Client Portal Web API gateway is connected and authenticated.")
                else:
                    log.warning("Client Portal Web API gateway is running but not authenticated.")
        except aiohttp.ClientError as e:
            log.error(f"Failed to connect to Client Portal Web API gateway: {e}")
            await self.disconnect_web_api()
            raise ConnectionError(f"Web API gateway connection failed: {e}")

    async def disconnect_web_api(self):
        """Closes the Web API session."""
        if self.web_session and not self.web_session.closed:
            log.info("Closing Web API session...")
            await self.web_session.close()
            self.web_session = None

    async def get_tws_client(self) -> IB:
        """Returns the connected ib_insync client."""
        if not self.ib.isConnected():
            await self.connect_tws()
        return self.ib

    async def get_web_session(self) -> Optional[aiohttp.ClientSession]:
        """Returns the aiohttp session for the Web API."""
        if not self.web_session or self.web_session.closed:
            await self.connect_web_api()
        return self.web_session

    @property
    def is_paper_account(self) -> Optional[bool]:
        return self._is_paper_account

# Example usage
async def main():
    manager = IBKRConnectionManager()
    try:
        # Connect to TWS
        await manager.connect_tws()
        log.info(f"Successfully connected. Is paper account? {manager.is_paper_account}")

        # Connect to Web API
        await manager.connect_web_api()

        # Keep running for a bit to see events
        await asyncio.sleep(10)

    finally:
        await manager.disconnect_tws()
        await manager.disconnect_web_api()

if __name__ == "__main__":
    # To run this example, you need to have TWS or Gateway running.
    # Also, make sure your .env file is configured correctly.
    # Note: ib_insync uses its own logger, which can be configured.
    # from ib_insync import util
    # util.logToConsole(logging.DEBUG)
    try:
        asyncio.run(main())
    except (ConnectionError, KeyboardInterrupt) as e:
        log.info(f"Shutting down: {e}")
