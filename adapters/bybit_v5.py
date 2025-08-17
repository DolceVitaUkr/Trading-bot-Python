from typing import Optional, Dict, Any
from pybit.unified_trading import HTTP
import config
from modules.Logger_Config import get_logger


class BybitV5Adapter:
    """
    Adapter for interacting with the Bybit v5 API (Unified Trading).
    """

    def __init__(self, product_name: str, mode: str = "paper"):
        """
        Initializes the Bybit v5 adapter. Always connects to Mainnet.
        The 'mode' parameter determines if trades are simulated or live.

        :param product_name: The name of the product branch (e.g., "CRYPTO_SPOT").
        :param mode: The trading mode ("paper" or "live").
        """
        self.log = get_logger(f"bybit_adapter.{product_name.lower()}")
        self.product_name = product_name
        self.mode = mode

        if not config.BYBIT_API_KEY or not config.BYBIT_API_SECRET:
            self.log.error("Bybit API key or secret is not configured.")
            raise ValueError("Bybit API credentials are not set.")

        # Always use Mainnet as per user requirements
        self.session = HTTP(
            testnet=False,
            api_key=config.BYBIT_API_KEY,
            api_secret=config.BYBIT_API_SECRET,
            base_url=config.BYBIT_V5_URL
        )
        self.log.info(f"Bybit v5 adapter initialized for {product_name} in {mode} mode (connected to Mainnet).")

    def get_wallet_balance(self, account_type: str = "UNIFIED") -> Dict[str, Any]:
        """
        Fetches the wallet balance for a specific account type.

        :param account_type: The type of account (e.g., "UNIFIED", "CONTRACT", "SPOT").
        :return: A dictionary containing the API response.
        """
        try:
            self.log.info(f"Fetching wallet balance for account type: {account_type}")
            response = self.session.get_wallet_balance(accountType=account_type)
            if response.get("retCode") == 0:
                self.log.info("Successfully fetched wallet balance.")
                return response['result']
            else:
                self.log.error(f"Error fetching wallet balance: {response}")
                return {}
        except Exception as e:
            self.log.error(f"An exception occurred while fetching wallet balance: {e}", exc_info=True)
            return {}

    def get_positions(self, category: str) -> Dict[str, Any]:
        """
        Fetches open positions for a specific category.

        :param category: The category of positions (e.g., "spot", "linear", "inverse", "option").
        :return: A dictionary containing the API response.
        """
        try:
            self.log.info(f"Fetching positions for category: {category}")
            response = self.session.get_positions(category=category)
            if response.get("retCode") == 0:
                self.log.info(f"Successfully fetched {len(response['result'].get('list', []))} positions.")
                return response['result']
            else:
                self.log.error(f"Error fetching positions: {response}")
                return {}
        except Exception as e:
            self.log.error(f"An exception occurred while fetching positions: {e}", exc_info=True)
            return {}

    def get_instrument_info(self, category: str) -> Dict[str, Any]:
        """
        Fetches instrument information (symbols metadata).

        :param category: The category of instruments (e.g., "spot", "linear").
        :return: A dictionary containing the API response.
        """
        try:
            self.log.info(f"Fetching instrument info for category: {category}")
            response = self.session.get_instruments_info(category=category)
            if response.get("retCode") == 0:
                self.log.info("Successfully fetched instrument info.")
                return response['result']
            else:
                self.log.error(f"Error fetching instrument info: {response}")
                return {}
        except Exception as e:
            self.log.error(f"An exception occurred while fetching instrument info: {e}", exc_info=True)
            return {}
