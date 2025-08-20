from ib_insync import Forex, Option
from typing import Dict, Optional
import aiohttp
import logging

from TradingBot.core.configmanager import config_manager
IBKR_CPAPI_GATEWAY_URL = config_manager.get_config().get("bot_settings", {}).get("ibkr_cpapi_gateway_url", "https://localhost:5000")

log = logging.getLogger(__name__)

# A simple in-memory cache for conIds to avoid repeated lookups
CONID_CACHE: Dict[str, int] = {}


def build_fx_spot_contract(pair: str) -> Forex:
    """
    Creates an IBKR Forex (CASH) contract for a given currency pair.

    Args:
        pair: The currency pair in 'EUR/USD' or 'EURUSD' format.

    Returns:
        An ib_insync Forex contract object.
    """
    pair = pair.replace("/", "").upper()
    if len(pair) != 6:
        raise ValueError("Invalid Forex pair format. Must be 6 characters, e.g., EURUSD.")

    symbol = pair[:3]
    currency = pair[3:]

    return Forex(pair, exchange="IDEALPRO", symbol=symbol, currency=currency)


async def get_conid_by_symbol(
    web_session: aiohttp.ClientSession,
    symbol: str,
    sec_type: str = "STK"
) -> Optional[int]:
    """
    Searches for a contract's conId using the Client Portal Web API.
    This is useful for finding options, futures, etc.

    Args:
        web_session: The aiohttp session for making requests.
        symbol: The underlying symbol (e.g., "SPY", "AAPL", "EUR").
        sec_type: The security type (e.g., "STK", "OPT", "CASH").

    Returns:
        The conId if found, otherwise None.
    """
    cache_key = f"{symbol}_{sec_type}"
    if cache_key in CONID_CACHE:
        log.debug(f"Returning cached conId for {cache_key}: {CONID_CACHE[cache_key]}")
        return CONID_CACHE[cache_key]

    url = f"{IBKR_CPAPI_GATEWAY_URL}/v1/api/iserver/secdef/search"
    params = {"symbol": symbol, "secType": sec_type}

    log.info(f"Searching for conId for {symbol} ({sec_type}) via {url}")
    try:
        async with web_session.get(url, params=params, ssl=False) as response:
            response.raise_for_status()
            results = await response.json()

            if results and isinstance(results, list):
                # Often, the first result is the most relevant one.
                # More complex logic might be needed for ambiguous symbols.
                conid = results[0].get("conid")
                if conid:
                    log.info(f"Found conId {conid} for {symbol}. Caching result.")
                    CONID_CACHE[cache_key] = conid
                    return conid

            log.warning(f"No conId found for symbol: {symbol}")
            return None

    except aiohttp.ClientError as e:
        log.error(f"Error searching for conId for {symbol}: {e}")
        return None


def build_fx_option_contract(
    symbol: str,
    last_trade_date: str, # YYYYMMDD format
    strike: float,
    right: str, # 'C' for Call, 'P' for Put
    exchange: str = "IDEALPRO"
) -> Option:
    """
    Creates an IBKR Option contract for a Forex pair.

    Args:
        symbol: The underlying currency pair, e.g., 'EURUSD'.
        last_trade_date: The expiration date in YYYYMMDD format.
        strike: The strike price.
        right: 'C' for Call or 'P' for Put.
        exchange: The exchange, defaults to IDEALPRO for FX options.

    Returns:
        An ib_insync Option contract object.
    """
    if right.upper() not in ['C', 'P']:
        raise ValueError("'right' must be 'C' or 'P'.")

    # For FX options, the symbol is the base currency.
    base_currency = symbol[:3].upper()
    quote_currency = symbol[3:].upper()

    return Option(
        symbol=base_currency,
        lastTradeDateOrContractMonth=last_trade_date,
        strike=strike,
        right=right.upper(),
        exchange=exchange,
        tradingClass=base_currency, # Often the same as the symbol for FX options
        currency=quote_currency,
        multiplier='1' # Typically 1 for FX options, but should be verified
    )
