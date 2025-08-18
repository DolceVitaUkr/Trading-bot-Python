import logging
from typing import List, Optional, Dict
import pandas as pd
from pathlib import Path

from ib_insync import IB, Contract, util

from trading_bot.brokers.Connect_IBKR_API import IBKRConnectionManager
from trading_bot.brokers.Contracts_IBKR import build_fx_spot_contract, get_conid_by_symbol
from trading_bot.core.rate_limiter import ibkr_rate_limiter
from trading_bot.core.Config_Manager import config_manager

log = logging.getLogger(__name__)

CACHE_DIR = Path("data/ibkr/historical")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class IBKRMarketDataFetcher:
    """
    Handles fetching of market data from Interactive Brokers, including
    historical bars and streaming data, with proper rate limiting and caching.
    """

    def __init__(self, conn_manager: IBKRConnectionManager):
        self.conn_manager = conn_manager
        self.ib: IB = None
        self.active_tickers = {} # To track streaming data subscriptions

    async def _ensure_connected(self):
        """Ensures the IB client is connected before making a request."""
        if not self.ib or not self.ib.isConnected():
            self.ib = await self.conn_manager.get_tws_client()

    async def market_data_check(self):
        """
        Performs a pre-flight check to verify market data subscriptions.
        Raises an exception if required subscriptions are likely missing.
        """
        await self._ensure_connected()
        log.info("Performing market data subscription pre-flight check...")

        # Check for Forex data if enabled
        if config_manager.get_config().get('bot_settings', {}).get('md_subscriptions', {}).get('FX'):
            try:
                contract = build_fx_spot_contract("EURUSD")
                # reqHeadTimeStamp is a lightweight request to check data availability
                await self.ib.reqHeadTimeStampAsync(contract, whatToShow="MIDPOINT", useRTH=True)
                log.info("Forex (FX) market data subscription check PASSED for EURUSD.")
            except Exception as e:
                if "No market data permissions" in str(e):
                    log.error("Forex (FX) market data subscription check FAILED. Please ensure you have the required subscriptions in Account Management.")
                    raise PermissionError("Missing Forex market data subscription.")
                else:
                    log.warning(f"An unexpected error occurred during FX data check: {e}")

        # Check for Options data if enabled
        if config_manager.get_config().get('bot_settings', {}).get('md_subscriptions', {}).get('OPTIONS'):
            try:
                # A common, liquid option like SPY
                contract = Contract(symbol="SPY", secType="OPT", exchange="SMART", currency="USD")
                await self.ib.reqHeadTimeStampAsync(contract, whatToShow="TRADES", useRTH=True)
                log.info("Options market data subscription check PASSED for SPY.")
            except Exception as e:
                if "No market data permissions" in str(e):
                    log.error("Options market data subscription check FAILED. Please ensure you have the required subscriptions for US Options.")
                    raise PermissionError("Missing Options market data subscription.")
                else:
                    log.warning(f"An unexpected error occurred during Options data check: {e}")

        log.info("Market data pre-flight check complete.")

    async def fetch_historical_bars(
        self,
        contract: Contract,
        duration: str, # e.g., "30 D", "1 M", "1 Y"
        bar_size: str, # e.g., "1 min", "5 mins", "1 hour", "1 day"
        what_to_show: str = "MIDPOINT",
        use_rth: bool = True,
        force_refresh: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Fetches historical bar data with rate limiting and caching.
        """
        await self._ensure_connected()

        cache_filename = f"{contract.symbol}_{contract.currency}_{bar_size.replace(' ','')}_{duration.replace(' ','')}.parquet"
        cache_path = CACHE_DIR / cache_filename

        if not force_refresh and cache_path.exists():
            log.info(f"Loading historical data from cache: {cache_path}")
            return pd.read_parquet(cache_path)

        log.info(f"Fetching historical data for {contract.localSymbol}: {duration} / {bar_size}")

        # Wait for our rate limiter
        await ibkr_rate_limiter.wait_for_historical_slot()

        try:
            bars = await self.ib.reqHistoricalDataAsync(
                contract=contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=use_rth,
                formatDate=2 # Return as UTC Posix time
            )

            if not bars:
                log.warning(f"No historical data returned for {contract.localSymbol}.")
                return None

            df = util.df(bars)
            df["date"] = pd.to_datetime(df["date"], unit="s", utc=True)
            df.set_index("date", inplace=True)

            log.info(f"Saving {len(df)} bars to cache: {cache_path}")
            df.to_parquet(cache_path)

            return df

        except Exception as e:
            log.error(f"Error fetching historical data for {contract.localSymbol}: {e}")
            if "pacing violation" in str(e):
                log.error("Pacing violation detected. The rate limiter may need adjustment.")
            return None

    async def fetch_option_chain(
        self, symbol: str, sec_type: str = "STK"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetches the entire option chain for a given underlying symbol.
        For Forex, use the pair symbol (e.g., 'EURUSD') and sec_type='CASH'.
        """
        await self._ensure_connected()

        # First, we need the conId of the underlying
        web_session = await self.conn_manager.get_web_session()
        # For CASH, the symbol to search for is the base currency
        underlying_symbol_for_search = symbol[:3] if sec_type == 'CASH' else symbol

        underlying_conid = await get_conid_by_symbol(web_session, underlying_symbol_for_search, sec_type)

        if not underlying_conid:
            log.error(f"Could not find conId for underlying {symbol}. Cannot fetch option chain.")
            return {}

        log.info(f"Fetching option chain for {symbol} (conId: {underlying_conid})...")

        # Wait for our rate limiter
        await ibkr_rate_limiter.wait_for_historical_slot()

        chains = await self.ib.reqSecDefOptParamsAsync(
            underlyingSymbol=symbol,
            futFopExchange="", # Not needed for STK/CASH
            underlyingSecType=sec_type,
            underlyingConId=underlying_conid
        )

        if not chains:
            log.warning(f"No option chain data returned for {symbol}.")
            return {}

        # Group by exchange and expiry
        chain_data = {}
        for chain in chains:
            key = f"{chain.exchange}:{chain.tradingClass}:{chain.multiplier}"
            df = util.df(chain.expirations)
            if df is None or df.empty:
                continue
            df['strikes'] = [util.df(s) for s in chain.strikes]
            chain_data[key] = df

        return chain_data

    async def fetch_greeks_for_contract(self, contract: Contract, timeout: int = 10) -> Optional[Dict]:
        """
        Fetches the greeks (Delta, Gamma, Vega, Theta, IV) for a single option contract.
        This requires a real-time data subscription.
        """
        await self._ensure_connected()

        log.info(f"Fetching greeks for {contract.localSymbol}...")

        # Wait for a general request token, as this is a streaming request
        await ibkr_rate_limiter.wait_for_token()

        # Request streaming data, wait for a tick, then cancel.
        # Use snapshot=True for a single update. Generic tick list 258 is for Option Greeks.
        ticker = await self.ib.reqMktDataAsync(contract, "258", snapshot=True, timeout=timeout)

        if not ticker or not ticker.modelGreeks:
            log.warning(f"Could not fetch model greeks for {contract.localSymbol}. Check market data subscriptions.")
            return None

        greeks = ticker.modelGreeks
        return {
            "iv": greeks.iv,
            "delta": greeks.delta,
            "gamma": greeks.gamma,
            "vega": greeks.vega,
            "theta": greeks.theta,
            "timestamp": ticker.time
        }
