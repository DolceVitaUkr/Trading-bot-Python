import logging
import pandas as pd

from modules.brokers.ibkr.Fetch_IBKR_MarketData import IBKRMarketDataFetcher
from modules.brokers.ibkr.Contracts_IBKR import build_fx_option_contract, build_fx_spot_contract

log = logging.getLogger(__name__)

class ForexOptionsTrainer:
    """
    A training pipeline for Forex options strategies.
    This initial version focuses on fetching and analyzing greeks for an ATM option.
    """

    def __init__(self, market_data_fetcher: IBKRMarketDataFetcher):
        self.market_data = market_data_fetcher

    async def analyze_atm_greeks(self, pair: str):
        """
        Fetches the option chain, finds the ATM option, and analyzes its greeks.

        Returns:
            A tuple of (model_artifact, metrics).
        """
        log.info(f"Starting Forex Options analysis for {pair}.")

        # 1. Fetch option chain
        # For FX, the sec_type is CASH and the symbol is the pair, e.g., EURUSD
        chain = await self.market_data.fetch_option_chain(pair, sec_type="CASH")
        if not chain:
            log.error(f"Could not fetch option chain for {pair}. Aborting.")
            return None, None

        # Assume we're interested in the first chain returned (e.g., from IDEALPRO)
        chain_key = list(chain.keys())[0]
        expirations_df = chain[chain_key]

        # 2. Select an expiry and strike
        # For this example, let's pick the front-month expiration
        if expirations_df.empty:
            log.error(f"No expirations found for {pair} in chain {chain_key}.")
            return None, None
        target_expiry = expirations_df['expirations'].iloc[0]

        # Get current spot price to find ATM strike
        spot_contract = build_fx_spot_contract(pair)
        # Fetch a recent bar to get a proxy for the spot price
        spot_df = await self.market_data.fetch_historical_bars(spot_contract, "1 D", "1 min")
        if spot_df is None or spot_df.empty:
            log.error(f"Could not get spot price for {pair}.")
            return None, None

        spot_price = spot_df['close'].iloc[-1]

        # Find the strike closest to the spot price from the list of available strikes for that expiry
        all_strikes_for_expiry = expirations_df[expirations_df['expirations'] == target_expiry]['strikes'].iloc[0]
        if all_strikes_for_expiry is None or all_strikes_for_expiry.empty:
            log.error(f"No strikes found for expiry {target_expiry}.")
            return None, None

        atm_strike = all_strikes_for_expiry['strikes'].iloc[(all_strikes_for_expiry['strikes'] - spot_price).abs().argmin()]

        log.info(f"Spot price: {spot_price:.5f}, Target expiry: {target_expiry}, ATM strike: {atm_strike}")

        # 3. Build the ATM call option contract
        atm_call_contract = build_fx_option_contract(
            symbol=pair,
            last_trade_date=target_expiry.replace('-', ''),
            strike=atm_strike,
            right='C'
        )

        # 4. Fetch and analyze greeks
        greeks = await self.market_data.fetch_greeks_for_contract(atm_call_contract)

        if not greeks:
            log.error(f"Could not fetch greeks for {atm_call_contract.localSymbol}.")
            return None, None

        metrics = {
            "product": "FOREX_OPTIONS",
            "pair": pair,
            "expiry": target_expiry,
            "strike": atm_strike,
            **greeks
        }
        log.info(f"Greeks analysis complete. Metrics: {metrics}")

        # 5. "Model" artifact is the greeks data
        model_artifact = {
            "strategy_name": "atm_greeks_snapshot",
            "parameters": {"pair": pair, "expiry": target_expiry, "strike": atm_strike},
            "greeks": greeks
        }

        return model_artifact, metrics
