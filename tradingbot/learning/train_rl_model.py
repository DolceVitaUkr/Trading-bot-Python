import logging

import pandas_ta as ta  # registers DataFrame.ta accessor

from tradingbot.brokers.contractsibkr import build_fx_spot_contract
from tradingbot.brokers.fetchibkrmarketdata import IBKRMarketDataFetcher

log = logging.getLogger(__name__)

class ForexSpotTrainer:
    """
    A training pipeline for Forex spot strategies.
    """

    def __init__(self, market_data_fetcher: IBKRMarketDataFetcher):
        self.market_data = market_data_fetcher

    async def train_strategy(self, pair: str, timeframe: str, duration: str):
        """
        Fetches data, engineers features, and simulates a simple strategy.

        Returns:
            A tuple of (trained_model_artifact, metrics).
        """
        log.info(f"Starting Forex Spot training for {pair} on {timeframe} timeframe.")

        # 1. Fetch Data
        contract = build_fx_spot_contract(pair)
        df = await self.market_data.fetch_historical_bars(
            contract=contract,
            duration=duration,
            bar_size=timeframe
        )

        if df is None or df.empty:
            log.error(f"Could not fetch historical data for {pair}. Aborting training.")
            return None, None

        # 2. Feature Engineering
        df.ta.sma(length=20, append=True)
        df.ta.sma(length=50, append=True)
        df.ta.rsi(length=14, append=True)
        df.dropna(inplace=True)

        log.info(f"Feature engineering complete. Data shape: {df.shape}")

        # 3. Simulate a Simple Crossover Strategy
        df['signal'] = 0
        df.loc[df['SMA_20'] > df['SMA_50'], 'signal'] = 1
        df.loc[df['SMA_20'] < df['SMA_50'], 'signal'] = -1

        # Calculate positions based on signal changes
        df['position'] = df['signal'].diff().fillna(0)

        # 4. Calculate Metrics
        # This is a very simplified simulation of PnL
        df['returns'] = df['close'].pct_change()
        # Assume we enter on the close of the signal bar, so PnL is calculated on the next bar's return
        df['strategy_returns'] = df['signal'].shift(1) * df['returns']

        simulated_pnl = df['strategy_returns'].sum()
        num_trades = (df['position'] != 0).sum()

        metrics = {
            "product": "FOREX_SPOT",
            "pair": pair,
            "simulated_pnl": simulated_pnl,
            "num_trades": num_trades,
            "sharpe_ratio": (df['strategy_returns'].mean() / df['strategy_returns'].std()) * (252**0.5) if df['strategy_returns'].std() != 0 else 0 # Annualized
        }
        log.info(f"Simulation complete. Metrics: {metrics}")

        # 5. "Train" a model (in this case, just return the parameters)
        trained_model = {
            "strategy_name": "sma_crossover",
            "parameters": {"fast_ma": 20, "slow_ma": 50},
            "features_used": ["SMA_20", "SMA_50"]
        }

        return trained_model, metrics
