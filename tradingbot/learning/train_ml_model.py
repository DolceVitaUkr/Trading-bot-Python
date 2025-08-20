from tradingbot.core.loggerconfig import get_logger
# In a real scenario, this would depend on a market data interface
# from tradingbot.core.interfaces import IMarketDataFetcher
try:  # pragma: no cover - defensive import
    from tradingbot.brokers.exchangebybit import ExchangeBybit
except Exception:  # pragma: no cover
    ExchangeBybit = object  # type: ignore
import numpy as np
import pandas as pd


class TrainMLModel:
    """
    A trainer class for cryptocurrency strategies (Spot and Futures).
    """
    def __init__(self, market_fetcher: ExchangeBybit):
        """
        Initializes the crypto trainer.

        :param market_fetcher: An instance of a market data fetcher, like ExchangeBybit.
        """
        self.log = get_logger("crypto_trainer")
        self.market_fetcher = market_fetcher
        self.log.info("Crypto trainer initialized.")

    async def run_crypto_strategy(self, symbol: str):
        """
        A placeholder method for running a crypto trading strategy analysis or training.

        In a real implementation, this would involve:
        - Fetching historical data using the market_fetcher.
        - Applying technical indicators.
        - Backtesting a strategy.
        - Returning performance metrics and a trained model artifact.
        """
        self.log.info(f"Running training/analysis for crypto strategy on symbol: {symbol}")

        # Example: Fetch some data
        instrument_info = self.market_fetcher.get_instrument_info(category="spot" if "SPOT" in self.market_fetcher.product_name else "linear")

        # In a real scenario, you'd do much more here.
        # For now, we'll just log a message and return dummy data.
        if instrument_info:
            self.log.info(f"Found {len(instrument_info.get('list', []))} instruments.")
        else:
            self.log.warning("Could not retrieve instrument info.")

        # Dummy return values to match the structure expected by the pipeline
        model_artifact = {"strategy_name": "dummy_crypto_strategy", "symbol": symbol}
        metrics = {"num_trades": 10, "win_rate": 0.6} # Dummy metrics

        self.log.info("Crypto strategy training/analysis complete.")
        return model_artifact, metrics


# ---------------------------------------------------------------------------
# Utility functions used in unit tests
# ---------------------------------------------------------------------------

def triple_barrier_label(
    prices: pd.Series,
    upper_pct: float,
    lower_pct: float,
    max_holding: int,
) -> pd.Series:
    """Apply the triple‑barrier method to generate labels.

    Parameters
    ----------
    prices:
        Series of prices indexed by time.
    upper_pct, lower_pct:
        Multipliers defining the take‑profit and stop‑loss barriers relative to
        the price at *t0*.
    max_holding:
        Number of periods after which the position is closed regardless of
        price.

    Returns
    -------
    pandas.Series
        Labels of 1 (upper barrier hit), -1 (lower barrier hit) or 0 (vertical
        barrier / none).
    """

    prices = prices.reset_index(drop=True)
    n = len(prices)
    labels = np.zeros(n, dtype=int)
    for i in range(n):
        start = prices[i]
        upper = start * (1 + upper_pct)
        lower = start * (1 - lower_pct)
        end = min(n, i + max_holding + 1)
        for j in range(i + 1, end):
            p = prices[j]
            if p >= upper:
                labels[i] = 1
                break
            if p <= lower:
                labels[i] = -1
                break
    return pd.Series(labels, index=prices.index)


def purged_train_test_split(
    n_samples: int, test_size: float = 0.2, embargo_pct: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return indices for a purged train/test split.

    ``embargo_pct`` removes observations from the end of the train set to
    reduce leakage.  The function works purely with lengths so it can be used
    with any indexable structure.
    """

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    test_start = int(n_samples * (1 - test_size))
    embargo = int(n_samples * embargo_pct)
    train_end = max(0, test_start - embargo)
    train_idx = np.arange(0, train_end)
    test_idx = np.arange(test_start, n_samples)
    return train_idx, test_idx
