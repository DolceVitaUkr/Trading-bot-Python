import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

class NewsAgent:
    """
    (Mock) Fetches and analyzes news sentiment to influence trading decisions.
    Also includes a macro calendar filter to pause trading during high-impact events.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the NewsAgent.

        Args:
            config (Dict[str, Any]): Configuration for the news agent.
        """
        self.config = config
        self.high_impact_events = config.get("high_impact_events", ["CPI", "NFP", "FOMC", "ECB"])
        self.event_window_minutes = config.get("event_imminent_window_minutes", 15)

        # --- Mock Data for Simulation ---
        self.mock_imminent_event: Optional[Tuple[str, int]] = None  # (event_name, minutes_until_event)
        self.mock_sentiments: Dict[str, float] = {
            "BTC/USDT": 0.7,   # Strong bullish
            "ETH/USDT": -0.6,  # Strong bearish
            "SOL/USDT": 0.2,   # Mildly bullish
            "EUR/USD": -0.4,   # Mildly bearish
        }
        # --------------------------------

    def get_sentiment(self, symbols: List[str]) -> Dict[str, float]:
        """
        (Mock) Gets the news sentiment for a list of symbols.

        Args:
            symbols (List[str]): The list of symbols to get sentiment for.

        Returns:
            Dict[str, float]: A dictionary mapping each symbol to a sentiment score
                              (-1.0 bearish to +1.0 bullish).
        """
        sentiments = {}
        for symbol in symbols:
            # Use the mock sentiment if available, otherwise default to neutral (0.0)
            sentiments[symbol] = self.mock_sentiments.get(symbol, 0.0)

        logger.debug(f"Fetched mock sentiments: {sentiments}")
        return sentiments

    def is_high_impact_event_imminent(self) -> Tuple[bool, str]:
        """
        (Mock) Checks if a high-impact macroeconomic event is scheduled soon.

        Returns:
            Tuple[bool, str]: (is_event_imminent, event_name)
        """
        if self.mock_imminent_event:
            event_name, minutes_away = self.mock_imminent_event
            if 0 < minutes_away <= self.event_window_minutes:
                logger.warning(f"High-impact event '{event_name}' is imminent ({minutes_away} mins).")
                return True, event_name

        logger.debug("No high-impact events imminent.")
        return False, ""

    def get_news_bias(self, symbol: str) -> str:
        """
        Provides a trading bias based on news sentiment.

        Args:
            symbol (str): The symbol to check.

        Returns:
            str: 'long', 'short', or 'neutral'
        """
        sentiment_score = self.get_sentiment([symbol]).get(symbol, 0.0)

        # Get thresholds from config or use defaults
        long_threshold = self.config.get("sentiment_long_threshold", 0.5)
        short_threshold = self.config.get("sentiment_short_threshold", -0.5)

        if sentiment_score >= long_threshold:
            return 'long'
        elif sentiment_score <= short_threshold:
            return 'short'
        return 'neutral'
