"""
News adapter that parses RSS feeds for sentiment and macro events.
"""
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import List, Dict
import datetime
import re

from core.interfaces import NewsFeed

# --- Configuration ---
RSS_FEEDS = {
    "forex": "https://www.investing.com/rss/news_285.rss",
    "crypto": "https://www.coindesk.com/arc/outboundfeeds/rss/",
}

# Keywords that trigger a macro blocker
MACRO_KEYWORDS = {
    "USD": ["Fed", "FOMC", "CPI", "NFP", "Non-farm", "GDP"],
    "EUR": ["ECB", "Lagarde"],
    "GBP": ["BoE", "Bailey"],
    "JPY": ["BoJ", "Kuroda"],
}

# How long to block trading after a macro event is detected
BLOCK_DURATION_HOURS = 4


class NewsRssAdapter(NewsFeed):
    """
    Implementation of the NewsFeed interface using RSS and VADER.
    """

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.last_macro_events = {} # symbol -> (timestamp, title)

    def _fetch_and_parse(self, feed_url: str):
        """Fetches and parses an RSS feed."""
        try:
            feed = feedparser.parse(feed_url)
            return feed.entries
        except Exception as e:
            print(f"Error fetching RSS feed {feed_url}: {e}")
            return []

    def sentiment(self, symbols: List[str]) -> Dict[str, float]:
        """
        Calculates sentiment for symbols.
        For simplicity, we'll average the sentiment of recent news for each asset class.
        """
        sentiments = {}
        # A real implementation would map symbols to asset classes (forex/crypto)
        # For now, we'll just get a general sentiment from one feed.
        entries = self._fetch_and_parse(RSS_FEEDS["crypto"])

        avg_sentiment = 0.0
        if entries:
            for entry in entries[:10]: # Analyze last 10 entries
                vs = self.analyzer.polarity_scores(entry.title)
                avg_sentiment += vs['compound']
            avg_sentiment /= len(entries[:10])

        for symbol in symbols:
            sentiments[symbol] = avg_sentiment # Apply same sentiment to all for now

        return sentiments

    def macro_blockers(self, symbols: List[str]) -> Dict[str, bool]:
        """
        Checks for macro-economic events that could block trading.
        """
        blockers = {symbol: False for symbol in symbols}
        now = datetime.datetime.now(datetime.timezone.utc)

        # 1. Check for new macro events from RSS feeds
        forex_entries = self._fetch_and_parse(RSS_FEEDS["forex"])
        for entry in forex_entries:
            for currency, keywords in MACRO_KEYWORDS.items():
                if any(re.search(r'\b' + keyword + r'\b', entry.title, re.IGNORECASE) for keyword in keywords):
                    # Found a macro event, store its timestamp
                    self.last_macro_events[currency] = (now, entry.title)
                    print(f"Detected new macro event for {currency}: {entry.title}")

        # 2. Check if any symbols should be blocked
        for symbol in symbols:
            # This requires parsing the symbol, e.g., "EUR/USD" -> ["EUR", "USD"]
            currencies = symbol.split('/')
            for currency in currencies:
                if currency in self.last_macro_events:
                    event_time, event_title = self.last_macro_events[currency]
                    if now - event_time < datetime.timedelta(hours=BLOCK_DURATION_HOURS):
                        blockers[symbol] = True
                        print(f"Blocking {symbol} due to recent event: '{event_title}'")
                        break # No need to check other currency in pair

        return blockers
