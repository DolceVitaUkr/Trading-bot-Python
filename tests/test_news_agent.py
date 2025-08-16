import unittest
from modules.News_Agent import NewsAgent

class TestNewsAgent(unittest.TestCase):

    def setUp(self):
        """Set up a new NewsAgent instance for each test."""
        config = {
            "news_api_key": "test_key",
            "high_impact_events": ["CPI", "NFP"]
        }
        self.news_agent = NewsAgent(config)

    def test_get_sentiment_simulation(self):
        """Test the simulated sentiment scores."""
        symbols = ["BTCUSDT", "ETHUSDT", "EURUSD"]
        sentiments = self.news_agent.get_sentiment(symbols)

        self.assertIn("BTCUSDT", sentiments)
        self.assertEqual(sentiments["BTCUSDT"], 0.6) # Bullish simulation
        self.assertIn("EURUSD", sentiments)
        self.assertEqual(sentiments["EURUSD"], -0.4) # Bearish simulation
        self.assertIn("ETHUSDT", sentiments)
        self.assertEqual(sentiments["ETHUSDT"], 0.0) # Neutral default

    def test_get_news_bias(self):
        """Test the translation of sentiment scores to trading biases."""
        # Mock the get_sentiment method to return specific scores
        self.news_agent.get_sentiment = lambda symbols: {
            "BULL": 0.8,
            "BEAR": -0.7,
            "NEUTRAL": 0.2
        }

        self.assertEqual(self.news_agent.get_news_bias("BULL"), "long")
        self.assertEqual(self.news_agent.get_news_bias("BEAR"), "short")
        self.assertEqual(self.news_agent.get_news_bias("NEUTRAL"), "neutral")

    def test_high_impact_event_check(self):
        """Test the high-impact event check (currently a stub)."""
        # The stub always returns (False, ""), this test ensures it runs without error
        is_event, event_name = self.news_agent.is_high_impact_event_imminent()
        self.assertFalse(is_event)
        self.assertEqual(event_name, "")

if __name__ == '__main__':
    unittest.main()
