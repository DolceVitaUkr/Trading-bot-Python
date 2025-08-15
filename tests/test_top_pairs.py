import pytest
from unittest.mock import MagicMock
from modules.top_pairs import TopPairs, PairStats

@pytest.fixture
def mock_exchange():
    exchange = MagicMock()
    exchange.markets = {
        "BTC/USDT": {"spot": True, "active": True, "base": "BTC", "quote": "USDT"},
        "ETH/USDT": {"spot": True, "active": True, "base": "ETH", "quote": "USDT"},
        "SOL/USDT": {"spot": True, "active": True, "base": "SOL", "quote": "USDT"},
        "XRP/USDT": {"spot": True, "active": False, "base": "XRP", "quote": "USDT"}, # Inactive
        "BTC/USDC": {"spot": True, "active": True, "base": "BTC", "quote": "USDC"}, # Different quote
    }

    tickers = {
        "BTC/USDT": {"quoteVolume": 500000, "percentage": 2.5, "ask": 50001, "bid": 50000},
        "ETH/USDT": {"quoteVolume": 800000, "percentage": 1.5, "ask": 3001, "bid": 3000},
        "SOL/USDT": {"quoteVolume": 100000, "percentage": 5.0, "ask": 151, "bid": 150}, # Below min volume
    }
    exchange.fetch_tickers.return_value = tickers

    return exchange

@pytest.fixture
def top_pairs_instance(mock_exchange):
    return TopPairs(
        exchange=mock_exchange,
        quote="USDT",
        min_volume_usd_24h=200000,
        max_pairs=10
    )

def test_get_top_pairs(top_pairs_instance):
    pairs = top_pairs_instance.get_top_pairs(force=True)
    assert pairs == ["ETH/USDT", "BTC/USDT"]

def test_get_top_pairs_with_stats(top_pairs_instance):
    stats = top_pairs_instance.get_top_pairs_with_stats(force=True)
    assert len(stats) == 2
    assert stats[0].symbol == "ETH/USDT"
    assert stats[1].symbol == "BTC/USDT"
    assert isinstance(stats[0], PairStats)

def test_caching(top_pairs_instance):
    top_pairs_instance.get_top_pairs(force=True)
    assert top_pairs_instance.exchange.fetch_tickers.call_count == 1

    # Should use cache
    top_pairs_instance.get_top_pairs()
    assert top_pairs_instance.exchange.fetch_tickers.call_count == 1

    # Force refresh
    top_pairs_instance.get_top_pairs(force=True)
    assert top_pairs_instance.exchange.fetch_tickers.call_count == 2
