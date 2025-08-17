import unittest
from unittest.mock import patch, MagicMock
from adapters.bybit_v5 import BybitV5Adapter
import config

class TestBybitV5Adapter(unittest.TestCase):

    @patch('adapters.bybit_v5.HTTP')
    def test_initialization_live_mode(self, mock_http):
        """Test adapter initialization in live mode."""
        adapter = BybitV5Adapter(product_name="CRYPTO_SPOT", mode="live")

        mock_http.assert_called_with(
            testnet=False,
            api_key=config.BYBIT_API_KEY,
            api_secret=config.BYBIT_API_SECRET,
            base_url=config.BYBIT_V5_URL
        )
        self.assertEqual(adapter.mode, "live")
        self.assertIsNotNone(adapter.session)

    @patch('adapters.bybit_v5.HTTP')
    def test_initialization_paper_mode(self, mock_http):
        """Test adapter initialization in paper mode (still uses mainnet)."""
        adapter = BybitV5Adapter(product_name="CRYPTO_FUTURES", mode="paper")

        mock_http.assert_called_with(
            testnet=False, # Should always be false as per user requirement
            api_key=config.BYBIT_API_KEY,
            api_secret=config.BYBIT_API_SECRET,
            base_url=config.BYBIT_V5_URL
        )
        self.assertEqual(adapter.mode, "paper")

    @patch('adapters.bybit_v5.HTTP')
    def test_get_wallet_balance_success(self, mock_http):
        """Test successful fetching of wallet balance."""
        mock_session = MagicMock()
        mock_session.get_wallet_balance.return_value = {
            "retCode": 0,
            "result": {"list": [{"coin": [{"coin": "USDT", "equity": "1000"}]}]}
        }
        mock_http.return_value = mock_session

        adapter = BybitV5Adapter(product_name="CRYPTO_SPOT", mode="live")
        balance = adapter.get_wallet_balance(account_type="UNIFIED")

        mock_session.get_wallet_balance.assert_called_with(accountType="UNIFIED")
        self.assertIn("list", balance)

    @patch('adapters.bybit_v5.HTTP')
    def test_get_wallet_balance_api_error(self, mock_http):
        """Test handling of an API error when fetching balance."""
        mock_session = MagicMock()
        mock_session.get_wallet_balance.return_value = {"retCode": 10001, "retMsg": "Error"}
        mock_http.return_value = mock_session

        adapter = BybitV5Adapter(product_name="CRYPTO_SPOT", mode="live")
        balance = adapter.get_wallet_balance(account_type="UNIFIED")

        self.assertEqual(balance, {})

    @patch('adapters.bybit_v5.HTTP')
    def test_get_positions_success(self, mock_http):
        """Test successful fetching of positions."""
        mock_session = MagicMock()
        mock_session.get_positions.return_value = {
            "retCode": 0,
            "result": {"list": [{"symbol": "BTCUSDT", "size": "1"}]}
        }
        mock_http.return_value = mock_session

        adapter = BybitV5Adapter(product_name="CRYPTO_FUTURES", mode="live")
        positions = adapter.get_positions(category="linear")

        mock_session.get_positions.assert_called_with(category="linear")
        self.assertEqual(len(positions.get("list", [])), 1)
        self.assertEqual(positions["list"][0]["symbol"], "BTCUSDT")

if __name__ == '__main__':
    unittest.main()
