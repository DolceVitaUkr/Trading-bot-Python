"""
Bybit Wallet Sync Adapter.
"""
import os
import time
import hmac
import hashlib
import aiohttp
from typing import Dict

from core.interfaces import WalletSync

class BybitWalletSync(WalletSync):
    """
    Implementation of the WalletSync interface for Bybit.
    Fetches Spot and Futures (Unified Margin) account balances.
    """

    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.bybit.com"

    async def _get_signature(self, payload: str, timestamp: str) -> str:
        """Generates the HMAC-SHA256 signature for Bybit API."""
        param_str = timestamp + self.api_key + "5000" + payload
        return hmac.new(bytes(self.api_secret, "utf-8"), param_str.encode("utf-8"), hashlib.sha256).hexdigest()

    async def subledger_equity(self) -> Dict[str, float]:
        """
        Fetches equity for SPOT and FUTURES (UNIFIED) sub-ledgers.
        """
        # We need to call /v5/account/wallet-balance
        # This requires authentication.

        timestamp = str(int(time.time() * 1000))
        # For GET requests, the payload is the query string
        query_string = "accountType=UNIFIED" # or SPOT

        headers = {
            'X-BAPI-API-KEY': self.api_key,
            'X-BAPI-TIMESTAMP': timestamp,
            'X-BAPI-RECV-WINDOW': '5000',
        }

        # We'll make two calls, one for SPOT and one for UNIFIED (Futures)
        results = {"SPOT": 0.0, "FUTURES": 0.0}

        async with aiohttp.ClientSession() as session:
            for account_type, subledger_name in [("SPOT", "SPOT"), ("UNIFIED", "FUTURES")]:
                query = f"accountType={account_type}"
                signature = await self._get_signature(query, timestamp)
                headers['X-BAPI-SIGN'] = signature

                url = f"{self.base_url}/v5/account/wallet-balance?{query}"

                try:
                    async with session.get(url, headers=headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data.get("retCode") == 0:
                                # Find the total equity from the response
                                total_equity = float(data['result']['list'][0]['totalEquity'])
                                results[subledger_name] = total_equity
                            else:
                                print(f"Bybit API error for {account_type}: {data.get('retMsg')}")
                        else:
                            print(f"HTTP Error fetching Bybit balance for {account_type}: {response.status}")
                except Exception as e:
                    print(f"Exception fetching Bybit balance for {account_type}: {e}")

        return results
