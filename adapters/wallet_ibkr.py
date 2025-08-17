"""
IBKR Wallet Sync Adapter.
"""
from ib_insync import IB
from typing import Dict

from core.interfaces import WalletSync

class IbkrWalletSync(WalletSync):
    """
    Implementation of the WalletSync interface for Interactive Brokers.
    Fetches cash balance to represent FX sub-ledger equity.
    """

    def __init__(self, ib_client: IB):
        self.ib = ib_client

    async def subledger_equity(self) -> Dict[str, float]:
        """
        Fetches the total cash value from the account summary.
        This will represent the "FX" sub-ledger.
        """
        if not self.ib.isConnected():
            raise ConnectionError("IBKR client is not connected.")

        # We'll fetch the account summary and look for 'TotalCashValue'
        # This requires TWS to be set up to send the summary.
        account_values = await self.ib.reqAccountSummaryAsync()

        fx_equity = 0.0
        for value in account_values:
            if value.tag == 'TotalCashValue' and value.currency == 'BASE':
                # 'BASE' means it's in the account's base currency.
                fx_equity = float(value.value)
                break

        # This adapter only provides FX equity. The other types are zero.
        return {"SPOT": 0.0, "FUTURES": 0.0, "FX": fx_equity}
