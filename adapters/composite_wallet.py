"""
Composite Wallet Sync Adapter.
"""
from typing import List, Dict

from core.interfaces import WalletSync

class CompositeWalletSync(WalletSync):
    """
    A composite adapter that merges results from multiple WalletSync adapters.
    """

    def __init__(self, wallets: List[WalletSync]):
        self.wallets = wallets

    async def subledger_equity(self) -> Dict[str, float]:
        """
        Fetches equity from all configured wallets and merges them.
        """
        combined_equity = {"SPOT": 0.0, "FUTURES": 0.0, "FX": 0.0}

        for wallet in self.wallets:
            try:
                equity = await wallet.subledger_equity()
                for subledger, value in equity.items():
                    if value > 0: # Only update if the adapter provides a value
                        combined_equity[subledger] = value
            except Exception as e:
                print(f"Error fetching equity from wallet {type(wallet).__name__}: {e}")
                # Decide if we should continue or fail. For now, we'll just log and continue.

        return combined_equity
