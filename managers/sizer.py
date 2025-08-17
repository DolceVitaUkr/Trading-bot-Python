"""
Wallet-aware Sizer Manager.
"""
from typing import Dict
from core.interfaces import WalletSync

# Placeholder for asset-specific sizing rules
# A real implementation would have a more detailed config
ASSET_RULES = {
    "Crypto": {"lot_step": 0.001},
    "Forex": {"lot_step": 1000},
}

class Sizer:
    """
    Calculates order sizes based on sub-ledger equity and risk parameters.
    """

    def __init__(self, wallet_sync: WalletSync):
        self.wallet_sync = wallet_sync

    async def get_order_qty(
        self,
        asset_class: str,
        price: float,
        risk_fraction: float = 0.01, # Risk 1% of equity by default
        max_exposure_pct: float = 0.10 # Max 10% of equity in one position
    ) -> float:
        """
        Calculates the order quantity based on the asset class and risk settings.
        """
        subledger_map = {
            "Crypto": "SPOT", # Defaulting to SPOT for crypto
            "Forex": "FX",
        }
        subledger = subledger_map.get(asset_class)
        if not subledger:
            print(f"Warning: No subledger mapping for asset class '{asset_class}'. Sizing will be zero.")
            return 0.0

        all_equity = await self.wallet_sync.subledger_equity()
        equity = all_equity.get(subledger, 0.0)

        if equity <= 0:
            print(f"Warning: No equity available in subledger '{subledger}'. Cannot size order.")
            return 0.0

        # 1. Calculate notional size based on risk fraction
        # This is a simplified model. A real one would use stop-loss distance.
        # For now, we'll just size based on a fraction of total equity.
        target_notional = equity * risk_fraction

        # 2. Enforce max exposure cap
        max_notional = equity * max_exposure_pct
        notional_size = min(target_notional, max_notional)

        # 3. Convert notional to quantity
        if price <= 0:
            print("Warning: Price is zero or negative. Cannot calculate quantity.")
            return 0.0

        quantity = notional_size / price

        # 4. Apply lot step rounding
        lot_step = ASSET_RULES.get(asset_class, {}).get("lot_step")
        if lot_step:
            quantity = (quantity // lot_step) * lot_step

        return round(quantity, 8) # Round to a reasonable precision
