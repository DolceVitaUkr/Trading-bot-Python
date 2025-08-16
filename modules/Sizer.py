import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class Sizer:
    """
    Calculates trade size and leverage based on a flexible policy,
    incorporating equity, risk, and signal quality.
    """

    def __init__(self, policy: Dict[str, Any]):
        """
        Initializes the Sizer with a sizing policy.

        Args:
            policy (Dict[str, Any]): The sizing policy, typically loaded from a JSON file.
        """
        self.policy = policy
        self.global_policy = policy.get("global", {})
        self.leverage_tiers = policy.get("leverage_tiers", [])
        self.asset_caps = policy.get("asset_caps", {})

    @classmethod
    def from_json(cls, policy_path: str) -> 'Sizer':
        """
        Factory method to create a Sizer instance from a JSON policy file.

        Args:
            policy_path (str): The file path to the sizing policy JSON.

        Returns:
            Sizer: A new instance of the Sizer.
        """
        try:
            with open(policy_path, 'r') as f:
                policy = json.load(f)
            return cls(policy)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.exception(f"Failed to load sizing policy from {policy_path}: {e}")
            raise

    def propose(self,
                equity: float,
                asset_class: str,
                mode: str,
                atr: float,
                price: float,
                pair_cap_pct: float,
                signal_score: float = 0.5,
                good_setup: bool = False) -> Optional[Dict[str, float]]:
        """
        Proposes a trade size, leverage, and stop loss based on the sizing policy.

        Args:
            equity (float): The current equity of the sub-ledger.
            asset_class (str): The asset class (e.g., 'SPOT', 'PERP').
            mode (str): The trading strategy mode (e.g., 'SCALP', 'TREND').
            atr (float): The Average True Range value for stop loss calculation.
            price (float): The current price of the asset.
            pair_cap_pct (float): The max percentage of equity to allocate to one pair.
            signal_score (float, optional): The score of the trading signal (0-1). Defaults to 0.5.
            good_setup (bool, optional): Flag indicating if the setup is considered high quality. Defaults to False.

        Returns:
            Optional[Dict[str, float]]: A dictionary with {size_usd, leverage, sl_distance}
                                        or None if a trade cannot be sized.
        """
        if price <= 0 or equity <= 0:
            return None

        # 1. Determine Stop-Loss distance
        # Ensure we don't divide by zero and handle cases with no ATR
        if atr > 0 and price > 0:
            atr_sl_pct = (self.global_policy['atr_mult_sl'] * atr) / price
        else:
            # Fallback if ATR is not available, use only the minimum
            atr_sl_pct = self.global_policy['min_stop_distance_pct']

        min_sl_pct = self.global_policy['min_stop_distance_pct']
        sl_pct = max(min_sl_pct, atr_sl_pct)
        sl_distance = sl_pct * price

        # 2. Determine sizing mode (fixed or %-risk)
        if equity < self.global_policy['equity_threshold_usd']:
            size_usd = self.global_policy['fixed_trade_usd']
        else:
            # Use %-risk sizing
            is_good_setup = good_setup and signal_score >= self.global_policy['good_setup_score_min']
            risk_pct = self.global_policy['max_risk_pct_good_setup'] if is_good_setup else self.global_policy['max_risk_pct']

            risk_per_trade_usd = equity * risk_pct

            # Calculate size based on risk and stop loss
            if sl_pct > 0:
                size_usd = risk_per_trade_usd / sl_pct
            else:
                logger.warning("Stop loss percentage is zero, cannot calculate size.")
                return None

        # 3. Clamp by pair allocation cap
        pair_equity_cap = equity * pair_cap_pct
        size_usd = min(size_usd, pair_equity_cap)

        # 4. Determine Leverage
        leverage = self._get_leverage(equity, asset_class, mode)

        # 5. Final checks (e.g., minimum notional)
        # Using a hardcoded minimum of 10 USD notional for now.
        if size_usd * leverage < 10.0:
             logger.warning(f"Proposed size {size_usd:.2f} with leverage {leverage} is below min notional. Skipping.")
             return None

        return {
            "size_usd": round(size_usd, 4),
            "leverage": round(leverage, 2),
            "sl_distance": round(sl_distance, 8), # High precision for crypto
        }

    def _get_leverage(self, equity: float, asset_class: str, mode: str) -> float:
        """
        Determines the appropriate leverage based on equity, asset class, and mode.
        """
        leverage = 1.0

        # Tier-based leverage (assumes tiers are sorted by equity_max)
        for tier in self.leverage_tiers:
            if equity <= tier['equity_max']:
                # Find a mode-specific leverage, fall back to a generic 'DEFAULT' or 1.0
                leverage = tier.get(mode, tier.get('DEFAULT', 1.0))
                break

        # Clamp by asset class maximum leverage
        asset_cap_info = self.asset_caps.get(asset_class)
        if asset_cap_info:
            max_leverage_for_asset = asset_cap_info.get('max_leverage', 1.0)
            leverage = min(leverage, max_leverage_for_asset)

        return leverage
