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
        if atr > 0 and price > 0:
            atr_sl_pct = (self.global_policy['atr_mult_sl'] * atr) / price
        else:
            atr_sl_pct = self.global_policy['min_stop_distance_pct']
        min_sl_pct = self.global_policy['min_stop_distance_pct']
        sl_pct = max(min_sl_pct, atr_sl_pct)
        sl_distance = sl_pct * price

        # 2. Determine sizing based on phased equity model
        size_usd = 0.0

        # Phase 1: Equity <= $1,000
        if equity <= 1000:
            size_usd = 10.0
            logger.info(f"Phase 1 (Equity <= $1k): Fixed $10 trade. Equity: ${equity:.2f}")

        # Phase 2: $1,000 < Equity <= $5,000
        elif equity <= 5000:
            # Scale trade size from 0.5% to 1% of equity based on signal score
            size_pct = 0.005 + (signal_score * 0.005)
            size_usd = equity * size_pct
            logger.info(f"Phase 2 ($1k < E <= $5k): Direct equity size ({size_pct:.2%}). Equity: ${equity:.2f}, Size: ${size_usd:.2f}")

        # Phase 3: $5,000 < Equity <= $20,000
        elif equity <= 20000:
            # Percentage risk sizing, scaling from 0.5% to 5% risk based on signal score
            base_risk_pct = 0.005
            max_risk_pct = 0.05
            risk_pct = base_risk_pct + (signal_score * (max_risk_pct - base_risk_pct))

            risk_per_trade_usd = equity * risk_pct
            if sl_pct > 0:
                size_usd = risk_per_trade_usd / sl_pct
            else:
                logger.warning("Stop loss percentage is zero, cannot calculate size for Phase 3.")
                return None
            logger.info(f"Phase 3 ($5k < E <= $20k): %-Risk sizing ({risk_pct:.2%}). Equity: ${equity:.2f}, Risk: ${risk_per_trade_usd:.2f}, Size: ${size_usd:.2f}")

        # Phase 4: Equity > $20,000
        else:
            # Advanced %-risk sizing with good setup boosts and caps
            is_good_setup = good_setup and signal_score >= self.global_policy.get('good_setup_score_min', 0.8)

            # Use risk percentages from the policy file
            base_risk = self.global_policy.get('max_risk_pct', 0.01) # Default to 1%
            good_setup_risk = self.global_policy.get('max_risk_pct_good_setup', 0.02) # Default to 2%
            risk_pct = good_setup_risk if is_good_setup else base_risk

            risk_per_trade_usd = equity * risk_pct
            if sl_pct > 0:
                size_usd = risk_per_trade_usd / sl_pct
            else:
                logger.warning("Stop loss percentage is zero, cannot calculate size for Phase 4.")
                return None

            logger.info(f"Phase 4 (E > $20k): Advanced %-Risk ({risk_pct:.2%}). Equity: ${equity:.2f}, Good Setup: {is_good_setup}, Size: ${size_usd:.2f}")

            # If it's a good setup, cap the total size at 10% of equity
            if is_good_setup:
                equity_cap_for_good_setup = equity * 0.10
                if size_usd > equity_cap_for_good_setup:
                    logger.info(f"Capping good setup trade size from ${size_usd:.2f} to ${equity_cap_for_good_setup:.2f} (10% of equity).")
                    size_usd = equity_cap_for_good_setup

        # 3. Clamp by global pair allocation cap
        pair_equity_cap = equity * pair_cap_pct
        if size_usd > pair_equity_cap:
            logger.info(f"Clamping trade size from ${size_usd:.2f} to pair cap ${pair_equity_cap:.2f} ({pair_cap_pct:.2%}).")
            size_usd = pair_equity_cap

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
