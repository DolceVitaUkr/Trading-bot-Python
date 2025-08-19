import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple, Any

from TradingBot.core.configmanager import config_manager
from TradingBot.core.killswitch import KillSwitch

logger = logging.getLogger(__name__)


@dataclass
class PositionRisk:
    """Represents the risk associated with a single open position."""

    symbol: str
    quantity: float
    entry_price: float
    dollar_risk: float
    value_usd: float


class RiskManager:
    """
    Handles all non-sizing related risk checks, such as cooldowns,
    daily loss limits, and final proposal validation.
    Integrates KillSwitch and funding/carry cost filters.
    """

    def __init__(self,
                 account_balance: float,
                 sizing_policy: Dict[str, Any],
                 kill_switch: KillSwitch,
                 data_provider: Any, # Using Any to avoid circular dependency
                 notifier=None):
        """
        Initializes the Risk_Manager.

        Args:
            account_balance (float): The starting total equity.
            sizing_policy (Dict[str, Any]): The sizing policy dictionary.
            kill_switch (KillSwitch): The KillSwitch instance.
            data_provider (Any): The data provider for fetching funding rates.
            notifier ([type], optional): Notifier instance. Defaults to None.
        """
        self.equity = float(account_balance)
        self.sizing_policy = sizing_policy
        self.kill_switch = kill_switch
        self.data_provider = data_provider
        self.notifier = notifier
        self.asset_caps = sizing_policy.get("asset_caps", {})

        # --- State Tracking ---
        self.peak_equity = self.equity
        # NOTE: This is now for high-level tracking. Portfolio_Manager is the source of truth for exposure.
        self.open_positions: Dict[str, PositionRisk] = {}

        # --- Daily Loss Limit ---
        kpi_targets = config_manager.get_config().get("kpi_targets", {})
        self.daily_loss_limit_pct = kpi_targets.get("daily_loss_limit", 0.03)
        self.daily_start_equity = self.equity
        self.last_trade_day = datetime.now(timezone.utc).date()

        # --- Progressive Cooldown ---
        self.consecutive_losses = 0
        self.in_cooldown_until = None
        self.cooldown_rules = {3: {"duration_minutes": 30}, 5: {"duration_minutes": -1}}

    # --- Public Methods ---

    def allow(self,
              proposal: Dict[str, Any],
              asset_class: str,
              symbol: str,
              side: str,
              price: float,
              mode: Optional[str] = None,
              session: Optional[str] = None) -> Tuple[bool, str]:
        """
        Performs final risk validation on a trade proposal.

        Args:
            proposal (Dict[str, Any]): The trade proposal from the Sizer.
            asset_class (str): The asset class of the trade (e.g., 'PERP').
            symbol (str): The symbol for the trade.
            side (str): The side of the trade ('buy' or 'sell').
            price (float): The current price of the asset.
            mode (str, optional): The trading mode.
            session (str, optional): The market session.

        Returns:
            Tuple[bool, str]: (is_allowed, reason_string)
        """
        # 1. Kill Switch Check
        if self.kill_switch.is_active(asset_class):
            reason = self.kill_switch.active_kill_switches[asset_class]
            logger.critical(f"Trade for {symbol} rejected: Kill switch is active for {asset_class}. Reason: {reason}", extra={'action': 'reject', 'symbol': symbol, 'reason': f"Kill switch: {reason}"})
            return False, f"Kill switch active for {asset_class}: {reason}"

        # 2. Standard Cooldown and Drawdown Checks
        self._check_daily_reset()
        if not self._is_not_in_cooldown():
            reason = f"In cooldown until {self.in_cooldown_until}"
            logger.warning(f"Trade disallowed for {symbol}: {reason}", extra={'action': 'reject', 'symbol': symbol, 'reason': reason})
            return False, reason
        if not self._is_within_daily_loss_limit():
            reason = f"Daily loss limit of {self.daily_loss_limit_pct:.2%} hit"
            logger.warning(f"Trade disallowed for {symbol}: {reason}", extra={'action': 'reject', 'symbol': symbol, 'reason': reason})
            return False, reason

        # 3. Funding/Carry Filter
        if asset_class in ["PERP", "FOREX"]:
            funding_rate = self.data_provider.get_funding_rate(symbol)
            funding_rate_threshold = self.asset_caps.get(asset_class, {}).get("funding_rate_threshold", -0.0002)

            if side == 'buy' and funding_rate < funding_rate_threshold:
                reason = f"High negative funding rate: {funding_rate:.4%}"
                logger.warning(f"Trade for {symbol} (long) rejected: {reason}", extra={'action': 'reject', 'symbol': symbol, 'reason': reason})
                return False, reason

            if side == 'sell' and funding_rate > abs(funding_rate_threshold):
                reason = f"High positive funding rate: {funding_rate:.4%}"
                logger.warning(f"Trade for {symbol} (short) rejected: {reason}", extra={'action': 'reject', 'symbol': symbol, 'reason': reason})
                return False, reason

        # 4. Leverage and Liquidation Buffer Checks
        asset_cap_info = self.asset_caps.get(asset_class, {})
        max_leverage = asset_cap_info.get('max_leverage', 1.0)
        if proposal['leverage'] > max_leverage:
            return False, f"Proposed leverage {proposal['leverage']:.1f}x exceeds asset class cap of {max_leverage:.1f}x."

        if asset_class == "PERP":
            liq_buffer_pct = asset_cap_info.get('liq_buffer_pct', 0.0)
            if price > 0:
                sl_distance_pct = proposal['sl_distance'] / price
                if sl_distance_pct < liq_buffer_pct:
                    return False, f"Stop loss distance {sl_distance_pct:.2%} is below required liquidation buffer of {liq_buffer_pct:.2%}."

        return True, "Trade is allowed."

    def record_trade_closure(self, pnl: float, new_equity: float):
        """Records the outcome of a closed trade to update risk states."""
        self.equity = new_equity
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        if pnl < 0:
            self.consecutive_losses += 1
            self._check_for_cooldown()
        else:
            if self.consecutive_losses > 0:
                logger.info("Winning trade recorded, resetting consecutive loss counter.")
                self.consecutive_losses = 0

    def register_position(self, symbol: str, quantity: float, entry_price: float, dollar_risk: float):
        """(May be deprecated) Adds an open position to the local tracker."""
        if symbol in self.open_positions:
            logger.warning(f"Position for {symbol} already registered. Overwriting.")
        self.open_positions[symbol] = PositionRisk(
            symbol=symbol, quantity=quantity, entry_price=entry_price,
            dollar_risk=dollar_risk, value_usd=quantity * entry_price
        )
        logger.info(f"Registered new position in Risk_Manager: {self.open_positions[symbol]}")

    def unregister_position(self, symbol: str):
        """(May be deprecated) Removes a closed position from the local tracker."""
        if symbol in self.open_positions:
            logger.info(f"Unregistering position from Risk_Manager for {symbol}.")
            del self.open_positions[symbol]

    # --- Private Helper Methods ---

    def _check_daily_reset(self):
        """Resets daily loss tracking if a new day has started."""
        current_day = datetime.now(timezone.utc).date()
        if current_day > self.last_trade_day:
            logger.info(f"New day. Resetting daily equity from {self.daily_start_equity} to {self.equity}.")
            self.daily_start_equity = self.equity
            self.last_trade_day = current_day
            if self.in_cooldown_until:
                self.in_cooldown_until = None
                logger.info("New session started, cooldown lifted.")

    def _is_not_in_cooldown(self) -> bool:
        """Checks if the bot is currently in a cooldown period."""
        if self.in_cooldown_until:
            if self.in_cooldown_until.year == 9999: # Sentinel for session-long stop
                return False
            if datetime.now(timezone.utc) < self.in_cooldown_until:
                return False
            self.in_cooldown_until = None # Cooldown has expired
        return True

    def _is_within_daily_loss_limit(self) -> bool:
        """Checks if the daily loss is within the defined limit."""
        if self.daily_start_equity <= 0:
            return True  # Avoid division by zero
        current_loss_pct = (self.daily_start_equity - self.equity) / self.daily_start_equity
        if current_loss_pct >= self.daily_loss_limit_pct:
            logger.critical(f"Daily loss limit hit! Loss: {current_loss_pct:.2%}")
            return False
        return True

    def _check_for_cooldown(self):
        """Triggers a cooldown if consecutive loss streak hits a threshold."""
        if self.consecutive_losses in self.cooldown_rules:
            rule = self.cooldown_rules[self.consecutive_losses]
            duration = rule['duration_minutes']
            if duration == -1:
                self.in_cooldown_until = datetime(9999, 1, 1, tzinfo=timezone.utc)
                msg = f"STOP FOR SESSION cooldown triggered after {self.consecutive_losses} losses."
            else:
                self.in_cooldown_until = datetime.now(timezone.utc) + timedelta(minutes=duration)
                msg = f"{duration}-min cooldown triggered after {self.consecutive_losses} losses."
            logger.warning(msg)
            if self.notifier:
                self.notifier.send_notification(f"Risk Alert: {msg}")
