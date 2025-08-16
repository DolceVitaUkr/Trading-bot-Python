import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import config
from modules.error_handler import RiskViolationError

logger = logging.getLogger(__name__)

@dataclass
class PositionRisk:
    symbol: str
    quantity: float
    entry_price: float
    dollar_risk: float
    value_usd: float

class RiskManager:
    """
    Handles all risk-related checks and calculations.
    """

    def __init__(self, account_balance: float, notifier=None, per_pair_cap_pct=None, portfolio_cap_pct=None):
        self.equity = float(account_balance)
        self.notifier = notifier

        # --- State Tracking ---
        self.peak_equity = self.equity
        self.open_positions: Dict[str, PositionRisk] = {}

        # --- Daily Loss Limit ---
        self.daily_loss_limit_pct = config.KPI_TARGETS.get("daily_loss_limit", 0.03)
        self.daily_start_equity = self.equity
        self.last_trade_day = datetime.now(timezone.utc).date()

        # --- Progressive Cooldown ---
        self.consecutive_losses = 0
        self.in_cooldown_until = None
        self.cooldown_rules = {3: {"duration_minutes": 30}, 5: {"duration_minutes": -1}}

        # --- Risk Parameters ---
        self.base_per_trade_risk_pct = config.PER_TRADE_RISK_PERCENT
        self.drawdown_reduction_threshold = 0.06 # 6% drawdown

        # --- Portfolio Caps ---
        caps = config.RISK_CAPS.get("crypto_spot", {})
        self.per_pair_cap_pct = per_pair_cap_pct if per_pair_cap_pct is not None else caps.get("per_pair_cap_pct", 0.15)
        self.portfolio_cap_pct = portfolio_cap_pct if portfolio_cap_pct is not None else caps.get("portfolio_concurrent_pct", 0.30)

    # --- Public Methods ---

    def is_trade_allowed(self) -> bool:
        """Checks if a new trade is permitted based on master risk rules."""
        self._check_daily_reset()
        return self._is_not_in_cooldown() and self._is_within_daily_loss_limit()

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

    def calculate_position_size(self, symbol: str, entry_price: float, sl_price: float) -> Optional[tuple[float, float]]:
        """
        Calculates position size in units, applying all risk checks.
        Returns a tuple of (quantity, dollar_risk) or (None, None).
        """
        if not self.is_trade_allowed():
            return None, None

        # 1. Determine effective risk for this trade
        effective_risk_pct = self._get_effective_risk_pct()
        dollar_risk = self.equity * effective_risk_pct

        # 2. Calculate initial quantity based on risk
        price_risk_per_unit = abs(entry_price - sl_price)
        if price_risk_per_unit <= 1e-9:
            logger.error("Stop loss cannot be the same as entry price.")
            return None, None
        quantity = dollar_risk / price_risk_per_unit

        # 3. Apply portfolio and per-pair caps
        quantity = self._apply_caps(symbol, quantity, entry_price)

        # 4. Enforce minimum trade size
        if quantity * entry_price < config.MIN_TRADE_AMOUNT_USD:
            logger.warning(f"Sized position for {symbol} is below minimum trade value. Aborting.")
            return None, None

        return quantity, dollar_risk

    def register_position(self, symbol: str, quantity: float, entry_price: float, dollar_risk: float):
        """Adds an open position to the tracker."""
        if symbol in self.open_positions:
            logger.warning(f"Position for {symbol} already registered. Overwriting.")
        self.open_positions[symbol] = PositionRisk(
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            dollar_risk=dollar_risk,
            value_usd=quantity * entry_price
        )
        logger.info(f"Registered new position: {self.open_positions[symbol]}")

    def unregister_position(self, symbol: str):
        """Removes a closed position."""
        if symbol in self.open_positions:
            logger.info(f"Unregistering position for {symbol}.")
            del self.open_positions[symbol]

    # --- Private Helper Methods ---

    def _get_effective_risk_pct(self) -> float:
        """Gets the per-trade risk, adjusted for current drawdown."""
        drawdown = (self.peak_equity - self.equity) / self.peak_equity
        if drawdown > self.drawdown_reduction_threshold:
            logger.warning(
                f"Drawdown of {drawdown:.2%} exceeds threshold of {self.drawdown_reduction_threshold:.2%}. "
                f"Reducing trade risk by 50%."
            )
            return self.base_per_trade_risk_pct / 2
        return self.base_per_trade_risk_pct

    def _apply_caps(self, symbol: str, quantity: float, entry_price: float) -> float:
        """Reduces quantity if it violates portfolio or per-pair caps."""
        desired_usd = quantity * entry_price

        # Per-pair cap
        pair_cap_usd = self.equity * self.per_pair_cap_pct
        current_pair_exposure = self.open_positions.get(symbol, PositionRisk("",0,0,0,0)).value_usd
        allowed_for_pair = max(0, pair_cap_usd - current_pair_exposure)

        # Portfolio cap
        portfolio_cap_usd = self.equity * self.portfolio_cap_pct
        current_portfolio_exposure = sum(p.value_usd for p in self.open_positions.values())
        allowed_for_portfolio = max(0, portfolio_cap_usd - current_portfolio_exposure)

        allowed_usd = min(desired_usd, allowed_for_pair, allowed_for_portfolio)

        if allowed_usd < desired_usd:
            logger.warning(f"Position size for {symbol} reduced from {desired_usd:.2f} USD to {allowed_usd:.2f} USD due to exposure caps.")
            return allowed_usd / entry_price
        return quantity

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
            if self.in_cooldown_until.year == 9999:
                return False
            if datetime.now(timezone.utc) < self.in_cooldown_until:
                return False
            self.in_cooldown_until = None
        return True

    def _is_within_daily_loss_limit(self) -> bool:
        """Checks if the daily loss is within the defined limit."""
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
