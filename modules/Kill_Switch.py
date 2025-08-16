import logging
from typing import Dict, Any, List
from collections import defaultdict

logger = logging.getLogger(__name__)

class KillSwitch:
    """
    Monitors portfolio health and triggers a circuit breaker for specific asset classes
    if critical risk limits are breached.
    """

    def __init__(self, config: Dict[str, Any], portfolio_manager: Any):
        """
        Initializes the KillSwitch.
        """
        self.config = config
        self.portfolio_manager = portfolio_manager

        self.daily_dd_limit = config.get("daily_drawdown_limit", 0.05)
        self.monthly_dd_limit = config.get("monthly_drawdown_limit", 0.15)
        self.max_slippage_events = config.get("max_slippage_events", 3)
        self.max_api_errors = config.get("max_api_errors", 10)

        self.active_kill_switches: Dict[str, str] = {}  # {asset_class: reason}

    def check_drawdowns(self):
        """
        Checks for daily and monthly drawdown breaches for each asset class.
        This method requires historical equity data from the portfolio manager.
        """
        asset_classes = self.portfolio_manager.get_all_asset_classes()
        for asset in asset_classes:
            if self.is_active(asset): continue

            history = self.portfolio_manager.get_equity_history(asset, days=30)
            if not history or len(history) < 2:
                continue # Not enough data to calculate drawdown

            current_equity = history[-1]

            # Daily Drawdown
            yesterday_equity = history[-2]
            daily_dd = (yesterday_equity - current_equity) / yesterday_equity if yesterday_equity > 0 else 0
            if daily_dd > self.daily_dd_limit:
                reason = f"Daily drawdown limit breached ({daily_dd:.2%} > {self.daily_dd_limit:.2%})"
                self.activate(asset, reason)
                continue # No need to check monthly if daily is hit

            # Monthly Drawdown
            peak_30d_equity = max(history)
            monthly_dd = (peak_30d_equity - current_equity) / peak_30d_equity if peak_30d_equity > 0 else 0
            if monthly_dd > self.monthly_dd_limit:
                reason = f"Monthly drawdown limit breached ({monthly_dd:.2%} > {self.monthly_dd_limit:.2%})"
                self.activate(asset, reason)

    def check_slippage(self, slippage_events: List[Dict[str, Any]]):
        """
        Checks for excessive slippage events per asset class.
        Expects a list of dicts, e.g., [{'asset_class': 'PERP', 'slippage_pct': 0.5}, ...]
        """
        counts = defaultdict(int)
        for event in slippage_events:
            asset_class = event.get('asset_class')
            if asset_class:
                counts[asset_class] += 1

        for asset, count in counts.items():
            if count >= self.max_slippage_events:
                reason = f"Excessive slippage: {count} events in last 24h"
                self.activate(asset, reason)

    def check_api_errors(self, api_error_counts: Dict[str, int]):
        """
        Checks for a spike in API errors from the exchange per asset class.
        """
        for asset, count in api_error_counts.items():
            if count > self.max_api_errors:
                reason = f"High API error count: {count}"
                self.activate(asset, reason)

    def activate(self, asset_class: str, reason: str):
        """
        Activates the kill switch for a given asset class.
        """
        if asset_class not in self.active_kill_switches:
            log_details = {'action': 'kill_switch_activated', 'asset_class': asset_class, 'reason': reason}
            logger.critical(f"KILL SWITCH ACTIVATED for {asset_class}. Reason: {reason}", extra=log_details)
            self.active_kill_switches[asset_class] = reason
            # TODO: Integrate with a notification manager to send an urgent alert

    def reset(self, asset_class: str):
        """
        Resets the kill switch for a given asset class.
        """
        if asset_class in self.active_kill_switches:
            logger.warning(f"Resetting kill switch for {asset_class}.")
            del self.active_kill_switches[asset_class]

    def is_active(self, asset_class: str) -> bool:
        """
        Checks if the kill switch is active for a given asset class,
        also checking for a global 'ALL' switch.
        """
        if "ALL" in self.active_kill_switches:
            return True
        return asset_class in self.active_kill_switches
