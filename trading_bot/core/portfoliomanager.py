import logging
from typing import Dict, Optional, List, Any

from trading_bot.brokers.Exchange_Bybit import Exchange_Bybit
from trading_bot.brokers.Exchange_IBKR import Exchange_IBKR
from trading_bot.core.schemas import PortfolioState

logger = logging.getLogger(__name__)


class PortfolioManager:
    """
    Aggregates portfolio information from multiple broker adapters.
    Provides a unified view of the entire portfolio state and equity curves.
    """

    def __init__(self, bybit_adapter: Optional[Exchange_Bybit] = None, ibkr_adapter: Optional[Exchange_IBKR] = None):
        """
        Initializes the Portfolio_Manager with broker adapters.
        """
        self.bybit_adapter = bybit_adapter
        self.ibkr_adapter = ibkr_adapter
        # Store historical equity points for the UI graph
        self.equity_history: Dict[str, List[Dict[str, Any]]] = {
            "bybit": [],
            "ibkr": [],
            "total": []
        }

    async def get_total_portfolio_state(self) -> PortfolioState:
        """
        Fetches portfolio states from all brokers and aggregates them.
        """
        total_balance = 0
        total_available = 0
        total_margin = 0
        total_unrealized_pnl = 0
        all_positions = []

        if self.bybit_adapter:
            bybit_state = await self.bybit_adapter.get_wallet_balance()
            if bybit_state:
                total_balance += bybit_state.total_balance_usd
                total_available += bybit_state.available_balance_usd
                total_margin += bybit_state.margin_used
                total_unrealized_pnl += bybit_state.unrealized_pnl
                all_positions.extend(await self.bybit_adapter.get_positions("linear")) # Example category

        if self.ibkr_adapter:
            ibkr_state = await self.ibkr_adapter.get_wallet_balance()
            if ibkr_state:
                total_balance += ibkr_state.total_balance_usd
                total_available += ibkr_state.available_balance_usd
                total_margin += ibkr_state.margin_used
                total_unrealized_pnl += ibkr_state.unrealized_pnl
                all_positions.extend(ibkr_state.positions)

        # In a real scenario, you'd handle realized PnL tracking more robustly
        return PortfolioState(
            total_balance_usd=total_balance,
            available_balance_usd=total_available,
            margin_used=total_margin,
            unrealized_pnl=total_unrealized_pnl,
            realized_pnl=0, # This would be tracked across trades
            positions=all_positions
        )

    def record_equity_snapshot(self, timestamp: float, portfolio_state: PortfolioState):
        """Records a snapshot of the total equity for the UI graph."""
        self.equity_history["total"].append({
            "timestamp": timestamp,
            "equity": portfolio_state.total_balance_usd
        })

    def get_equity_curve(self, account: str = "total") -> List[Dict[str, Any]]:
        """Returns the equity history for a given account or the total portfolio."""
        return self.equity_history.get(account, [])
