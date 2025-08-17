import logging
from typing import Dict, Any, Optional

from modules.exchange import ExchangeAPI
from modules.Strategy_Manager import Decision

logger = logging.getLogger(__name__)

class TradeExecutor:
    """
    Executes trades in live or simulation mode, calculates net P&L,
    and returns a detailed trade receipt.
    """

    def __init__(
        self,
        sizing_policy: Dict[str, Any],
        simulation_mode: bool = True,
        notifier: Optional[Any] = None,
        risk_manager: Optional[Any] = None,
        exchange: Optional[Any] = None,
    ):
        """
        Initializes the TradeExecutor.
        """
        self.simulation_mode = simulation_mode
        self.sizing_policy = sizing_policy
        self.global_policy = sizing_policy.get("global", {})

        self.exchange = exchange or ExchangeAPI()
        self.risk_manager = risk_manager
        self.notifier = notifier

        # Simulation state
        self.total_trades = 0
        self.winning_trades = 0
        self.realized_pnl = 0.0

    def execute_order(
        self,
        decision: Decision,
        size_usd: float,
        leverage: float,
        price: float,
        order_type: str = "market",
    ) -> Dict[str, Any]:
        """
        Execute a trade based on a decision and a sizing proposal.
        """
        if not self.simulation_mode:
            logger.warning("Live trading execution is not fully implemented in this refactor.")
            return {"status": "skipped", "reason": "Live mode not implemented"}

        return self._handle_simulated_trade(decision, size_usd, leverage, price)

    def _handle_simulated_trade(
        self,
        decision: Decision,
        size_usd: float,
        leverage: float,
        entry_price: float,
    ) -> Dict[str, Any]:
        """
        Simulates a full trade cycle (entry and exit) and returns a receipt.
        This is a simplified synchronous simulation assuming the trade hits the stop loss.
        """
        symbol = decision.meta["symbol"]
        side = decision.signal

        # 1. Calculate trade parameters
        notional_size_usd = size_usd * leverage
        quantity = notional_size_usd / entry_price if entry_price > 0 else 0

        # 2. Simulate slippage on entry
        slippage_bps = self.global_policy.get("slippage_bps", 0)
        slippage_mult = 1 + (slippage_bps / 10000) if side == "buy" else 1 - (slippage_bps / 10000)
        effective_entry_price = entry_price * slippage_mult

        # 3. Simulate exit price (using SL from decision)
        # For simulation, we assume the trade hits the stop loss.
        exit_price = decision.sl

        # 4. Calculate Gross P&L
        pnl_per_unit = exit_price - effective_entry_price if side == "buy" else effective_entry_price - exit_price
        pnl_gross_usd = pnl_per_unit * quantity

        # 5. Calculate Fees
        fee_bps = self.global_policy.get("fee_bps", 0)
        fee_rate = fee_bps / 10000
        entry_fees_usd = (quantity * effective_entry_price) * fee_rate
        exit_fees_usd = (quantity * exit_price) * fee_rate
        total_fees_usd = entry_fees_usd + exit_fees_usd

        # 6. Calculate Net P&L
        pnl_net_usd = pnl_gross_usd - total_fees_usd

        self.total_trades += 1
        if pnl_net_usd > 0:
            self.winning_trades += 1
        self.realized_pnl += pnl_net_usd

        receipt = {
            "status": "closed",
            "symbol": symbol,
            "side": side,
            "size_usd": size_usd,
            "pnl_gross_usd": round(pnl_gross_usd, 4),
            "fees_usd": round(total_fees_usd, 4),
            "pnl_net_usd": round(pnl_net_usd, 4),
            "effective_leverage": leverage,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "quantity": quantity,
        }

        logger.info(f"SIMULATED TRADE: {receipt}")

        return receipt
