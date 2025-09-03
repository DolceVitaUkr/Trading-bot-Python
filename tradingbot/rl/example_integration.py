from __future__ import annotations

# Unified close-time reward integration that delegates to core.reward_system
from typing import Dict, Any, Optional
from pathlib import Path
from ..core.order_events import emit_event
from ..core.reward_system import RewardSystem, TradeContext

class RLExampleIntegration:
    def __init__(self, base_dir: Path, notifier=None, asset_type: Optional[str] = None):
        self.base_dir = base_dir
        self.notifier = notifier
        self.asset_type = asset_type
        self._reward = RewardSystem()

    async def on_trade_opened(self, trade: Dict[str, Any]) -> None:
        emit_event(self.base_dir, "paper", trade.get("asset",""), {"event":"TRADE_OPEN", **trade})

    def on_trade_closed(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        leverage: float,
        fees_paid: float,
        slippage: float,
        holding_time_seconds: float,
        current_equity: float,
        open_exposure: float,
    ) -> float:
        ctx = TradeContext(
            symbol=symbol, side=side,
            entry_price=float(entry_price), exit_price=float(exit_price),
            quantity=float(quantity), leverage=float(leverage),
            fees_paid=float(fees_paid), holding_time_seconds=float(holding_time_seconds),
            current_equity=float(current_equity), open_exposure=float(open_exposure),
            asset_type=self.asset_type
        )
        reward = self._reward.compute_reward(ctx)
        emit_event(self.base_dir, "paper", "", {
            "event":"TRADE_CLOSE", "symbol": symbol, "reward": reward,
            "entry_price": entry_price, "exit_price": exit_price, "qty": quantity
        })
        if self.notifier:
            try:
                # Async-friendly: notify if available
                msg = f"Closed {symbol}  Reward={reward:+.2f}"
                # If notifier has async send, ignore here to keep sync signature
            except Exception:
                pass
        return float(reward)

# Keep alias so paper_trader imports remain valid
TradingBotRewardIntegration = RLExampleIntegration