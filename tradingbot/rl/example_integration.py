from __future__ annotations
from typing import Dict, Any, List, Optional
from pathlib import Path
from ..core.order_events import emit_event
from ..training.reward_shaping import compute_reward

class RLExampleIntegration:
    def __init__(self, base_dir: Path, notifier=None):
        self.base_dir = base_dir
        self.notifier = notifier
        self.episode_trades: List[Dict[str, Any]] = []
    async def on_trade_opened(self, trade: Dict[str, Any]) -> None:
        self.episode_trades.append({"open": trade, "close": None})
        emit_event(self.base_dir, "paper", trade.get("asset",""), {"event":"TRADE_OPEN", **trade})
    async def on_trade_closed(self, trade: Dict[str, Any]) -> None:
        for rec in reversed(self.episode_trades):
            if rec["open"] and not rec["close"] and rec["open"].get("id") == trade.get("id"):
                rec["close"] = trade
                break
        pnl = float(trade.get("realized_pnl", 0.0))
        dd = float(trade.get("max_dd_pct", 0.0))
        fees = float(trade.get("fees", 0.0))
        exp = float(trade.get("exposure", 1.0))
        reward = compute_reward(pnl, dd, fees, exp)
        emit_event(self.base_dir, "paper", trade.get("asset",""), {"event":"TRADE_CLOSE", **trade, "reward": reward})
        if self.notifier:
            await self.notifier.send_message_async(f"Closed {trade.get('symbol')} PnL={pnl:.2f} Rwd={reward:.4f}")