import random, asyncio
from typing import Dict, Any, Optional
from .loggerconfig import get_logger

log = get_logger(__name__)

class PaperExecutionModel:
    def __init__(self, slippage_bps: int = 8, partial_prob: float = 0.2, min_latency_ms: int = 100, max_latency_ms: int = 300):
        self.slippage_bps = slippage_bps
        self.partial_prob = partial_prob
        self.min_latency_ms = min_latency_ms
        self.max_latency_ms = max_latency_ms

    async def execute(self, symbol: str, side: str, qty: float, price: Optional[float], **extra) -> Dict[str, Any]:
        delay = random.randint(self.min_latency_ms, self.max_latency_ms) / 1000.0
        await asyncio.sleep(delay)
        p = float(price) if price is not None else float(extra.get('last', 0))
        if p <= 0:
            raise ValueError("PaperExecutionModel requires a positive price or last quote")
        slip = p * (self.slippage_bps / 10000.0)
        filled_price = p + slip if side.lower() == "buy" else p - slip
        filled_qty = qty
        status = "FILLED"
        if random.random() < self.partial_prob:
            filled_qty = max(qty * random.uniform(0.4, 0.9), 0.0)
            status = "PARTIAL"
        return {
            "symbol": symbol,
            "side": side,
            "avg_price": round(filled_price, 8),
            "filled_qty": round(filled_qty, 8),
            "status": status,
            "latency_s": delay,
            "slippage_bps": self.slippage_bps
        }