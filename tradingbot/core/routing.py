from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Literal, Optional
from pathlib import Path
from .exchange_conformance import clamp_order_if_needed
from .runtime_flags import get_flags
from .gates import can_go_live
from .idempotency import IdempotencyStore

from .loggerconfig import get_logger

log = get_logger(__name__)

Reject = Dict[str, Any]

def _reject(code: str, reason: str, details: Optional[Dict[str, Any]] = None) -> Reject:
    return {"rejected": True, "code": code, "reason": reason, "details": details or {}}

@dataclass(slots=True)
class OrderContext:
    strategy_id: str
    asset: str
    venue: str
    symbol: str
    side: Literal["BUY","SELL"]
    qty: float
    price: Optional[float]
    type: Literal["LIMIT","MARKET"]
    time_in_force: Literal["GTC","IOC","FOK"] = "GTC"
    extra: Dict[str, Any] = field(default_factory=dict)

class PaperRouter:
    def __init__(self, submit_fn: Callable[[OrderContext], Awaitable[Dict[str, Any]]]):
        self.submit_fn = submit_fn
    async def submit(self, oc: OrderContext) -> Dict[str, Any] | Reject:
        oc = clamp_order_if_needed(oc)
        return await self.submit_fn(oc)

class LiveRouter:
    def __init__(self, submitters: Dict[str, Callable[[OrderContext], Awaitable[Dict[str, Any]]]]):
        self.submitters = submitters
        self._idemp = IdempotencyStore(Path("tradingbot/state/idempotency.jsonl"))
    async def submit(self, venue: str, oc: OrderContext) -> Dict[str, Any] | Reject:
        flags = get_flags()
        if flags.get("kill_switch"):
            return _reject("KILL_SWITCH", "Live trading disabled")
        if flags.get("close_only") and oc.extra.get("open_new", True):
            return _reject("CLOSE_ONLY", "System is close-only")
        # Metrics gate (strategy approval)
        metrics = oc.extra.get("metrics")
        if not metrics:
            try:
                from .strategy_development_manager import get_strategy_metrics
                metrics = get_strategy_metrics(oc.strategy_id) or {"state": "DEVELOPING"}
            except Exception:
                metrics = {"state": "DEVELOPING"}
        if not can_go_live(metrics):
            return _reject("GATE_BLOCKED", f"State={metrics.get('state')}")
        oc = clamp_order_if_needed(oc)
        if venue not in self.submitters:
            return _reject("NO_SUBMITTER", f"No submitter for venue={venue}")
        cid = (oc.extra or {}).get("client_order_id")
        if cid and self._idemp.seen(cid):
            return _reject("DUPLICATE", "Duplicate client_order_id", {"client_order_id": cid})
        res = await self.submitters[venue](oc)
        if cid and not res.get("rejected"):
            self._idemp.record(cid)
        return res