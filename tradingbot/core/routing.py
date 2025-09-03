from typing import Callable, Dict, Any
from dataclasses import dataclass
from .loggerconfig import get_logger
from .exchange_conformance import clamp_order_if_needed
from .runtime_flags import get_flags
from .gates import can_go_live
from .idempotency import IdempotencyStore
from pathlib import Path

log = get_logger(__name__)

@dataclass
class OrderContext:
    asset: str
    symbol: str
    side: str
    qty: float
    price: float | None
    extra: Dict[str, Any]

class PaperRouter:
    """
    Paper mode: uses mainnet quotes for read-only; executes orders in local simulator.
    The simulator function must be injected; this keeps Bybit/IBKR logic out of here.
    """
    def __init__(self, simulate_order: Callable[[OrderContext], Dict[str, Any]]):
        self.simulate_order = simulate_order

    async def submit(self, oc: OrderContext) -> Dict[str, Any]:
        oc = clamp_order_if_needed(oc)
        result = await self.simulate_order(oc)
        return result

class LiveRouter:
    """
    Live mode: forwards orders to the provided broker submitters.
    You inject per-venue submit functions, e.g., {'bybit': bybit_submit, 'ibkr': ibkr_submit}
    """
    def __init__(self, submitters: Dict[str, Callable[[OrderContext], Any]]):
        self.submitters = submitters
        self._idemp = IdempotencyStore(Path("tradingbot/state/idempotency.jsonl"))

    async def submit(self, venue: str, oc: OrderContext) -> Any:
        flags = get_flags()
        if flags.get("kill_switch") or flags.get("close_only"):
            raise RuntimeError("Live routing disabled (kill_switch/close_only)")
        if venue not in self.submitters:
            raise ValueError(f"No live submitter configured for venue: {venue}")
        oc = clamp_order_if_needed(oc)
        cid = oc.extra.get("client_order_id") if oc.extra else None
        if cid and self._idemp.seen(cid):
            raise RuntimeError("Duplicate client_order_id detected; live submit aborted")
        return await self.submitters[venue](oc)