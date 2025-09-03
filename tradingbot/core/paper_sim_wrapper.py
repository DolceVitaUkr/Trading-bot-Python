from typing import Dict, Any
from .loggerconfig import get_logger

log = get_logger(__name__)

async def simulate_with_paper_trader(paper_trader, oc) -> Dict[str, Any]:
    """Use the existing paper trader to simulate an order based on OrderContext."""
    # Expect paper_trader to expose an async submit_order(symbol, side, qty, price, **extra)
    return await paper_trader.submit_order(oc.symbol, oc.side, oc.qty, oc.price, **oc.extra)