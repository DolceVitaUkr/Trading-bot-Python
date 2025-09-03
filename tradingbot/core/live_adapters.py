from typing import Any, Dict
from .loggerconfig import get_logger
from .rate_limits import get_bucket
from .brackets import for_bybit, for_ibkr

log = get_logger(__name__)

async def bybit_submit_wrapper(adapter, oc) -> Dict[str, Any]:
    """Call the Bybit adapter with a normalized OrderContext."""
    # Expect the adapter to have an async submit_order(symbol, side, qty, price, **extra)
    await get_bucket("bybit","trade").acquire(1)
    extra = dict(oc.extra or {})
    extra.update(for_bybit(extra))
    return await adapter.submit_order(oc.symbol, oc.side, oc.qty, oc.price, **extra)

async def ibkr_submit_wrapper(adapter, oc) -> Dict[str, Any]:
    """Call the IBKR adapter with a normalized OrderContext."""
    # Expect the adapter to have an async place_order(contract, action, quantity, price=None, **extra)
    # Here we assume oc.extra contains a constructed IBKR Contract when asset is forex/options
    await get_bucket("ibkr","trade").acquire(1)
    extra = dict(oc.extra or {})
    extra.update(for_ibkr(extra))
    return await adapter.place_order(oc.extra.get('contract'), oc.side.upper(), oc.qty, oc.price, **extra)