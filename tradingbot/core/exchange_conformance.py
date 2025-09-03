from dataclasses import replace
from typing import Dict, Any
from .loggerconfig import get_logger
from .contract_catalog import ContractCatalog
log = get_logger(__name__)

_catalog_instance = None
def set_catalog_instance(c):
    global _catalog_instance
    _catalog_instance = c

def _snap(value: float, step: float) -> float:
    if step is None or step == 0: return value
    return round((value / step)) * step

def clamp_order_if_needed(oc):
    """
    Best-effort clamp for qty/price against known increments.
    Looks for tick/step in oc.extra or contract metadata.
    """
    meta = None
    if _catalog_instance and oc.extra and 'contract_id' in oc.extra:
        meta = _catalog_instance.find(oc.extra['contract_id'])
    extra = dict(oc.extra or {})
    if meta:
        extra.setdefault('tick_size', meta.get('tick_size'))
        extra.setdefault('step_size', meta.get('lot_size'))
        extra.setdefault('min_notional', meta.get('min_notional'))
        # For options, premium-based sizing often applies; multiplier informs notional math elsewhere.
        if 'multiplier' in meta:
            extra.setdefault('multiplier', meta.get('multiplier'))
        oc = replace(oc, extra=extra)
    tick = extra.get('tick_size') or extra.get('tickSize')
    step = extra.get('step_size') or extra.get('stepSize')
    min_notional = extra.get('min_notional') or extra.get('minNotional')
    price = oc.price
    qty = oc.qty
    changed = False
    if price is not None and tick:
        new_price = _snap(price, float(tick))
        if new_price != price:
            price = new_price; changed = True
    if step:
        new_qty = _snap(qty, float(step))
        if new_qty != qty:
            qty = new_qty; changed = True
    if min_notional and price is not None:
        if qty * price < float(min_notional):
            # Bump qty up to minimum
            required = float(min_notional) / max(price, 1e-12)
            qty = required; changed = True
    if changed:
        log.info(f"CLAMP {oc.symbol}: qty {oc.qty}->{qty}, price {oc.price}->{price}")
        oc = replace(oc, qty=qty, price=price)
    return oc