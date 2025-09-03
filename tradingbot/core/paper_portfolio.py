from typing import Dict, Any
from .loggerconfig import get_logger
log = get_logger(__name__)

def apply_option_expiry_effects(position: Dict[str, Any], underlying_last: float) -> Dict[str, Any]:
    """Return delta changes to apply to portfolio on option expiry (paper).

    Caller is responsible for mutating paper state using this delta.
    """
    right = str(position.get('right','C')).upper()
    qty = float(position.get('qty',0))
    strike = float(position.get('strike',0))
    side = str(position.get('side','BUY')).upper()
    mult = float(position.get('multiplier', position.get('extra',{}).get('multiplier', 100)))
    intrinsic = max(0.0, underlying_last - strike) if right=='C' else max(0.0, strike - underlying_last)
    underlying_qty_delta = 0.0
    cash_delta = 0.0
    if intrinsic > 0:
        if side == 'BUY':
            underlying_qty_delta = qty if right=='C' else -qty
            cash_delta = -intrinsic * abs(qty) * mult
        else:  # SOLD option assigned
            underlying_qty_delta = -qty if right=='C' else qty
            cash_delta = intrinsic * abs(qty) * mult
    return {
        "underlying_qty_delta": underlying_qty_delta,
        "cash_delta": cash_delta,
        "intrinsic": intrinsic
    }