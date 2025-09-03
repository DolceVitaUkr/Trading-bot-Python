from typing import Dict, Any, Optional

def build_bracket(tp_price: Optional[float], sl_price: Optional[float], reduce_only: bool = True, tif: str = "GTC") -> Dict[str, Any]:
    return {
        "tp": tp_price,
        "sl": sl_price,
        "reduce_only": reduce_only,
        "tif": tif
    }

def for_bybit(oc_extra: Dict[str, Any]) -> Dict[str, Any]:
    if oc_extra is None: return {}
    tp = oc_extra.get("tp") or oc_extra.get("take_profit")
    sl = oc_extra.get("sl") or oc_extra.get("stop_loss")
    if tp is None and sl is None:
        return {}
    return {"bracket": build_bracket(tp, sl, reduce_only=True)}

def for_ibkr(oc_extra: Dict[str, Any]) -> Dict[str, Any]:
    tp = oc_extra.get("tp") or oc_extra.get("take_profit")
    sl = oc_extra.get("sl") or oc_extra.get("stop_loss")
    if tp is None and sl is None:
        return {}
    return {"bracket": build_bracket(tp, sl, reduce_only=True)}