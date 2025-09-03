import datetime as dt
from typing import Optional, Dict, Any
from pathlib import Path
from .loggerconfig import get_logger
from .order_events import emit_event
from .paper_portfolio import apply_option_expiry_effects
from .fx_converter import convert_value

log = get_logger(__name__)

class OptionsLifecycle:
    def __init__(self, catalog, base_dir: Path):
        self.catalog = catalog
        self.base_dir = base_dir

    def _days_to_expiry(self, contract_id: str) -> Optional[int]:
        meta = self.catalog.find(contract_id)
        if not meta or not meta.get('expiry'):
            return None
        expiry = dt.datetime.fromisoformat(meta['expiry'].replace('Z','+00:00'))
        now = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
        return (expiry - now).days

    def allow_open(self, contract_id: str, min_dte: int) -> (bool, str):
        dte = self._days_to_expiry(contract_id)
        if dte is None:
            return True, ""
        if dte < min_dte:
            return False, f"DTE {dte} < min {min_dte}"
        return True, ""

    def handle_expiry(self, asset: str, position: Dict[str, Any], underlying_last: float, itm_threshold: float = 0.01):
        """On expiry (or just after), emit EXPIRY + EXERCISE/ASSIGNMENT when applicable (paper).

        position requires: {'contract_id','underlying','strike','right','qty','side','avg_price'}
        """
        cid = position.get('contract_id')
        dte = self._days_to_expiry(cid)
        if dte is None or dte > 0:
            return
        strike = float(position.get('strike', 0))
        right = str(position.get('right', 'C')).upper()
        qty = float(position.get('qty', 0))
        side = str(position.get('side', 'BUY')).upper()
        intrinsic = 0.0
        if right == 'C':
            intrinsic = max(0.0, underlying_last - strike)
        else:
            intrinsic = max(0.0, strike - underlying_last)
        intrinsic_pct = intrinsic / max(strike, 1e-9)
        # Expiry event
        emit_event(self.base_dir, 'paper', asset, {
            "event": "EXPIRY", "contract_id": cid, "strike": strike, "right": right,
            "underlying_last": underlying_last, "intrinsic": intrinsic
        })
        if intrinsic_pct >= itm_threshold and intrinsic > 0:
            # ITM: exercise/assignment simulation
            if side == 'BUY':
                emit_event(self.base_dir, 'paper', asset, {
                    "event": "EXERCISE", "contract_id": cid, "strike": strike, "right": right,
                    "qty": qty, "underlying_qty_delta": qty if right=='C' else -qty
                })
            else:
                emit_event(self.base_dir, 'paper', asset, {
                    "event": "ASSIGNMENT", "contract_id": cid, "strike": strike, "right": right,
                    "qty": qty, "underlying_qty_delta": -qty if right=='C' else qty
                })
            # Portfolio delta (paper)
            delta = apply_option_expiry_effects(position, underlying_last)
            emit_event(self.base_dir, 'paper', asset, {
                "event": "PORTFOLIO_ADJUSTMENT", "contract_id": cid, **delta
            })